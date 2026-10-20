"""Recall backtest + preventable-window safety correlation.  [AI-LAYER]

Mission check for SkewAi: if a fault had been flagged early, how many
crash-, fire- and injury-linked complaints were filed between that flag and
the official recall? This script answers strictly from public NHTSA data:

1. Runs the EXISTING per-recall lead-time backtest
   (``src.backtest.recall_lead.recall_lead_report``, received-date clock,
   each recall scored only on complaints NHTSA had already received). The
   per-stamp ``run_backtest`` engine pass is skipped (``run_backtests=False``);
   it does not affect hit/miss or lead time.
2. Counts flag-level precision: every flagged slice-week in the full series
   is "confirmed" when a later recall of the same make + component exists
   (the module's own ``_later_recall`` rule), else a false alarm.
3. Streams the raw ODI ``FLAT_CMPL.zip`` and, for every hit, counts the
   complaints of the same make + component received in the preventable window
   ``[Monday of first flagged week, recall report date)`` whose official
   CRASH / FIRE fields are "Y" or INJURED / DEATHS are > 0.

These are correlations, not proof that an earlier flag would have prevented
any crash: complaints are self-reported, unverified, may describe incidents
that occurred earlier, and a flag is not a repair.

    DOMAIN_DB_PATH=/tmp/skew-nhtsa-full python -m scripts.ai_safety_backtest \\
        --zip /workspace/nhtsa_data/FLAT_CMPL.zip --makes HONDA,TOYOTA,FORD,HYUNDAI,KIA \\
        --year-min 2015 --year-max 2024
"""

from __future__ import annotations

import argparse
import io
import json
import statistics
import sys
import time
import zipfile
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import REPO_ROOT, is_default_pilot_domain_path  # noqa: E402

OUT_MD = REPO_ROOT / "reports" / "nhtsa" / "backtest_safety_2015_2024.md"
OUT_JSON = REPO_ROOT / "reports" / "nhtsa" / "backtest_safety.json"

# Positions in the official FLAT_CMPL layout (scripts/ingest_nhtsa._FLAT_FIELDS).
_I = {"ODINO": 1, "MAKETXT": 3, "MODELTXT": 4, "YEARTXT": 5, "CRASH": 6, "FIRE": 8,
      "INJURED": 9, "DEATHS": 10, "COMPDESC": 11, "DATEA": 15}


def _int(v: str) -> int:
    try:
        return int(str(v).strip() or 0)
    except ValueError:
        return 0


def _date(v: str) -> date | None:
    v = str(v or "").strip()
    try:
        return datetime.strptime(v[:8], "%Y%m%d").date()
    except ValueError:
        return None


def iter_flat(zip_path: Path) -> Iterable[list[str]]:
    with zipfile.ZipFile(zip_path) as zf:
        name = [n for n in zf.namelist() if not n.endswith("/")][0]
        with zf.open(name) as raw:
            for line in io.TextIOWrapper(raw, encoding="latin-1", errors="replace"):
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) > _I["DATEA"]:
                    yield parts


def safety_rows(lines: Iterable[list[str]], makes: set[str], year_min: int | None, year_max: int | None,
                slices: set[tuple[str, str]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """(component, make) -> complaint rows with the official safety fields."""
    from src.domains.nhtsa_components import normalize_component

    out: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for p in lines:
        make = p[_I["MAKETXT"]].strip().upper()
        if makes and make not in makes:
            continue
        yr = _int(p[_I["YEARTXT"]])
        if (year_min and yr < year_min) or (year_max and yr > year_max):
            continue
        comp = normalize_component(p[_I["COMPDESC"]]).strip().upper()
        if (comp, make) not in slices:
            continue
        d = _date(p[_I["DATEA"]])
        if d is None:
            continue
        out[(comp, make)].append({
            "odino": p[_I["ODINO"]].strip(), "datea": d,
            "crash": p[_I["CRASH"]].strip().upper() == "Y", "fire": p[_I["FIRE"]].strip().upper() == "Y",
            "injured": _int(p[_I["INJURED"]]), "deaths": _int(p[_I["DEATHS"]]),
        })
    return out


def _monday(iso_week: str) -> date | None:
    try:
        y, w = str(iso_week).split("-W")
        return date.fromisocalendar(int(y), int(w), 1)
    except (ValueError, TypeError):
        return None


def window_counts(rows: list[dict[str, Any]], start: date, end: date) -> tuple[dict[str, int], set[str]]:
    """Distinct ODI complaints received in [start, end) and their safety flags."""
    seen: dict[str, dict[str, Any]] = {}
    for r in rows:
        if start <= r["datea"] < end:
            prev = seen.get(r["odino"])
            if prev is None:
                seen[r["odino"]] = dict(r)
            else:  # an ODINO can repeat across lines; OR the flags, max the counts
                prev["crash"] |= r["crash"]
                prev["fire"] |= r["fire"]
                prev["injured"] = max(prev["injured"], r["injured"])
                prev["deaths"] = max(prev["deaths"], r["deaths"])
    vals = list(seen.values())
    return {
        "complaints": len(vals),
        "crash": sum(1 for r in vals if r["crash"]),
        "fire": sum(1 for r in vals if r["fire"]),
        "injury_complaints": sum(1 for r in vals if r["injured"] > 0),
        "injured_persons_reported": sum(r["injured"] for r in vals),
        "death_complaints": sum(1 for r in vals if r["deaths"] > 0),
        "deaths_reported": sum(r["deaths"] for r in vals),
        "crash_fire_or_injury": sum(1 for r in vals if r["crash"] or r["fire"] or r["injured"] > 0 or r["deaths"] > 0),
    }, set(seen)


def flag_precision(pack_id: str, model_grain: bool = False) -> dict[str, Any]:
    from src.backtest.recall_lead import _later_recall, _load, _score_received

    advisories, records = _load(pack_id)
    full = _score_received(records, pack_id=pack_id, as_of=None, model_grain=model_grain)
    flags = [r for r in full if r.get("is_anomaly")]
    confirmed = [r for r in flags if _later_recall(r, advisories, model_grain=model_grain)]
    # episodes: consecutive flagged weeks of one slice count once
    by_slice: dict[tuple, list[date]] = defaultdict(list)
    conf_keys = {(r.get("iso_week"), r.get("category"), r.get("entity_2")) for r in confirmed}
    conf_slice_weeks: dict[tuple, set[date]] = defaultdict(set)
    for r in flags:
        m = _monday(str(r.get("iso_week")))
        if m is None:
            continue
        key = (r.get("category"), r.get("entity_2"))
        by_slice[key].append(m)
        if (r.get("iso_week"), r.get("category"), r.get("entity_2")) in conf_keys:
            conf_slice_weeks[key].add(m)
    episodes = conf_eps = 0
    for key, weeks in by_slice.items():
        weeks.sort()
        prev = None
        ep_conf = False
        for w in weeks + [None]:
            if w is None or (prev is not None and (w - prev).days > 7):
                episodes += 1
                conf_eps += int(ep_conf)
                ep_conf = False
                if w is None:
                    break
            ep_conf = ep_conf or (w in conf_slice_weeks[key])
            prev = w
    return {
        "records": len(records), "advisories": len(advisories),
        "flagged_slice_weeks": len(flags), "confirmed_slice_weeks": len(confirmed),
        "false_alarm_slice_weeks": len(flags) - len(confirmed),
        "precision_slice_weeks": round(len(confirmed) / len(flags), 3) if flags else None,
        "flag_episodes": episodes, "confirmed_episodes": conf_eps,
        "precision_episodes": round(conf_eps / episodes, 3) if episodes else None,
    }


def summarise(report: dict[str, Any], safety: dict[tuple[str, str], list[dict[str, Any]]]) -> dict[str, Any]:
    hits, misses = report["hits"], report["misses"]
    leads = [int(h["lead_weeks"]) for h in hits]
    per_hit = []
    global_ids: set[str] = set()
    global_rows: dict[str, dict[str, Any]] = {}
    for h in hits:
        start = _monday(str(h["first_flagged_week"]))
        end = date.fromisoformat(str(h["issued_at"]))
        key = (str(h["scope_category"] or "").upper(), str(h["scope_entity_2"] or "").upper())
        rows = safety.get(key, [])
        counts, ids = window_counts(rows, start, end) if start else ({}, set())
        for r in rows:
            if r["odino"] in ids:
                global_rows.setdefault(f"{key}:{r['odino']}", r)
        global_ids |= {f"{key}:{i}" for i in ids}
        per_hit.append({**{k: h[k] for k in ("advisory_id", "issued_at", "scope_category", "scope_entity_2",
                                               "scope_entity_3", "first_flagged_week", "lead_weeks")},
                        "window": counts})
    uniq = list(global_rows.values())
    totals = {
        "complaints": len(uniq),
        "crash": sum(1 for r in uniq if r["crash"]),
        "fire": sum(1 for r in uniq if r["fire"]),
        "injury_complaints": sum(1 for r in uniq if r["injured"] > 0),
        "injured_persons_reported": sum(r["injured"] for r in uniq),
        "death_complaints": sum(1 for r in uniq if r["deaths"] > 0),
        "deaths_reported": sum(r["deaths"] for r in uniq),
        "crash_fire_or_injury": sum(1 for r in uniq if r["crash"] or r["fire"] or r["injured"] > 0 or r["deaths"] > 0),
    }
    n = len(hits) + len(misses)
    hero = max(per_hit, key=lambda r: (r["window"].get("crash_fire_or_injury", 0), r["lead_weeks"]), default=None)
    return {
        "recalls_scored": n, "hits": len(hits), "misses": len(misses),
        "recall_rate": round(len(hits) / n, 3) if n else None,
        "lead_weeks_median": statistics.median(leads) if leads else None,
        "lead_weeks_mean": round(statistics.mean(leads), 1) if leads else None,
        "lead_weeks_p25": statistics.quantiles(leads, n=4)[0] if len(leads) >= 4 else None,
        "lead_weeks_p75": statistics.quantiles(leads, n=4)[2] if len(leads) >= 4 else None,
        "unconfirmed_flags": report["unconfirmed_count"],
        "preventable_window_totals_dedup": totals,
        "per_hit": per_hit, "hero": hero,
    }


def render(meta: dict[str, Any], s: dict[str, Any], prec: dict[str, Any]) -> str:
    t = s["preventable_window_totals_dedup"]
    L = [
        "# Recall backtest and preventable-window safety correlation (NHTSA 2015-2024)",
        "",
        "Generated by `scripts/ai_safety_backtest.py` from public NHTSA ODI data. Every number below was",
        "computed by that script on this run; nothing is estimated or extrapolated.",
        "",
        "## Scope",
        "",
        f"- Complaints: FLAT_CMPL, makes {meta['makes']}, model years {meta['year_min']}-{meta['year_max']},",
        f"  {prec['records']} complaint records in the warehouse (received-date clock).",
        f"- Recalls: FLAT_RCL_POST_2010 vehicle campaigns for the same makes/years, {prec['advisories']} advisory rows,",
        f"  {s['recalls_scored']} scored make+component recalls.",
        f"- Grain: make + component (`model_grain={meta['model_grain']}`). Run time {meta['elapsed_s']} s on the build box.",
        "",
        "## Early warning vs official recall",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Recalls flagged early (hit, >= 1 week lead) | {s['hits']} |",
        f"| Recalls missed | {s['misses']} |",
        f"| Recall (hits / scored recalls) | {s['recall_rate']} |",
        f"| Median lead time, weeks (hits) | {s['lead_weeks_median']} |",
        f"| Mean lead time, weeks (hits) | {s['lead_weeks_mean']} |",
        f"| Lead time IQR, weeks | {s['lead_weeks_p25']} - {s['lead_weeks_p75']} |",
        f"| Flagged slice-weeks (full series) | {prec['flagged_slice_weeks']} |",
        f"| ...confirmed by a later recall of same make+component | {prec['confirmed_slice_weeks']} |",
        f"| ...false alarms (no later recall) | {prec['false_alarm_slice_weeks']} |",
        f"| Precision, slice-weeks | {prec['precision_slice_weeks']} |",
        f"| Flag episodes (consecutive weeks merged) | {prec['flag_episodes']} |",
        f"| Precision, episodes | {prec['precision_episodes']} |",
        "",
        "## Preventable window (correlation, not proof)",
        "",
        "For every hit: distinct ODI complaints of the same make + component received between the Monday of the",
        "first flagged week and the recall report date, using NHTSA's own CRASH / FIRE / INJURED / DEATHS fields.",
        "Deduplicated across overlapping recalls of the same slice.",
        "",
        "| Field | Complaints in preventable windows |",
        "| --- | --- |",
        f"| All complaints | {t['complaints']} |",
        f"| CRASH = Y | {t['crash']} |",
        f"| FIRE = Y | {t['fire']} |",
        f"| INJURED > 0 | {t['injury_complaints']} (persons reported: {t['injured_persons_reported']}) |",
        f"| DEATHS > 0 | {t['death_complaints']} (deaths reported: {t['deaths_reported']}) |",
        f"| Crash, fire, injury or death | {t['crash_fire_or_injury']} |",
        "",
    ]
    if s["hero"]:
        h = s["hero"]
        L += [
            "### Largest single window",
            "",
            f"{h['scope_entity_2']} {h['scope_category']}: first flagged {h['first_flagged_week']}, recall {h['advisory_id']}",
            f"report date {h['issued_at']} ({h['lead_weeks']} weeks later). In between, NHTSA received",
            f"{h['window'].get('complaints')} complaints on that slice, {h['window'].get('crash')} marked CRASH,",
            f"{h['window'].get('fire')} FIRE, {h['window'].get('injury_complaints')} with injuries.",
            "",
        ]
    L += [
        "## Limitations (read before quoting any number)",
        "",
        "- Correlation only. A complaint in the window is not a crash an earlier flag would have prevented; the",
        "  incident may predate the flag, the recall may not cover that failure, and a flag is not a repair.",
        "- Complaints are self-reported and unverified; CRASH/FIRE/INJURED are owner-entered fields.",
        "- Matching is make + component (not model / model year), so a slice can mix unrelated defects and one",
        "  recall can match a flag driven by a different fault. Precision and recall are at that grain.",
        "- 'False alarm' means no later recall of that make+component in this data window; it can be a real",
        "  defect that was never recalled, or one recalled after the data window ends (right-censoring).",
        "- Recall dates are RCDATE (Part 573 report received). Manufacturers often know earlier (TSBs, warranty).",
        "- Only five makes and model years 2015-2024 were ingested; other makes may behave differently.",
        "- The detector is the existing deterministic weekly z-score scan; the optional Meta AI layer is not",
        "  involved in these numbers.",
        "",
        "## Per-recall hits",
        "",
        "| Advisory | Report date | Slice | First flag | Lead wk | Complaints | Crash | Fire | Injury |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for h in sorted(s["per_hit"], key=lambda r: -r["window"].get("crash_fire_or_injury", 0)):
        w = h["window"]
        L.append(f"| {h['advisory_id']} | {h['issued_at']} | {h['scope_entity_2']} {h['scope_category']} | "
                 f"{h['first_flagged_week']} | {h['lead_weeks']} | {w.get('complaints')} | {w.get('crash')} | "
                 f"{w.get('fire')} | {w.get('injury_complaints')} |")
    L += ["", "Reproduce:", "", "```bash", meta["command"], "```", ""]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--pack", default="automotive_nhtsa")
    p.add_argument("--zip", required=True, help="local FLAT_CMPL.zip")
    p.add_argument("--makes", default="HONDA,TOYOTA,FORD,HYUNDAI,KIA")
    p.add_argument("--year-min", type=int, default=2015)
    p.add_argument("--year-max", type=int, default=2024)
    p.add_argument("--model-grain", action="store_true")
    p.add_argument("--out-md", default=str(OUT_MD))
    p.add_argument("--out-json", default=str(OUT_JSON))
    args = p.parse_args(argv)
    if is_default_pilot_domain_path():
        print("refusing the pilot data/domains directory; set DOMAIN_DB_PATH to a side directory", file=sys.stderr)
        return 2
    from src.backtest.recall_lead import recall_lead_report

    t0 = time.time()
    report = recall_lead_report(args.pack, model_grain=args.model_grain, run_backtests=False)
    print(f"lead report: hits={report['hit_count']} misses={report['miss_count']} "
          f"unconfirmed={report['unconfirmed_count']} ({time.time() - t0:.0f}s)", flush=True)
    prec = flag_precision(args.pack, model_grain=args.model_grain)
    makes = {m.strip().upper() for m in args.makes.split(",") if m.strip()}
    slices = {(str(h["scope_category"] or "").upper(), str(h["scope_entity_2"] or "").upper()) for h in report["hits"]}
    safety = safety_rows(iter_flat(Path(args.zip)), makes, args.year_min, args.year_max, slices)
    s = summarise(report, safety)
    meta = {
        "makes": ",".join(sorted(makes)), "year_min": args.year_min, "year_max": args.year_max,
        "model_grain": args.model_grain, "elapsed_s": int(time.time() - t0),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": (f"DOMAIN_DB_PATH=<side dir> python -m scripts.ai_safety_backtest --zip FLAT_CMPL.zip "
                    f"--makes {args.makes} --year-min {args.year_min} --year-max {args.year_max}"),
    }
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(render(meta, s, prec), encoding="utf-8")
    t = s["preventable_window_totals_dedup"]
    headline = None
    if s["hits"]:
        headline = {
            "recalls_scored": s["recalls_scored"], "hits": s["hits"], "misses": s["misses"],
            "recall_rate": s["recall_rate"], "lead_weeks_median": s["lead_weeks_median"],
            "lead_weeks_mean": s["lead_weeks_mean"], "precision_episodes": prec["precision_episodes"],
            "preventable_window_crash": t["crash"], "preventable_window_fire": t["fire"],
            "preventable_window_injury_complaints": t["injury_complaints"],
            "preventable_window_crash_fire_or_injury": t["crash_fire_or_injury"],
            "hero": s["hero"], "caveat": "correlation from self-reported NHTSA complaints; not proof of prevention",
            "source": str(Path(args.out_md).relative_to(REPO_ROOT)) if Path(args.out_md).resolve().is_relative_to(REPO_ROOT) else args.out_md,
        }
    Path(args.out_json).write_text(json.dumps({"meta": meta, "precision": prec, "headline": headline,
                                               "summary": {k: v for k, v in s.items() if k != "per_hit"},
                                               "per_hit": s["per_hit"]}, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"headline": headline, "precision": prec}, default=str, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
