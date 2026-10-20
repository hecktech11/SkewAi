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
OUT_LEAD = REPO_ROOT / "reports" / "nhtsa" / "recall_lead_time_2015_2024.md"

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


_G: dict[str, Any] = {}


def _rows_for_stamps(stamps: list[Any]) -> dict[int, dict[str, Any]]:
    """Score each as-of stamp once and build the per-advisory rows for it."""
    from src.backtest.engine import _weeks_between
    from src.backtest.recall_lead import _first_flag, _score_received
    from src.ml_runtime.anomalies import _week_monday

    advisories, records, by_stamp = _G["advisories"], _G["records"], _G["by_stamp"]
    pack_id, model_grain, progress = _G["pack_id"], _G["model_grain"], _G["progress"]
    rows: dict[int, dict[str, Any]] = {}
    recent: dict[int, dict[str, Any] | None] = {}
    t0 = time.time()
    for n, issued in enumerate(stamps, 1):
        series = [r for r in _score_received(records, pack_id=pack_id, as_of=issued, model_grain=model_grain)
                  if r.get("is_anomaly")]
        for i in by_stamp[issued]:
            advisory = advisories[i]
            recent[i] = _recent_episode(series, advisory, issued, model_grain=model_grain)
            flag = _first_flag(series, advisory, model_grain=model_grain)
            if flag is None:
                lead, first_week, count, baseline, z_score = 0, None, None, None, None
            else:
                start = _week_monday(str(flag["iso_week"]))
                lead = _weeks_between(start, issued) if start is not None else 0
                first_week = flag["iso_week"]
                count = flag.get("record_count")
                baseline = flag.get("baseline_mean")
                z_score = flag.get("z_score")
            rows[i] = {
                "advisory_id": advisory["advisory_id"],
                "issued_at": issued.date().isoformat(),
                "scope_category": advisory.get("scope_category"),
                "scope_entity_2": advisory.get("scope_entity_2"),
                "scope_entity_3": advisory.get("scope_entity_3"),
                "first_flagged_week": first_week,
                "lead_weeks": lead,
                "outcome": "hit" if lead > 0 else "miss",
                "record_count": count,
                "baseline_mean": baseline,
                "z_score": z_score,
                "backtest_matched": False,
                "backtest_best_lead": 0,
            }
        if progress and n % 25 == 0:
            print(f"  worker scored {n}/{len(stamps)} as-of dates ({time.time() - t0:.0f}s)", flush=True)
    return {"rows": rows, "recent": recent}


RECENT_WINDOW_WEEKS = 52


def _recent_episode(series: list[dict[str, Any]], advisory: dict[str, Any], issued: Any, *,
                    model_grain: bool) -> dict[str, Any] | None:
    """The flag episode an engineer would actually act on: the LATEST run of
    consecutive flagged weeks for the recall's slice, if its last week is
    within RECENT_WINDOW_WEEKS before the recall. Returns the episode start
    and lead, or None. Uses the same as-of series as the first-flag metric
    (nothing after the recall date is seen)."""
    from src.backtest.engine import _weeks_between
    from src.backtest.recall_lead import _slice_match
    from src.ml_runtime.anomalies import _week_monday

    weeks: dict[Any, str] = {}
    for row in series:
        if not _slice_match(row, advisory, model_grain=model_grain):
            continue
        m = _week_monday(str(row.get("iso_week") or ""))
        if m is not None and m < issued:
            weeks[m] = str(row["iso_week"])
    if not weeks:
        return None
    ordered = sorted(weeks)
    last = ordered[-1]
    if _weeks_between(last, issued) > RECENT_WINDOW_WEEKS:
        return None
    start = last
    for m in reversed(ordered[:-1]):
        if (start - m).days > 7:
            break
        start = m
    lead = _weeks_between(start, issued)
    if lead < 1:
        return None
    return {"episode_start_week": weeks[start], "last_flag_week": weeks[last], "lead_weeks": lead}


def lead_report_streaming(pack_id: str, *, model_grain: bool = False, progress: bool = False,
                          workers: int = 1) -> dict[str, Any]:
    """Same result as ``recall_lead_report(pack_id, run_backtests=False)``.

    It calls the same building blocks (``_load``, ``_score_received``,
    ``_first_flag``, ``_later_recall``) with the same as-of clock. Only memory
    and wall time differ. The existing function caches every as-of series
    (749 distinct dates x 219k records ran the box out of RAM), while this
    keeps one series per worker at a time. ``workers`` > 1 forks processes over
    disjoint sets of as-of dates. Rows come back in the original advisory order.
    The existing module is not modified. Equality is asserted in
    tests/frontline/ai_layer/test_safety_backtest.py.
    """
    from src.backtest.recall_lead import _later_recall, _load, _score_received

    advisories, records = _load(pack_id)
    by_stamp: dict[Any, list[int]] = defaultdict(list)
    for i, a in enumerate(advisories):
        by_stamp[a["issued_at"]].append(i)
    stamps = sorted(by_stamp)
    _G.update(advisories=advisories, records=records, by_stamp=by_stamp, pack_id=pack_id,
              model_grain=model_grain, progress=progress)
    rows: dict[int, dict[str, Any]] = {}
    recent: dict[int, Any] = {}
    workers = max(1, int(workers))
    if workers == 1 or len(stamps) < 2:
        parts = [_rows_for_stamps(stamps)]
    else:
        import multiprocessing as mp

        chunks = [stamps[k::workers] for k in range(workers)]  # round-robin balances early/late cost
        with mp.get_context("fork").Pool(workers) as pool:
            parts = pool.map(_rows_for_stamps, [c for c in chunks if c])
    for part in parts:
        rows.update(part["rows"])
        recent.update(part["recent"])
    recalls = [rows[i] for i in range(len(advisories))]
    full = _score_received(records, pack_id=pack_id, as_of=None, model_grain=model_grain)
    unconfirmed = []
    for row in full:
        if not row.get("is_anomaly") or _later_recall(row, advisories, model_grain=model_grain):
            continue
        unconfirmed.append({
            "iso_week": row.get("iso_week"), "category": row.get("category"), "entity_2": row.get("entity_2"),
            "entity_3": row.get("entity_3") or "", "record_count": row.get("record_count"),
            "baseline_mean": row.get("baseline_mean"), "z_score": row.get("z_score"), "status": "unconfirmed",
        })
    unconfirmed.sort(key=lambda r: (str(r["iso_week"]), str(r["category"]), str(r["entity_2"])))
    hits = [r for r in recalls if r["outcome"] == "hit"]
    misses = [r for r in recalls if r["outcome"] == "miss"]
    return {
        "pack_id": pack_id, "model_grain": model_grain, "recalls": recalls, "hits": hits, "misses": misses,
        "unconfirmed": unconfirmed, "backtest_rows": [], "hit_count": len(hits), "miss_count": len(misses),
        "unconfirmed_count": len(unconfirmed),
        # Extra (not in recall_lead_report): latest flag episode within 52 weeks, aligned with "recalls".
        "recent_episode": [recent.get(i) for i in range(len(advisories))],
    }


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


def campaign_level(recalls: list[dict[str, Any]]) -> dict[str, Any]:
    """NHTSA recall rows are campaign x model x model-year scope rows
    (advisory_id ``24V123000|2019|HONDA|CR-V|SERVICE BRAKES``). Roll them up to
    the campaign number: a campaign is a hit when any of its rows is, with its
    best (largest) lead."""
    camp: dict[str, dict[str, Any]] = {}
    for r in recalls:
        cid = str(r.get("advisory_id") or "").split("|")[0]
        c = camp.setdefault(cid, {"hit": False, "lead": 0})
        if r.get("outcome") == "hit":
            c["hit"] = True
            c["lead"] = max(c["lead"], int(r.get("lead_weeks") or 0))
    leads = [c["lead"] for c in camp.values() if c["hit"]]
    n = len(camp)
    return {
        "campaigns": n, "hits": len(leads), "misses": n - len(leads),
        "recall_rate": round(len(leads) / n, 3) if n else None,
        "lead_weeks_median": statistics.median(leads) if leads else None,
        "lead_weeks_mean": round(statistics.mean(leads), 1) if leads else None,
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
    recent = _summarise_recent(report, safety)
    hero = max(per_hit, key=lambda r: (r["window"].get("crash_fire_or_injury", 0), r["lead_weeks"]), default=None)
    return {
        "recalls_scored": n, "hits": len(hits), "misses": len(misses),
        "recall_rate": round(len(hits) / n, 3) if n else None,
        "lead_weeks_median": statistics.median(leads) if leads else None,
        "lead_weeks_mean": round(statistics.mean(leads), 1) if leads else None,
        "lead_weeks_p25": statistics.quantiles(leads, n=4)[0] if len(leads) >= 4 else None,
        "lead_weeks_p75": statistics.quantiles(leads, n=4)[2] if len(leads) >= 4 else None,
        "unconfirmed_flags": report["unconfirmed_count"],
        "campaign_level": campaign_level(list(hits) + list(misses)),
        "preventable_window_totals_dedup": totals,
        "per_hit": per_hit, "hero": hero, "recent": recent,
    }


def _summarise_recent(report: dict[str, Any], safety: dict[tuple[str, str], list[dict[str, Any]]]) -> dict[str, Any] | None:
    """Actionable-lead metric: recalls whose slice had a flag episode ending
    within RECENT_WINDOW_WEEKS before the recall; lead from that episode's start."""
    eps = report.get("recent_episode")
    if eps is None:
        return None
    recalls = report["recalls"]
    rows = []
    for r, ep in zip(recalls, eps):
        rows.append({**r, "outcome": "hit" if ep else "miss", "lead_weeks": ep["lead_weeks"] if ep else 0,
                     "first_flagged_week": ep["episode_start_week"] if ep else None})
    hits = [r for r in rows if r["outcome"] == "hit"]
    leads = [int(r["lead_weeks"]) for r in hits]
    per_hit = []
    uniq: dict[str, dict[str, Any]] = {}
    for h in hits:
        start = _monday(str(h["first_flagged_week"]))
        end = date.fromisoformat(str(h["issued_at"]))
        key = (str(h["scope_category"] or "").upper(), str(h["scope_entity_2"] or "").upper())
        srows = safety.get(key, [])
        counts, ids = window_counts(srows, start, end) if start else ({}, set())
        for x in srows:
            if x["odino"] in ids:
                uniq.setdefault(f"{key}:{x['odino']}", x)
        per_hit.append({**{k: h[k] for k in ("advisory_id", "issued_at", "scope_category", "scope_entity_2",
                                               "scope_entity_3", "first_flagged_week", "lead_weeks")},
                        "window": counts})
    u = list(uniq.values())
    totals = {
        "complaints": len(u),
        "crash": sum(1 for r in u if r["crash"]),
        "fire": sum(1 for r in u if r["fire"]),
        "injury_complaints": sum(1 for r in u if r["injured"] > 0),
        "injured_persons_reported": sum(r["injured"] for r in u),
        "death_complaints": sum(1 for r in u if r["deaths"] > 0),
        "deaths_reported": sum(r["deaths"] for r in u),
        "crash_fire_or_injury": sum(1 for r in u if r["crash"] or r["fire"] or r["injured"] > 0 or r["deaths"] > 0),
    }
    n = len(rows)
    hero = max(per_hit, key=lambda r: (r["window"].get("crash_fire_or_injury", 0), r["lead_weeks"]), default=None)
    return {
        "window_weeks": RECENT_WINDOW_WEEKS,
        "recalls_scored": n, "hits": len(hits), "misses": n - len(hits),
        "recall_rate": round(len(hits) / n, 3) if n else None,
        "lead_weeks_median": statistics.median(leads) if leads else None,
        "lead_weeks_mean": round(statistics.mean(leads), 1) if leads else None,
        "lead_weeks_p25": statistics.quantiles(leads, n=4)[0] if len(leads) >= 4 else None,
        "lead_weeks_p75": statistics.quantiles(leads, n=4)[2] if len(leads) >= 4 else None,
        "campaign_level": campaign_level(rows),
        "preventable_window_totals_dedup": totals,
        "hero": hero,
        "per_hit": per_hit,
    }


def _render_recent(r: dict[str, Any] | None) -> list[str]:
    if not r:
        return []
    c, t, h = r["campaign_level"], r["preventable_window_totals_dedup"], r["hero"]
    L = [
        f"## Actionable early warning (headline metric): flag episode within {r['window_weeks']} weeks before the recall",
        "",
        f"A recall counts as flagged early only if its make + component slice had a flag episode (consecutive",
        f"flagged weeks) whose last week falls within {r['window_weeks']} weeks before the recall report date, scored",
        "only on complaints NHTSA had received by that date. Lead = weeks from that episode's first week to the",
        "recall report date. This is the signal an engineer would act on.",
        "",
        "| Metric | Campaigns | Scope rows |",
        "| --- | --- | --- |",
        f"| Scored | {c['campaigns']} | {r['recalls_scored']} |",
        f"| Flagged early | {c['hits']} | {r['hits']} |",
        f"| Missed | {c['misses']} | {r['misses']} |",
        f"| Recall (hits / scored) | {c['recall_rate']} | {r['recall_rate']} |",
        f"| Median lead time, weeks | {c['lead_weeks_median']} | {r['lead_weeks_median']} |",
        f"| Mean lead time, weeks | {c['lead_weeks_mean']} | {r['lead_weeks_mean']} |",
        f"| Lead time IQR, weeks (scope rows) | | {r['lead_weeks_p25']} - {r['lead_weeks_p75']} |",
        "",
        "Campaign lead is the campaign's best scope-row lead.",
        "",
        "### Preventable window (correlation, not proof)",
        "",
        "Distinct ODI complaints on the flagged make + component received between the start of that episode and",
        "the recall report date, from NHTSA's own CRASH / FIRE / INJURED / DEATHS fields, deduplicated across",
        "overlapping recalls.",
        "",
        "| Field | Complaints |",
        "| --- | --- |",
        f"| All complaints | {t['complaints']} |",
        f"| CRASH = Y | {t['crash']} |",
        f"| FIRE = Y | {t['fire']} |",
        f"| INJURED > 0 | {t['injury_complaints']} (persons reported: {t['injured_persons_reported']}) |",
        f"| DEATHS > 0 | {t['death_complaints']} (deaths reported: {t['deaths_reported']}) |",
        f"| Crash, fire, injury or death | {t['crash_fire_or_injury']} |",
        "",
    ]
    if h:
        w = h["window"]
        L += [
            f"Largest window: {h['scope_entity_2']} {h['scope_category']}, episode from {h['first_flagged_week']},",
            f"{h['lead_weeks']} weeks before recall {h['advisory_id']} ({h['issued_at']}): {w.get('complaints')} complaints,",
            f"{w.get('crash')} CRASH, {w.get('fire')} FIRE, {w.get('injury_complaints')} with injuries.",
            "",
        ]
    return L


def render(meta: dict[str, Any], s: dict[str, Any], prec: dict[str, Any]) -> str:
    t = s["preventable_window_totals_dedup"]
    c = s["campaign_level"]
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
    ]
    L += _render_recent(s.get("recent"))
    L += [
        "## First-ever flag vs official recall (upper bound; read the caveat)",
        "",
        "Lead here runs from the FIRST week the make+component slice was ever flagged, which for broad slices",
        "(e.g. FORD ENGINE) is years before an unrelated later recall. These leads are inflated; the",
        "actionable metric above is the one to quote. NHTSA recall rows are campaign x model x model-year",
        "scope rows; both grains are shown.",
        "",
        "| Metric | Campaigns | Scope rows |",
        "| --- | --- | --- |",
        f"| Scored | {c['campaigns']} | {s['recalls_scored']} |",
        f"| Flagged early (hit, >= 1 week lead) | {c['hits']} | {s['hits']} |",
        f"| Missed | {c['misses']} | {s['misses']} |",
        f"| Recall (hits / scored) | {c['recall_rate']} | {s['recall_rate']} |",
        f"| Median lead time, weeks (hits) | {c['lead_weeks_median']} | {s['lead_weeks_median']} |",
        f"| Mean lead time, weeks (hits) | {c['lead_weeks_mean']} | {s['lead_weeks_mean']} |",
        "",
        "| Metric (scope rows, flags) | Value |",
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
        "## Preventable window, first-ever flag (upper bound; correlation, not proof)",
        "",
        "For every first-flag hit: distinct ODI complaints of the same make + component received between the",
        "Monday of the first flagged week and the recall report date, using NHTSA's own CRASH / FIRE / INJURED /",
        "DEATHS fields. Deduplicated across overlapping recalls of the same slice. Inflated by the long leads above.",
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
            "### Largest single first-flag window",
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
        "- The 52-week actionable window is a chosen parameter, not tuned; a recent flag on a broad slice can",
        "  still come from a different defect than the one recalled.",
        "- Flag precision is computed on the full series (all data), which uses later weeks in each baseline.",
        "- Only five makes and model years 2015-2024 were ingested; other makes may behave differently.",
        "- The detector is the existing deterministic weekly z-score scan; the optional Meta AI layer is not",
        "  involved in these numbers.",
        "",
        "## Per-recall hits (first-ever flag)",
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
    p.add_argument("--workers", type=int, default=1, help="processes for as-of scoring (same result)")
    p.add_argument("--cache", default=None, help="JSON cache of the lead report + precision (written after scoring; "
                   "reused when present so the report can be re-rendered without re-scoring)")
    p.add_argument("--out-md", default=str(OUT_MD))
    p.add_argument("--out-json", default=str(OUT_JSON))
    p.add_argument("--out-lead", default=str(OUT_LEAD), help="existing-format per-recall lead-time report")
    args = p.parse_args(argv)
    if is_default_pilot_domain_path():
        print("refusing the pilot data/domains directory; set DOMAIN_DB_PATH to a side directory", file=sys.stderr)
        return 2
    from src.backtest.recall_lead import format_recall_lead_report

    t0 = time.time()
    cache = Path(args.cache) if args.cache else None
    if cache and cache.exists():
        cached = json.loads(cache.read_text())
        report, prec = cached["report"], cached["precision"]
        print(f"reusing cached scoring from {cache}", flush=True)
    else:
        report = lead_report_streaming(args.pack, model_grain=args.model_grain, progress=True, workers=args.workers)
        report.pop("backtest_rows", None)
        print(f"lead report: hits={report['hit_count']} misses={report['miss_count']} "
              f"unconfirmed={report['unconfirmed_count']} ({time.time() - t0:.0f}s)", flush=True)
        prec = flag_precision(args.pack, model_grain=args.model_grain)
        print(f"precision: {prec} ({time.time() - t0:.0f}s)", flush=True)
        if cache:
            cache.write_text(json.dumps({"report": report, "precision": prec}, default=str))
    Path(args.out_lead).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_lead).write_text(format_recall_lead_report(report), encoding="utf-8")
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
    r = s.get("recent")
    if r and r["hits"]:
        rt = r["preventable_window_totals_dedup"]
        headline = {
            "metric": f"flag episode within {r['window_weeks']} weeks before the recall (scored as of recall date)",
            "recalls_scored": r["recalls_scored"], "hits": r["hits"], "misses": r["misses"],
            "recall_rate": r["recall_rate"], "lead_weeks_median": r["lead_weeks_median"],
            "lead_weeks_mean": r["lead_weeks_mean"], "campaign_level": r["campaign_level"],
            "precision_episodes": prec["precision_episodes"],
            "preventable_window_crash": rt["crash"], "preventable_window_fire": rt["fire"],
            "preventable_window_injury_complaints": rt["injury_complaints"],
            "preventable_window_crash_fire_or_injury": rt["crash_fire_or_injury"],
            "hero": r["hero"],
            "first_flag": {"hits": s["hits"], "recall_rate": s["recall_rate"], "lead_weeks_median": s["lead_weeks_median"],
                           "note": "first-ever slice flag; inflated by broad make+component slices"},
            "caveat": "correlation from self-reported NHTSA complaints; not proof of prevention",
            "source": str(Path(args.out_md).relative_to(REPO_ROOT)) if Path(args.out_md).resolve().is_relative_to(REPO_ROOT) else args.out_md,
        }
    elif s["hits"]:
        headline = {
            "recalls_scored": s["recalls_scored"], "hits": s["hits"], "misses": s["misses"],
            "recall_rate": s["recall_rate"], "lead_weeks_median": s["lead_weeks_median"],
            "campaign_level": s["campaign_level"],
            "lead_weeks_mean": s["lead_weeks_mean"], "precision_episodes": prec["precision_episodes"],
            "preventable_window_crash": t["crash"], "preventable_window_fire": t["fire"],
            "preventable_window_injury_complaints": t["injury_complaints"],
            "preventable_window_crash_fire_or_injury": t["crash_fire_or_injury"],
            "hero": s["hero"], "caveat": "correlation from self-reported NHTSA complaints; not proof of prevention",
            "source": str(Path(args.out_md).relative_to(REPO_ROOT)) if Path(args.out_md).resolve().is_relative_to(REPO_ROOT) else args.out_md,
        }
    Path(args.out_json).write_text(json.dumps({"meta": meta, "precision": prec, "headline": headline,
                                               "summary": {k: ({kk: vv for kk, vv in v.items() if kk != "per_hit"} if k == "recent" and v else v)
                                                           for k, v in s.items() if k != "per_hit"},
                                               "recent_per_hit": (s.get("recent") or {}).get("per_hit"),
                                               "per_hit": s["per_hit"]}, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"headline": headline, "precision": prec}, default=str, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
