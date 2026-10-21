"""Out-of-sample recall backtest: tune flag thresholds on 2015-2020, freeze, test ONCE on 2021-2024.  [AI-LAYER]

The in-sample safety backtest (``scripts/ai_safety_backtest.py``) scores the
EXISTING detector (``src.ml_runtime.anomalies.score_weekly_slices``, constants
z >= 2.0, BH q <= 0.05, absolute lift >= 5, relative lift >= 50%) on all of
2015-2024. This script asks whether thresholds chosen on older data hold up
on newer data:

Phase ``tune`` (2015-01-01 .. 2020-12-31 ONLY)
    * recalls: advisories whose report date (RCDATE) is in the train window;
    * complaints: only rows NHTSA had received by 2020-12-31 (the record list
      is cut before scoring), and each recall is scored as of its own date;
    * precision: flags in train weeks, confirmed only by recalls issued by
      2020-12-31.
    For each recall date the existing scorer runs ONCE; every grid config is
    then applied as a re-gate on the scorer's own row fields (z_score, p_bh,
    absolute_lift, relative_lift, method). For the production config the
    re-gate is asserted equal to the scorer's ``is_anomaly`` on every row
    (``regate_mismatches`` must be 0). The objective (fixed BEFORE the run):
    maximise the harmonic mean of the actionable campaign recall rate and the
    episode precision; ties -> higher precision, then the production config.
    The winner is frozen to ``reports/oos/oos_config.json`` with a sha256 of
    its canonical body.

Phase ``test`` (2021-01-01 .. 2024-12-31), ONE run
    * refuses to run without a frozen config, refuses if the config hash does
      not verify, and refuses a second test run unless ``--force-retest``
      (which is logged);
    * recalls issued in the test window, scored as of their own date (complaint
      history before 2021 is allowed as baseline; nothing after the recall
      date is seen); complaints received after 2024-12-31 are cut;
    * precision: flags in test weeks, confirmed by recalls issued by 2024-12-31.

``tune`` refuses to run when a frozen config exists unless ``--force-retune``
(logged). Every invocation is appended to ``reports/oos/run_log.jsonl``.

Metrics are the existing backtest's: ``ai_safety_backtest.summarise`` (first
flag + 52-week actionable episode, scope rows and campaigns, lead times,
preventable-window complaint counts) and the same episode / slice-week
precision definitions.

    DOMAIN_DB_PATH=<side dir> python -m scripts.ai_oos_backtest --phase tune --zip FLAT_CMPL.zip --workers 6
    DOMAIN_DB_PATH=<side dir> python -m scripts.ai_oos_backtest --phase test --zip FLAT_CMPL.zip --workers 6
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import REPO_ROOT, is_default_pilot_domain_path  # noqa: E402

OUT_DIR = REPO_ROOT / "reports" / "oos"
PROD = {"z": 2.0, "q": 0.05, "abs_lift": 5, "rel_lift": 0.5}
GRID = {"z": [2.0, 2.5, 3.0], "q": [0.05, 0.01], "abs_lift": [3, 5, 8], "rel_lift": [0.5, 1.0]}
OBJECTIVE = ("maximise harmonic mean of (actionable 52-week campaign recall rate, episode precision) on the train "
             "window; ties -> higher episode precision, then the production config")


class Refused(Exception):
    pass


# ── gate (mirror of score_weekly_slices' is_anomaly, parameterised) ─────────


def gate(r: dict[str, Any], cfg: dict[str, Any]) -> bool:
    if r.get("method") == "insufficient-history":
        return False
    # zero-baseline rows get z := ANOMALY_Z inside the scorer, so they pass the z gate at any threshold
    z_ok = r.get("method") == "zero-baseline" or float(r.get("z_score") or 0.0) >= cfg["z"]
    if not z_ok or float(r.get("p_bh", 1.0)) > cfg["q"]:
        return False
    rel = r.get("relative_lift") or 0.0
    if r.get("normalized"):
        return rel >= cfg["rel_lift"]
    return float(r.get("absolute_lift") or 0.0) >= cfg["abs_lift"] and (
        float(r.get("baseline_mean") or 0.0) <= 0 or rel >= cfg["rel_lift"])


def cfg_id(cfg: dict[str, Any]) -> str:
    return f"z{cfg['z']}_q{cfg['q']}_abs{cfg['abs_lift']}_rel{cfg['rel_lift']}"


def grid_configs() -> list[dict[str, Any]]:
    keys = list(GRID)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(GRID[k] for k in keys))]


def canonical_hash(body: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _dt(s: str, end: bool = False) -> datetime:
    d = date.fromisoformat(s)
    return datetime(d.year, d.month, d.day, 23, 59, 59) if end else datetime(d.year, d.month, d.day)


# ── per-stamp scoring (one scorer pass per recall date, all configs re-gated) ─

_G: dict[str, Any] = {}


def _norm(v: Any) -> str:
    return str(v or "").strip().upper()


def _score_stamps(stamps: list[Any]) -> dict[str, Any]:
    from src.backtest.engine import _weeks_between
    from src.backtest.recall_lead import _first_flag, _score_received
    from src.ml_runtime.anomalies import _week_monday
    from scripts.ai_safety_backtest import _recent_episode

    advisories, records, by_stamp, cfgs = _G["advisories"], _G["records"], _G["by_stamp"], _G["cfgs"]
    out: dict[int, dict[str, Any]] = {}
    mismatches = rows_checked = 0
    t0 = time.time()
    for n, issued in enumerate(stamps, 1):
        idx = by_stamp[issued]
        keys = {(_norm(advisories[i]["scope_category"]), _norm(advisories[i]["scope_entity_2"])) for i in idx}
        series = [r for r in _score_received(records, pack_id=_G["pack_id"], as_of=issued, model_grain=False)
                  if (_norm(r.get("category")), _norm(r.get("entity_2"))) in keys]
        for r in series:
            rows_checked += 1
            mismatches += int(gate(r, PROD) != bool(r.get("is_anomaly")))
        for i in idx:
            out[i] = {}
        for cfg in cfgs:
            flagged = [dict(r, is_anomaly=True) for r in series if gate(r, cfg)]
            for i in idx:
                adv = advisories[i]
                flag = _first_flag(flagged, adv, model_grain=False)
                lead, first_week = 0, None
                if flag is not None:
                    start = _week_monday(str(flag["iso_week"]))
                    lead = _weeks_between(start, issued) if start is not None else 0
                    first_week = flag["iso_week"]
                ep = _recent_episode(flagged, adv, issued, model_grain=False)
                out[i][cfg_id(cfg)] = (first_week, lead, ep)
        if _G.get("progress") and n % 25 == 0:
            print(f"  worker scored {n}/{len(stamps)} as-of dates ({time.time() - t0:.0f}s)", flush=True)
    return {"rows": out, "mismatches": mismatches, "rows_checked": rows_checked}


def score_period(pack_id: str, advisories: list[dict[str, Any]], records: list[dict[str, Any]],
                 cfgs: list[dict[str, Any]], *, workers: int, progress: bool) -> dict[str, Any]:
    by_stamp: dict[Any, list[int]] = defaultdict(list)
    for i, a in enumerate(advisories):
        by_stamp[a["issued_at"]].append(i)
    stamps = sorted(by_stamp)
    _G.update(advisories=advisories, records=records, by_stamp=by_stamp, cfgs=cfgs, pack_id=pack_id,
              progress=progress)
    if workers <= 1 or len(stamps) < 2:
        parts = [_score_stamps(stamps)]
    else:
        import multiprocessing as mp

        chunks = [stamps[k::workers] for k in range(workers)]
        with mp.get_context("fork").Pool(workers) as pool:
            parts = pool.map(_score_stamps, [c for c in chunks if c])
    rows: dict[int, dict[str, Any]] = {}
    for p in parts:
        rows.update(p["rows"])
    return {"rows": rows, "stamps": len(stamps), "regate_mismatches": sum(p["mismatches"] for p in parts),
            "regate_rows_checked": sum(p["rows_checked"] for p in parts)}


def build_report(advisories: list[dict[str, Any]], scored: dict[int, dict[str, Any]], cid: str,
                 unconfirmed_count: int) -> dict[str, Any]:
    """Same shape ``ai_safety_backtest.summarise`` consumes."""
    recalls, recent = [], []
    for i, a in enumerate(advisories):
        first_week, lead, ep = scored[i][cid]
        recalls.append({
            "advisory_id": a["advisory_id"], "issued_at": a["issued_at"].date().isoformat(),
            "scope_category": a.get("scope_category"), "scope_entity_2": a.get("scope_entity_2"),
            "scope_entity_3": a.get("scope_entity_3"), "first_flagged_week": first_week, "lead_weeks": lead,
            "outcome": "hit" if lead > 0 else "miss",
        })
        recent.append(ep)
    hits = [r for r in recalls if r["outcome"] == "hit"]
    misses = [r for r in recalls if r["outcome"] == "miss"]
    return {"recalls": recalls, "hits": hits, "misses": misses, "unconfirmed_count": unconfirmed_count,
            "recent_episode": recent}


def recent_campaign_recall(report: dict[str, Any]) -> dict[str, Any]:
    from scripts.ai_safety_backtest import campaign_level

    rows = [{**r, "outcome": "hit" if ep else "miss", "lead_weeks": ep["lead_weeks"] if ep else 0}
            for r, ep in zip(report["recalls"], report["recent_episode"])]
    return campaign_level(rows)


# ── precision (same definitions as ai_safety_backtest.flag_precision, period-bounded) ─


def full_series(pack_id: str, records: list[dict[str, Any]], as_of: datetime) -> list[dict[str, Any]]:
    from src.backtest.recall_lead import _score_received

    return _score_received(records, pack_id=pack_id, as_of=as_of, model_grain=False)


def precision(series: list[dict[str, Any]], cfg: dict[str, Any], label_pool: list[dict[str, Any]],
              start: datetime, end: datetime) -> dict[str, Any]:
    from src.backtest.recall_lead import _later_recall
    from src.ml_runtime.anomalies import _week_monday

    flags = []
    for r in series:
        m = _week_monday(str(r.get("iso_week") or ""))
        if m is not None and start <= m <= end and gate(r, cfg):
            flags.append(r)
    confirmed = [r for r in flags if _later_recall(r, label_pool, model_grain=False)]
    conf_keys = {(r.get("iso_week"), r.get("category"), r.get("entity_2")) for r in confirmed}
    by_slice: dict[tuple, list] = defaultdict(list)
    conf_slice_weeks: dict[tuple, set] = defaultdict(set)
    for r in flags:
        m = _week_monday(str(r.get("iso_week")))
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
        "flagged_slice_weeks": len(flags), "confirmed_slice_weeks": len(confirmed),
        "false_alarm_slice_weeks": len(flags) - len(confirmed),
        "precision_slice_weeks": round(len(confirmed) / len(flags), 3) if flags else None,
        "flag_episodes": episodes, "confirmed_episodes": conf_eps,
        "precision_episodes": round(conf_eps / episodes, 3) if episodes else None,
        "unconfirmed_slice_weeks": len(flags) - len(confirmed),
    }


def objective(recall_rate: float | None, prec: float | None) -> float:
    r, p = float(recall_rate or 0.0), float(prec or 0.0)
    return round(2 * r * p / (r + p), 4) if (r + p) > 0 else 0.0


def choose(results: list[dict[str, Any]]) -> dict[str, Any]:
    def key(x: dict[str, Any]):
        return (x["objective"], x["precision_episodes"] or 0.0, int(x["config"] == PROD))
    return max(results, key=key)


# ── I/O helpers ─────────────────────────────────────────────────────────────


def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def log_run(out_dir: Path, entry: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    entry = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_head": _git_head(), **entry}
    with (out_dir / "run_log.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")


def load_frozen(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise Refused(f"no frozen config at {path}; run --phase tune first")
    cfg = json.loads(path.read_text())
    body = cfg.get("body")
    if not isinstance(body, dict) or canonical_hash(body) != cfg.get("sha256"):
        raise Refused(f"frozen config hash does not verify ({path}); refusing to evaluate")
    return cfg


def check_tune_allowed(config_path: Path, force: bool) -> None:
    if config_path.exists() and not force:
        raise Refused(f"frozen config exists at {config_path}; refusing to re-tune (pass --force-retune to override, "
                      "which is logged)")


def check_test_allowed(test_path: Path, force: bool) -> None:
    if test_path.exists() and not force:
        raise Refused(f"test results already exist at {test_path}; the out-of-sample test runs once "
                      "(pass --force-retest to override, which is logged)")


def _zip_fingerprint(p: Path | None) -> dict[str, Any] | None:
    if not p or not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return {"name": p.name, "bytes": p.stat().st_size, "sha256": h.hexdigest()}


def _slice_records(records: list[dict[str, Any]], end: datetime) -> list[dict[str, Any]]:
    from src.ml_runtime.anomalies import _parse_dt

    out = []
    for r in records:
        t = _parse_dt(r.get("received_at"))
        if t is not None and t <= end:
            out.append(r)
    return out


def period_inputs(pack_id: str, start: str, end: str) -> tuple[list, list, list, datetime, datetime]:
    from src.backtest.recall_lead import _load

    advisories, records = _load(pack_id)
    s, e = _dt(start), _dt(end, end=True)
    recs = _slice_records(records, e)
    period_advs = [a for a in advisories if s <= a["issued_at"] <= e]
    label_pool = [a for a in advisories if a["issued_at"] <= e]
    return period_advs, recs, label_pool, s, e


def evaluate(pack_id: str, start: str, end: str, cfgs: list[dict[str, Any]], *, workers: int,
             progress: bool) -> dict[str, Any]:
    advs, recs, pool, s, e = period_inputs(pack_id, start, end)
    t0 = time.time()
    scored = score_period(pack_id, advs, recs, cfgs, workers=workers, progress=progress)
    series = full_series(pack_id, recs, e)
    per_cfg = []
    for cfg in cfgs:
        cid = cfg_id(cfg)
        rep = build_report(advs, scored["rows"], cid, 0)
        prec = precision(series, cfg, pool, s, e)
        camp = recent_campaign_recall(rep)
        per_cfg.append({"config": cfg, "id": cid, "recent_campaign": camp,
                        "recent_scope_hits": sum(1 for x in rep["recent_episode"] if x),
                        "first_flag_scope_hits": len(rep["hits"]), "precision": prec,
                        "precision_episodes": prec["precision_episodes"],
                        "objective": objective(camp["recall_rate"], prec["precision_episodes"])})
    return {"advisories": advs, "records_n": len(recs), "scored": scored, "series": series, "pool": pool,
            "start": s, "end": e, "per_cfg": per_cfg, "elapsed_s": int(time.time() - t0)}


def full_metrics(ev: dict[str, Any], cfg: dict[str, Any], zip_path: Path | None, makes: set[str],
                 year_min: int, year_max: int) -> dict[str, Any]:
    from scripts.ai_safety_backtest import iter_flat, safety_rows, summarise

    cid = cfg_id(cfg)
    prec = precision(ev["series"], cfg, ev["pool"], ev["start"], ev["end"])
    rep = build_report(ev["advisories"], ev["scored"]["rows"], cid, prec["unconfirmed_slice_weeks"])
    if zip_path and zip_path.exists():
        slices = {(_norm(h["scope_category"]), _norm(h["scope_entity_2"])) for h in rep["hits"]}
        safety = safety_rows(iter_flat(zip_path), makes, year_min, year_max, slices)
        window_note = "computed from FLAT_CMPL.zip"
    else:
        safety, window_note = {}, "not computed (no --zip given): preventable-window counts are 0 placeholders"
    s = summarise(rep, safety)
    recent = s.get("recent") or {}
    return {
        "config_id": cid, "records_received_by_period_end": ev["records_n"], "advisories": len(ev["advisories"]),
        "as_of_dates": ev["scored"]["stamps"], "precision": prec,
        "first_flag": {k: s[k] for k in ("recalls_scored", "hits", "misses", "recall_rate", "lead_weeks_median",
                                         "lead_weeks_mean", "lead_weeks_p25", "lead_weeks_p75", "campaign_level",
                                         "preventable_window_totals_dedup")},
        "actionable_52w": {k: recent.get(k) for k in ("window_weeks", "recalls_scored", "hits", "misses", "recall_rate",
                                                       "lead_weeks_median", "lead_weeks_mean", "lead_weeks_p25",
                                                       "lead_weeks_p75", "campaign_level",
                                                       "preventable_window_totals_dedup")},
        "preventable_window_source": window_note,
        "regate_mismatches_vs_scorer": ev["scored"]["regate_mismatches"],
        "regate_rows_checked": ev["scored"]["regate_rows_checked"],
    }


# ── phases ──────────────────────────────────────────────────────────────────


def run_tune(args) -> dict[str, Any]:
    out = Path(args.out_dir)
    cfg_path = out / "oos_config.json"
    check_tune_allowed(cfg_path, args.force_retune)
    cfgs = grid_configs()
    ev = evaluate(args.pack, args.train_start, args.train_end, cfgs, workers=args.workers, progress=True)
    best = choose(ev["per_cfg"])
    prod = next(x for x in ev["per_cfg"] if x["config"] == PROD)
    makes = {m.strip().upper() for m in args.makes.split(",") if m.strip()}
    zp = Path(args.zip) if args.zip else None
    insample = full_metrics(ev, best["config"], zp, makes, args.year_min, args.year_max)
    insample_prod = full_metrics(ev, PROD, zp, makes, args.year_min, args.year_max) if best["config"] != PROD else None
    body = {
        "thresholds": best["config"], "config_id": best["id"], "production_thresholds": PROD,
        "objective": OBJECTIVE, "grid": GRID, "train_window": [args.train_start, args.train_end],
        "test_window": [args.test_start, args.test_end], "pack": args.pack,
        "label_rule": "recall report date (RCDATE); precision label = later recall of same make+component issued by "
                      "the period end",
        "detector": "src.ml_runtime.anomalies.score_weekly_slices (received clock, make+component grain), re-gated",
        "data": {"flat_cmpl": _zip_fingerprint(zp), "records_received_by_train_end": ev["records_n"],
                 "train_advisories": len(ev["advisories"])},
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    frozen = {"sha256": canonical_hash(body), "body": body}
    out.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(frozen, indent=2, default=str))
    tune = {"grid_results": [{k: v for k, v in x.items()} for x in ev["per_cfg"]], "chosen": best["id"],
            "production": prod["id"], "regate_mismatches_vs_scorer": ev["scored"]["regate_mismatches"],
            "regate_rows_checked": ev["scored"]["regate_rows_checked"], "elapsed_s": ev["elapsed_s"],
            "in_sample_frozen": insample, "in_sample_production": insample_prod, "config_sha256": frozen["sha256"]}
    (out / "oos_tune_results.json").write_text(json.dumps(tune, indent=2, default=str))
    log_run(out, {"phase": "tune", "forced": bool(args.force_retune), "chosen": best["id"],
                  "config_sha256": frozen["sha256"], "elapsed_s": ev["elapsed_s"]})
    return tune


def run_test(args) -> dict[str, Any]:
    out = Path(args.out_dir)
    frozen = load_frozen(out / "oos_config.json")
    test_path = out / "oos_test_results.json"
    check_test_allowed(test_path, args.force_retest)
    cfg = frozen["body"]["thresholds"]
    ws, we = frozen["body"]["test_window"]
    log_run(out, {"phase": "test_start", "forced": bool(args.force_retest), "config_sha256": frozen["sha256"],
                  "config_id": cfg_id(cfg)})
    ev = evaluate(args.pack, ws, we, [cfg], workers=args.workers, progress=True)
    makes = {m.strip().upper() for m in args.makes.split(",") if m.strip()}
    res = full_metrics(ev, cfg, Path(args.zip) if args.zip else None, makes, args.year_min, args.year_max)
    res.update({"config_sha256": frozen["sha256"], "test_window": [ws, we], "elapsed_s": ev["elapsed_s"],
                "note": "single evaluation of the frozen config; no thresholds were changed after this run"})
    test_path.write_text(json.dumps(res, indent=2, default=str))
    log_run(out, {"phase": "test", "forced": bool(args.force_retest), "config_sha256": frozen["sha256"],
                  "elapsed_s": ev["elapsed_s"]})
    return res


def _fmt(v: Any) -> str:
    return "n/a" if v is None else str(v)


def render(out_dir: Path) -> str:
    frozen = json.loads((out_dir / "oos_config.json").read_text())
    tune = json.loads((out_dir / "oos_tune_results.json").read_text())
    test_p = out_dir / "oos_test_results.json"
    test = json.loads(test_p.read_text()) if test_p.exists() else None
    ins = tune["in_sample_frozen"]
    b = frozen["body"]
    L = ["# Out-of-sample recall backtest (tune 2015-2020, frozen, test 2021-2024)", "",
         "Generated by `scripts/ai_oos_backtest.py` from the saved JSON in `reports/oos/`. Every number is copied from",
         "`oos_tune_results.json` / `oos_test_results.json`; nothing is estimated.", "",
         f"- Frozen config: `reports/oos/oos_config.json`, sha256 `{frozen['sha256']}`",
         f"- Chosen thresholds: `{b['config_id']}` (production: `{cfg_id(b['production_thresholds'])}`)",
         f"- Objective (fixed before tuning): {b['objective']}",
         f"- Grid: {json.dumps(b['grid'])} ({len(tune['grid_results'])} configs)",
         f"- Re-gate check (production config vs scorer `is_anomaly`): {tune['regate_mismatches_vs_scorer']} mismatches "
         f"over {tune['regate_rows_checked']} rows (train)"
         + (f"; {test['regate_mismatches_vs_scorer']} over {test['regate_rows_checked']} rows (test)" if test else ""),
         ""]
    if not test:
        L += ["**Test phase not run yet.**", ""]

    def col(m, path):
        cur = m
        for p in path:
            cur = (cur or {}).get(p) if isinstance(cur, dict) else None
        return _fmt(cur)

    rows = [
        ("Recall rows scored (scope rows)", ("actionable_52w", "recalls_scored")),
        ("Campaigns scored", ("actionable_52w", "campaign_level", "campaigns")),
        ("Actionable 52-wk: campaigns flagged early", ("actionable_52w", "campaign_level", "hits")),
        ("Actionable 52-wk: campaign recall rate", ("actionable_52w", "campaign_level", "recall_rate")),
        ("Actionable 52-wk: campaign median lead, wk", ("actionable_52w", "campaign_level", "lead_weeks_median")),
        ("Actionable 52-wk: scope-row hits", ("actionable_52w", "hits")),
        ("Actionable 52-wk: scope-row recall rate", ("actionable_52w", "recall_rate")),
        ("Actionable 52-wk: scope-row median lead, wk", ("actionable_52w", "lead_weeks_median")),
        ("Actionable 52-wk: scope-row mean lead, wk", ("actionable_52w", "lead_weeks_mean")),
        ("Flag episodes", ("precision", "flag_episodes")),
        ("Precision, episodes", ("precision", "precision_episodes")),
        ("Flagged slice-weeks", ("precision", "flagged_slice_weeks")),
        ("Precision, slice-weeks", ("precision", "precision_slice_weeks")),
        ("First-flag (upper bound): campaign recall rate", ("first_flag", "campaign_level", "recall_rate")),
        ("First-flag (upper bound): scope-row median lead, wk", ("first_flag", "lead_weeks_median")),
        ("Preventable window (52-wk): complaints", ("actionable_52w", "preventable_window_totals_dedup", "complaints")),
        ("Preventable window (52-wk): crash", ("actionable_52w", "preventable_window_totals_dedup", "crash")),
        ("Preventable window (52-wk): fire", ("actionable_52w", "preventable_window_totals_dedup", "fire")),
        ("Preventable window (52-wk): injury complaints", ("actionable_52w", "preventable_window_totals_dedup", "injury_complaints")),
        ("Preventable window (52-wk): crash/fire/injury/death", ("actionable_52w", "preventable_window_totals_dedup", "crash_fire_or_injury")),
    ]
    L += ["## Side by side (frozen config)", "",
          f"| Metric | In-sample {b['train_window'][0][:4]}-{b['train_window'][1][:4]} | Out-of-sample {b['test_window'][0][:4]}-{b['test_window'][1][:4]} |",
          "| --- | --- | --- |"]
    for label, path in rows:
        L.append(f"| {label} | {col(ins, path)} | {col(test, path) if test else 'not run'} |")
    L += ["", f"Preventable-window source: in-sample {ins['preventable_window_source']}"
          + (f"; test {test['preventable_window_source']}" if test else ""), ""]
    if tune.get("in_sample_production"):
        p = tune["in_sample_production"]
        L += ["In-sample with the production thresholds for reference: actionable campaign recall "
              f"{col(p, ('actionable_52w', 'campaign_level', 'recall_rate'))}, episode precision "
              f"{col(p, ('precision', 'precision_episodes'))}.", ""]
    L += ["## Tuning grid (train window only)", "",
          "| Config | Campaign recall (52-wk) | Episode precision | Episodes | Objective |", "| --- | --- | --- | --- | --- |"]
    for g in sorted(tune["grid_results"], key=lambda x: -x["objective"]):
        mark = " (chosen)" if g["id"] == tune["chosen"] else (" (production)" if g["id"] == tune["production"] else "")
        L.append(f"| `{g['id']}`{mark} | {g['recent_campaign']['recall_rate']} | {g['precision_episodes']} | "
                 f"{g['precision']['flag_episodes']} | {g['objective']} |")
    L += ["", "## Limitations (read before quoting)", "",
          "- Labels are recall REPORT dates (RCDATE, Part 573 received). Manufacturers often know earlier, so a",
          "  'lead' is measured against the public date, not the date the defect was known.",
          "- Right-censoring: flags late in each window (late 2020 for train, 2023-24 for test) can only be confirmed",
          "  by recalls issued by that window's end. Recalls for 2023-24 faults that NHTSA has not issued yet count",
          "  as false alarms, so late-window precision is biased DOWN; the same applies to late-2020 train flags.",
          "- Training uses only complaints received by 2020-12-31 and recalls issued by then. The test window still",
          "  uses pre-2021 complaints as baseline history (as the live detector would), but no complaint received",
          "  after a recall's date is seen when that recall is scored, and nothing after 2024-12-31.",
          "- NHTSA complaints are self-reported, unverified and subject to reporting bias (media coverage, prior",
          "  recalls and investigations drive spikes; some owners never file). Volume is raw counts, not rates:",
          "  no sales/fleet denominator and NO MILEAGE field, so exposure and age effects are not controlled.",
          "- Matching is make + component, not model / model year; a slice can mix unrelated defects.",
          "- Only five makes and model years 2015-2024 were ingested. Model-year coverage differs by window",
          "  (newer model years have little history in 2015-2020).",
          "- The grid is small (36 configs) and the objective is one documented choice; another objective could",
          "  pick different thresholds. The test window was evaluated once with the frozen config.",
          "- Preventable-window counts are complaint counts (correlation), not crashes prevented.", ""]
    if test:
        L += [f"Test run recorded in `reports/oos/run_log.jsonl` (config sha256 `{test['config_sha256']}`).", ""]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--phase", choices=["tune", "test", "render"], required=True)
    p.add_argument("--pack", default="automotive_nhtsa")
    p.add_argument("--zip", default=None, help="FLAT_CMPL.zip for preventable-window counts")
    p.add_argument("--makes", default="HONDA,TOYOTA,FORD,HYUNDAI,KIA")
    p.add_argument("--year-min", type=int, default=2015)
    p.add_argument("--year-max", type=int, default=2024)
    p.add_argument("--train-start", default="2015-01-01")
    p.add_argument("--train-end", default="2020-12-31")
    p.add_argument("--test-start", default="2021-01-01")
    p.add_argument("--test-end", default="2024-12-31")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--force-retune", action="store_true")
    p.add_argument("--force-retest", action="store_true")
    args = p.parse_args(argv)
    out = Path(args.out_dir)
    if args.phase != "render" and is_default_pilot_domain_path():
        print("refusing the pilot data/domains directory; set DOMAIN_DB_PATH to a side directory", file=sys.stderr)
        return 2
    try:
        if args.phase == "tune":
            res = run_tune(args)
            print(json.dumps({"chosen": res["chosen"], "config_sha256": res["config_sha256"],
                              "regate_mismatches": res["regate_mismatches_vs_scorer"]}, indent=2))
        elif args.phase == "test":
            res = run_test(args)
            print(json.dumps({k: res[k] for k in ("config_id", "precision", "actionable_52w")}, indent=2, default=str))
        (out / "OOS_BACKTEST.md").write_text(render(out), encoding="utf-8")
    except Refused as e:
        log_run(out, {"phase": args.phase, "refused": str(e)})
        print(f"REFUSED: {e}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
