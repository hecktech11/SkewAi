"""Per-recall lead time from complaints NHTSA had already received.

``run_backtest(as_of=issued_at)`` censors cluster members, but it still reads
whatever ``weekly_anomalies`` already contains. Those rows were scored with
later weeks in the baseline. This report scores each recall on
``received_at <= issued_at`` first, persists that series, then calls
``run_backtest`` so the engine and the report see the same clock.

A recall is a hit only when the first flagged week leads by at least one
week. Misses stay in the report. A flagged week that no later recall matches
is listed as unconfirmed, not dropped.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from src.backtest.engine import _parse_ts, _weeks_between, run_backtest
from src.data.warehouse import apply_domain_schema, domain_con
from src.ml_runtime.anomalies import (
    _week_monday,
    count_weekly_slices,
    persist_scored_anomalies,
    recompute_weekly_anomalies,
    score_weekly_slices,
)


def _norm(val: Any) -> str:
    return str(val or "").strip().upper()


def _slice_match(row: dict[str, Any], advisory: dict[str, Any], *, model_grain: bool) -> bool:
    if _norm(row.get("category")) != _norm(advisory.get("scope_category")):
        return False
    if _norm(row.get("entity_2")) != _norm(advisory.get("scope_entity_2")):
        return False
    if model_grain and _norm(advisory.get("scope_entity_3")):
        return _norm(row.get("entity_3")) == _norm(advisory.get("scope_entity_3"))
    return True


def _load(pack_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    with domain_con(pack_id, read_only=False) as con:
        apply_domain_schema(con)
        adv_rows = con.execute(
            """
            SELECT advisory_id, issued_at, scope_category, scope_entity_2, scope_entity_3
            FROM advisories
            """
        ).fetchall()
        rec_rows = con.execute(
            """
            SELECT received_at, occurred_at, category, entity_2, entity_3, source,
                   COALESCE(event_count, 1)
            FROM records
            WHERE provenance IS NULL OR provenance NOT IN ('inferred', 'fixture')
            """
        ).fetchall()
    advisories = [
        {
            "advisory_id": r[0],
            "issued_at": _parse_ts(r[1]),
            "scope_category": r[2],
            "scope_entity_2": r[3],
            "scope_entity_3": r[4],
        }
        for r in adv_rows
        if _parse_ts(r[1]) is not None
    ]
    records = [
        {
            "received_at": r[0],
            "occurred_at": r[1],
            "category": r[2],
            "entity_2": r[3],
            "entity_3": r[4],
            "source": r[5],
            "event_count": r[6] if r[6] is not None else 1,
        }
        for r in rec_rows
    ]
    return advisories, records


def _score_received(
    records: list[dict[str, Any]],
    *,
    pack_id: str,
    as_of: datetime | None,
    model_grain: bool,
) -> list[dict[str, Any]]:
    counts = count_weekly_slices(
        records, clock="received", as_of=as_of, model_grain=model_grain
    )
    return score_weekly_slices(counts, pack_id=pack_id)


def _first_flag(
    scored: list[dict[str, Any]],
    advisory: dict[str, Any],
    *,
    model_grain: bool,
) -> dict[str, Any] | None:
    issued = advisory["issued_at"]
    found: dict[str, Any] | None = None
    for row in scored:
        if not row.get("is_anomaly"):
            continue
        if not _slice_match(row, advisory, model_grain=model_grain):
            continue
        start = _week_monday(str(row.get("iso_week") or ""))
        if start is None or not (start < issued):
            continue
        if found is None or str(row["iso_week"]) < str(found["iso_week"]):
            found = row
    return found


def _later_recall(
    flag: dict[str, Any],
    advisories: list[dict[str, Any]],
    *,
    model_grain: bool,
) -> bool:
    start = _week_monday(str(flag.get("iso_week") or ""))
    if start is None:
        return False
    after = start + timedelta(days=7)
    for advisory in advisories:
        if not _slice_match(flag, advisory, model_grain=model_grain):
            continue
        issued = advisory.get("issued_at")
        if issued is not None and issued >= after:
            return True
    return False


def recall_lead_report(
    pack_id: str,
    *,
    model_grain: bool = False,
    run_backtests: bool = True,
    restore_live_scores: bool = True,
) -> dict[str, Any]:
    """Score every advisory as of its own receipt date. Misses stay in the list."""
    advisories, records = _load(pack_id)
    by_day: dict[str, list[dict[str, Any]]] = {}
    recalls: list[dict[str, Any]] = []
    backtests: list[dict[str, Any]] = []

    def scored_on(as_of: datetime | None) -> list[dict[str, Any]]:
        key = "all" if as_of is None else as_of.isoformat()
        if key not in by_day:
            by_day[key] = _score_received(
                records, pack_id=pack_id, as_of=as_of, model_grain=model_grain
            )
        return by_day[key]

    # One backtest per distinct receipt timestamp. The engine overwrites
    # backtest_results, so the return value is captured before the next call.
    issued_stamps = sorted({a["issued_at"] for a in advisories})
    backtest_by_stamp: dict[str, list[dict[str, Any]]] = {}
    unconfirmed: list[dict[str, Any]] = []
    # persist_scored_anomalies replaces weekly_anomalies wholesale, so every
    # exit from here on has to put the live board back. A raise in the middle
    # of the loop would otherwise leave the pack showing one recall's censored
    # series as if it were current.
    try:
        if run_backtests and issued_stamps:
            for as_of in issued_stamps:
                series = scored_on(as_of)
                persist_scored_anomalies(pack_id, series)
                rows = run_backtest(pack_id, as_of=as_of)
                backtest_by_stamp[as_of.isoformat()] = rows
                backtests.extend(rows)

        for advisory in advisories:
            issued = advisory["issued_at"]
            series = scored_on(issued)
            flag = _first_flag(series, advisory, model_grain=model_grain)
            if flag is None:
                lead = 0
                first_week = None
                count = None
                baseline = None
                z_score = None
            else:
                start = _week_monday(str(flag["iso_week"]))
                lead = _weeks_between(start, issued) if start is not None else 0
                first_week = flag["iso_week"]
                count = flag.get("record_count")
                baseline = flag.get("baseline_mean")
                z_score = flag.get("z_score")
            outcome = "hit" if lead > 0 else "miss"
            bt_rows = [
                r for r in backtest_by_stamp.get(issued.isoformat(), [])
                if r.get("advisory_id") == advisory["advisory_id"]
            ]
            matched_bt = [r for r in bt_rows if r.get("matched")]
            recalls.append(
                {
                    "advisory_id": advisory["advisory_id"],
                    "issued_at": issued.date().isoformat(),
                    "scope_category": advisory.get("scope_category"),
                    "scope_entity_2": advisory.get("scope_entity_2"),
                    "scope_entity_3": advisory.get("scope_entity_3"),
                    "first_flagged_week": first_week,
                    "lead_weeks": lead,
                    "outcome": outcome,
                    "record_count": count,
                    "baseline_mean": baseline,
                    "z_score": z_score,
                    "backtest_matched": bool(matched_bt),
                    "backtest_best_lead": max(
                        (int(r.get("lead_time_weeks") or 0) for r in matched_bt),
                        default=0,
                    ),
                }
            )

        full = scored_on(None)
        for row in full:
            if not row.get("is_anomaly"):
                continue
            if _later_recall(row, advisories, model_grain=model_grain):
                continue
            unconfirmed.append(
                {
                    "iso_week": row.get("iso_week"),
                    "category": row.get("category"),
                    "entity_2": row.get("entity_2"),
                    "entity_3": row.get("entity_3") or "",
                    "record_count": row.get("record_count"),
                    "baseline_mean": row.get("baseline_mean"),
                    "z_score": row.get("z_score"),
                    "status": "unconfirmed",
                }
            )
        unconfirmed.sort(
            key=lambda r: (str(r["iso_week"]), str(r["category"]), str(r["entity_2"]))
        )
    finally:
        if restore_live_scores:
            # The last as-of series is not the live board. Put the event clock back.
            recompute_weekly_anomalies(pack_id, model_grain=model_grain)

    hits = [r for r in recalls if r["outcome"] == "hit"]
    misses = [r for r in recalls if r["outcome"] == "miss"]
    return {
        "pack_id": pack_id,
        "model_grain": model_grain,
        "recalls": recalls,
        "hits": hits,
        "misses": misses,
        "unconfirmed": unconfirmed,
        "backtest_rows": backtests,
        "hit_count": len(hits),
        "miss_count": len(misses),
        "unconfirmed_count": len(unconfirmed),
    }


def format_recall_lead_report(report: dict[str, Any]) -> str:
    """Markdown the board and the eval can both read. Misses are not folded away."""
    lines = [
        "# Recall lead time",
        "",
        "Complaints are counted by `received_at` only, as of each recall's report-received date.",
        "A hit needs a flagged week at least one week before that date.",
        "Misses and flags with no later recall stay in this report.",
        "",
        f"Pack `{report.get('pack_id')}`. "
        f"Hits {report.get('hit_count', 0)}. "
        f"Misses {report.get('miss_count', 0)}. "
        f"Unconfirmed flags {report.get('unconfirmed_count', 0)}.",
        "",
        "## Recalls",
        "",
        "| Advisory | Issued | Slice | First flagged week | Lead weeks | Result |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in report.get("recalls") or []:
        slice_ = " ".join(
            p for p in (
                str(row.get("scope_entity_2") or ""),
                str(row.get("scope_entity_3") or ""),
                str(row.get("scope_category") or ""),
            ) if p
        )
        lines.append(
            "| {advisory} | {issued} | {slice} | {week} | {lead} | {outcome} |".format(
                advisory=row.get("advisory_id"),
                issued=row.get("issued_at"),
                slice=slice_ or "(none)",
                week=row.get("first_flagged_week") or "",
                lead=row.get("lead_weeks"),
                outcome=row.get("outcome"),
            )
        )
    lines += [
        "",
        "## Misses",
        "",
    ]
    misses = report.get("misses") or []
    if not misses:
        lines.append("No misses in this set.")
    else:
        for row in misses:
            lines.append(
                f"- {row.get('advisory_id')} issued {row.get('issued_at')} "
                f"({row.get('scope_entity_2')} {row.get('scope_category')}): miss"
            )
    lines += [
        "",
        "## Unconfirmed flags",
        "",
        "| Week | Slice | Count | Baseline | z | Status |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    flags = report.get("unconfirmed") or []
    if not flags:
        lines.append("|  |  |  |  |  | none |")
    for row in flags:
        slice_ = " ".join(
            p for p in (
                str(row.get("entity_2") or ""),
                str(row.get("entity_3") or ""),
                str(row.get("category") or ""),
            ) if p
        )
        baseline = row.get("baseline_mean")
        z_score = row.get("z_score")
        lines.append(
            "| {week} | {slice} | {count} | {base} | {z} | unconfirmed |".format(
                week=row.get("iso_week"),
                slice=slice_ or "(none)",
                count=row.get("record_count"),
                base=f"{float(baseline):.2f}" if isinstance(baseline, (int, float)) else "",
                z=f"{float(z_score):.2f}" if isinstance(z_score, (int, float)) else "",
            )
        )
    lines += [
        "",
        "## How to read this file",
        "",
        "Complaint rows are counted on `received_at` (NHTSA DATEA).",
        "Recall `issued_at` is RCDATE, the Part 573 report-received date.",
        "A hit is the first flagged week of the same category and make, at least one week before `issued_at`.",
        "The Honda and Ford sample in `eval/frontline/fixtures/recall_lead/` uses real campaign numbers. It is not a five-make download.",
        "Regenerate that sample without touching `data/domains`:",
        "",
        "```bash",
        "DOMAIN_DB_PATH=/tmp/skew-recall-lead python -m scripts.recall_lead_sample",
        "```",
        "",
        "A larger public window is manual. Point `DOMAIN_DB_PATH` at a side",
        "directory so the pilot warehouse is not replaced:",
        "",
        "```bash",
        "DOMAIN_DB_PATH=/tmp/skew-nhtsa make ingest-nhtsa N=200000 MAKES=HONDA,TOYOTA,FORD,HYUNDAI,KIA YEAR_MIN=2015 YEAR_MAX=2024",
        "DOMAIN_DB_PATH=/tmp/skew-nhtsa make ingest-recalls MAKES=HONDA,TOYOTA,FORD,HYUNDAI,KIA YEAR_MIN=2015 YEAR_MAX=2024 RCL_TYPE=V",
        "DOMAIN_DB_PATH=/tmp/skew-nhtsa python -m scripts.recall_lead_report --pack automotive_nhtsa --out reports/nhtsa/recall_lead_time_2015_2024.md",
        "```",
        "",
    ]
    return "\n".join(lines)


def write_recall_lead_report(
    pack_id: str,
    dest: str | Path,
    *,
    model_grain: bool = False,
    run_backtests: bool = True,
) -> dict[str, Any]:
    report = recall_lead_report(
        pack_id, model_grain=model_grain, run_backtests=run_backtests
    )
    path = Path(dest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_recall_lead_report(report), encoding="utf-8")
    report["path"] = str(path)
    return report


__all__ = [
    "recall_lead_report",
    "format_recall_lead_report",
    "write_recall_lead_report",
]
