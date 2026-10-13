"""Fix-effectiveness: before vs after record rates around a recorded fix."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from src.data.timeutil import to_iso_z, to_naive_utc, utc_now
from src.data.warehouse import domain_con, ops_con
from src.ids import new_ulid


def _as_dt(val: Any) -> datetime | None:
    if val is None:
        return None
    if isinstance(val, datetime):
        return to_naive_utc(val)
    try:
        return to_naive_utc(datetime.fromisoformat(str(val).replace("Z", "+00:00")))
    except ValueError:
        return None


def _ensure_fix_columns(con) -> None:
    """Older ops files created recorded_fixes before the control slice existed."""
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS recorded_fixes (
            fix_id             VARCHAR PRIMARY KEY,
            pack_id            VARCHAR NOT NULL,
            investigation_id   VARCHAR,
            entity_2           VARCHAR,
            entity_3           VARCHAR,
            category           VARCHAR,
            fixed_at           TIMESTAMP NOT NULL,
            note               TEXT,
            control_entity_2   VARCHAR,
            control_entity_3   VARCHAR
        )
        """
    )
    for ddl in (
        "ALTER TABLE recorded_fixes ADD COLUMN control_entity_2 VARCHAR",
        "ALTER TABLE recorded_fixes ADD COLUMN control_entity_3 VARCHAR",
    ):
        try:
            con.execute(ddl)
        except Exception:
            try:
                con.rollback()
            except Exception:
                pass
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS fix_measurements (
            measurement_id     VARCHAR PRIMARY KEY,
            fix_id             VARCHAR NOT NULL,
            measured_at        TIMESTAMP NOT NULL,
            status             VARCHAR NOT NULL,
            before_rate        DOUBLE,
            after_rate         DOUBLE,
            before_count       INTEGER,
            after_count        INTEGER,
            diff_in_diff       DOUBLE,
            event_weeks_after  INTEGER,
            window_days        INTEGER
        )
        """
    )


def record_fix(
    *,
    pack_id: str,
    fixed_at: datetime,
    category: str | None = None,
    entity_2: str | None = None,
    entity_3: str | None = None,
    investigation_id: str | None = None,
    note: str | None = None,
    control_entity_2: str | None = None,
    control_entity_3: str | None = None,
) -> dict[str, Any]:
    """Persist a fix on an affected slice. Returns the stored row."""
    fid = "fix_" + new_ulid()
    ts = to_naive_utc(fixed_at)
    with ops_con() as con:
        _ensure_fix_columns(con)
        con.execute(
            """
            INSERT INTO recorded_fixes
            (fix_id, pack_id, investigation_id, entity_2, entity_3, category,
             fixed_at, note, control_entity_2, control_entity_3)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                fid,
                pack_id,
                investigation_id,
                entity_2,
                entity_3,
                category,
                ts,
                note,
                control_entity_2,
                control_entity_3,
            ],
        )
    return {
        "fix_id": fid,
        "pack_id": pack_id,
        "investigation_id": investigation_id,
        "entity_2": entity_2,
        "entity_3": entity_3,
        "category": category,
        "fixed_at": to_iso_z(ts),
        "note": note,
        "control_entity_2": control_entity_2,
        "control_entity_3": control_entity_3,
    }


def measure_effectiveness(
    *,
    pack_id: str,
    fixed_at: datetime,
    category: str | None = None,
    entity_2: str | None = None,
    entity_3: str | None = None,
    window_days: int = 30,
    control_entity_2: str | None = None,
    min_post_weeks: int = 8,
    exposure_units: float | None = None,
    control_exposure_units: float | None = None,
) -> dict[str, Any]:
    """Count matching ``records`` in [fixed_at-window, fixed_at) vs (fixed_at, fixed_at+window].

    Rate = count / window_days when unnormalized, or (count / exposure_units) * 1000
    when exposure is provided. Does not use fixture literals.

    Causal honesty (audit 8.4): a raw before/after drop can be seasonality,
    a sales dip, or reporting lag. Pass ``control_entity_2`` (a sibling
    slice the fix should NOT move) for a diff-in-diff estimate; results
    with a post window shorter than ``min_post_weeks`` event-clock weeks
    are ``inconclusive``, never "resolved".
    """
    if window_days < 1:
        raise ValueError("window_days must be >= 1")
    pivot = to_naive_utc(fixed_at)
    before_start = pivot - timedelta(days=window_days)
    after_end = pivot + timedelta(days=window_days)

    with domain_con(pack_id, read_only=True) as con:
        rows = con.execute(
            "SELECT occurred_at, received_at, category, entity_2, entity_3 FROM records"
        ).fetchall()
        cols = [d[0] for d in con.description]
        records = [dict(zip(cols, r)) for r in rows]

    def _match(r: dict[str, Any]) -> bool:
        if category and str(r.get("category") or "") != category:
            return False
        if entity_2 and str(r.get("entity_2") or "") != entity_2:
            return False
        if entity_3 and str(r.get("entity_3") or "") != entity_3:
            return False
        return True

    def _match_control(r: dict[str, Any], control_ent: str) -> bool:
        # Same category, sibling entity the fix should not move.
        if category and str(r.get("category") or "") != category:
            return False
        return str(r.get("entity_2") or "") == control_ent

    before_n = after_n = 0
    ctrl_before_n = ctrl_after_n = 0
    lags: list[int] = []
    latest_rec_ts: datetime | None = None

    for r in records:
        ts = _as_dt(r.get("received_at"))
        occ_ts = _as_dt(r.get("occurred_at"))
        if ts is None:
            continue
        if latest_rec_ts is None or ts > latest_rec_ts:
            latest_rec_ts = ts
        if occ_ts and ts >= occ_ts:
            lags.append(max(0, int((ts - occ_ts).total_seconds() / 86400.0)))
        if _match(r):
            if before_start <= ts < pivot:
                before_n += 1
            elif pivot < ts <= after_end:
                after_n += 1
        if control_entity_2 and _match_control(r, control_entity_2):
            if before_start <= ts < pivot:
                ctrl_before_n += 1
            elif pivot < ts <= after_end:
                ctrl_after_n += 1

    # Exposure normalization (audit 3.2): rate per 1,000 units when exposure is present
    normalized = bool(exposure_units and exposure_units > 0)
    if normalized:
        assert exposure_units is not None
        before_rate = (before_n / float(exposure_units)) * 1000.0
        after_rate = (after_n / float(exposure_units)) * 1000.0
        rate_unit = "per_1000_exposure_units"
    else:
        before_rate = before_n / float(window_days)
        after_rate = after_n / float(window_days)
        rate_unit = "per_day"

    # Post-period maturity (audit 8.4): fewer than min_post_weeks of
    # event-clock coverage after the fix → inconclusive, not resolved.
    post_weeks = window_days / 7.0
    conclusive = post_weeks >= min_post_weeks
    improved = after_rate < before_rate

    # Diff-in-diff vs the control slice: did the fixed slice fall FASTER
    # than background? None when no control was supplied.
    did: float | None = None
    if control_entity_2:
        if control_exposure_units and control_exposure_units > 0:
            ctrl_before = (ctrl_before_n / float(control_exposure_units)) * 1000.0
            ctrl_after = (ctrl_after_n / float(control_exposure_units)) * 1000.0
        else:
            ctrl_before = ctrl_before_n / float(window_days)
            ctrl_after = ctrl_after_n / float(window_days)
        did = (after_rate - before_rate) - (ctrl_after - ctrl_before)

    # Reporting lag / nowcasting (audit 4.1): estimate empirical lag distribution
    reporting_lag_completion = 1.0
    reporting_lag_immature = False
    nowcasted_after_count = float(after_n)
    if lags and latest_rec_ts:
        elapsed_days = max(1, int((latest_rec_ts - pivot).total_seconds() / 86400.0))
        cum_completed = sum(1 for lag in lags if lag <= elapsed_days) / float(len(lags))
        reporting_lag_completion = round(min(1.0, max(0.01, cum_completed)), 3)
        reporting_lag_immature = reporting_lag_completion < 0.5
        if 0.05 <= reporting_lag_completion < 1.0:
            nowcasted_after_count = round(after_n / reporting_lag_completion, 2)

    return {
        "pack_id": pack_id,
        "category": category,
        "entity_2": entity_2,
        "entity_3": entity_3,
        "fixed_at": to_iso_z(pivot),
        "window_days": window_days,
        "before_count": before_n,
        "after_count": after_n,
        "before_rate": before_rate,
        "after_rate": after_rate,
        "improved": improved and conclusive,
        "conclusive": conclusive,
        "min_post_weeks": min_post_weeks,
        "control_entity_2": control_entity_2,
        "control_before_count": ctrl_before_n,
        "control_after_count": ctrl_after_n,
        "diff_in_diff": did,
        "normalized": normalized,
        "exposure_units": exposure_units,
        "rate_unit": rate_unit,
        "reporting_lag_completion": reporting_lag_completion,
        "reporting_lag_immature": reporting_lag_immature,
        "nowcasted_after_count": nowcasted_after_count,
        "measured_at": to_iso_z(utc_now()),
    }


def reopen_window_days() -> int:
    """Reopen definition (audit 8.4): same entity_key + category recurring
    within 90 days of the fix is a reopen, not a new issue."""
    return 90


def is_reopen(
    pack_id: str,
    *,
    entity_key: str | None,
    category: str | None,
    fixed_at: datetime,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Check whether the fixed slice recurred inside the reopen window."""
    from src.data.timeutil import utc_now as _now

    pivot = to_naive_utc(fixed_at)
    end = to_naive_utc(now) if now is not None else _now().replace(tzinfo=None)
    if not entity_key or not category:
        return {"reopened": False, "reason": "no entity_key/category to match"}
    with domain_con(pack_id, read_only=True) as con:
        try:
            rows = con.execute(
                """
                SELECT received_at FROM records
                WHERE category = ? AND entity_key = ? AND received_at > ?
                  AND (provenance IS NULL OR provenance NOT IN ('inferred', 'fixture'))
                """,
                [category, entity_key, pivot],
            ).fetchall()
        except Exception:
            return {"reopened": False, "reason": "query failed"}
    hits = [r[0] for r in rows if r[0] is not None]
    window_end = pivot + timedelta(days=reopen_window_days())
    recurred = [h for h in hits if pivot < _as_dt(h) <= min(window_end, end)]
    return {
        "reopened": bool(recurred),
        "recurrences": len(recurred),
        "window_days": reopen_window_days(),
        "entity_key": entity_key,
        "category": category,
    }


# A fix stays waiting until this many distinct event-weeks have been
# observed after fixed_at. The rate window is the same length.
MIN_EVENT_WEEKS = 8
FIX_WINDOW_DAYS = MIN_EVENT_WEEKS * 7


def _iso_week(ts: datetime) -> str:
    year, week, _ = ts.isocalendar()
    return f"{year}-W{week:02d}"


def _event_time(row: dict[str, Any]) -> datetime | None:
    return _as_dt(row.get("occurred_at")) or _as_dt(row.get("received_at"))


def _slice_rows(pack_id: str) -> list[dict[str, Any]]:
    with domain_con(pack_id, read_only=True) as con:
        rows = con.execute(
            """
            SELECT occurred_at, received_at, category, entity_2, entity_3, entity_key
            FROM records
            WHERE provenance IS NULL OR provenance NOT IN ('inferred', 'fixture')
            """
        ).fetchall()
    return [
        {
            "occurred_at": r[0],
            "received_at": r[1],
            "category": r[2],
            "entity_2": r[3],
            "entity_3": r[4],
            "entity_key": r[5],
        }
        for r in rows
    ]


def evaluate_fix(
    *,
    pack_id: str,
    fixed_at: datetime,
    category: str | None = None,
    entity_2: str | None = None,
    entity_3: str | None = None,
    control_entity_2: str | None = None,
    control_entity_3: str | None = None,
    records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Before/after rates once eight event-weeks exist after the fix.

    Fewer than eight distinct event-weeks after ``fixed_at`` stays
    ``waiting``. Eight or more reports ``improved`` or ``not improved``.
    The control slice is a sibling make, or a sibling model on the same
    make. Fixture and inferred rows are not evidence.
    """
    pivot = to_naive_utc(fixed_at)
    window = timedelta(days=FIX_WINDOW_DAYS)
    before_start = pivot - window
    after_end = pivot + window
    try:
        rows = records if records is not None else _slice_rows(pack_id)
    except Exception:
        rows = []

    def _treated(row: dict[str, Any]) -> bool:
        if category and str(row.get("category") or "") != category:
            return False
        if entity_2 and str(row.get("entity_2") or "") != entity_2:
            return False
        if entity_3 and str(row.get("entity_3") or "") != entity_3:
            return False
        return True

    def _control(row: dict[str, Any]) -> bool:
        if category and str(row.get("category") or "") != category:
            return False
        if control_entity_2:
            return str(row.get("entity_2") or "") == control_entity_2
        if control_entity_3:
            if entity_2 and str(row.get("entity_2") or "") != entity_2:
                return False
            return str(row.get("entity_3") or "") == control_entity_3
        return False

    before_n = after_n = 0
    ctrl_before_n = ctrl_after_n = 0
    weeks: set[str] = set()
    for row in rows:
        ts = _event_time(row)
        if ts is None:
            continue
        if _treated(row):
            if ts > pivot:
                weeks.add(_iso_week(ts))
            if before_start <= ts < pivot:
                before_n += 1
            elif pivot < ts <= after_end:
                after_n += 1
        if (control_entity_2 or control_entity_3) and _control(row):
            if before_start <= ts < pivot:
                ctrl_before_n += 1
            elif pivot < ts <= after_end:
                ctrl_after_n += 1

    days = float(FIX_WINDOW_DAYS)
    before_rate = before_n / days
    after_rate = after_n / days
    did: float | None = None
    if control_entity_2 or control_entity_3:
        did = (after_rate - before_rate) - (
            (ctrl_after_n / days) - (ctrl_before_n / days)
        )
    event_weeks = len(weeks)
    if event_weeks < MIN_EVENT_WEEKS:
        status = "waiting"
    elif after_rate < before_rate:
        status = "improved"
    else:
        status = "not improved"
    return {
        "pack_id": pack_id,
        "category": category,
        "entity_2": entity_2,
        "entity_3": entity_3,
        "control_entity_2": control_entity_2,
        "control_entity_3": control_entity_3,
        "fixed_at": to_iso_z(pivot),
        "window_days": FIX_WINDOW_DAYS,
        "event_weeks_after": event_weeks,
        "before_count": before_n,
        "after_count": after_n,
        "before_rate": before_rate,
        "after_rate": after_rate,
        "diff_in_diff": did,
        "status": status,
        "improved": status == "improved",
        "conclusive": status != "waiting",
        "measured_at": to_iso_z(utc_now()),
    }


def store_measurement(fix_id: str, result: dict[str, Any]) -> dict[str, Any]:
    """Append one measurement. Earlier rows stay, so a weekly job keeps history."""
    mid = "fxm_" + new_ulid()
    with ops_con() as con:
        _ensure_fix_columns(con)
        con.execute(
            """
            INSERT INTO fix_measurements (
                measurement_id, fix_id, measured_at, status,
                before_rate, after_rate, before_count, after_count,
                diff_in_diff, event_weeks_after, window_days
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                mid,
                fix_id,
                utc_now(),
                result.get("status") or "waiting",
                result.get("before_rate"),
                result.get("after_rate"),
                result.get("before_count"),
                result.get("after_count"),
                result.get("diff_in_diff"),
                result.get("event_weeks_after"),
                result.get("window_days"),
            ],
        )
    stored = dict(result)
    stored["measurement_id"] = mid
    stored["fix_id"] = fix_id
    return stored


def measure_recorded_fixes(pack_id: str | None = None) -> dict[str, Any]:
    """Re-measure every recorded fix and store each result."""
    with ops_con() as con:
        _ensure_fix_columns(con)
        try:
            rows = con.execute(
                """
                SELECT fix_id, pack_id, category, entity_2, entity_3, fixed_at,
                       control_entity_2, control_entity_3
                FROM recorded_fixes
                ORDER BY fixed_at
                """
            ).fetchall()
        except Exception:
            rows = []
    results: list[dict[str, Any]] = []
    for fix_id, pack, category, e2, e3, fixed_at, ctrl2, ctrl3 in rows:
        if pack_id and str(pack) != pack_id:
            continue
        if fixed_at is None:
            continue
        result = evaluate_fix(
            pack_id=str(pack),
            fixed_at=fixed_at,
            category=str(category or "") or None,
            entity_2=str(e2 or "") or None,
            entity_3=str(e3 or "") or None,
            control_entity_2=str(ctrl2 or "") or None,
            control_entity_3=str(ctrl3 or "") or None,
        )
        results.append(store_measurement(str(fix_id), result))
    return {"ok": True, "measured": len(results), "results": results}


def latest_measurement(fix_id: str) -> dict[str, Any] | None:
    with ops_con(read_only=True) as con:
        try:
            row = con.execute(
                """
                SELECT measurement_id, status, before_rate, after_rate,
                       before_count, after_count, diff_in_diff,
                       event_weeks_after, window_days, measured_at
                FROM fix_measurements
                WHERE fix_id = ?
                ORDER BY measured_at DESC
                LIMIT 1
                """,
                [fix_id],
            ).fetchone()
        except Exception:
            return None
    if not row:
        return None
    measured_at = row[9]
    return {
        "measurement_id": row[0],
        "fix_id": fix_id,
        "status": row[1],
        "before_rate": row[2],
        "after_rate": row[3],
        "before_count": row[4],
        "after_count": row[5],
        "diff_in_diff": row[6],
        "event_weeks_after": row[7],
        "window_days": row[8],
        "measured_at": to_iso_z(measured_at) if isinstance(measured_at, datetime) else measured_at,
        "improved": row[1] == "improved",
        "conclusive": row[1] != "waiting",
    }


def entity_keys_before_fix(
    pack_id: str,
    *,
    category: str | None,
    entity_2: str | None,
    fixed_at: datetime,
    limit: int = 20,
) -> list[str]:
    """Keys seen on the slice before the fix. Recurrence of one of these is a reopen."""
    pivot = to_naive_utc(fixed_at)
    try:
        with domain_con(pack_id, read_only=True) as con:
            rows = con.execute(
                """
                SELECT DISTINCT entity_key FROM records
                WHERE entity_key IS NOT NULL AND entity_key <> ''
                  AND (? IS NULL OR category = ?)
                  AND (? IS NULL OR entity_2 = ?)
                  AND received_at < ?
                  AND (provenance IS NULL OR provenance NOT IN ('inferred', 'fixture'))
                LIMIT ?
                """,
                [category, category, entity_2, entity_2, pivot, int(limit)],
            ).fetchall()
    except Exception:
        return []
    return [str(r[0]) for r in rows if r and r[0]]


__all__ = [
    "record_fix",
    "measure_effectiveness",
    "evaluate_fix",
    "store_measurement",
    "measure_recorded_fixes",
    "latest_measurement",
    "entity_keys_before_fix",
    "is_reopen",
    "reopen_window_days",
    "MIN_EVENT_WEEKS",
    "FIX_WINDOW_DAYS",
]
