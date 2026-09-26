"""Age-based retention for unbounded ops tables (pilot hygiene).

Deletes rows older than a configurable retention window. Safe to re-run.
Preserves audit history: agent_actions is NEVER hard-deleted (retention
policy declares 2,555 days with tombstone-only handling to preserve hash-chain
integrity). interaction_turns respects the 90-day retention policy (or
pack/class-specific override).
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any

from src.compliance.retention import retention_days
from src.data.timeutil import utc_now
from src.data.warehouse import ops_con

# Default retention days (env override: FRONTLINE_PRUNE_DAYS)
_DEFAULT_DAYS = 30

# Tables / column / data_class mapping used for age filter
# NOTE: agent_actions is NEVER hard-deleted; audit history is tombstone-only.
_PRUNE_TARGETS: list[tuple[str, str, str | None]] = [
    ("alert_dedup", "fired_at", "alerts"),
    ("alert_dead_letter", "created_at", "alerts"),
    ("connector_deliveries", "created_at", None),
    ("risk_snapshots", "ts", None),
    ("interaction_turns", "ts", "turns"),
]


def prune_days(data_class: str | None = None) -> int:
    if data_class:
        return retention_days(data_class)
    raw = os.getenv("FRONTLINE_PRUNE_DAYS", "").strip()
    if raw.isdigit():
        return max(1, int(raw))
    return _DEFAULT_DAYS


def prune_ops_tables(
    *,
    older_than_days: int | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Delete aged rows from growth tables. Returns counts deleted per table.

    Respects retention policies per data class:
    - agent_actions is NEVER hard-deleted (audit integrity preserved).
    - interaction_turns uses retention_days('turns') (default 90d) unless overridden.
    - alert tables use retention_days('alerts') (default 180d) unless overridden.
    """
    ref_time = now or utc_now()
    from src.security.sql_ident import safe_column, safe_table

    deleted: dict[str, int] = {}
    cutoffs: dict[str, str] = {}
    with ops_con() as con:
        for table, col, data_class in _PRUNE_TARGETS:
            if older_than_days is not None:
                days = older_than_days
            elif data_class is not None:
                days = retention_days(data_class)
            else:
                days = prune_days()

            cutoff = ref_time - timedelta(days=days)
            if cutoff.tzinfo is not None:
                cutoff = cutoff.astimezone(timezone.utc).replace(tzinfo=None)
            cutoffs[table] = cutoff.isoformat() + "Z"

            try:
                t = safe_table(table)
                c = safe_column(col)
                before = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                con.execute(
                    f"DELETE FROM {t} WHERE {c} IS NOT NULL AND {c} < ?",
                    [cutoff],
                )
                after = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                deleted[table] = int(before) - int(after)
            except Exception as e:
                deleted[table] = -1  # table missing or column mismatch
                deleted[f"{table}_error"] = f"{type(e).__name__}:{e}"

    default_cutoff = ref_time - timedelta(
        days=older_than_days if older_than_days is not None else _DEFAULT_DAYS
    )
    if default_cutoff.tzinfo is not None:
        default_cutoff = default_cutoff.astimezone(timezone.utc).replace(tzinfo=None)

    return {
        "cutoff": default_cutoff.isoformat() + "Z",
        "cutoffs": cutoffs,
        "older_than_days": older_than_days if older_than_days is not None else _DEFAULT_DAYS,
        "deleted": deleted,
    }


__all__ = ["prune_ops_tables", "prune_days"]
