"""Configurable alert rules engine (feature #23).

Rules live in ops table ``alert_rules`` (JSON conditions). Evaluator is pure and
testable; side effects (open investigation / fire_alert) are explicit.
"""

from __future__ import annotations

import json
from typing import Any

from src.data.timeutil import utc_now
from src.data.warehouse import ops_con
from src.ids import new_ulid


def ensure_alert_rules_table() -> None:
    with ops_con() as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS alert_rules (
                rule_id            VARCHAR PRIMARY KEY,
                pack_id            VARCHAR,
                name               VARCHAR NOT NULL,
                enabled            BOOLEAN NOT NULL DEFAULT TRUE,
                min_cases          INTEGER NOT NULL DEFAULT 5,
                window_days        INTEGER NOT NULL DEFAULT 7,
                min_severity       VARCHAR NOT NULL DEFAULT 'Medium',
                action             VARCHAR NOT NULL DEFAULT 'alert',  -- alert | investigation | both
                created_at         TIMESTAMP NOT NULL
            )
            """
        )


_SEV_RANK = {"Low": 1, "Medium": 2, "Critical": 3}


def create_rule(
    *,
    name: str,
    pack_id: str | None = None,
    min_cases: int = 5,
    window_days: int = 7,
    min_severity: str = "Medium",
    action: str = "alert",
    enabled: bool = True,
) -> dict[str, Any]:
    ensure_alert_rules_table()
    rid = "rule_" + new_ulid()
    now = utc_now()
    with ops_con() as con:
        con.execute(
            """
            INSERT INTO alert_rules
            (rule_id, pack_id, name, enabled, min_cases, window_days, min_severity, action, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                rid,
                pack_id,
                name,
                enabled,
                int(min_cases),
                int(window_days),
                min_severity,
                action,
                now,
            ],
        )
    return get_rule(rid)  # type: ignore[return-value]


def get_rule(rule_id: str) -> dict[str, Any] | None:
    ensure_alert_rules_table()
    with ops_con(read_only=True) as con:
        cur = con.execute("SELECT * FROM alert_rules WHERE rule_id = ?", [rule_id])
        row = cur.fetchone()
        if not row:
            return None
        cols = [d[0] for d in cur.description]
        return dict(zip(cols, row))


def list_rules(*, pack_id: str | None = None) -> list[dict[str, Any]]:
    ensure_alert_rules_table()
    with ops_con(read_only=True) as con:
        if pack_id:
            cur = con.execute(
                "SELECT * FROM alert_rules WHERE pack_id IS NULL OR pack_id = ? ORDER BY created_at",
                [pack_id],
            )
        else:
            cur = con.execute("SELECT * FROM alert_rules ORDER BY created_at")
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def evaluate_cluster_rule(
    rule: dict[str, Any],
    *,
    pack_id: str,
    cluster_id: int,
    case_count: int,
    max_severity: str = "Low",
) -> dict[str, Any]:
    """Pure evaluation against a cluster snapshot."""
    if not rule.get("enabled", True):
        return {"triggered": False, "reason": "disabled"}
    if rule.get("pack_id") and rule["pack_id"] != pack_id:
        return {"triggered": False, "reason": "pack_mismatch"}
    need = int(rule.get("min_cases") or 5)
    if case_count < need:
        return {"triggered": False, "reason": f"cases {case_count}<{need}"}
    need_sev = _SEV_RANK.get(str(rule.get("min_severity") or "Medium"), 2)
    got_sev = _SEV_RANK.get(str(max_severity or "Low"), 1)
    if got_sev < need_sev:
        return {"triggered": False, "reason": f"severity {max_severity}<{rule.get('min_severity')}"}
    return {
        "triggered": True,
        "reason": "ok",
        "action": rule.get("action") or "alert",
        "rule_id": rule.get("rule_id"),
        "cluster_id": cluster_id,
        "pack_id": pack_id,
        "case_count": case_count,
        "max_severity": max_severity,
    }


async def apply_triggered_rule(eval_result: dict[str, Any], *, title: str = "") -> dict[str, Any]:
    """Fire alert and/or open investigation for a triggered evaluation."""
    if not eval_result.get("triggered"):
        return {"applied": False, **eval_result}
    action = eval_result.get("action") or "alert"
    pack_id = eval_result.get("pack_id")
    cluster_id = int(eval_result.get("cluster_id") or 0)
    out: dict[str, Any] = {"applied": True, "alert": False, "investigation_id": None}

    if action in ("alert", "both"):
        try:
            from src.frontline.alerts import fire_alert

            await fire_alert(
                event="early_warning_threshold",
                summary=(
                    title
                    or f"Rule {eval_result.get('rule_id')}: cluster {cluster_id} "
                    f"has {eval_result.get('case_count')} cases"
                ),
                ref_id=str(cluster_id),
                pack_id=pack_id,
                extra=eval_result,
            )
            out["alert"] = True
        except Exception as e:
            out["alert_error"] = f"{type(e).__name__}:{e}"

    if action in ("investigation", "both"):
        try:
            from src.ids import new_ulid
            from src.data.timeutil import utc_now

            inv_id = "inv_" + new_ulid()[:12]
            now = utc_now()
            with ops_con() as con:
                con.execute(
                    """
                    INSERT INTO investigations
                    (investigation_id, pack_id, cluster_id, title, status, opened_at, last_case_at, case_count)
                    VALUES (?, ?, ?, ?, 'open', ?, ?, ?)
                    """,
                    [
                        inv_id,
                        pack_id,
                        cluster_id,
                        title or f"Rule-triggered cluster {cluster_id}",
                        now,
                        now,
                        int(eval_result.get("case_count") or 0),
                    ],
                )
            out["investigation_id"] = inv_id
        except Exception as e:
            out["investigation_error"] = f"{type(e).__name__}:{e}"
    return out


def fetch_cluster_metrics(
    pack_id: str,
    cluster_id: int,
    *,
    window_days: int | None = None,
) -> tuple[int, str]:
    """Fetch live case count and maximum severity for cluster from the ops DB (FU08).

    Severity ordering: Critical (4) > High (3) > Medium (2) > Low (1) > 0.
    If window_days is specified, cases older than window_days are excluded.
    """
    from datetime import timedelta
    from typing import Any

    from src.data.timeutil import utc_now

    ensure_alert_rules_table()
    params: list[Any] = [pack_id, cluster_id, str(cluster_id)]
    time_filter = ""
    if window_days is not None and int(window_days) > 0:
        cutoff = utc_now().replace(tzinfo=None) - timedelta(days=int(window_days))
        time_filter = "AND created_at >= ?"
        params.append(cutoff)

    with ops_con(read_only=True) as con:
        cur = con.execute(
            f"""
            SELECT count(*),
                   coalesce(
                       max(
                           CASE upper(trim(coalesce(severity, 'Low')))
                               WHEN 'CRITICAL' THEN 4
                               WHEN 'HIGH' THEN 3
                               WHEN 'MEDIUM' THEN 2
                               WHEN 'LOW' THEN 1
                               ELSE 0
                           END
                       ), 0
                   )
            FROM cases
            WHERE (pack_id = ? OR pack_id IS NULL)
              AND (cluster_match_id = ? OR cluster_match_id = ?)
              {time_filter}
            """,
            params,
        )
        row = cur.fetchone()
        if row and row[0] is not None and row[0] > 0:
            rank = int(row[1] or 0)
            rank_map = {4: "Critical", 3: "High", 2: "Medium", 1: "Low"}
            return int(row[0]), rank_map.get(rank, "Low")
    return 0, "Low"


__all__ = [
    "ensure_alert_rules_table",
    "create_rule",
    "get_rule",
    "list_rules",
    "evaluate_cluster_rule",
    "apply_triggered_rule",
    "fetch_cluster_metrics",
]
