"""Four-eyes approval on high-impact actions (feature #38)."""

from __future__ import annotations

import json
from typing import Any

from src.data.timeutil import utc_now
from src.data.warehouse import ops_con
from src.ids import new_ulid
from src.ledger import AgentAction, record_action

HIGH_IMPACT = frozenset(
    {
        "open_investigation",
        "close_p1_case",
        "send_customer_followup",
        "export_audit_bundle",
    }
)


def _ensure(con) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS approvals (
            approval_id VARCHAR PRIMARY KEY,
            action_type VARCHAR NOT NULL,
            resource_id VARCHAR,
            requested_by VARCHAR,
            status VARCHAR,
            reviewer VARCHAR,
            reason VARCHAR,
            created_at TIMESTAMP DEFAULT current_timestamp,
            decided_at TIMESTAMP,
            payload_json VARCHAR
        )
        """
    )


def request_approval(
    action_type: str,
    *,
    resource_id: str,
    requested_by: str,
    reason: str = "",
    payload: dict[str, Any] | None = None,
    interaction_id: str | None = None,
) -> dict[str, Any]:
    if action_type not in HIGH_IMPACT:
        return {
            "required": False,
            "action_type": action_type,
            "status": "auto_allowed",
            "message": "not a high-impact action",
        }
    aid = f"appr_{new_ulid()}"
    with ops_con() as con:
        _ensure(con)
        con.execute(
            """
            INSERT INTO approvals
              (approval_id, action_type, resource_id, requested_by, status, reason, payload_json)
            VALUES (?, ?, ?, ?, 'pending', ?, ?)
            """,
            [
                aid,
                action_type,
                resource_id,
                requested_by,
                reason,
                json.dumps(payload or {}),
            ],
        )
    if interaction_id:
        record_action(
            AgentAction(
                interaction_id=interaction_id,
                agent="four_eyes",
                action_type="approval_requested",
                input_summary=action_type,
                output_summary=aid,
                evidence_ids=[resource_id],
            )
        )
    return {
        "required": True,
        "approval_id": aid,
        "status": "pending",
        "action_type": action_type,
        "resource_id": resource_id,
        "requested_by": requested_by,
    }


def decide_approval(
    approval_id: str,
    *,
    reviewer: str,
    approve: bool,
    interaction_id: str | None = None,
) -> dict[str, Any]:
    status = "approved" if approve else "rejected"
    if not reviewer or reviewer.strip() == "":
        raise ValueError("reviewer identity required")
    with ops_con() as con:
        _ensure(con)
        row = con.execute(
            "SELECT requested_by, action_type, resource_id, status FROM approvals WHERE approval_id = ?",
            [approval_id],
        ).fetchone()
        if not row:
            raise KeyError(approval_id)
        if row[0] and row[0] == reviewer:
            raise ValueError("four-eyes: reviewer must differ from requester")
        if row[3] != "pending":
            return {"approval_id": approval_id, "status": row[3], "already_decided": True}
        con.execute(
            """
            UPDATE approvals SET status = ?, reviewer = ?, decided_at = ?
            WHERE approval_id = ?
            """,
            [status, reviewer, utc_now(), approval_id],
        )
    if interaction_id:
        record_action(
            AgentAction(
                interaction_id=interaction_id,
                agent="four_eyes",
                action_type="approval_decided",
                input_summary=approval_id,
                output_summary=status,
                evidence_ids=[row[2] or ""],
            )
        )
    return {
        "approval_id": approval_id,
        "status": status,
        "reviewer": reviewer,
        "action_type": row[1],
        "resource_id": row[2],
    }


def is_approved(action_type: str, resource_id: str) -> bool:
    with ops_con(read_only=True) as con:
        try:
            row = con.execute(
                """
                SELECT 1 FROM approvals
                WHERE action_type = ? AND resource_id = ? AND status = 'approved'
                LIMIT 1
                """,
                [action_type, resource_id],
            ).fetchone()
            return bool(row)
        except Exception:
            return False
