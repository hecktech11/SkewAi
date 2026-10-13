"""Investigation workspace: assignee, SLA, hypotheses, comments.

Thin store over ops DuckDB. Write then read through these functions is the
shipped path the tests drive.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from src.data.timeutil import to_iso_z, utc_now
from src.data.warehouse import ops_con
from src.ids import new_ulid

HYPOTHESIS_STATUSES = frozenset({"open", "confirmed", "rejected", "superseded"})


def _ensure_workspace_schema(con) -> None:
    for stmt in (
        "ALTER TABLE investigations ADD COLUMN assignee VARCHAR",
        "ALTER TABLE investigations ADD COLUMN sla_due_at TIMESTAMP",
        "ALTER TABLE investigations ADD COLUMN slice_category VARCHAR",
        "ALTER TABLE investigations ADD COLUMN slice_entity_2 VARCHAR",
        "ALTER TABLE investigations ADD COLUMN slice_entity_3 VARCHAR",
    ):
        try:
            con.execute(stmt)
        except Exception:
            pass
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS investigation_hypotheses (
            hypothesis_id      VARCHAR PRIMARY KEY,
            investigation_id   VARCHAR NOT NULL,
            body               TEXT NOT NULL,
            status             VARCHAR NOT NULL DEFAULT 'open',
            created_at         TIMESTAMP NOT NULL
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS investigation_comments (
            comment_id       VARCHAR PRIMARY KEY,
            investigation_id VARCHAR NOT NULL,
            author           VARCHAR NOT NULL,
            body             TEXT NOT NULL,
            created_at       TIMESTAMP NOT NULL
        )
        """
    )
    try:
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_inv_comments_inv "
            "ON investigation_comments (investigation_id, created_at)"
        )
    except Exception:
        pass


def update_investigation(
    investigation_id: str,
    *,
    assignee: str | None = None,
    sla_due_at: datetime | None = None,
) -> dict[str, Any]:
    """Persist assignee and/or SLA. Raises LookupError if missing."""
    with ops_con() as con:
        _ensure_workspace_schema(con)
        exists = con.execute(
            "SELECT 1 FROM investigations WHERE investigation_id = ?",
            [investigation_id],
        ).fetchone()
        if not exists:
            raise LookupError(f"investigation not found: {investigation_id}")
        if assignee is not None:
            con.execute(
                "UPDATE investigations SET assignee = ? WHERE investigation_id = ?",
                [assignee, investigation_id],
            )
        if sla_due_at is not None:
            con.execute(
                "UPDATE investigations SET sla_due_at = ? WHERE investigation_id = ?",
                [sla_due_at, investigation_id],
            )
    return get_investigation(investigation_id)


def add_hypothesis(
    investigation_id: str,
    body: str,
    *,
    status: str = "open",
) -> dict[str, Any]:
    status = (status or "open").strip().lower()
    if status not in HYPOTHESIS_STATUSES:
        raise ValueError(f"unknown hypothesis status: {status}")
    hid = "hyp_" + new_ulid()
    with ops_con() as con:
        _ensure_workspace_schema(con)
        exists = con.execute(
            "SELECT 1 FROM investigations WHERE investigation_id = ?",
            [investigation_id],
        ).fetchone()
        if not exists:
            raise LookupError(f"investigation not found: {investigation_id}")
        con.execute(
            """
            INSERT INTO investigation_hypotheses
            (hypothesis_id, investigation_id, body, status, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            [hid, investigation_id, body.strip(), status, utc_now()],
        )
    return get_investigation(investigation_id)


def add_comment(
    investigation_id: str,
    body: str,
    *,
    author: str = "operator",
) -> dict[str, Any]:
    """Append an engineer comment to an investigation. Raises LookupError if missing."""
    text = (body or "").strip()
    if not text:
        raise ValueError("comment body required")
    cid = "icm_" + new_ulid()
    with ops_con() as con:
        _ensure_workspace_schema(con)
        exists = con.execute(
            "SELECT 1 FROM investigations WHERE investigation_id = ?",
            [investigation_id],
        ).fetchone()
        if not exists:
            raise LookupError(f"investigation not found: {investigation_id}")
        con.execute(
            """
            INSERT INTO investigation_comments
            (comment_id, investigation_id, author, body, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            [cid, investigation_id, (author or "operator")[:80], text[:2000], utc_now()],
        )
    return get_investigation(investigation_id)


def list_comments(investigation_id: str) -> list[dict[str, Any]]:
    """Engineer comments, oldest first."""
    with ops_con(read_only=True) as con:
        try:
            _ensure_workspace_schema(con)
            cur = con.execute(
                """
                SELECT comment_id, author, body, created_at
                FROM investigation_comments
                WHERE investigation_id = ?
                ORDER BY created_at
                """,
                [investigation_id],
            )
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        except Exception:
            return []
    for r in rows:
        if isinstance(r.get("created_at"), datetime):
            r["created_at"] = to_iso_z(r["created_at"])
    return rows


def get_investigation(investigation_id: str) -> dict[str, Any]:
    with ops_con(read_only=True) as con:
        try:
            cur = con.execute(
                """
                SELECT investigation_id, pack_id, cluster_id, title, status,
                       opened_at, last_case_at, case_count, assignee, sla_due_at
                FROM investigations WHERE investigation_id = ?
                """,
                [investigation_id],
            )
        except Exception:
            cur = con.execute(
                """
                SELECT investigation_id, pack_id, cluster_id, title, status,
                       opened_at, last_case_at, case_count
                FROM investigations WHERE investigation_id = ?
                """,
                [investigation_id],
            )
        row = cur.fetchone()
        if not row:
            raise LookupError(f"investigation not found: {investigation_id}")
        cols = [d[0] for d in cur.description]
        inv = dict(zip(cols, row))
        hyps: list[dict[str, Any]] = []
        try:
            hcur = con.execute(
                """
                SELECT hypothesis_id, body, status, created_at
                FROM investigation_hypotheses
                WHERE investigation_id = ?
                ORDER BY created_at
                """,
                [investigation_id],
            )
            hcols = [d[0] for d in hcur.description]
            hyps = [dict(zip(hcols, r)) for r in hcur.fetchall()]
        except Exception:
            hyps = []
    for k, v in list(inv.items()):
        if isinstance(v, datetime):
            inv[k] = to_iso_z(v)
    for h in hyps:
        if isinstance(h.get("created_at"), datetime):
            h["created_at"] = to_iso_z(h["created_at"])
    inv.setdefault("assignee", None)
    inv.setdefault("sla_due_at", None)
    inv["hypotheses"] = hyps
    try:
        inv["comments"] = list_comments(investigation_id)
    except Exception:
        inv["comments"] = []
    return inv


__all__ = [
    "HYPOTHESIS_STATUSES",
    "update_investigation",
    "add_hypothesis",
    "add_comment",
    "list_comments",
    "get_investigation",
]
