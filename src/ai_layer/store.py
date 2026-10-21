"""Persistence for the optional AI layer (own tables, created lazily).  [AI-LAYER]

Two tables in the ops warehouse, both created with ``CREATE TABLE IF NOT
EXISTS`` the first time the AI layer writes. No existing table or migration is
altered, and with no key + no evidence upload they are never created.

* ``ai_layer_events``: one row per AI decision, holding BOTH the deterministic
  value and the AI proposal, the accepted outcome and the reason. Customer text
  is PII-redacted before it is stored.
* ``case_evidence``: photo evidence attached to a case (and, through the case,
  to its cluster / investigation), with optional SAM + Muse Spark analysis.
"""

from __future__ import annotations

import json
from typing import Any

from src.data.timeutil import utc_now
from src.data.warehouse import ops_con
from src.ids import new_ulid

_DDL = [
    """
    CREATE TABLE IF NOT EXISTS ai_layer_events (
        event_id        VARCHAR PRIMARY KEY,
        interaction_id  VARCHAR,
        case_id         VARCHAR,
        kind            VARCHAR NOT NULL,
        mode            VARCHAR NOT NULL,
        deterministic   JSON,
        ai_proposal     JSON,
        accepted        JSON,
        reason          VARCHAR,
        model_id        VARCHAR,
        created_at      TIMESTAMP NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_ai_events_interaction ON ai_layer_events(interaction_id, created_at)",
    """
    CREATE TABLE IF NOT EXISTS case_evidence (
        evidence_id      VARCHAR PRIMARY KEY,
        case_id          VARCHAR,
        interaction_id   VARCHAR,
        pack_id          VARCHAR,
        cluster_id       INTEGER,
        investigation_id VARCHAR,
        filename         VARCHAR,
        mime             VARCHAR NOT NULL,
        size_bytes       INTEGER NOT NULL,
        width            INTEGER,
        height           INTEGER,
        sha256           VARCHAR NOT NULL,
        storage_path     VARCHAR NOT NULL,
        analysis_status  VARCHAR NOT NULL,
        analysis         JSON,
        mapped_category  VARCHAR,
        created_at       TIMESTAMP NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_case_evidence_case ON case_evidence(case_id, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_case_evidence_cluster ON case_evidence(pack_id, cluster_id)",
]

def ensure_tables(con) -> None:
    """Idempotent DDL, run on the writer's own connection (cheap; no caching,
    so a test or deployment that swaps the ops DB path never sees a stale flag)."""
    for stmt in _DDL:
        con.execute(stmt)


def _redacted(value: Any) -> Any:
    from src.security.pii import redact_pii

    if isinstance(value, str):
        return redact_pii(value)
    if isinstance(value, dict):
        return {k: _redacted(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redacted(v) for v in value]
    return value


def record_event(
    *,
    kind: str,
    interaction_id: str | None = None,
    case_id: str | None = None,
    mode: str = "enhanced",
    deterministic: Any = None,
    ai_proposal: Any = None,
    accepted: Any = None,
    reason: str = "",
    model_id: str = "",
) -> str | None:
    """Best-effort; never raises into the caller's path."""
    try:
        eid = "aie_" + new_ulid()
        with ops_con() as con:
            ensure_tables(con)
            con.execute(
                """
                INSERT INTO ai_layer_events (event_id, interaction_id, case_id, kind, mode,
                    deterministic, ai_proposal, accepted, reason, model_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    eid,
                    interaction_id,
                    case_id,
                    kind[:64],
                    mode[:32],
                    json.dumps(_redacted(deterministic), default=str),
                    json.dumps(_redacted(ai_proposal), default=str),
                    json.dumps(_redacted(accepted), default=str),
                    (reason or "")[:300],
                    (model_id or "")[:80],
                    utc_now(),
                ],
            )
        return eid
    except Exception:
        return None


def _relation_exists(con, name: str) -> bool:
    row = con.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_name = ?", [name]
    ).fetchone()
    return row is not None


def list_events(interaction_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    with ops_con() as con:
        if not _relation_exists(con, "ai_layer_events"):
            return []
        sql = "SELECT * FROM ai_layer_events"
        args: list[Any] = []
        if interaction_id:
            sql += " WHERE interaction_id = ?"
            args.append(interaction_id)
        sql += " ORDER BY created_at DESC LIMIT ?"
        args.append(max(1, min(500, int(limit))))
        cur = con.execute(sql, args)
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    for row in rows:
        for key in ("deterministic", "ai_proposal", "accepted"):
            try:
                row[key] = json.loads(row[key]) if isinstance(row[key], str) else row[key]
            except (TypeError, json.JSONDecodeError):
                pass
        if row.get("created_at") is not None:
            row["created_at"] = str(row["created_at"])
    return rows


def erase_interaction(con, interaction_id: str) -> dict[str, int]:
    """DSR erase hook: drop AI-layer rows (and evidence files) for one contact.

    Returns ``{}`` when the AI layer never created its tables, so the DSR
    result is identical to the pre-AI-layer behaviour.
    """
    deleted: dict[str, int] = {}
    if _relation_exists(con, "case_evidence"):
        rows = con.execute(
            "SELECT storage_path FROM case_evidence WHERE interaction_id = ?", [interaction_id]
        ).fetchall()
        from pathlib import Path

        for (path,) in rows:
            try:
                Path(path).unlink(missing_ok=True)
            except OSError:
                pass
        con.execute("DELETE FROM case_evidence WHERE interaction_id = ?", [interaction_id])
        deleted["case_evidence"] = len(rows)
    if _relation_exists(con, "ai_layer_events"):
        n = con.execute(
            "SELECT COUNT(*) FROM ai_layer_events WHERE interaction_id = ?", [interaction_id]
        ).fetchone()[0]
        con.execute("DELETE FROM ai_layer_events WHERE interaction_id = ?", [interaction_id])
        deleted["ai_layer_events"] = int(n)
    from src.ai_layer.video_evidence import erase_for_interaction as _erase_video

    if _relation_exists(con, "case_video_evidence"):
        deleted["case_video_evidence"] = _erase_video(con, interaction_id)
    return deleted


__all__ = ["ensure_tables", "erase_interaction", "list_events", "record_event"]
