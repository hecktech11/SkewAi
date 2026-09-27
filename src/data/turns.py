"""Durable interaction turn writes — mid-contact, not only at finalize."""

from __future__ import annotations

from typing import Any

from src.data.timeutil import to_naive_utc, utc_now
from src.data.warehouse import ops_con
from src.security.pii import ERASED_TEXT, reveal_subject_text


def persist_turn(interaction_id: str, turn: dict[str, Any]) -> None:
    """Insert one turn row. Idempotent on turn_id primary key."""
    turn_id = turn.get("turn_id")
    if not turn_id or not interaction_id:
        return
    raw_ts = turn.get("ts")
    ts = to_naive_utc(raw_ts) if raw_ts is not None else utc_now()
    speaker = turn.get("speaker") or "customer"
    text = turn.get("text") or ""
    if speaker == "customer" and text:
        from src.security.pii import encrypt_subject_pii

        text = encrypt_subject_pii(interaction_id, text)
        if not str(text).startswith("enc:v1:"):
            raise RuntimeError("customer_turn_encrypt_failed")

    with ops_con() as con:
        exists = con.execute(
            "SELECT 1 FROM interaction_turns WHERE turn_id = ?", [turn_id]
        ).fetchone()
        if exists:
            return
        con.execute(
            """
            INSERT INTO interaction_turns
            (turn_id, interaction_id, seq, speaker, text, ts, latency_ms, llm_used, frustration_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                turn_id,
                interaction_id,
                int(turn.get("seq") or 0),
                speaker,
                text,
                ts,
                turn.get("latency_ms"),
                bool(turn.get("llm_used", False)),
                turn.get("frustration_score"),
            ],
        )


ERASED_TURN = ERASED_TEXT


def reveal_turn_text(interaction_id: str, text: str | None) -> str:
    """Decrypt an at-rest turn payload for display / audit / DSR."""
    return reveal_subject_text(interaction_id, text)


def decrypt_turn_rows(interaction_id: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        if "text" in item:
            item["text"] = reveal_turn_text(interaction_id, item.get("text"))
        out.append(item)
    return out


__all__ = ["persist_turn", "reveal_turn_text", "decrypt_turn_rows", "ERASED_TURN"]
