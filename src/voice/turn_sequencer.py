"""Telephony sequence deduplication and idempotency layer."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Optional, Set

from src.security.exceptions import DuplicateTurnException


def _is_pk_collision(exc: BaseException) -> bool:
    """True only when the error proves the row already exists (true replay)."""
    msg = f"{type(exc).__name__}: {exc}".lower()
    return any(k in msg for k in (
        "duplicate", "unique", "primary key", "constraint",
        "already exists", "conflict",
    ))


class TelephonyTurnSequencer:
    """
    Prevents race conditions and duplicate turns caused by cellular network
    reconnects or packet replay. Uses atomic in-memory locking and persists to
    the durable turn_dedup table when available.
    """

    def __init__(self, interaction_id: str, db_con: Any = None) -> None:
        self.interaction_id = interaction_id
        self._con = db_con
        self.last_committed_seq = -1
        self.processed_turn_hashes: Set[str] = set()
        self._lock = asyncio.Lock()

    async def acquire_turn_execution_slot(
        self,
        turn_seq: int,
        turn_payload_hash: str,
        *,
        bypass_sequencing: bool = False,
    ) -> bool:
        """
        Validates monotonic turn ordering and uniqueness.
        If bypass_sequencing=True (e.g. non-ordered simulation tests), sequence
        ordering check is relaxed while preserving payload deduplication.
        """
        async with self._lock:
            # Monotonic order check (unless explicitly bypassed for synthetic tests)
            if not bypass_sequencing and turn_seq <= self.last_committed_seq:
                return False

            # In-memory payload identity check
            if turn_payload_hash in self.processed_turn_hashes:
                return False

            # Durable database deduplication check if warehouse is active.
            # Fail-OPEN: only a proven primary-key collision means duplicate.
            # Any other DB error (lock contention, missing table) must let the
            # customer's turn through — dropping input is worse than a replay.
            if turn_payload_hash:
                try:
                    from src.data.warehouse import ops_con

                    with ops_con() as con:
                        con.execute(
                            """
                            INSERT INTO turn_dedup (interaction_id, client_turn_id, seen_at)
                            VALUES (?, ?, ?)
                            """,
                            [self.interaction_id, turn_payload_hash, datetime.now(timezone.utc)],
                        )
                except Exception as e:
                    if _is_pk_collision(e):
                        return False
                    try:
                        from src.observability.metrics import inc as _inc

                        _inc("turn_dedup_db_error_fail_open")
                    except Exception:
                        pass
                    # Fall through to in-memory marking: still dedup this
                    # process's own retries while letting the turn through.

            # Mark processed
            self.last_committed_seq = max(self.last_committed_seq, turn_seq)
            self.processed_turn_hashes.add(turn_payload_hash)
            return True
