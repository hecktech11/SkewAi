"""Ledger-first spoken playback reconciler and audio frame tracker."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, List, Optional

from src.ledger.journal import AgentActionJournal


class TruncationPersistError(RuntimeError):
    """The durable spoken-turn truncation write failed.

    Callers must not treat the audible prefix as a successful reconciliation.
    """


@dataclass
class SpokenTokenMarker:
    word: str
    start_ms: int
    end_ms: int


class SpokenPlaybackReconciler:
    """
    Correlates TTS synthesized word timestamps with playback progress to guarantee
    the immutable ledger receives the true transcript of what was audible to the caller.
    """

    def __init__(self, action_journal: AgentActionJournal, interaction_id: str) -> None:
        self.journal = action_journal
        self.interaction_id = interaction_id
        self.current_action_id: Optional[str] = None
        self.word_markers: List[SpokenTokenMarker] = []
        self.playback_start_ts: Optional[float] = None
        self.total_audio_duration_ms: int = 0

    def register_planned_turn(self, action_id: str, text: str, word_alignment: List[dict[str, Any]]) -> None:
        self.current_action_id = action_id
        self.playback_start_ts = time.monotonic()
        self.word_markers = [
            SpokenTokenMarker(
                word=w["word"],
                start_ms=int(w["start"] * 1000) if isinstance(w["start"], float) else int(w["start"]),
                end_ms=int(w["end"] * 1000) if isinstance(w["end"], float) else int(w["end"]),
            )
            for w in word_alignment
        ]
        self.total_audio_duration_ms = self.word_markers[-1].end_ms if self.word_markers else 0

    def handle_barge_in(self) -> str:
        """
        Executed immediately when VAD flags customer speech during agent output.
        Calculates spoken boundary and reconciles the ledger action record.
        """
        if not self.playback_start_ts or not self.current_action_id:
            return ""

        elapsed_ms = int((time.monotonic() - self.playback_start_ts) * 1000)

        # A word counts only after its end timestamp has been reached.
        audible_words = [m.word for m in self.word_markers if m.end_ms <= elapsed_ms]
        if audible_words:
            actual_spoken_text = " ".join(audible_words) + " [INTERRUPTED]"
        else:
            actual_spoken_text = "[INTERRUPTED]"

        # Durable audit row first. A failed write is not an acknowledgement.
        truncation_id = self.journal.record_action_truncation(
            interaction_id=self.interaction_id,
            action_id=self.current_action_id,
            spoken_text=actual_spoken_text,
            interrupted_at_ms=elapsed_ms,
        )
        if not truncation_id:
            raise TruncationPersistError(
                f"spoken_turn_truncated write failed for {self.current_action_id}"
            )

        self.current_action_id = None
        self.playback_start_ts = None
        return actual_spoken_text
