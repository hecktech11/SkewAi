"""Barge-in must not acknowledge success when the truncation audit write fails."""

from __future__ import annotations

import pytest

from src.ledger.journal import AgentActionJournal
from src.ledger.writer import list_actions
from src.voice.interruption_reconciler import (
    SpokenPlaybackReconciler,
    TruncationPersistError,
)


_MARKERS = [
    {"word": "Please", "start": 0.0, "end": 0.3},
    {"word": "describe", "start": 0.3, "end": 0.7},
    {"word": "the", "start": 0.7, "end": 0.9},
    {"word": "noise", "start": 0.9, "end": 1.2},
]


def _fail_truncation_writes(monkeypatch) -> None:
    import src.ledger.writer as writer

    real = writer.record_action

    def _record(action, *, on_degraded=None):
        if getattr(action, "action_type", "") == "spoken_turn_truncated":
            raise RuntimeError("truncation store unavailable")
        return real(action, on_degraded=on_degraded)

    monkeypatch.setattr(writer, "record_action", _record)


def test_reconcile_barge_in_surfaces_truncation_write_failure(monkeypatch, reset_ops_db):
    from src.voice.wiring import reconcile_barge_in

    _fail_truncation_writes(monkeypatch)
    iid = "int_trunc_fail_reconcile"
    with pytest.raises(TruncationPersistError):
        reconcile_barge_in(
            iid,
            "act_planned",
            "Please describe the noise",
            _MARKERS,
            elapsed_ms=400,
        )
    rows = list_actions(iid)
    assert not any(r.get("action_type") == "spoken_turn_truncated" for r in rows)


def test_handle_barge_in_does_not_ack_or_store_success(monkeypatch):
    _fail_truncation_writes(monkeypatch)
    journal = AgentActionJournal()
    reconciler = SpokenPlaybackReconciler(journal, interaction_id="int_trunc_fail_direct")
    reconciler.register_planned_turn("act_planned", "Please describe the noise", _MARKERS)
    import time
    reconciler.playback_start_ts = time.monotonic() - 0.4
    with pytest.raises(TruncationPersistError):
        reconciler.handle_barge_in()
    assert journal.truncations == []
    assert reconciler.current_action_id == "act_planned"


@pytest.mark.asyncio
async def test_orchestrator_barge_in_surfaces_truncation_failure(
    orchestrator_factory, monkeypatch,
):
    orch, hooks = orchestrator_factory()
    await orch.start()
    orch.note_emitted_utterance(
        "Please describe the noise from the brakes",
        utterance_id="utt-fail",
        action_id="act_planned",
    )
    _fail_truncation_writes(monkeypatch)
    with pytest.raises(TruncationPersistError):
        await orch.handle_barge_in_interrupt(400, utterance_id="utt-fail")
    rows = list_actions(orch.ctx.interaction_id)
    assert not any(r.get("action_type") == "spoken_turn_truncated" for r in rows)
    assert not any(
        r.get("action_type") == "spoken_turn_reconciled" and r.get("ok") is not False
        for r in rows
        if "utt-fail" in str(r.get("input_summary") or "")
    )
    assert not any(
        a.get("action_type") == "spoken_turn_truncated" and a.get("ok") is True
        for a in hooks.activities
    )
