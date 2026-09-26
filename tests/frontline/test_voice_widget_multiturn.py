"""Widget multiturn invariants: short answers must survive, mic must pause."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HELPERS = REPO / "dashboard" / "src" / "voiceHelpers.js"
CALL = REPO / "dashboard" / "routes" / "CallWidget.jsx"


def _node_labels():
    js = r"""
import { CALL_STATE, callStateLabel, callPhaseHint } from './dashboard/src/voiceHelpers.js';
const out = {
  thinking: callStateLabel(CALL_STATE.THINKING),
  hint: callPhaseHint(CALL_STATE.THINKING),
  hasThinking: !!CALL_STATE.THINKING,
};
console.log(JSON.stringify(out));
"""
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", js],
        cwd=str(REPO), capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_thinking_state_exists_with_copy():
    out = _node_labels()
    assert out["hasThinking"] is True
    assert "Think" in out["thinking"]
    assert "mic paused" in out["hint"].lower() or "working" in out["hint"].lower()


def test_widget_pauses_mic_after_send_and_rescues_interim():
    text = CALL.read_text(encoding="utf-8")
    # After a final is sent the widget must enter thinking (mic paused).
    assert "enterThinking()" in text
    assert 'CALL_STATE.THINKING' in text
    # Short answers that only ever surface as interim must be rescued.
    assert "lastInterimRef" in text
    assert "flushCustomerUtterance()" in text
    # Thinking must not restart recognition until the agent answers.
    assert "stateRef.current === CALL_STATE.THINKING" in text
    # Agent answer clears the thinking timeout so TTS can start.
    assert "clearThinkingTimeout()" in text


def test_widget_thinking_timeout_recovers():
    text = CALL.read_text(encoding="utf-8")
    assert "thinkingTimeoutRef" in text
    assert "12000" in text
    assert "Still working" in text


def test_widget_keeps_the_call_live_for_handoff_and_speaks_supervisor_replies():
    text = CALL.read_text(encoding="utf-8")
    # Microphone permission starts in parallel with contact setup, rather than
    # leaving the caller on a Connecting screen until getUserMedia returns.
    assert "const micReady = setupMic" in text
    assert "void micReady.finally" in text
    # The requested console opens separately, so navigating to it cannot run
    # CallWidget's unmount cleanup and end the customer contact.
    assert 'window.open(target, "_blank", "noopener")' in text
    assert "?id=${encodeURIComponent(iid)}" in text
    # A supervisor's typed reply remains audible to the caller.
    assert 'speaker === "supervisor") setInfo("A human specialist is responding")' in text
    # The caller control asks the server to create a real queue entry.
    assert "body: JSON.stringify({ force: true })" in text


def test_widget_does_not_turn_a_phrase_confidence_into_an_entity_readback():
    text = CALL.read_text(encoding="utf-8")
    helpers = HELPERS.read_text(encoding="utf-8")
    # Browser SpeechRecognition gives one score for an utterance, never an
    # individual make/model/year. Sending that score caused the backend to
    # confirm the next generic entity on every partial phrase.
    assert "confidence: conf" not in text
    assert "STT_FINAL_DEBOUNCE_MS = 1400" in helpers
    # A recognition onend between Chrome result chunks must not flush before
    # the debounce timer can combine them into the caller's complete thought.
    assert "Keep the debounce window alive" in text
