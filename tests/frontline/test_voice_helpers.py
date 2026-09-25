"""Unit tests for shipped dashboard voice helpers (real import path)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
HELPERS = REPO / "dashboard" / "src" / "voiceHelpers.js"
CALL = REPO / "dashboard" / "routes" / "CallWidget.jsx"


def _load_helpers_via_node() -> dict:
    """Execute shipped voiceHelpers.js in Node and return exported results via checks."""
    import json
    import subprocess

    script = r"""
const h = require('./dashboard/src/voiceHelpers.js');
// ESM may not load via require — use dynamic import
"""
    # Prefer dynamic import for ESM
    js = r"""
import {
  callStateLabel,
  mergeTranscriptTurn,
  mapInteractionEnded,
  shouldAcceptSpeechResult,
  buildSlotEntries,
  slotProgress,
  capabilitySnapshot,
  shouldReconnectOnClose,
  shouldResumeListeningOnOpen,
  shouldAllowBargeIn,
  greetingSpeakText,
  callPhaseHint,
  CALL_STATE,
  BARGE_IN_GRACE_MS,
  looksLikeAgentEcho,
  POST_TTS_COOLDOWN_MS,
} from './dashboard/src/voiceHelpers.js';

const out = {};

out.labels = {
  idle: callStateLabel(CALL_STATE.IDLE),
  listening: callStateLabel(CALL_STATE.LISTENING),
  speaking: callStateLabel(CALL_STATE.AGENT_SPEAKING),
  greeting: callStateLabel(CALL_STATE.AGENT_SPEAKING, { speakPhase: 'greeting' }),
  ended: callStateLabel(CALL_STATE.ENDED),
};
out.hintGreeting = callPhaseHint(CALL_STATE.AGENT_SPEAKING, { speakPhase: 'greeting' });
out.hintListen = callPhaseHint(CALL_STATE.LISTENING);

const FULL = "Thanks for calling support. What's going on with your vehicle?";
out.greetingSpeak = greetingSpeakText(FULL);
out.greetingTrim = greetingSpeakText("  " + FULL + "  ");
// Full agent greeting survives merge (no interim replace of agent text)
let g = [];
g = mergeTranscriptTurn(g, { speaker: 'agent', text: FULL });
out.greetingInTranscript = g[0].text;
out.bargeGreetingBlocked = shouldAllowBargeIn({
  phase: 'greeting', elapsedMs: 5000, rms: 0.5, highRmsStreak: 10, streakNeeded: 4,
});
out.bargeTooEarly = shouldAllowBargeIn({
  phase: 'normal', elapsedMs: 200, graceMs: BARGE_IN_GRACE_MS, rms: 0.5, highRmsStreak: 10, streakNeeded: 4,
});
out.bargeOk = shouldAllowBargeIn({
  phase: 'normal', elapsedMs: BARGE_IN_GRACE_MS + 100, graceMs: BARGE_IN_GRACE_MS,
  rms: 0.2, highRmsStreak: 4, streakNeeded: 4,
});
out.bargeLowRms = shouldAllowBargeIn({
  phase: 'normal', elapsedMs: 5000, rms: 0.05, highRmsStreak: 10, streakNeeded: 4,
});

out.reconnectNormal = shouldReconnectOnClose({ intentionalClose: false, callEnded: false });
out.reconnectAfterEnded = shouldReconnectOnClose({ intentionalClose: true, callEnded: true });
out.reconnectCallEndedOnly = shouldReconnectOnClose({ intentionalClose: false, callEnded: true });
out.resumeAfterEnded = shouldResumeListeningOnOpen({
  intentionalClose: true,
  callEnded: true,
  isReconnect: true,
});
out.resumeLiveReconnect = shouldResumeListeningOnOpen({
  intentionalClose: false,
  callEnded: false,
  isReconnect: true,
});

let t = [];
t = mergeTranscriptTurn(t, { speaker: 'customer', text: 'hel', interim: true });
t = mergeTranscriptTurn(t, { speaker: 'customer', text: 'hello', interim: true });
t = mergeTranscriptTurn(t, { speaker: 'customer', text: 'hello there', interim: false });
t = mergeTranscriptTurn(t, { speaker: 'agent', text: 'Hi' });
out.transcript = t;
out.transcriptLen = t.length;
out.lastCustomer = t.find(x => x.speaker === 'customer' && !x.interim);

out.endedFlat = mapInteractionEnded({
  type: 'interaction_ended',
  case_id: 'case_abc',
  investigation_id: 'inv_1',
  investigation_opened: true,
  audit_pending: true,
});
out.endedNested = mapInteractionEnded({
  type: 'interaction_ended',
  payload: { case_id: 'case_nested', reason: 'done' },
});

out.acceptWhileSpeaking = shouldAcceptSpeechResult({ speaking: true, wsOpen: true });
out.acceptListening = shouldAcceptSpeechResult({ speaking: false, wsOpen: true });
out.acceptClosed = shouldAcceptSpeechResult({ speaking: false, wsOpen: false });
out.acceptDuringCooldown = shouldAcceptSpeechResult({
  speaking: false, wsOpen: true, listenReadyAt: Date.now() + 5000, now: Date.now(),
});
out.echoHurt = looksLikeAgentEcho('is anyone hurt', 'Is anyone hurt in this incident?');
out.echoNo = looksLikeAgentEcho('2019 Honda CR-V', 'Is anyone hurt in this incident?');
out.cooldownMs = POST_TTS_COOLDOWN_MS;

const slots = buildSlotEntries(
  { entity_labels: { entity_1: 'Make', entity_2: 'Model' } },
  { entity_1: 'Toyota', category: 'brakes' }
);
out.slots = slots;
out.progress = slotProgress(slots);

out.caps = capabilitySnapshot({
  hasSpeechRecognition: false,
  hasSpeechSynthesis: true,
  hasMicStream: false,
});

console.log(JSON.stringify(out));
"""
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", js],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=30,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"node failed: {proc.stderr or proc.stdout}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_voice_helpers_file_exists():
    assert HELPERS.is_file()
    assert CALL.is_file()


def test_call_widget_imports_helpers_and_map_ended():
    text = CALL.read_text(encoding="utf-8")
    assert "voiceHelpers" in text
    assert "mapInteractionEnded" in text
    assert "mergeTranscriptTurn" in text
    assert "shouldAcceptSpeechResult" in text
    assert "looksLikeAgentEcho" in text
    assert "enqueueSpeak" in text
    # Must not only use nested payload
    assert "mapInteractionEnded(msg)" in text
    # P1-4: typing during greeting must reset speakPhase or STT never restarts.
    assert "speakPhaseRef.current = \"normal\"" in text or "speakPhaseRef.current = 'normal'" in text


def test_shipped_helpers_via_node():
    out = _load_helpers_via_node()
    assert "Ready" in out["labels"]["idle"] or out["labels"]["idle"]
    assert "Listening" in out["labels"]["listening"]
    assert "speaking" in out["labels"]["speaking"].lower() or "Agent" in out["labels"]["speaking"]
    assert "Greeting" in out["labels"]["greeting"]
    assert "Barge-in off" in out["hintGreeting"]
    assert "Your turn" in out["hintListen"]

    assert out["transcriptLen"] == 2  # final customer + agent (interim replaced)
    assert out["lastCustomer"]["text"] == "hello there"
    assert out["lastCustomer"].get("interim") in (False, None)

    assert out["endedFlat"]["case_id"] == "case_abc"
    assert out["endedFlat"]["investigation_id"] == "inv_1"
    assert out["endedFlat"]["investigation_opened"] is True
    assert out["endedNested"]["case_id"] == "case_nested"

    assert out["acceptWhileSpeaking"] is False
    assert out["acceptListening"] is True
    assert out["acceptClosed"] is False
    assert out["acceptDuringCooldown"] is False
    assert out["echoHurt"] is True
    assert out["echoNo"] is False
    assert out["cooldownMs"] >= 400

    labels = {s["label"] for s in out["slots"]}
    assert "Make" in labels
    assert out["progress"]["filled"] >= 1
    assert out["progress"]["total"] >= 2

    assert out["caps"]["sttOk"] is False
    assert out["caps"]["textFallback"] is True

    assert out["reconnectNormal"] is True
    assert out["reconnectAfterEnded"] is False
    assert out["reconnectCallEndedOnly"] is False
    assert out["resumeAfterEnded"] is False
    assert out["resumeLiveReconnect"] is True

    full = "Thanks for calling support. What's going on with your vehicle?"
    assert out["greetingSpeak"] == full
    assert out["greetingTrim"] == full
    assert out["greetingInTranscript"] == full
    assert "Thanks for calling" in out["greetingInTranscript"]
    assert out["bargeGreetingBlocked"] is False
    assert out["bargeTooEarly"] is False
    assert out["bargeOk"] is True
    assert out["bargeLowRms"] is False


def test_stt_paused_during_tts_in_callwidget_source():
    text = CALL.read_text(encoding="utf-8")
    # Explicit stop/pause of recognition when speaking
    assert "stopRecognition" in text or "recogRef.current?.stop" in text
    assert "speakingRef.current = true" in text
    assert "shouldAcceptSpeechResult" in text
    # Typed barge-in must resume STT (cancel does not always fire onend)
    assert "startRecognitionSafe()" in text
    assert "markCallTerminal" in text


def test_interaction_ended_marks_terminal_before_close():
    """Regression: normal end must not reconnect and wipe case_id summary."""
    text = CALL.read_text(encoding="utf-8")
    idx = text.find('case "interaction_ended"')
    assert idx > 0
    block = text[idx : idx + 450]
    assert "markCallTerminal()" in block
    # markCallTerminal must set intentional close + clear reconnect URL
    assert "intentionalCloseRef.current = true" in text
    assert "callEndedRef.current = true" in text
    assert "shouldReconnectOnClose" in text


def test_callwidget_greeting_phase_disables_barge_in():
    """Start path must speak full greeting with phase greeting (no mid-word cancel)."""
    text = CALL.read_text(encoding="utf-8")
    assert 'phase: "greeting"' in text or "phase: 'greeting'" in text
    assert "greetingSpeakText" in text
    assert "shouldAllowBargeIn" in text
    # Barge watch must skip greeting phase
    assert 'speakPhaseRef.current === "greeting"' in text or "phase !== \"greeting\"" in text
