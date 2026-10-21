"""Speech finalization, turn acknowledgement, and playback on the shipped handlers."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from src.channels.twilio_media import carrier_audio_ready

REPO = Path(__file__).resolve().parents[2]


def _voice_events() -> dict:
    js = r"""
import {
  decideSpeechFinal,
  createTurnQueue,
  socketIsLive,
  advancePlayback,
  looksLikeAgentEcho,
  shouldRetainBargeInUtterance,
  IDLE_LISTEN_REPROMPT_MS,
} from './dashboard/src/voiceHelpers.js';

const out = {};
const t0 = 10_000;
out.whileSpeaking = decideSpeechFinal({
  lastInterimAt: t0,
  committedAt: t0 - 500,
  lastSpeechAt: t0,
  now: t0 + 400,
  debounceMs: 1400,
  listeningSince: t0 - 2000,
});
out.afterPause = decideSpeechFinal({
  lastInterimAt: t0,
  committedAt: t0,
  lastSpeechAt: t0,
  now: t0 + 1400,
  debounceMs: 1400,
  listeningSince: t0 - 3000,
});
out.idle = decideSpeechFinal({
  lastInterimAt: 0,
  committedAt: 0,
  lastSpeechAt: 0,
  now: t0 + IDLE_LISTEN_REPROMPT_MS,
  listeningSince: t0,
});
out.notIdleYet = decideSpeechFinal({
  now: t0 + 1000,
  listeningSince: t0,
});

const queue = createTurnQueue();
const noisy = "um uh the brakes are grindin real bad";
const numeric = "two zero one nine";
const first = queue.submit("turn-voice-1", noisy);
const retry = queue.submit("turn-voice-1", noisy);
const acked = queue.acknowledge("turn-voice-1", { status: "completed", turn_id: "turn-voice-1" });
const duplicate = queue.submit("turn-voice-1", "a different sentence");
const numbers = queue.submit("turn-voice-2", numeric);
out.queue = { first, retry, acked, duplicate, numbers };

out.socketOpenOnly = socketIsLive({ open: true, authenticated: false, attached: false });
out.socketAuthOnly = socketIsLive({ open: true, authenticated: true, attached: false });
out.socketReady = socketIsLive({ open: true, authenticated: true, attached: true });

let playback = { status: "", text: "", utteranceId: "" };
const states = [];
for (const status of ["queued", "started", "interrupted", "failed", "completed"]) {
  playback = advancePlayback(playback, {
    status,
    text: status === "queued" ? "Are you in a safe location?" : undefined,
    utterance_id: "utt-1",
  });
  states.push(playback.status);
}
playback = advancePlayback(playback, { status: "nope" });
out.playback = { states, final: playback.status, text: playback.text };

out.echoNumeric = looksLikeAgentEcho(numeric, "What model year is the vehicle?");
out.echoAccent = looksLikeAgentEcho(noisy, "Tell me what the brakes are doing.");
out.bargeNo = shouldRetainBargeInUtterance("No!", { interrupting: true });
out.bargeFire = shouldRetainBargeInUtterance("the car is on fire", { interrupting: true });
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
        raise RuntimeError(proc.stderr or proc.stdout)
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_scheduled_speech_turn_and_playback_events():
    out = _voice_events()
    assert out["whileSpeaking"]["flush"] is False
    assert out["whileSpeaking"]["reason"] == "interim"
    assert out["afterPause"]["flush"] is True
    assert out["afterPause"]["reason"] == "committed"
    assert out["idle"]["reprompt"] is True
    assert out["idle"]["flush"] is False
    assert out["notIdleYet"]["reprompt"] is False
    assert out["notIdleYet"]["flush"] is False

    assert out["queue"]["first"]["status"] == "pending"
    assert out["queue"]["first"]["accepted"] is False
    assert out["queue"]["retry"]["turnId"] == "turn-voice-1"
    assert out["queue"]["retry"]["accepted"] is False
    assert out["queue"]["acked"]["status"] == "completed"
    assert out["queue"]["duplicate"]["status"] == "duplicate"
    assert out["queue"]["duplicate"]["outcome"]["status"] == "completed"
    assert out["queue"]["duplicate"]["text"] == "um uh the brakes are grindin real bad"
    assert out["queue"]["numbers"]["text"] == "two zero one nine"

    assert out["socketOpenOnly"] is False
    assert out["socketAuthOnly"] is False
    assert out["socketReady"] is True
    assert out["playback"]["states"] == [
        "queued", "started", "interrupted", "failed", "completed",
    ]
    assert out["playback"]["final"] == "completed"
    assert "safe location" in out["playback"]["text"]
    assert out["echoNumeric"] is False
    assert out["echoAccent"] is False
    assert out["bargeNo"] is True
    assert out["bargeFire"] is True


def test_carrier_route_is_not_a_live_audio_conversation():
    assert carrier_audio_ready() is False


@pytest.mark.asyncio
async def test_duplicate_turn_id_returns_the_existing_outcome(orchestrator_factory):
    orch, _hooks = orchestrator_factory()
    await orch.start()
    await orch.handle_customer_turn("hello there friend", client_turn_id="turn-7")
    stored = dict(orch.last_turn_outcome)
    customers = [t for t in orch.ctx.turns if t["speaker"] == "customer"]
    again = await orch.handle_customer_turn(
        "a completely different sentence", client_turn_id="turn-7"
    )
    assert again["status"] == "duplicate"
    assert again["duplicate"] is True
    assert again["turn_id"] == "turn-7"
    assert again["outcome"]["turn_id"] == stored["turn_id"]
    assert again["outcome"]["status"] == stored["status"]
    assert [t for t in orch.ctx.turns if t["speaker"] == "customer"] == customers


@pytest.mark.asyncio
async def test_playback_states_and_unheard_prompt(orchestrator_factory):
    orch, _hooks = orchestrator_factory()
    await orch.start()
    queued = orch.note_playback({
        "status": "queued",
        "text": "Are you in a safe location?",
        "utterance_id": "utt-9",
    })
    assert queued["status"] == "queued"
    started = orch.note_playback({"status": "started"})
    assert started["status"] == "started"
    assert started["text"] == "Are you in a safe location?"
    assert started["utterance_id"] == "utt-9"
    failed = orch.note_playback({"status": "failed"})
    assert failed["status"] == "failed"
    interrupted = orch.note_playback({"status": "interrupted", "text": "No!"})
    assert interrupted["status"] == "interrupted"
    assert interrupted["text"] == "No!"
    kept = orch.note_playback({"status": "not-a-state"})
    assert kept["status"] == "interrupted"
    done = orch.note_playback({"status": "completed", "text": "Are you in a safe location?"})
    assert done["status"] == "completed"
    # An unheard prompt is whatever was queued and not completed. Re-queue it.
    pending = orch.note_playback({
        "status": "queued",
        "text": "Is anyone hurt?",
        "utterance_id": "utt-10",
    })
    playback = orch.ctx.slots["__playback__"]
    assert playback["status"] == "queued"
    assert playback["text"] == pending["text"]
    assert playback["status"] != "completed"
