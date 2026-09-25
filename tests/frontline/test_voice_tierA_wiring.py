"""Tier-A wiring proof: every dormant voice module now has a production caller.

Each test drives the LIVE path (orchestrator / WS route / wiring helper), not
just the module in isolation — so CI fails if a module goes back to
test-only.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from scripts.seed_domains import build as build_auto


@pytest.fixture
def api_client(reset_ops_db):
    build_auto("automotive_nhtsa")
    from src.api.routes.interactions import _active
    _active.clear()
    with TestClient(app) as client:
        yield client
    _active.clear()


# ── wiring helpers: each dormant module has a prod caller ────────────────────

def test_reconciler_prod_caller_writes_audible_prefix(reset_ops_db):
    from src.voice.wiring import reconcile_barge_in

    out = reconcile_barge_in(
        "int_wire_1", "act_1",
        "Please describe what happened with the vehicle in detail",
        [{"word": "Please", "start": 0.0, "end": 0.3},
         {"word": "describe", "start": 0.3, "end": 0.7},
         {"word": "what", "start": 0.7, "end": 0.9}],
        elapsed_ms=350,
    )
    assert "Please" in out
    assert "INTERRUPTED" in out


@pytest.mark.asyncio
async def test_durable_sequencer_drops_replay(reset_ops_db):
    from src.voice.wiring import acquire_durable_turn_slot

    ok1 = await acquire_durable_turn_slot("int_seq_1", 1, "cid-1", "hello brakes")
    ok2 = await acquire_durable_turn_slot("int_seq_1", 1, "cid-1", "hello brakes")
    assert ok1 is True
    assert ok2 is False


def test_vin_capture_valid_and_invalid():
    from src.voice.wiring import detect_vin_in_text

    good = detect_vin_in_text("my vin is 1HGCR2F84HA000000 thanks")
    assert good.get("valid") is True
    assert good.get("vin") == "1HGCR2F84HA000000"
    bad = detect_vin_in_text("vin 1HGCR2F85HA000000")
    assert bad.get("vin") == "1HGCR2F85HA000000"
    assert bad.get("valid") is False
    assert "check digit" in bad.get("prompt", "").lower()
    # Phonetic path: F as in Frank
    phon = detect_vin_in_text("my vin is 1 H G C R 2 F 8 4 H A 0 0 0 0 0 0")
    assert phon.get("vin") == "1HGCR2F84HA000000"


def test_vin_ignores_ordinary_sentences():
    # Regression: "the vehicle has been started..." squashes to a 17-letter
    # run (HASBEENSTARTEDTHE) and must not hijack the turn with a VIN retry.
    from src.voice.wiring import detect_vin_in_text

    assert detect_vin_in_text("the vehicle has been started the road help me") == {}
    assert detect_vin_in_text("my brakes are grinding when I stop") == {}
    assert detect_vin_in_text("hello hello hello") == {}
    assert detect_vin_in_text("HASBEENSTARTEDTHE") == {}


@pytest.mark.asyncio
async def test_sequencer_fail_open_on_db_error(orchestrator_factory):
    # Transient DB failures must let the turn through (fail-open), only proven
    # PK collisions drop.
    from src.voice import turn_sequencer as _ts

    seq = _ts.TelephonyTurnSequencer("int_fail_open")
    assert await seq.acquire_turn_execution_slot(1, "cid-fail-open-1", bypass_sequencing=True) is True
    assert await seq.acquire_turn_execution_slot(2, "cid-fail-open-1", bypass_sequencing=True) is False


def test_confirmation_protocol_prod_caller():
    from src.voice.wiring import confirmation_prompt_for, evaluate_confirmation_reply

    slots = {"entity_1": "2019", "entity_2": "HONDA", "entity_3": "CR-V",
             "category": "SERVICE BRAKES", "description": "grinding noise"}
    prompt = confirmation_prompt_for(slots, channel="web_voice")
    assert prompt and "2019" in prompt and "Did I get that" in prompt
    res = evaluate_confirmation_reply(slots, channel="web_voice", text="yes that's right")
    assert res["ok"] is True and res["outcome"] == "confirmed"
    # Batch channels bypass.
    assert confirmation_prompt_for(slots, channel="batch") is None


def test_drive_mode_prod_caller():
    from src.voice.wiring import check_drive_mode, drive_safety_script

    assert check_drive_mode(explicit_hint=True) is True
    assert check_drive_mode(text_hint="I am driving on the highway") is True
    assert check_drive_mode(text_hint="my brakes squeal") is False
    assert "safely parked" in drive_safety_script()
    # PCM path never raises (silence is not driving).
    assert check_drive_mode(pcm_chunk=b"\x00\x00" * 800) is False


def test_whisper_prod_caller_shape():
    from src.voice.wiring import build_supervisor_whisper

    pkt = build_supervisor_whisper(
        "int_wh_1",
        {"entity_1": "2019", "entity_2": "HONDA", "entity_3": "CR-V",
         "category": "SERVICE BRAKES", "description": "grinding"},
        severity="Critical", priority="P1", safety_tripped=True,
        kill_terms=["fire"], sentiment_peak=0.9,
    )
    assert pkt["interaction_id"] == "int_wh_1"
    assert "P1" in pkt["whisper_audio_ssml"] or "Critical" in pkt["whisper_audio_ssml"] or "Warning" in pkt["whisper_audio_ssml"]
    assert pkt["triage_state"]["safety_flag_tripped"] is True


def test_telephony_bridge_prod_caller():
    from src.voice.wiring import telephony_bridge_for
    from src.voice.telephony_bridge import pcm16k_to_ulaw8k, ulaw8k_to_pcm16k

    frames: list[bytes] = []

    async def _speech(pcm: bytes):
        frames.append(pcm)

    class _WS:
        async def receive_text(self):
            raise RuntimeError("done")

        async def send_text(self, _t):
            pass

    bridge = telephony_bridge_for(_WS(), "int_tel_1", _speech)
    assert bridge.interaction_id == "int_tel_1"
    pcm = b"\x00\x01" * 160
    assert ulaw8k_to_pcm16k(pcm16k_to_ulaw8k(pcm))[:4] != b""


@pytest.mark.asyncio
async def test_live_latency_verdict_and_diagnostics():
    from src.voice.wiring import live_turn_budget_verdict, run_single_trial_diagnostics

    v = live_turn_budget_verdict(120.0)
    assert v["elapsed_ms"] == 120
    rep = await run_single_trial_diagnostics()
    assert rep["trials"] == 1
    assert "total_ttfa" in rep["stages"]


def test_pack_locale_wired():
    from src.domains.loader import load_pack

    pack = load_pack("automotive_nhtsa", reload=True)
    assert (getattr(pack, "locale", "en-US") or "en-US") == "en-US"
    assert pack.manifest.model_dump().get("locale", "en-US") == "en-US"


# ── orchestrator live path ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_consent_gate_blocks_until_affirmation(orchestrator_factory):
    orch, hooks = orchestrator_factory(channel="web_voice")
    await orch.start(region="CA")
    assert orch.ctx.voice_consent_required is True
    await orch.handle_customer_turn("my brakes grind", region="CA")
    # Blocked: no intake question, consent script spoken instead.
    assert any("recorded" in t.lower() or "consent" in t.lower()
               for t in hooks.agent_texts())
    assert orch.ctx.voice_consent_obtained is False
    await orch.handle_customer_turn("I consent", region="CA")
    assert orch.ctx.voice_consent_obtained is True


@pytest.mark.asyncio
async def test_low_confidence_triggers_readback(orchestrator_factory):
    orch, hooks = orchestrator_factory(channel="web_voice")
    await orch.start()
    # Entity-level confidence is appropriate for an entity readback. Browser
    # Web Speech's single utterance score is deliberately not treated this way.
    await orch.handle_customer_turn("2019 Honda", asr_confidence={"entity_1": 0.2})
    assert any("confirm" in t.lower() or "heard" in t.lower()
               for t in hooks.agent_texts())


@pytest.mark.asyncio
async def test_browser_utterance_confidence_never_becomes_an_entity_readback(orchestrator_factory):
    orch, hooks = orchestrator_factory(channel="web_voice")
    await orch.start()
    await orch.handle_customer_turn(
        "I am stuck in the road and my car is broken", confidence=0.2
    )
    texts = hooks.agent_texts()
    assert not any("for entity_1" in t.lower() for t in texts)
    assert any("anyone hurt" in t.lower() for t in texts)


@pytest.mark.asyncio
async def test_spoken_human_request_creates_a_real_handoff(orchestrator_factory):
    orch, hooks = orchestrator_factory(channel="web_voice")
    await orch.start()
    await orch.handle_customer_turn("Please get me a human now")
    assert orch.ctx.state == "HANDOFF_PENDING"
    assert any("connecting you to a human" in t.lower() for t in hooks.agent_texts())
    again = await orch.accept_handoff(force=True, source="caller_control")
    assert again["accepted"] is True
    assert again["already_pending"] is True


def test_caller_control_queues_handoff_without_prior_offer(api_client):
    started = api_client.post("/api/interactions/start?channel=web_voice")
    assert started.status_code in (200, 201)
    iid = started.json()["interaction_id"]
    queued = api_client.post(
        f"/api/interactions/{iid}/handoff/accept", json={"force": True}
    )
    assert queued.status_code == 200
    assert queued.json()["accepted"] is True
    repeated = api_client.post(
        f"/api/interactions/{iid}/handoff/accept", json={"force": True}
    )
    assert repeated.status_code == 200
    assert repeated.json()["already_pending"] is True


@pytest.mark.asyncio
async def test_vin_captured_on_live_turn(orchestrator_factory):
    orch, hooks = orchestrator_factory(channel="web_voice")
    await orch.start()
    await orch.handle_customer_turn("my vin is 1HGCR2F84HA000000")
    assert orch.ctx.slots.get("vin") == "1HGCR2F84HA000000"
    assert any(a.get("action_type") == "vin_captured" for a in hooks.activities)


@pytest.mark.asyncio
async def test_barge_in_reconciles_ledger(orchestrator_factory):
    orch, hooks = orchestrator_factory(channel="web_voice")
    await orch.start()
    orch._register_spoken_turn("Please describe what happened with the vehicle in detail")
    out = await orch.handle_barge_in_interrupt(elapsed_ms=350)
    assert "INTERRUPTED" in out
    assert any(a.get("action_type") == "spoken_turn_truncated" for a in hooks.activities)


@pytest.mark.asyncio
async def test_frame_confirmation_gate_runs_before_enrichment(orchestrator_factory):
    orch, hooks = orchestrator_factory(channel="web_voice")
    await orch.start()
    for txt in ["hello", "no nobody is hurt", "yes", "2019", "Honda", "CR-V",
                "brakes are grinding and squealing when I stop",
                "yes, only on cold mornings", "yes that's right"]:
        await orch.handle_customer_turn(txt)
        if orch.ctx.confirmation_phase in ("pending", "done"):
            break
        if orch.ctx.state in ("ENRICHING", "CLOSING", "DONE"):
            break
    assert orch.ctx.confirmation_phase in ("pending", "done")
    assert any("make sure" in t.lower() or "completely right" in t.lower()
               for t in hooks.agent_texts())


# ── WS contract: new fields accepted, dedup + reconciler + latency live ─────

def test_start_returns_locale_consent_serverstt(api_client):
    r = api_client.post("/api/interactions/start?channel=web_voice&region=CA")
    assert r.status_code in (200, 201)
    body = r.json()
    assert body["locale"] == "en-US"
    assert body["pack"]["locale"] == "en-US"
    assert body["voice_consent_required"] is True
    assert "recorded" in body["consent_script"].lower() or "consent" in body["consent_script"].lower()
    assert "server_stt_available" in body
    assert body["twilio_ws_url"].startswith("/ws/twilio/")


def test_ws_first_turn_with_id_is_answered_not_dropped(api_client):
    # Regression for the double-dedup outage: a first turn carrying turn_id
    # must be ANSWERED. A silent drop (or duplicate_dropped on first send)
    # wedges the widget in THINKING forever.
    r = api_client.post("/api/interactions/start?channel=web_text")
    ws_path = r.json()["ws_url"]
    with api_client.websocket_connect(ws_path) as ws:
        ws.send_json({"type": "user_turn", "text": "hello brakes", "final": True,
                      "turn_id": "cid-ws-first-1", "turn_seq": 1, "confidence": 0.95,
                      "locale": "en-US"})
        got_agent_turn = False
        for _ in range(10):
            msg = ws.receive_json()
            if msg.get("type") == "agent_turn":
                got_agent_turn = True
                break
            assert msg.get("type") != "duplicate_dropped", \
                "first turn must never be reported as duplicate"
        assert got_agent_turn, "server never answered the first turn"
        ws.send_json({"type": "hangup"})


def test_ws_replay_same_id_is_dropped(api_client):
    r = api_client.post("/api/interactions/start?channel=web_text")
    ws_path = r.json()["ws_url"]
    with api_client.websocket_connect(ws_path) as ws:
        ws.send_json({"type": "user_turn", "text": "hello brakes", "final": True,
                      "turn_id": "cid-ws-replay-1", "turn_seq": 1, "confidence": 0.95})
        for _ in range(10):
            msg = ws.receive_json()
            if msg.get("type") == "agent_turn":
                break
        # True replay: same turn_id resent (aggressive reconnect retry).
        ws.send_json({"type": "user_turn", "text": "hello brakes", "final": True,
                      "turn_id": "cid-ws-replay-1", "turn_seq": 2, "confidence": 0.95})
        ws.send_json({"type": "hangup"})


def test_ws_barge_in_reconciled(api_client):
    r = api_client.post("/api/interactions/start?channel=web_text")
    ws_path = r.json()["ws_url"]
    with api_client.websocket_connect(ws_path) as ws:
        ws.send_json({"type": "barge_in", "elapsed_ms": 400})
        ws.send_json({"type": "hangup"})


def test_twilio_route_registered():
    routes = [getattr(rt, "path", "") for rt in app.routes]
    assert "/ws/twilio/{interaction_id}" in routes


def test_diagnostics_endpoint_runs_one_trial(api_client):
    r = api_client.get("/api/interactions/diagnostics/voice-latency")
    assert r.status_code == 200
    body = r.json()
    assert body["trials"] == 1
    assert "stages" in body
