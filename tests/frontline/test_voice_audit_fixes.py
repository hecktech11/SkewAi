"""Regression coverage for the voice-agent audit (F01–F20 backend paths)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.agents.base import TurnPersistenceError
from src.agents.orchestrator import _pack_captures_vin
from src.api.main import app
from src.api.rbac import issue_session
from src.domains.loader import load_pack
from src.ledger import list_actions
from src.voice.wiring import detect_vin_in_text


def test_digit_account_number_is_not_a_vin():
    assert detect_vin_in_text("My account number is 12345678901234567") == {}
    found = detect_vin_in_text("The VIN is 1HGCM82633A004352")
    assert found.get("vin")


def test_finance_pack_does_not_capture_vins():
    finance = load_pack("finance_cfpb")
    auto = load_pack("automotive_nhtsa")
    assert _pack_captures_vin(finance) is False
    assert _pack_captures_vin(auto) is True


def test_failed_turn_persist_is_not_accepted(orchestrator_factory, monkeypatch):
    orch, _hooks = orchestrator_factory()

    def _boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr("src.data.turns.persist_turn", _boom)
    with pytest.raises(TurnPersistenceError):
        orch.ctx.record_turn("customer", "hello")
    assert orch.ctx.turns == []


@pytest.mark.asyncio
async def test_consent_after_takeover_does_not_speak(orchestrator_factory):
    orch, hooks = orchestrator_factory()
    await orch.start()
    orch.ctx.voice_consent_required = True
    orch.ctx.region = "CA"
    await orch.takeover(claimed_by="supervisor_one")
    before = len(hooks.turns)
    await orch.handle_customer_turn("I consent")
    assert orch.ctx.voice_consent_obtained is True
    assert hooks.turns[before:] == []
    assert orch.ctx.state == "SUPERVISED"


@pytest.mark.asyncio
async def test_losing_supervisor_cannot_send_or_release(orchestrator_factory):
    orch, hooks = orchestrator_factory()
    await orch.start()
    await orch.takeover(claimed_by="supervisor_one")
    denied = await orch.human_turn("from two", actor="supervisor_two", message_id="m1")
    assert denied["ok"] is False
    assert denied["code"] == "not_owner"
    assert hooks.turns == [] or all(t.get("text") != "from two" for t in hooks.turns)
    released = await orch.release(actor="supervisor_two")
    assert released["ok"] is False
    assert orch.ctx.takeover_claimed_by == "supervisor_one"
    ok = await orch.human_turn("from one", actor="supervisor_one", message_id="m2")
    assert ok["ok"] is True
    again = await orch.human_turn("from one", actor="supervisor_one", message_id="m2")
    assert again.get("duplicate") is True
    spoken = [t for t in hooks.turns if t.get("text") == "from one"]
    assert len(spoken) == 1
    rows = [a for a in list_actions(orch.ctx.interaction_id) if a["action_type"] == "human_turn"]
    assert rows and "actor=supervisor_one" in (rows[-1].get("input_summary") or "")


@pytest.mark.asyncio
async def test_barge_in_keeps_the_playing_utterance(orchestrator_factory):
    orch, _hooks = orchestrator_factory()
    await orch.start()
    from src.ledger import AgentAction, record_action

    aid = record_action(AgentAction(
        interaction_id=orch.ctx.interaction_id,
        agent="orchestrator",
        action_type="question_asked",
        input_summary="A",
        output_summary="alpha utterance that is long enough",
        ok=True,
    ))
    orch.note_emitted_utterance(
        "alpha utterance that is long enough",
        utterance_id="utt-a",
        action_id=aid,
    )
    orch.note_emitted_utterance(
        "bravo utterance generated while alpha is still playing",
        utterance_id="utt-b",
        action_id="should-not-be-the-target",
    )
    audible = await orch.handle_barge_in_interrupt(400, utterance_id="utt-a")
    assert "alpha" in audible or audible
    assert orch.ctx.emitted_utterances["utt-a"]["action_id"] == aid
    assert "bravo" not in (orch.ctx.emitted_utterances["utt-a"]["text"])


@pytest.mark.asyncio
async def test_remedy_is_ledgered_before_emit_and_skipped_on_failure(orchestrator_factory, monkeypatch):
    orch, hooks = orchestrator_factory()
    await orch.start()
    offer = {
        "advisory_id": "adv-1",
        "customer_text": "Recommended next step: check the account fee.",
    }
    await orch._emit_ledgered_remedy(offer)
    assert hooks.turns[-1]["text"] == offer["customer_text"]
    rows = [a for a in list_actions(orch.ctx.interaction_id) if a["action_type"] == "remedy_offered"]
    assert rows and "account fee" in (rows[-1].get("output_summary") or "")

    def _boom(*_a, **_k):
        raise OSError("ledger down")

    monkeypatch.setattr("src.agents.orchestrator.record_action", _boom)
    before = len(hooks.turns)
    with pytest.raises(OSError):
        await orch._emit_ledgered_remedy(offer)
    assert len(hooks.turns) == before


@pytest.mark.asyncio
async def test_handoff_timeout_does_not_speak_after_takeover(orchestrator_factory):
    orch, hooks = orchestrator_factory()
    await orch.start()
    await orch.accept_handoff(force=True, source="test")
    assert orch.ctx.state == "HANDOFF_PENDING"
    from datetime import datetime, timedelta, timezone

    from src.data.warehouse import ops_con

    with ops_con() as con:
        con.execute(
            "UPDATE handoff_requests SET sla_due_at = ? WHERE interaction_id = ?",
            [
                datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(seconds=5),
                orch.ctx.interaction_id,
            ],
        )

    real_activity = hooks.emit_activity

    async def _claim_during_activity(payload):
        await real_activity(payload)
        await orch.takeover(claimed_by="supervisor_one")

    hooks.emit_activity = _claim_during_activity
    await orch.sweep_handoff_timeout()
    assert orch.ctx.state == "SUPERVISED"
    assert not any("sorry for the wait" in (t.get("text") or "").lower() for t in hooks.turns)


def test_auditor_cannot_write_customer_socket(reset_ops_db, seed_automotive_pack, monkeypatch):
    monkeypatch.setenv("FRONTLINE_AUTH_REQUIRED", "1")
    monkeypatch.setenv("FRONTLINE_API_KEY", "audit-fix-key-32-chars-minimum!")
    monkeypatch.setenv("SESSION_SECRET", "audit-fix-session-secret-32!!")
    monkeypatch.setenv("FRONTLINE_ENABLED", "1")
    monkeypatch.setenv("FRONTLINE_BOOTSTRAP_ADMIN", "1")
    with TestClient(app) as client:
        started = client.post(
            "/api/interactions/start",
            params={"channel": "web_text"},
            headers={"X-API-Key": "audit-fix-key-32-chars-minimum!"},
        )
        assert started.status_code == 200
        iid = started.json()["interaction_id"]
        auditor = issue_session("aud_1", "auditor", issuer_role="admin")["token"]
        with client.websocket_connect(
            f"/ws/interaction/{iid}",
            headers={"x-frontline-session": auditor},
        ) as ws:
            frame = ws.receive_json()
            assert frame.get("code") == "forbidden"


def test_malformed_frame_does_not_fail_the_contact(reset_ops_db, seed_automotive_pack, monkeypatch):
    monkeypatch.delenv("FRONTLINE_API_KEY", raising=False)
    monkeypatch.setenv("FRONTLINE_ENABLED", "1")
    with TestClient(app) as client:
        started = client.post("/api/interactions/start", params={"channel": "web_text"})
        iid = started.json()["interaction_id"]
        with client.websocket_connect(f"/ws/interaction/{iid}") as ws:
            ws.send_json([])
            frame = ws.receive_json()
            assert frame.get("code") == "bad_message"
            detail = client.get(f"/api/interactions/{iid}")
            assert detail.status_code == 200
            assert detail.json().get("status") != "failed"
