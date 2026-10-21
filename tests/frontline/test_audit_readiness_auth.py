"""Audit findings: agent console/replay, open-mode ingest, query-string keys.

These tests call the production console filter, the REST replay route, the
HTTP ingest route, and the API-key dependency. They do not reimplement them.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient


SECRET = "zebra-coach-991"


def _dump(value) -> str:
    return json.dumps(value, default=str)


def test_agent_console_frame_drops_supervisor_coaching():
    """The sender filter is what console clients receive."""
    from src.api.routes.interactions import _prepare_console_frame

    whisper = {
        "type": "supervisor_whisper",
        "interaction_id": "ix_coach",
        "action_type": "supervisor_whisper",
        "whisper_text": f"{SECRET} hold the line",
        "whisper_audio_ssml": f"<speak>{SECRET}</speak>",
        "diagnostic_summary": {"reported_symptom": f"{SECRET} brake fade"},
        "suggestions": [f"ask about {SECRET}"],
    }
    coach = {
        "type": "agent_activity",
        "interaction_id": "ix_coach",
        "action_type": "coach_whisper",
        "output_summary": f"{SECRET} private note",
        "text": f"{SECRET} spoken aside",
    }
    human = {
        "type": "human_turn",
        "interaction_id": "ix_coach",
        "speaker": "human",
        "text": f"{SECRET} takeover line",
    }
    public = {
        "type": "agent_turn",
        "interaction_id": "ix_coach",
        "speaker": "agent",
        "text": "public reply stays visible",
    }

    for frame in (whisper, coach, human):
        agent_frame = _prepare_console_frame(frame, "agent")
        assert SECRET not in _dump(agent_frame)
        assert agent_frame.get("restricted") is True
        sup_frame = _prepare_console_frame(frame, "supervisor")
        assert SECRET in _dump(sup_frame)

    # A role with export but without ledger read still loses the coaching.
    dsr_frame = _prepare_console_frame(whisper, "dsr_officer")
    assert SECRET not in _dump(dsr_frame)

    kept = _prepare_console_frame(public, "agent")
    assert kept["text"] == "public reply stays visible"
    assert "restricted" not in kept


def test_agent_rest_replay_hides_supervisor_text(reset_ops_db, monkeypatch):
    """GET /api/interactions/{id} is the REST replay path."""
    from src.api.main import app
    from src.api.rbac import issue_session
    from src.data.warehouse import ops_con
    from src.ledger.writer import AgentAction, record_action

    monkeypatch.setenv("FRONTLINE_AUTH_REQUIRED", "1")
    monkeypatch.delenv("FRONTLINE_OPEN_MODE", raising=False)
    monkeypatch.setenv("SESSION_SECRET", "super-secret-key-for-test-32bytes")
    monkeypatch.setenv("FRONTLINE_BOOTSTRAP_ADMIN", "1")

    iid = "ix_replay_coach"
    now = datetime.now(timezone.utc)
    with ops_con() as con:
        con.execute(
            """
            INSERT INTO interactions
            (interaction_id, pack_id, pack_version, started_at, channel,
             status, supervised, llm_calls)
            VALUES (?, 'automotive_nhtsa', '1', ?, 'web_text', 'active', FALSE, 0)
            """,
            [iid, now],
        )
        con.execute(
            """
            INSERT INTO interaction_turns
            (turn_id, interaction_id, seq, speaker, text, ts)
            VALUES ('turn_sup_1', ?, 1, 'supervisor', ?, ?)
            """,
            [iid, f"{SECRET} supervisor said this", now],
        )
    record_action(
        AgentAction(
            interaction_id=iid,
            agent="supervisor",
            action_type="coach_whisper",
            output_summary=f"{SECRET} coaching script",
            input_summary=f"{SECRET} customer context",
        )
    )

    agent = issue_session("desk", "agent")["token"]
    supervisor = issue_session("lead", "supervisor", issuer_role="admin")["token"]
    officer = issue_session("privacy", "dsr_officer", issuer_role="admin")["token"]

    with TestClient(app) as client:
        agent_body = client.get(
            f"/api/interactions/{iid}",
            headers={"X-Frontline-Session": agent},
        )
        assert agent_body.status_code == 200, agent_body.text
        assert SECRET not in agent_body.text

        officer_body = client.get(
            f"/api/interactions/{iid}",
            params={"scrub_pii": "false"},
            headers={"X-Frontline-Session": officer},
        )
        assert officer_body.status_code == 200, officer_body.text
        assert SECRET not in officer_body.text

        sup_body = client.get(
            f"/api/interactions/{iid}",
            params={"scrub_pii": "false"},
            headers={"X-Frontline-Session": supervisor},
        )
        assert sup_body.status_code == 200, sup_body.text
        assert f"{SECRET} supervisor said this" in sup_body.text
        assert f"{SECRET} coaching script" in sup_body.text


def test_open_mode_ingest_requires_admin_while_other_routes_stay_open(
    reset_ops_db, monkeypatch
):
    from src.api.main import app

    monkeypatch.setenv("FRONTLINE_OPEN_MODE", "1")
    monkeypatch.delenv("FRONTLINE_AUTH_REQUIRED", raising=False)
    monkeypatch.setenv("FRONTLINE_API_KEY", "pilot-service-key")
    monkeypatch.setenv("FRONTLINE_ENABLED", "1")
    monkeypatch.setenv("SESSION_SECRET", "super-secret-key-for-test-32bytes")

    with TestClient(app) as client:
        packs = client.get("/api/packs")
        assert packs.status_code == 200, packs.text

        anon = client.post("/api/frontline/pack-builder/ingest", json={})
        assert anon.status_code == 401

        service = client.post(
            "/api/frontline/pack-builder/ingest",
            headers={"X-API-Key": "pilot-service-key"},
            json={"pack_id": "csv_http_proof"},
        )
        assert service.status_code == 403

        from src.api.rbac import issue_session

        # Open mode refuses to mint an elevated session, so this stays an agent.
        forced = issue_session("boss", "admin", issuer_role="admin")["token"]
        elevated = client.post(
            "/api/frontline/pack-builder/ingest",
            headers={"X-Frontline-Session": forced},
            json={"pack_id": "csv_http_proof"},
        )
        assert elevated.status_code == 403

    monkeypatch.delenv("FRONTLINE_OPEN_MODE", raising=False)
    monkeypatch.setenv("FRONTLINE_AUTH_REQUIRED", "1")
    monkeypatch.setenv("FRONTLINE_BOOTSTRAP_ADMIN", "1")
    from src.api.rbac import issue_session

    admin = issue_session("boss", "admin", issuer_role="admin")["token"]
    with TestClient(app) as client:
        passed = client.post(
            "/api/frontline/pack-builder/ingest",
            headers={"X-Frontline-Session": admin},
            json={},
        )
        assert passed.status_code == 400, passed.text
        assert passed.status_code not in (401, 403)


def test_query_string_api_key_does_not_authenticate(monkeypatch):
    import anyio

    from src.api.auth import require_api_key, require_ws_api_key
    from src.api.main import app

    monkeypatch.setenv("FRONTLINE_ALLOW_QUERY_KEY", "1")
    monkeypatch.setenv("FRONTLINE_AUTH_REQUIRED", "1")
    monkeypatch.delenv("FRONTLINE_OPEN_MODE", raising=False)
    monkeypatch.setenv("FRONTLINE_API_KEY", "header-only-secret")

    with pytest.raises(HTTPException) as ws_exc:
        anyio.run(require_ws_api_key, "header-only-secret", None, None)
    assert ws_exc.value.status_code == 401

    class _Req:
        def __init__(self):
            self.headers = {}
            self.query_params = {"api_key": "header-only-secret"}
            self.cookies = {}

            class _State:
                pass

            self.state = _State()

    with pytest.raises(HTTPException) as http_exc:
        anyio.run(require_api_key, _Req(), None, None)
    assert http_exc.value.status_code == 401

    with TestClient(app) as client:
        query_only = client.get("/api/packs?api_key=header-only-secret")
        assert query_only.status_code == 401, query_only.text
        header = client.get(
            "/api/packs",
            headers={"X-API-Key": "header-only-secret"},
        )
        assert header.status_code == 200, header.text
