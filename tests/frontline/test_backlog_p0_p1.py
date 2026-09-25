"""Gating tests for the 2026-08-16 P0/P1 backlog (safety, keys, RBAC, console)."""

from __future__ import annotations

import io
import json
import re
import urllib.request
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.agents.base import InteractionContext
from src.agents.intake import (
    IntakeAgent,
    _check_kill_switch,
    classify_yes_no,
    escalate_on_for_prompt,
)
from src.api.main import app
from src.api.rbac import PERMS, issue_session
from src.api.routes.interactions import normalize_activity_payload
from src.domains.loader import Gazetteer


def _ctx(pack) -> InteractionContext:
    return InteractionContext(interaction_id="int_backlog_safety", pack=pack)


# ── P0-2 safety answers ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_yes_to_anyone_hurt_escalates_async(pack):
    ctx = _ctx(pack)
    agent = IntakeAgent(ctx)
    r0 = await agent.run(customer_turn="My 2019 Honda CR-V grinds when I brake.")
    assert "hurt" in (r0.get("question") or "").lower()
    r1 = await agent.run(customer_turn="yes")
    assert r1.get("kill_switch")
    assert ctx.safety_flags.get("escalation") is True


@pytest.mark.asyncio
async def test_i_have_got_hurt_escalates(pack):
    ctx = _ctx(pack)
    agent = IntakeAgent(ctx)
    await agent.run(customer_turn="calling about my car")
    r = await agent.run(customer_turn="I have got hurt")
    assert r.get("kill_switch")
    assert ctx.safety_flags.get("escalation") is True


@pytest.mark.asyncio
async def test_not_safe_location_escalates(pack):
    ctx = _ctx(pack)
    agent = IntakeAgent(ctx)
    await agent.run(customer_turn="My 2019 Honda CR-V grinds when I brake.")
    r = await agent.run(customer_turn="No, nobody is hurt.")
    assert not r.get("kill_switch")
    assert "safe location" in (r.get("question") or "").lower()
    r2 = await agent.run(customer_turn="no")
    assert r2.get("kill_switch")
    assert ctx.safety_flags.get("escalation") is True


@pytest.mark.asyncio
async def test_nobody_hurt_then_safe_does_not_escalate(pack):
    ctx = _ctx(pack)
    agent = IntakeAgent(ctx)
    await agent.run(customer_turn="My 2019 Honda CR-V grinds when I brake.")
    await agent.run(customer_turn="No, nobody is hurt.")
    r = await agent.run(customer_turn="Yes, I'm in a safe location.")
    assert not r.get("kill_switch")
    assert not ctx.safety_flags.get("escalation")


def test_hurt_in_lexicon_but_negation_does_not_trip(pack):
    ctx = _ctx(pack)
    assert _check_kill_switch("I have got hurt", ctx) == "hurt"
    assert _check_kill_switch("No, nobody is hurt", ctx) is None
    assert classify_yes_no("yes") == "yes"
    assert classify_yes_no("No, nobody is hurt.") == "no"
    assert escalate_on_for_prompt("Is anyone hurt?") == "yes"
    assert escalate_on_for_prompt("Are you in a safe location right now?") == "no"


def test_safety_state_lives_in_slots_not_attr(pack):
    src = Path("src/agents/intake.py").read_text(encoding="utf-8")
    assert "__safety_questions_asked__" in src
    assert "__safety_pending__" in src
    assert "setdefault(\"__safety_questions_asked__\", \"\")" not in src


# ── P0-3 provider keys ───────────────────────────────────────────────────────


def test_claude_key_is_not_sent_to_openai(monkeypatch):
    monkeypatch.setenv("FRONTLINE_LLM_ENABLED", "1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    # The base-URL overrides are ambient on developer machines; a test that
    # asserts the default endpoint must not inherit them.
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("FRONTLINE_LLM_ALLOWED_HOSTS", raising=False)
    monkeypatch.setenv("CLAUDE_API_KEY", "sk-ant-secret-should-not-leave")
    from src.ai import provider as prov

    monkeypatch.setattr(prov, "_openai_key", lambda: "")
    monkeypatch.setattr(prov, "_claude_key", lambda: "sk-ant-secret-should-not-leave")

    captured: dict = {}

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, *a):
            return json.dumps(
                {"content": [{"type": "text", "text": "hello from claude"}]}
            ).encode()

    def _urlopen(req, timeout=8.0):
        captured["url"] = req.full_url
        captured["headers"] = {k.lower(): v for k, v in req.headers.items()}
        captured["body"] = json.loads(req.data.decode())
        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
    text = prov._http_chat("sys", "user", "")
    assert text == "hello from claude"
    assert "api.anthropic.com" in captured["url"]
    assert "openai.com" not in captured["url"]
    assert captured["headers"].get("x-api-key") == "sk-ant-secret-should-not-leave"
    assert "authorization" not in captured["headers"]
    assert captured["headers"].get("anthropic-version") == "2023-06-01"


def test_openai_path_does_not_fall_back_to_claude_key(monkeypatch):
    from src.ai import provider as prov

    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("FRONTLINE_LLM_ALLOWED_HOSTS", raising=False)
    monkeypatch.setattr(prov, "_openai_key", lambda: "sk-openai")
    monkeypatch.setattr(prov, "_claude_key", lambda: "sk-ant-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("CLAUDE_API_KEY", "sk-ant-secret")
    captured: dict = {}

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, *a):
            return json.dumps(
                {"choices": [{"message": {"content": "ok"}}]}
            ).encode()

    def _urlopen(req, timeout=8.0):
        captured["url"] = req.full_url
        captured["headers"] = {k.lower(): v for k, v in req.headers.items()}
        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
    prov._http_chat("sys", "user", "gpt-4o-mini")
    assert "openai.com" in captured["url"] or captured["url"].endswith("/chat/completions")
    assert captured["headers"]["authorization"] == "Bearer sk-openai"
    assert "sk-ant-secret" not in captured["headers"].get("authorization", "")


# ── P0-4 RBAC ────────────────────────────────────────────────────────────────


def test_every_defined_perm_has_require_perm_site():
    root = Path("src")
    blob = "\n".join(p.read_text(encoding="utf-8") for p in root.rglob("*.py"))
    found = set(re.findall(r'require_perm(?:_dep)?\(\s*(?:role,\s*)?"([^"]+)"', blob))
    found |= set(re.findall(r"require_perm(?:_dep)?\(\s*(?:role,\s*)?'([^']+)'", blob))
    defined: set[str] = set()
    for grants in PERMS.values():
        defined |= set(grants)
    defined.discard("*")
    missing = sorted(defined - found)
    assert missing == [], f"PERMS without require_perm site: {missing}"


def test_agent_session_cannot_takeover(reset_ops_db, seed_automotive_pack, monkeypatch):
    monkeypatch.setenv("FRONTLINE_API_KEY", "pilot-secret-test")
    monkeypatch.setenv("SESSION_SECRET", "test-session-secret-takeover")
    monkeypatch.setenv("FRONTLINE_AUTH_REQUIRED", "1")
    monkeypatch.delenv("FRONTLINE_OPEN_MODE", raising=False)
    agent_tok = issue_session("alice", "agent", issuer_role="admin")["token"]
    sup_tok = issue_session("bob", "supervisor", issuer_role="admin")["token"]
    with TestClient(app) as client:
        started = client.post(
            "/api/interactions/start",
            params={"channel": "web_text"},
            headers={"X-API-Key": "pilot-secret-test"},
        )
        assert started.status_code == 200
        iid = started.json()["interaction_id"]
        denied = client.post(
            f"/api/interactions/{iid}/takeover",
            headers={
                "X-API-Key": "pilot-secret-test",
                "X-Frontline-Session": agent_tok,
            },
        )
        assert denied.status_code == 403
        allowed = client.post(
            f"/api/interactions/{iid}/takeover",
            headers={
                "X-API-Key": "pilot-secret-test",
                "X-Frontline-Session": sup_tok,
            },
        )
        assert allowed.status_code == 200
        assert allowed.json()["supervised"] is True
        rel = client.post(
            f"/api/interactions/{iid}/release",
            headers={
                "X-API-Key": "pilot-secret-test",
                "X-Frontline-Session": agent_tok,
            },
        )
        assert rel.status_code == 403


def test_auditor_cannot_write_case(reset_ops_db, monkeypatch):
    monkeypatch.setenv("FRONTLINE_API_KEY", "pilot-secret-test")
    monkeypatch.setenv("SESSION_SECRET", "test-session-secret-case")
    monkeypatch.setenv("FRONTLINE_AUTH_REQUIRED", "1")
    monkeypatch.delenv("FRONTLINE_OPEN_MODE", raising=False)
    tok = issue_session("aud", "auditor", issuer_role="admin")["token"]
    from src.data.warehouse import ops_con
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    with ops_con() as con:
        con.execute(
            """
            INSERT INTO cases (
                case_id, interaction_id, pack_id, created_at,
                category, description_summary, onset, severity,
                severity_source, priority, safety_flags,
                advisory_match_id, cluster_match_id, similar_record_count,
                investigation_id, status, followup_draft
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                "case_rbac_1",
                "int_rbac_1",
                "automotive_nhtsa",
                now,
                "ENGINE",
                "x",
                now,
                "Low",
                "rules",
                3,
                "{}",
                None,
                None,
                0,
                None,
                "open",
                "",
            ],
        )
    with TestClient(app) as client:
        r = client.patch(
            "/api/frontline/cases/case_rbac_1",
            json={"status": "closed"},
            headers={
                "X-API-Key": "pilot-secret-test",
                "X-Frontline-Session": tok,
            },
        )
        assert r.status_code == 403


# ── P1-3 activity wire shape ─────────────────────────────────────────────────


def test_activity_payload_has_both_summary_keys():
    live = normalize_activity_payload(
        {"agent": "triage", "action_type": "severity_scored", "summary": "severity=Low"},
        interaction_id="int_x",
    )
    assert live["summary"] == "severity=Low"
    assert live["output_summary"] == "severity=Low"
    polled = normalize_activity_payload(
        {
            "action_id": "act_1",
            "agent": "triage",
            "action_type": "severity_scored",
            "output_summary": "severity=Low",
        },
        interaction_id="int_x",
    )
    assert polled["summary"] == "severity=Low"
    assert polled["output_summary"] == "severity=Low"


# ── P1-6 short gazetteer tokens ──────────────────────────────────────────────


def test_lowercase_is_does_not_match_lexus_is():
    gaz = Gazetteer(
        name="models",
        values=("IS", "CR-V"),
        value_set=frozenset({"is", "cr-v"}),
        canonical={"is": "IS", "cr-v": "CR-V"},
    )
    assert gaz.match_substring("I am facing my Indian is not working") is None
    assert gaz.match_substring("The Lexus IS overheats") == "IS"
    assert gaz.match_substring("My CR-V grinds") == "CR-V"


# ── P1-1 / P1-2 / P1-4 / P1-5 source contracts ───────────────────────────────


def test_hot_async_paths_offload_ops_io():
    orch = Path("src/agents/orchestrator.py").read_text(encoding="utf-8")
    assert "ops_in_thread" in orch
    inter = Path("src/api/routes/interactions.py").read_text(encoding="utf-8")
    assert "ops_in_thread" in inter
    assert "FRONTLINE_CONSOLE_POLL_S" in inter
    assert "emit_heard_turn" in inter
    case = Path("src/agents/case_agent.py").read_text(encoding="utf-8")
    assert "ops_in_thread" in case


def test_call_widget_resets_speak_phase_on_text_fallback():
    src = Path("dashboard/routes/CallWidget.jsx").read_text(encoding="utf-8")
    # submitTextFallback must clear greeting phase, not only speakingRef.
    fn = src.split("function submitTextFallback")[1].split("function ")[0]
    assert "speakPhaseRef.current" in fn
    assert '"normal"' in fn or "'normal'" in fn


def test_live_console_renders_customer_turns_and_dedupes():
    src = Path("dashboard/routes/LiveContactConsole.jsx").read_text(encoding="utf-8")
    assert 'msg.type === "customer_turn"' in src or "customer_turn" in src
    assert "output_summary || msg.summary" in src or "output_summary || a.summary" in src


# ── P2 hygiene ───────────────────────────────────────────────────────────────


def test_p2_interactions_router_has_no_dead_code():
    src = Path("src/api/routes/interactions.py").read_text(encoding="utf-8")
    assert "check_api_key" not in src
    assert "async def _get_active" not in src
    assert "noqa: F401" not in src
    assert 'detail=f"{type(e).__name__}: {e}"' not in src


def test_ingest_does_not_echo_internal_errors(reset_ops_db, seed_automotive_pack, monkeypatch):
    from fastapi.testclient import TestClient

    from src.api.main import app
    from src.frontline import ingest as ingest_mod

    async def _boom(_payload):
        raise RuntimeError("/secret/db/path duckdb row leaked")

    monkeypatch.setattr(ingest_mod, "ingest_complaint", _boom)
    with TestClient(app) as client:
        r = client.post("/api/interactions/ingest", json={"description": "x"})
    assert r.status_code == 400
    detail = str(r.json().get("detail") or "")
    assert "secret" not in detail
    assert "duckdb" not in detail.lower()
    assert "RuntimeError" not in detail


def test_call_widget_reconnect_does_not_spam_ws_error():
    src = Path("dashboard/routes/CallWidget.jsx").read_text(encoding="utf-8")
    onerror = src.split("ws.onerror")[1].split("ws.onclose")[0]
    assert "reconnectAttemptRef" in onerror
    assert "isReconnect" in onerror


def test_start_call_resets_stale_call_state():
    src = Path("dashboard/routes/CallWidget.jsx").read_text(encoding="utf-8")
    fn = src.split("async function startCall()")[1].split("let startRes")[0]
    assert "setMicGranted(null)" in fn
    assert 'setWsStatus("connecting")' in fn
    assert "speakPhaseRef.current" in fn


def test_pack_install_does_not_write_tracked_registry(tmp_path, monkeypatch):
    from src.config import REPO_ROOT
    from src.domains.marketplace import pack_install, registry_path

    tracked = REPO_ROOT / "domains" / "registry.json"
    before = tracked.read_text(encoding="utf-8") if tracked.is_file() else ""
    dest = tmp_path / "reg.json"
    dest.write_text(before, encoding="utf-8")
    monkeypatch.setenv("PACK_REGISTRY_PATH", str(dest))
    pack_install("automotive_nhtsa")
    after = tracked.read_text(encoding="utf-8") if tracked.is_file() else ""
    assert after == before
    assert "pack-install" in dest.read_text(encoding="utf-8")


def test_export_locker_honors_isolated_dir(tmp_path, monkeypatch, reset_ops_db):
    from src.qubot.locker import export_locker, locker_dir

    monkeypatch.setenv("QUBOT_LOCKER_DIR", str(tmp_path / "lockers"))
    from src.data.warehouse import ops_con
    from src.data.timeutil import utc_now

    iid = "int_locker_iso"
    with ops_con() as con:
        con.execute(
            """
            INSERT INTO interactions
            (interaction_id, pack_id, pack_version, started_at, channel, status, supervised, llm_calls)
            VALUES (?, 'automotive_nhtsa', 't', ?, 'web_text', 'completed', FALSE, 0)
            """,
            [iid, utc_now()],
        )
    path = export_locker(iid)
    assert path.parent == locker_dir()
    assert str(tmp_path) in str(path)
    tracked = Path("reports/qubot/lockers") / f"{iid}.json"
    assert not tracked.exists()


@pytest.mark.asyncio
async def test_webhook_ingest_does_not_invent_safety_answers(reset_ops_db, seed_automotive_pack):
    from src.frontline.ingest import ingest_complaint

    out = await ingest_complaint(
        {
            "pack_id": "automotive_nhtsa",
            "entity_1": "2019",
            "entity_2": "HONDA",
            "entity_3": "CR-V",
            "category": "SERVICE BRAKES",
            "description": "grinding when braking at low speed",
        }
    )
    assert out["state"] in ("DONE", "CLOSING")
    from src.qubot.retrievers import contact_audit

    audit = contact_audit(out["interaction_id"])
    customer = " ".join(
        t["text"] for t in audit["turns"] if t.get("speaker") == "customer"
    )
    assert "Nobody is hurt" not in customer
    assert "grinding" in customer.lower()
    assert not (out.get("severity") == "Critical" and "hurt" in customer.lower())


@pytest.mark.asyncio
async def test_webhook_ingest_still_escalates_on_injury_language(
    reset_ops_db, seed_automotive_pack
):
    from src.frontline.ingest import ingest_complaint

    out = await ingest_complaint(
        {
            "pack_id": "automotive_nhtsa",
            "entity_1": "2019",
            "entity_2": "HONDA",
            "entity_3": "CR-V",
            "category": "SERVICE BRAKES",
            "description": "the passenger is hurt and bleeding after the crash",
        }
    )
    assert out["severity"] == "Critical"
    from src.data.warehouse import ops_con

    with ops_con(read_only=True) as con:
        row = con.execute(
            "SELECT outcome FROM interactions WHERE interaction_id = ?",
            [out["interaction_id"]],
        ).fetchone()
    assert row is not None
    assert row[0] == "escalated_safety"


@pytest.mark.asyncio
async def test_car_seat_synonym_maps_to_gazetteer(pack):
    ctx = _ctx(pack)
    agent = IntakeAgent(ctx)
    res = await agent.run(
        customer_turn="The car seat latch failed on my 2019 Honda CR-V"
    )
    assert res["extracted"].get("category") == "CHILD SEAT"


def test_rate_limit_key_splits_principals():
    from src.api.limiter import rate_limit_key

    class _Req:
        def __init__(self, headers, cookies=None):
            self.headers = headers
            self.cookies = cookies or {}

    a = rate_limit_key(_Req({"x-api-key": "alice-secret"}))
    b = rate_limit_key(_Req({"x-api-key": "bob-secret"}))
    assert a != b
    assert a.startswith("key:")
    assert "alice-secret" not in a
    sess = rate_limit_key(_Req({"x-frontline-session": "tok-1"}))
    assert sess.startswith("sess:")
    assert "tok-1" not in sess

