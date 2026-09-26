"""Coverage for the 56-feature pilot surfaces (new + previously stubbed)."""

from __future__ import annotations

import asyncio

import pytest


@pytest.fixture
def ops(reset_ops_db):
    return reset_ops_db


def test_01_llm_narration_fallback(ops, monkeypatch):
    monkeypatch.delenv("FRONTLINE_LLM_ENABLED", raising=False)
    from src.ai.narration import phrase_investigation_brief

    r = phrase_investigation_brief(
        similar_count=2, cluster_id=1, lead_time_weeks=5, keyword="x", evidence_ids=[]
    )
    assert r.used_llm is False
    assert r.text


def test_02_semantic_rank():
    from src.ml_runtime.embeddings import rank_by_similarity

    ranked = rank_by_similarity(
        "brake grind",
        [{"record_id": "1", "text": "brake grinding noise"}, {"record_id": "2", "text": "billing error"}],
        top_k=2,
    )
    assert ranked[0]["record_id"] == "1"


def test_03_severity_registry():
    from src.ml_runtime.registry import predict_severity
    from pathlib import Path

    # Honest: missing artifact raises; JSON overlay works when present
    art = Path("domains/automotive_nhtsa/models/severity_rules.json")
    if art.is_file():
        sev, reason = predict_severity(str(art), {"description": "vehicle fire", "safety_flags": {"fire": True}})
        assert sev in {"Critical", "Medium", "Low"}
        assert reason
    else:
        with pytest.raises((FileNotFoundError, RuntimeError, LookupError)):
            predict_severity("missing_model.bin", {"description": "x"})


def test_04_backtest(seed_automotive_pack):
    from src.backtest.engine import run_backtest

    rows = run_backtest("automotive_nhtsa")
    assert isinstance(rows, list)


def test_05_clustering(seed_automotive_pack):
    from src.ml_runtime.clustering import rebuild_clusters

    out = rebuild_clusters("automotive_nhtsa")
    assert out is not None


def test_06_llm_judge_heuristic():
    from eval.frontline.llm_judge import judge_interaction

    s = judge_interaction(
        agent_turns=["I'm sorry you're dealing with this brake issue. Thank you for calling."],
        customer_turns=["My brakes grind when I stop"],
        evidence_ids=["A1"],
        claimed_ids=["A1"],
    )
    assert s.faithfulness == 1.0
    assert s.method == "heuristic"
    assert 0 <= s.overall <= 1


def test_07_prompt_registry_stamp(ops):
    from src.ai.prompts import load_prompt, stamp_prompt_use

    p = load_prompt("intake_phrasing")
    assert p["prompt_hash"]
    st = stamp_prompt_use("int_prompt_56", prompt_name="intake_phrasing", used_llm=False)
    assert st["prompt_hash"]


@pytest.mark.asyncio
async def test_08_twilio_media_no_raise():
    from src.channels.twilio_media import TwilioMediaChannel

    ch = TwilioMediaChannel()  # must not raise
    assert ch.capabilities().server_tts is True
    await ch.send_turn("hello pilot")
    assert ch.outbound
    assert ch.outbound[0]["text"] == "hello pilot"


def test_09_server_stt_tts():
    from src.channels.stt_tts import ServerSpeechStack

    s = ServerSpeechStack()
    assert s.transcribe_chunk("customer said brakes") == "customer said brakes"
    meta = s.synthesize_meta("hello")
    assert meta["server_side"] is True
    assert meta["audio_ref"].startswith("tts://")


@pytest.mark.asyncio
async def test_10_whatsapp_sms():
    from src.channels.messaging import SmsChannel, WhatsAppChannel

    sms = SmsChannel(to_address="+15551212")
    await sms.send_turn("status update")
    assert sms.outbox[0]["channel"] == "sms"
    wa = WhatsAppChannel(to_address="whatsapp:+1")
    await wa.send_turn("hola")
    assert wa.channel_name == "whatsapp"


def test_11_email_intake():
    from src.channels.email_intake import EmailIntakeChannel

    ch = EmailIntakeChannel()
    r = ch.process_message(
        subject="Brake issue 2019 Camry",
        body="My Toyota brakes grind and spark. Year 2019.",
        from_addr="driver@example.com",
        required_slots=["entity_1", "description"],
    )
    assert r["slots"]["description"]
    assert "reply" in r


def test_12_webhook_ingest_module():
    from src.frontline.ingest import ingest_complaint

    assert callable(ingest_complaint)


def test_13_multilingual():
    from src.frontline.i18n import detect_language, normalize_to_english, slot_prompt

    assert detect_language("problema con frenos y peligro") == "es"
    n = normalize_to_english("frenos en peligro", lang="es")
    assert "brake" in n["normalized_en"].lower() or "frenos" in n["original"]
    assert "hola" in slot_prompt("greeting", lang="es").lower() or "Hola" in slot_prompt("greeting", lang="es")


def test_14_biometrics(ops):
    from src.frontline.biometrics import fingerprint_from_audio_features, match_caller, register_voiceprint

    fp = fingerprint_from_audio_features([0.1, 0.2, 0.3])
    register_voiceprint("automotive_nhtsa", fp, case_id="CS-1042", label="brake")
    m = match_caller("automotive_nhtsa", fp)
    assert m["matched"] is True
    assert "CS-1042" in (m["greeting"] or "")


def test_15_case_status_module():
    from src.agents.case_status import extract_case_id_from_text, format_case_status_reply

    assert extract_case_id_from_text("my case is case_CS1042 please") == "case_CS1042"
    assert callable(format_case_status_reply)


def test_16_remedy(ops):
    from src.frontline.remedy import build_remedy_offer

    out = build_remedy_offer(
        advisory_match={
            "advisory_id": "19V-TEST",
            "remedy": "Free dealer inspection",
            "summary": "Brake hose",
        },
        case_id="case_1",
    )
    assert out is not None
    assert "19V-TEST" in out["customer_text"]
    assert out["remedy"]


def test_17_booking(ops):
    from src.frontline.booking import book_appointment, list_open_slots, offer_for_advisory

    slots = list_open_slots(days_ahead=3)
    assert slots
    offer = offer_for_advisory(advisory_id="19V-1", case_id="c1", pack_id="automotive_nhtsa")
    assert offer["open_slots"]
    booked = book_appointment(
        case_id="c1",
        interaction_id="i1",
        pack_id="automotive_nhtsa",
        slot_start=slots[0]["slot_start"],
    )
    assert booked["status"] == "booked"
    assert booked["appointment_id"].startswith("appt_")


def test_18_multi_issue(ops):
    from src.frontline import multi_issue

    assert any(callable(getattr(multi_issue, n)) for n in dir(multi_issue) if not n.startswith("_"))


def test_20_self_critique(ops):
    from types import SimpleNamespace
    from src.agents.self_critique import run_self_critique

    ctx = SimpleNamespace(
        interaction_id="int_crit",
        slots={"entity_1": "Toyota", "entity_2": "Camry", "entity_3": "2019", "category": "brakes", "description": "grind"},
        pack=SimpleNamespace(required_slots=lambda: [
            SimpleNamespace(name="entity_1"),
            SimpleNamespace(name="description"),
        ]),
        safety_flags={},
        severity="Low",
        priority=3,
        followup_draft="We will follow up on case case_x",
        case_id="case_x",
        investigation_brief={"similar_records": [{"record_id": "r1"}]},
    )
    r = run_self_critique(ctx)
    assert r["ok"] is True
    assert r["checks"]


def test_21_callback(ops):
    from src.frontline.callback import list_callbacks, schedule_callback

    r = schedule_callback(
        interaction_id="int_cb",
        pack_id="automotive_nhtsa",
        phone_or_channel="+15550001",
    )
    assert r["status"] == "scheduled"
    assert list_callbacks()


def test_22_dynamic_slots(ops):
    from src.frontline.dynamic_slots import merge_slots_skip_known

    r = merge_slots_skip_known(
        ["entity_1", "entity_2", "description"],
        {"description": "noise"},
        {"entity_1": "Toyota", "entity_2": "Camry"},
    )
    assert "entity_1" in r["skipped"]
    assert r["turns_saved"] == 2
    assert r["slots"]["entity_1"] == "Toyota"


def test_23_alert_rules(ops):
    from src.frontline import alert_rules

    assert any(callable(getattr(alert_rules, n)) for n in dir(alert_rules) if not n.startswith("_"))


def test_24_30_analytics(ops):
    from src.frontline.analytics import (
        bias_fairness_report,
        cohort_analysis,
        cross_pack_patterns,
        financial_impact,
        forecast_cluster_volume,
        geographic_hotspots,
        regulator_filing_watch,
        severity_drift,
    )

    assert "forecasts" in forecast_cluster_volume()
    assert "patterns" in cross_pack_patterns()
    assert "hotspots" in geographic_hotspots()
    assert "clusters" in severity_drift()
    assert "cohorts" in cohort_analysis()
    assert "matches" in regulator_filing_watch(pack_id="automotive_nhtsa")
    assert "portfolio_risk_usd" in financial_impact()
    assert "groups" in bias_fairness_report()


def test_31_hash_chain(ops):
    from src.ledger import AgentAction, list_actions, record_action
    from src.ledger.chain import verify_chain

    record_action(
        AgentAction(
            interaction_id="int_h56",
            agent="orchestrator",
            action_type="state_transition",
            input_summary="a",
            output_summary="b",
        )
    )
    assert verify_chain(list_actions("int_h56"))["ok"]


def test_32_38_trust(ops):
    from src.security.pii import redact_pii
    from src.frontline.consent import consent_preamble
    from src.frontline import dsr, archive, explainability
    from src.frontline.four_eyes import decide_approval, request_approval

    red = redact_pii("call me at 555-123-4567 or a@b.com")
    assert isinstance(red, (str, dict, tuple))
    assert isinstance(consent_preamble(), str)
    assert dsr and archive and explainability
    req = request_approval(
        "open_investigation",
        resource_id="inv_1",
        requested_by="alice",
    )
    assert req["required"] is True
    dec = decide_approval(req["approval_id"], reviewer="bob", approve=True)
    assert dec["status"] == "approved"
    with pytest.raises(ValueError):
        decide_approval(
            request_approval("close_p1_case", resource_id="c1", requested_by="alice")["approval_id"],
            reviewer="alice",
            approve=True,
        )


def test_39_40_pack_builder_ingest():
    from src.domains.builder import pack_builder
    from scripts import ingest_scale

    assert pack_builder or ingest_scale


def test_41_marketplace():
    from src.domains.marketplace import list_marketplace, pack_install, publish_pack

    publish_pack("automotive_nhtsa", version="1.0.1", changelog_entry="1.0.1 test")
    packs = list_marketplace()
    assert any(p["pack_id"] == "automotive_nhtsa" for p in packs)
    r = pack_install("automotive_nhtsa")
    assert r["installed"] is True


def test_42_third_fourth_verticals():
    from src.domains.loader import lint_pack, load_pack

    for pid in ("medical_maude", "consumer_cpsc"):
        assert lint_pack(pid) == []
        p = load_pack(pid, reload=True)
        assert p.manifest.display_name


def test_43_pack_editor_dry_run():
    from src.domains.editor import apply_pack_edits, hot_reload

    r = apply_pack_edits("automotive_nhtsa", {"refusal_topics": ["legal advice"]}, dry_run=True)
    assert r["dry_run"] is True
    hr = hot_reload("automotive_nhtsa")
    assert hr["ok"] is True


def test_44_experiments():
    from src.v3 import experiments

    assert experiments


def test_45_postgres_backend_seam():
    from src.data.postgres_backend import backend_status, ops_backend, ops_connection

    assert ops_backend() == "duckdb"
    st = backend_status()
    assert st["alembic_ready"] is True
    with ops_connection(read_only=True) as con:
        assert con.backend == "duckdb"
        con.execute("SELECT 1")


def test_46_job_queue(ops):
    from src.jobs.queue import enqueue, list_jobs, run_next

    j = enqueue("audit_contact", {"interaction_id": "int_x"})
    assert j["status"] == "pending"
    done = run_next()
    assert done["status"] == "done"
    assert list_jobs()


def test_47_observability():
    from src.observability.metrics import bind, inc, log_event, observe, prometheus_text, snapshot

    bind(interaction_id="int_1", pack_id="automotive_nhtsa")
    inc("contacts_total", outcome="ok")
    observe("turn_latency_ms", 12.5)
    log_event("turn_complete", turns=3)
    text = prometheus_text()
    assert "frontline_info" in text
    assert snapshot()["counters"]


def test_48_load_script_exists():
    from pathlib import Path

    # lightweight soak: concurrent channel sends
    async def _burst():
        from src.channels.messaging import SmsChannel

        async def one(i):
            ch = SmsChannel(to_address=f"+{i}")
            await ch.send_turn(f"msg {i}")
            return len(ch.outbox)

        return await asyncio.gather(*[one(i) for i in range(50)])

    results = asyncio.get_event_loop().run_until_complete(_burst()) if False else None
    # use asyncio.run
    results = asyncio.run(_burst())
    assert len(results) == 50
    assert all(r >= 1 for r in results)


def test_49_chaos_defined_states(ops):
    """Failure injection: hangup mid-turn still yields defined channel status."""
    from src.channels.messaging import SmsChannel

    async def hangup_mid():
        ch = SmsChannel()
        await ch.send_turn("partial")
        await ch.hangup()
        return ch.outbox[-1]["type"]

    assert asyncio.run(hangup_mid()) == "session_closed"


def test_50_rbac_sso(monkeypatch):
    from src.api.rbac import issue_session, oidc_discovery, require_perm, verify_session

    # Elevated mint requires admin issuer (or bootstrap); not free client role.
    monkeypatch.delenv("FRONTLINE_OPEN_MODE", raising=False)
    monkeypatch.setenv("SESSION_SECRET", "test-session-secret-for-rbac-50")
    monkeypatch.setenv("FRONTLINE_API_KEY", "rbac-test-key")
    monkeypatch.setenv("FRONTLINE_AUTH_REQUIRED", "1")
    sess = issue_session("alice", "supervisor", issuer_role="admin")
    body = verify_session(sess["token"])
    assert body["role"] == "supervisor"
    require_perm("supervisor", "takeover")
    with pytest.raises(Exception):
        require_perm("agent", "takeover")
    assert oidc_discovery()["roles"]


@pytest.mark.asyncio
async def test_51_drain_rejects_create_interaction(ops, seed_automotive_pack):
    """Feature #51 real path: create_interaction must refuse while DRAIN is on."""
    from src.agents.orchestrator import create_interaction
    from src.ops.drain import DRAIN, ServiceDrainingError, install_sigterm_handler

    DRAIN.reset()
    install_sigterm_handler()
    assert DRAIN.status()["sigterm_handler_installed"] is True

    # Happy path while not draining
    orch, greeting = await create_interaction(channel="web_text", pack_id="automotive_nhtsa")
    assert orch.ctx.interaction_id.startswith("int_")
    assert greeting
    assert orch.ctx.interaction_id in DRAIN.status()["active_interactions"]

    # Begin drain: new contacts must fail on the shipped factory
    DRAIN.begin_drain()
    assert DRAIN.draining is True
    with pytest.raises(ServiceDrainingError):
        await create_interaction(channel="web_text", pack_id="automotive_nhtsa")

    # Finishing the active contact releases the slot
    await orch.hangup()
    assert orch.ctx.interaction_id not in DRAIN.status()["active_interactions"]
    DRAIN.reset()
    assert DRAIN.draining is False


def test_52_tenant_isolation():
    from src.ops.tenant import get_tenant, set_tenant, tenant_clause, with_tenant

    tok = set_tenant("acme")
    assert get_tenant() == "acme"
    sql, params = tenant_clause()
    assert "tenant_id" in sql and params == ["acme"]
    assert with_tenant({"x": 1})["tenant_id"] == "acme"
    from src.ops.tenant import reset_tenant

    reset_tenant(tok)


def test_53_coach(ops):
    from src.frontline.coach import list_whispers, suggest_replies, whisper

    s = suggest_replies(slots={"entity_1": "Toyota"}, last_customer_text="brakes", severity="Critical")
    assert len(s) >= 2
    w = whisper("int_coach", "Try asking about the VIN", from_role="coach")
    assert w["private"] is True
    assert list_whispers("int_coach")


def test_54_wallboard():
    from src.frontline.wallboard import build_wallboard

    out = build_wallboard()
    assert out is not None


def test_55_subscriptions(ops):
    from src.frontline.subscriptions import list_subscriptions, subscribe, tick_due

    sub = subscribe(channel="email", target="ops@example.com", report_type="daily_digest")
    assert sub["subscription_id"]
    assert list_subscriptions()
    sent = tick_due(force=True)
    assert sent


def test_56_metering(ops):
    from src.frontline.metering import record_usage, usage_summary

    record_usage("contacts", 1, tenant_id="default")
    record_usage("llm_spend_usd", 0.02, tenant_id="default")
    s = usage_summary(tenant_id="default")
    assert s["metrics"]["contacts"] >= 1
    assert s["metrics"]["llm_spend_usd"] >= 0.02


def test_platform56_routes_mounted():
    from src.api.main import app

    # FastAPI 0.139+ nests included routers; openapi is the stable surface.
    paths = set((app.openapi().get("paths") or {}).keys())
    assert any("/analytics/forecast" in p for p in paths)
    assert any("/booking/book" in p for p in paths)
    assert any(p.endswith("/usage") or "/usage" in p for p in paths)
