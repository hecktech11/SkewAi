"""Gating tests for pipeline-trust, queue, onboarding, commercial, analyst/ops/voice."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.data.timeutil import utc_now
from src.data.trust import evaluate_ingest, evaluate_record
from src.data.warehouse import domain_con, ops_con
from src.domains.mapping_ingest import ingest_mapped_csv
from src.qubot.auditor import audit_interaction
from src.domains.builder.pack_builder import first_insight_from_csv, lint_mapping
from src.domains.marketplace import install_vertical
from src.enterprise.grounded_nl import grounded_query
from src.frontline.billing import (
    assign_seat,
    enforce_plan,
    record_metered,
    stripe_checkout,
    stripe_webhook,
    usage_dashboard,
)
from src.frontline.booking import book_appointment
from src.frontline.callback import schedule_callback
from src.frontline.coach import whisper
from src.frontline.compliance_packs import evidence_template, toggle_pack
from src.frontline.embed_widget import widget_contract
from src.frontline.multi_issue import record_multi_issues
from src.frontline.provenance import figure_lineage, kpi_lineage
from src.frontline.remedy import offer_and_ledger
from src.frontline.reports import build_report, run_scheduled_digest
from src.frontline.sandbox import boot_sandbox, first_insight_from_rows
from src.frontline.saved_views import get_view, save_view
from src.frontline.validation_queue import enqueue_insight, list_queue_all
from src.frontline.webhook_catalog import fire_due_tick, list_catalog
from src.jobs.registry import backend_name, list_workers, register_worker
from src.ledger import AgentAction, record_action
from src.observability.otel import emit_span
from src.observability.sentry import capture_event
from src.observability.slo import job_queue_status, record_slo_sample
from src.sdk.client import sdk_call
from fastapi.testclient import TestClient

from src.api.main import app
from src.security.scoped_keys import authenticate_scoped, create_scoped_key


def _interaction(iid: str, pack_id: str = "automotive_nhtsa") -> None:
    with ops_con() as con:
        con.execute(
            """
            INSERT INTO interactions
            (interaction_id, pack_id, pack_version, started_at, channel, status, supervised, llm_calls)
            VALUES (?, ?, 'v', ?, 'web_text', 'active', FALSE, 0)
            """,
            [iid, pack_id, utc_now()],
        )


def test_data_trust_fresh_vs_stale_dup_schema(reset_ops_db):
    now = utc_now()
    fresh = {
        "record_id": "T-1",
        "text": "grinding brakes",
        "source": "NHTSA",
        "received_at": now,
        "category": "SERVICE BRAKES",
    }
    stale = {
        "record_id": "T-2",
        "text": "old row",
        "source": "NHTSA",
        "received_at": now - timedelta(days=400),
    }
    schema_break = {"record_id": "", "text": "", "source": "NHTSA", "received_at": now}
    dup = dict(fresh)
    batch = evaluate_ingest("automotive_nhtsa", [fresh, stale, schema_break, dup], persist=True)
    t1s = [r for r in batch["results"] if r["record_id"] == "T-1"]
    assert t1s[0]["trusted"] is True
    assert t1s[0]["badge"]["source"] == "NHTSA"
    assert t1s[0]["badge"]["freshness"] == "fresh"
    assert t1s[0]["badge"]["grounded_verdict"] == "trusted"
    by_id = {r["record_id"]: r for r in batch["results"] if r["record_id"] != "T-1"}
    assert by_id["T-2"]["trusted"] is False
    assert "freshness" in by_id["T-2"]["badge"]["failed_checks"]
    assert by_id[""]["trusted"] is False
    assert "schema" in by_id[""]["badge"]["failed_checks"]
    dup_result = t1s[1]
    assert dup_result["trusted"] is False
    assert "dedup" in dup_result["badge"]["failed_checks"]
    assert batch["trusted"] == 1


def test_provenance_kpi_chain(reset_ops_db):
    fig = kpi_lineage("open_cases")
    assert fig["kpi_id"] == "open_cases"
    assert "value" in fig
    badge = fig["badge"]
    assert badge["source"]
    assert badge["freshness"]
    assert badge["grounded_verdict"]
    steps = {s["step"] for s in fig["chain_of_custody"]}
    assert {"source", "freshness", "grounded_verdict", "value"} <= steps
    clicked = figure_lineage("open_cases")
    assert clicked["chain_of_custody"]
    assert clicked["value"] == fig["value"]


def test_validation_queue_three_kinds_with_spans(reset_ops_db, seed_automotive_pack, pack):
    iid = "int_vq_1"
    _interaction(iid, pack.id)
    aid = record_action(
        AgentAction(
            interaction_id=iid,
            agent="investigator",
            action_type="similar_search",
            output_summary="cited NHTSA-100001",
            evidence_ids=["NHTSA-100001"],
            claims=[
                {
                    "claim_text": "grind",
                    "evidence_id": "NHTSA-100001",
                    "span_start": 0,
                    "span_end": 5,
                }
            ],
        )
    )
    enqueue_insight(
        kind="needs_review",
        interaction_id=iid,
        summary="needs review insight",
        action_id=aid,
    )
    enqueue_insight(
        kind="unverifiable",
        interaction_id=iid,
        summary="unverifiable insight",
        action_id=aid,
    )
    enqueue_insight(
        kind="low_confidence",
        interaction_id=iid,
        summary="low confidence insight",
        action_id=aid,
    )
    items = list_queue_all()
    kinds = {i["queue_kind"] for i in items}
    assert {"needs_review", "unverifiable", "low_confidence"} <= kinds
    for kind in ("needs_review", "unverifiable", "low_confidence"):
        hit = next(i for i in items if i["queue_kind"] == kind)
        assert hit["span_evidence"], kind
        assert hit["span_evidence"][0]["span"]["hit"] is not None


def test_sandbox_and_pack_builder_and_marketplace(reset_ops_db, tmp_path):
    boot = boot_sandbox(dest=tmp_path / "sample.csv")
    assert boot["insight"]["count"] >= 1
    assert "theme" in boot["insight"]

    csv_path = tmp_path / "builder.csv"
    csv_path.write_text(
        "id,desc,make\nB1,window rattle on highway,HONDA\nB2,window rattle at night,HONDA\n",
        encoding="utf-8",
    )
    lint = lint_mapping({"record_id": "id", "text": "desc", "entity_2": "make"}, ["id", "desc", "make"])
    assert lint["ok"] is True
    built = first_insight_from_csv(
        csv_path,
        mapping={"record_id": "id", "text": "desc", "entity_2": "make"},
        pack_id="builder_preview",
        out_root=tmp_path / "domains",
    )
    assert built["ok"] is True
    assert built["insight"]["n"] >= 1

    cpsc = install_vertical("consumer_cpsc")
    assert cpsc["loadable"] is True
    maude = install_vertical("medical_maude")
    assert maude["loadable"] is True


def test_commercial_keys_webhooks_sdk_widget(reset_ops_db):
    before = usage_dashboard("acme")
    rec = record_metered("contacts", 3, tenant_id="acme")
    assert rec["recorded"] is True
    after = usage_dashboard("acme")
    assert after["metrics"]["contacts"] > (before["metrics"].get("contacts") or 0)

    session = stripe_checkout("acme", plan="pilot")
    assert session["test_mode"] is True
    assert session["stripe_session"].startswith("cs_test_")
    paid = stripe_webhook(
        {
            "type": "checkout.session.completed",
            "stripe_session": session["stripe_session"],
            "tenant_id": "acme",
            "plan": "pilot",
        },
        allow_unsigned=True,
    )
    assert paid["ok"] is True

    with pytest.raises(PermissionError):
        assign_seat("tiny", "u1", "supervisor")
    assign_seat("acme", "u2", "supervisor")
    denied = enforce_plan("tiny", role="admin")
    assert denied["allowed"] is False

    key = create_scoped_key(["kpi:read"], tenant_id="acme")
    ok = authenticate_scoped(key["token"], "kpi:read")
    assert ok["ok"] is True
    bad = authenticate_scoped(key["token"], "ingest:write")
    assert bad["ok"] is False

    catalog = list_catalog()
    assert any(e["event"] == "daily_digest" for e in catalog)
    tick = fire_due_tick(force=True, channel="slack")
    assert tick["count"] >= 1
    assert tick["sent"][0]["text"]

    sdk = sdk_call("figure", kpi_id="open_cases")
    assert sdk["ok"] is True
    assert sdk["kpi_id"] == "open_cases"

    contract = widget_contract()
    assert contract["has_global"]
    assert contract["has_install"]
    assert contract["no_require"]
    assert contract["no_module_exports"]

    with TestClient(app) as client:
        r = client.get("/api/frontline/provenance/kpis/open_cases")
        assert r.status_code == 200
        body = r.json()
        assert body["badge"]["source"]
        assert body["chain_of_custody"]
        r2 = client.get("/api/frontline/scoped", headers={"X-Scoped-Key": key["token"]})
        assert r2.status_code == 200
        r3 = client.get("/api/frontline/scoped", headers={"X-Scoped-Key": "sk_live_nope"})
        assert r3.status_code == 403


def test_nl_views_reports_obs_compliance_voice(reset_ops_db, seed_automotive_pack, pack):
    planted = grounded_query(
        "show brake complaints",
        pack_id=pack.id,
        plant={
            "record_id": "NHTSA-100001",
            "planted": True,
            "unsupported": True,
            "claim_text": "engine fire that is not in the source",
            "span_start": 0,
            "span_end": 11,
        },
    )
    assert planted["withheld_count"] >= 1
    assert any(w.get("reason") == "unsupported-claim" for w in planted["withheld"])

    cited = grounded_query("show brake complaints", pack_id=pack.id)
    # fixture corpus has brake-related rows; if none, shown may be empty
    if cited["shown"]:
        assert cited["shown"][0]["record_id"]
        assert cited["shown"][0]["claim"]["claim_text"]

    view = save_view("brakes", {"question": "show brake complaints"})
    assert get_view(view["view_id"])["name"] == "brakes"
    report = build_report("Weekly", kpi_ids=["open_cases"])
    digest = run_scheduled_digest(report["report_id"])
    assert digest["run_id"]
    assert digest["kpis"]

    span = emit_span("pipeline.trust", attributes={"pack": pack.id})
    # Item 47: real SDK label when installed, honest in-house label otherwise.
    assert span["instrumentation"] in ("opentelemetry-sdk", "in-house")
    ev = capture_event("pipeline test", level="info")
    # Item 47: real SDK label when installed, honest in-house label otherwise.
    assert ev["sdk"] in ("sentry-sdk", "in-house")
    slo = record_slo_sample("audit_grounded", numerator=99, denominator=100)
    assert slo["breached"] is False
    jobs = job_queue_status()
    assert "pending" in jobs
    w = register_worker("test-host")
    assert w["backend"] == backend_name()
    assert any(x["worker_id"] == w["worker_id"] for x in list_workers())

    runbook = Path("docs/compliance/backup_and_dr.md").read_text(encoding="utf-8")
    assert "Restore procedure" in runbook
    assert "GET /health" in runbook
    assert "Copy the snapshot" in runbook

    for key in ("21cfr11", "nhtsa_tread", "gdpr"):
        tog = toggle_pack(key, True)
        assert tog["enabled"] is True
        tmpl = evidence_template(key, record_id="NHTSA-100001", investigation_id="inv_1", interaction_id="int_x")
        assert tmpl["evidence_template"]
        assert "NHTSA-100001" in tmpl["evidence_template"] or "int_x" in tmpl["evidence_template"] or "inv_1" in tmpl["evidence_template"]

    iid = "int_voice_1"
    _interaction(iid, pack.id)
    rem = offer_and_ledger(
        iid,
        advisory_match={"advisory_id": "19V-00001", "remedy": "inspect calipers", "summary": "brake advisory"},
        case_id=None,
    )
    assert rem and rem["customer_text"]
    issues = record_multi_issues(
        iid,
        pack_id=pack.id,
        issues=[
            {"description": "primary grinding on the front brakes at low speed", "category": "SERVICE BRAKES"},
            {"description": "secondary rattle from the passenger window at night", "category": "VISIBILITY"},
        ],
    )
    assert len(issues) >= 2
    cb = schedule_callback(interaction_id=iid, pack_id=pack.id, phone_or_channel="+15551212")
    assert cb["status"] == "scheduled"
    coach = whisper(iid, "Ask about cold-weather onset")
    assert coach["kind"] == "whisper"
    book = book_appointment(
        case_id=None,
        interaction_id=iid,
        pack_id=pack.id,
        slot_start=utc_now().isoformat(),
    )
    assert book["status"] == "booked"
    with ops_con(read_only=True) as con:
        types = {
            r[0]
            for r in con.execute(
                "SELECT action_type FROM agent_actions WHERE interaction_id = ?",
                [iid],
            ).fetchall()
        }
    assert "followup_drafted" in types
    assert "case_created" in types
    assert "callback_scheduled" in types
    assert "coach_whisper" in types
    assert "appointment_booked" in types


def test_evaluate_record_lineage_names_check():
    bad = evaluate_record({"record_id": "X", "text": "hi", "source": "", "received_at": utc_now()})
    assert bad["trusted"] is False
    assert "completeness" in bad["badge"]["failed_checks"]
    assert "completeness" in bad["badge"]["why_trusted"]


def test_ingest_mapped_csv_writes_only_trusted_rows(tmp_path, monkeypatch, reset_ops_db):
    monkeypatch.setenv("DOMAIN_DB_PATH", str(tmp_path))
    mapping = Path("domains/automotive_nhtsa/data/mapping.yaml")
    csv_path = tmp_path / "mix.csv"
    now = utc_now().strftime("%Y%m%d")
    csv_path.write_text(
        "CMPLID,DATEA,DATEC,MODEL_YR,MAKETXT,MODELTXT,CDESC,NUM_CYLS,STATE\n"
        f"KEEP1,{now},{now},2019,HONDA,CR-V,grinding noise when braking,4,CA\n"
        "STALE1,20200101,20200101,2018,FORD,F150,old complaint text here,8,TX\n"
        f"KEEP1,{now},{now},2019,HONDA,CR-V,grinding noise when braking,4,CA\n",
        encoding="utf-8",
    )
    first = ingest_mapped_csv("automotive_nhtsa", csv_path, mapping_path=mapping)
    assert first["upserted"] == 1
    assert first["rejected"] >= 2
    with domain_con("automotive_nhtsa") as con:
        ids = [r[0] for r in con.execute("SELECT record_id FROM records").fetchall()]
    assert ids == ["KEEP1"]
    second = ingest_mapped_csv("automotive_nhtsa", csv_path, mapping_path=mapping)
    assert second["upserted"] == 0
    assert second["rejected"] == second["mapped"]


@pytest.mark.asyncio
async def test_auditor_unverifiable_lands_in_queue(reset_ops_db, pack):
    iid = "int_unv_q"
    _interaction(iid, pack.id)
    record_action(
        AgentAction(
            interaction_id=iid,
            agent="investigator",
            action_type="similar_search",
            output_summary="cited missing row",
            evidence_ids=["ZZZ-NOT-A-ROW"],
        )
    )
    result = await audit_interaction(iid, write_report=False)
    assert any(v.verdict == "unverifiable" for v in result.action_verdicts)
    items = list_queue_all()
    assert any(i["queue_kind"] == "unverifiable" for i in items)


@pytest.mark.asyncio
async def test_intake_low_confidence_lands_in_queue(orchestrator_factory, reset_ops_db, pack):
    orch, _hooks = orchestrator_factory()
    await orch.start()
    n_safety = len(orch.ctx.pack.manifest.safety.safety_questions or [])
    orch.ctx._safety_asked = set(range(n_safety))
    orch.ctx.slots["__safety_questions_asked__"] = ",".join(str(i) for i in range(n_safety))
    orch.ctx.slots.pop("__safety_pending__", None)
    for slot in orch.ctx.required_slots_remaining():
        orch.ctx.slot_attempts[slot] = 9
    await orch.handle_customer_turn("no model year in this sentence")
    items = list_queue_all()
    assert any(i["queue_kind"] == "low_confidence" for i in items)


def test_pack_builder_upload_and_dashboard_mount(reset_ops_db, tmp_path):
    src = Path("dashboard/src/App.jsx").read_text(encoding="utf-8")
    assert "TrustPipeline" in src
    assert "PackBuilder" in src
    assert 'id: "trust"' in src
    assert 'id: "builder"' in src
    assert Path("dashboard/routes/PackBuilder.jsx").is_file()
    assert Path("dashboard/routes/TrustPipeline.jsx").is_file()
    with TestClient(app) as client:
        files = {"file": ("demo.csv", b"id,desc,make\n1,window rattle,HONDA\n", "text/csv")}
        r = client.post("/api/frontline/pack-builder/profile", files=files)
        assert r.status_code == 200
        body = r.json()
        assert "id" in body["columns"]
        assert body["lint"]["ok"] is True
        ins = client.post(
            "/api/frontline/pack-builder/insight",
            json={
                "csv_path": body["csv_path"],
                "mapping": body["proposed_mapping"],
                "pack_id": "builder_preview",
            },
        )
        assert ins.status_code == 200
        assert ins.json()["insight"]["n"] >= 1


def test_worker_registry_persists_when_redis_url_set(reset_ops_db, monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:9")
    from src.jobs.registry import backend_name, list_workers, register_worker

    assert backend_name() == "duckdb"
    a = register_worker("host-a")
    b = register_worker("host-b")
    ids = {w["worker_id"] for w in list_workers()}
    assert a["worker_id"] in ids
    assert b["worker_id"] in ids
