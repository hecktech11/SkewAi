"""Acceptance tests for the five live-pilot engineering items.

FRONTLINE_EMBEDDING_MODE stays legacy throughout. No human labels, no cutover.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.agents.base import InteractionContext, count_turn
from src.agents.orchestrator import BUDGET_WRAP_SCRIPT, Orchestrator
from src.config import settings
from src.data.warehouse import domain_con, ops_con
from src.ids import new_ulid
from src.ml_runtime.embedding_runtime import embedding_mode


def test_embedding_mode_remains_legacy():
    assert embedding_mode() == "legacy"


# ── 0.1 readiness ────────────────────────────────────────────────────────────


def test_health_ready_ok(reset_ops_db, seed_automotive_pack, monkeypatch):
    from src.api.main import app

    monkeypatch.delenv("FRONTLINE_API_KEY", raising=False)
    with TestClient(app) as c:
        # Unauthenticated request receives minimal public response
        r_anon = c.get("/health/ready")
        assert r_anon.status_code == 200
        assert r_anon.json() == {"status": "ready", "ready": True}

        # Authenticated request receives full inspection checks
        monkeypatch.setenv("FRONTLINE_API_KEY", "health_check_key")
        r = c.get("/health/ready", headers={"X-API-Key": "health_check_key"})
        assert r.status_code == 200
        body = r.json()
        assert body["ready"] is True
        assert body["failing_component"] is None
        checks = body["checks"]
        assert checks["pack"]["ok"] is True
        assert checks["gazetteers"]["ok"] is True
        assert checks["gazetteer_lookup"]["ok"] is True
        assert checks["database"]["ok"] is True
        assert checks["domain_warehouse"]["ok"] is True


def test_health_ready_503_names_component(reset_ops_db, seed_automotive_pack, monkeypatch):
    from src.api.main import app
    from src.domains.active_pack import clear_active_pack_override, set_active_pack_id

    monkeypatch.delenv("FRONTLINE_API_KEY", raising=False)
    set_active_pack_id("pack_does_not_exist")
    try:
        with TestClient(app) as c:
            # Unauthenticated request receives minimal public response
            r_anon = c.get("/health/ready")
            assert r_anon.status_code == 503
            assert r_anon.json() == {"status": "not_ready", "ready": False}

            # Authenticated request receives component failure details
            monkeypatch.setenv("FRONTLINE_API_KEY", "health_check_key")
            r = c.get("/health/ready", headers={"X-API-Key": "health_check_key"})
            assert r.status_code == 503
            body = r.json()
            assert body["ready"] is False
            assert body["failing_component"]
            assert body["failing_component"] in body["failing_components"]
            assert "pack" in body["failing_components"]
    finally:
        clear_active_pack_override()


def test_startup_readiness_refuses_when_pack_missing(
    reset_ops_db, seed_automotive_pack, monkeypatch
):
    from src.api.main import _assert_startup_readiness
    from src.domains.active_pack import clear_active_pack_override, set_active_pack_id

    set_active_pack_id("pack_does_not_exist")
    try:
        with pytest.raises(RuntimeError, match="failing_component"):
            _assert_startup_readiness()
    finally:
        clear_active_pack_override()


# ── 2.1 unified turn budget ──────────────────────────────────────────────────


def test_count_turn_excludes_supervisor_and_agent():
    ctx = InteractionContext(
        interaction_id="int_x",
        pack=SimpleNamespace(required_slots=lambda: []),
        channel="web_text",
    )
    ctx.record_turn("customer", "one")
    ctx.record_turn("agent", "ask")
    ctx.record_turn("customer", "two")
    ctx.record_turn("supervisor", "I will take this")
    ctx.record_turn("customer", "three")
    assert count_turn(ctx) == 3
    assert ctx.count_turn() == 3


@pytest.mark.asyncio
async def test_turn_budget_pathological_loop_wraps_exactly(
    orchestrator_factory, monkeypatch
):
    from src.agents import intake as intake_mod
    from src.agents import orchestrator as orch_mod

    capped = replace(settings, max_turns=4)
    monkeypatch.setattr(intake_mod, "settings", capped)
    monkeypatch.setattr(orch_mod, "settings", capped)

    orch, hooks = orchestrator_factory()
    await orch.start()
    low = {"entity_2": 0.05, "description": 0.1}
    for i in range(4):
        if i == 2:
            orch.ctx.record_turn("supervisor", "watching")
        await orch.handle_customer_turn(
            "um not sure really, maybe the thing?",
            final=True,
            asr_confidence=low,
        )
        if orch.ctx.state in ("DONE", "CLOSING", "ABANDONED"):
            break

    assert orch.ctx.count_turn() == 4
    assert orch.ctx.state in ("DONE", "CLOSING", "ABANDONED") or orch.ctx.case_id
    texts = " ".join(hooks.agent_texts())
    assert BUDGET_WRAP_SCRIPT in texts

    prior_state = orch.ctx.state
    prior_count = orch.ctx.count_turn()
    await orch.handle_customer_turn("still going", final=True, asr_confidence=low)
    assert orch.ctx.count_turn() in (prior_count, prior_count + 1)
    assert orch.ctx.state in ("DONE", "CLOSING", "ABANDONED") or prior_state in (
        "DONE",
        "CLOSING",
        "ABANDONED",
    )


# ── 4.1 enrichment_partial ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_enrichment_timeout_sets_partial_flag(
    reset_ops_db, seed_automotive_pack, pack, monkeypatch
):
    from src.agents import orchestrator as orch_mod
    from tests.frontline.conftest import _RecordingHooks

    monkeypatch.setattr(
        orch_mod, "settings", replace(settings, enrich_timeout_s=0)
    )

    async def _slow(self) -> None:
        await asyncio.sleep(0.05)

    monkeypatch.setattr(Orchestrator, "_run_enrichment", _slow)

    iid = "int_partial_" + new_ulid()[:8]
    with ops_con() as con:
        con.execute(
            """
            INSERT INTO interactions
            (interaction_id, pack_id, pack_version, started_at, channel, status)
            VALUES (?, ?, ?, ?, 'web_text', 'active')
            """,
            [iid, pack.id, pack.pack_version, datetime.now(timezone.utc)],
        )
    orch = Orchestrator(iid, pack, channel="web_text", hooks=_RecordingHooks())
    orch.ctx.state = "COLLECTING"
    orch.ctx.slots.update({
        "entity_1": "2019",
        "entity_2": "HONDA",
        "entity_3": "CR-V",
        "category": "SERVICE BRAKES",
        "description": "grinding",
    })
    await orch._enter_enriching()
    assert orch.ctx.enrichment_partial is True
    with ops_con(read_only=True) as con:
        row = con.execute(
            "SELECT enrichment_partial FROM interactions WHERE interaction_id = ?",
            [iid],
        ).fetchone()
    assert row is not None and (row[0] is True or row[0] == 1)


def test_console_and_qubot_surface_enrichment_partial():
    live = Path("dashboard/routes/LiveContactConsole.jsx").read_text(encoding="utf-8")
    assert "enrichment_partial" in live
    assert "partial enrichment" in live
    auditor = Path("src/qubot/auditor.py").read_text(encoding="utf-8")
    assert "enrichment_partial" in auditor
    assert "PARTIAL" in auditor


# ── 6.1 abandoned-with-slots inferred projection ─────────────────────────────


@pytest.mark.asyncio
async def test_abandoned_with_slots_projects_inferred_not_observed(
    orchestrator_factory,
):
    orch, _ = orchestrator_factory()
    await orch.start()
    orch.ctx.slots.update({
        "entity_1": "2019",
        "entity_2": "HONDA",
        "entity_3": "CR-V",
        "category": "SERVICE BRAKES",
        "description": "grinding when braking",
    })
    orch.ctx.safety_flags.clear()
    orch.ctx.enrichment_done = False
    orch.ctx.case_id = None
    await orch.hangup()
    assert orch.ctx.state == "ABANDONED"
    with ops_con(read_only=True) as con:
        outcome = con.execute(
            "SELECT outcome FROM interactions WHERE interaction_id = ?",
            [orch.ctx.interaction_id],
        ).fetchone()[0]
    assert outcome == "abandoned_with_slots"
    rid = f"FRONTLINE-{orch.ctx.interaction_id}"
    with domain_con(orch.ctx.pack.id) as con:
        row = con.execute(
            "SELECT provenance FROM records WHERE record_id = ?", [rid]
        ).fetchone()
    assert row is not None and row[0] == "inferred"


def test_inferred_records_excluded_from_anomaly_recompute(seed_automotive_pack):
    from src.ml_runtime.anomalies import recompute_weekly_anomalies

    pack_id = "automotive_nhtsa"
    cat = "INFERRED_ONLY_SLICE_XYZ"
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    with domain_con(pack_id, read_only=False) as con:
        con.execute(
            """
            INSERT INTO records
            (record_id, occurred_at, received_at, entity_2, category, text,
             source, provenance)
            VALUES (?, ?, ?, 'HONDA', ?, 'inferred only', 'FRONTLINE', 'inferred')
            """,
            ["FRONTLINE-inferred-only", now, now, cat],
        )
    rows = recompute_weekly_anomalies(pack_id, category=cat)
    assert all((r.get("category") or "") != cat for r in rows) or rows == []


# ── 7.2 erasure drill ────────────────────────────────────────────────────────


def test_erasure_drill_passes_and_writes_ops_row(reset_ops_db):
    from src.compliance.erasure_drill import run_erasure_drill
    from src.jobs.queue import ALLOWED_JOB_TYPES

    assert "erasure_drill" in ALLOWED_JOB_TYPES
    report = run_erasure_drill()
    assert report["passed"] is True
    assert report["chain_ok"] is True
    assert report["payload_unreadable"] is True
    assert report["dek_destroyed"] is True
    with ops_con(read_only=True) as con:
        row = con.execute(
            "SELECT passed, chain_ok, payload_unreadable, dek_destroyed "
            "FROM erasure_drill_reports ORDER BY ran_at DESC LIMIT 1"
        ).fetchone()
    assert row is not None
    assert all(bool(x) for x in row)
