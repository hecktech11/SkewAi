"""Handoff promises, specialist outcomes, and narration follow what actually happened."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.agents.base import InteractionContext
from src.agents.investigator import InvestigatorAgent
from src.ai.provider import NarrationResult
from src.data.warehouse import ops_con
from src.frontline.reenrich import backfill_brief


def _agent_blob(hooks) -> str:
    return " ".join(hooks.agent_texts()).lower()


@pytest.mark.asyncio
async def test_menu_digit_connects_only_after_queue(orchestrator_factory):
    orch, hooks = orchestrator_factory()
    await orch.start()
    await orch.handle_customer_turn("hello there")
    await orch.handle_customer_turn("no nobody is hurt")
    await orch.handle_customer_turn("yes")
    await orch.handle_customer_turn("2")

    assert orch.ctx.slots.get("__handoff_status__") == "queued"
    assert orch.ctx.state == "HANDOFF_PENDING"
    with ops_con(read_only=True) as con:
        row = con.execute(
            """
            SELECT status FROM handoff_requests
            WHERE interaction_id = ? ORDER BY created_at DESC LIMIT 1
            """,
            [orch.ctx.interaction_id],
        ).fetchone()
    assert row is not None
    assert row[0] == "pending"
    spoken = _agent_blob(hooks)
    assert "connecting you to a human" in spoken
    assert "not connected yet" not in spoken


@pytest.mark.asyncio
async def test_unqueued_handoff_does_not_promise_a_connection(orchestrator_factory):
    orch, hooks = orchestrator_factory()
    await orch.start()
    await orch.handle_customer_turn("hello there")
    await orch.handle_customer_turn("no nobody is hurt")
    await orch.handle_customer_turn("yes")
    orch.ctx.state = "SAFETY_ESCALATION"
    before = len(hooks.agent_texts())
    await orch.handle_customer_turn("2")
    added = " ".join(hooks.agent_texts()[before:]).lower()
    assert orch.ctx.slots.get("__handoff_status__") == "failed"
    assert "couldn't queue" in added
    assert "connecting you" not in added
    assert orch.ctx.state != "HANDOFF_PENDING"


@pytest.mark.asyncio
async def test_disabled_investigator_is_not_applicable(pack):
    ctx = InteractionContext(interaction_id="int_inv_skip", pack=pack)
    ctx.slots.update({
        "entity_1": "2019",
        "entity_2": "HONDA",
        "entity_3": "CR-V",
        "category": "SERVICE BRAKES",
        "description": "mild noise",
    })

    def _enabled(name: str) -> bool:
        return name != "investigator"

    import src.ops.pilot as pilot

    original = pilot.agent_enabled
    pilot.agent_enabled = _enabled
    try:
        result = await InvestigatorAgent(ctx).run()
    finally:
        pilot.agent_enabled = original
    assert result["skipped"] is True
    assert result["outcome"] == "not_applicable"


@pytest.mark.asyncio
async def test_failed_retry_keeps_case_evidence(orchestrator_factory, monkeypatch):
    orch, _hooks = orchestrator_factory()
    await orch.start()
    case_id = "case_keep_evidence"
    now = datetime.now(timezone.utc)
    with ops_con() as con:
        con.execute(
            """
            INSERT INTO cases (
                case_id, interaction_id, pack_id, created_at,
                severity, severity_source, priority,
                cluster_match_id, similar_record_count, status
            ) VALUES (?, ?, ?, ?, 'Low', 'rules', 3, 7, 4, 'open')
            """,
            [case_id, orch.ctx.interaction_id, orch.ctx.pack.id, now],
        )

    async def _skipped(self, **kwargs):
        return {"skipped": True, "outcome": "unavailable"}

    monkeypatch.setattr(InvestigatorAgent, "run", _skipped)
    report = backfill_brief(orch.ctx.interaction_id, case_id=case_id)
    assert report["ok"] is False
    assert report["preserved"] is True
    assert report["outcome"] == "unavailable"
    with ops_con(read_only=True) as con:
        row = con.execute(
            """
            SELECT cluster_match_id, similar_record_count
            FROM cases WHERE case_id = ?
            """,
            [case_id],
        ).fetchone()
    assert row == (7, 4)


def test_narration_unsupported_claim_stores_deterministic_text(pack, monkeypatch):
    ctx = InteractionContext(interaction_id="int_narr", pack=pack)
    ctx.slots["description"] = "grinding brakes"
    agent = InvestigatorAgent(ctx)
    brief = {
        "similar_record_count": 2,
        "cluster_id": None,
        "cluster_count": 0,
        "similar_records": [],
        "lead_time_weeks": None,
        "spikes": [],
    }

    def _bad(**kwargs):
        return NarrationResult(
            text="Found 9 similar records. We will replace your brakes.",
            ok=True,
            model_id="fake-narrator",
            prompt_hash="hash-abc",
            used_llm=True,
            reason="test",
        )

    monkeypatch.setattr("src.ai.narration.phrase_investigation_brief", _bad)
    shown = agent._narrate(brief)
    assert "Found 2 similar" in shown
    assert "replace" not in shown.lower()
    assert brief["narration_shown"] == shown
    assert brief["model_version"] == "fake-narrator"
    assert brief["prompt_hash"] == "hash-abc"
    assert brief["narration_violations"]
    assert brief["narration_audited"] is True
