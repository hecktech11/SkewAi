"""Recognized emergencies preempt routine conversation gates.

Each test drives the production turn entry with one unhedged lexicon hit
while a gate that used to answer first is armed. The turn must escalate.
Hedged and negated hits must not.
"""

from __future__ import annotations

from datetime import datetime, timezone

from src.agents.intake import IntakeAgent
from src.agents.orchestrator import DONE, HANDOFF_PENDING
from src.data.warehouse import ops_con


_EMERGENCY = "the car is on fire"


def _assert_escalated(orch, hooks) -> None:
    assert orch.ctx.state == DONE
    assert orch.ctx.state != HANDOFF_PENDING
    assert orch.ctx.safety_flags.get("escalation") is True
    assert orch.ctx.severity == "Critical"
    assert orch.ctx.priority == 1
    spoken = " ".join(hooks.agent_texts()).lower()
    assert "safety specialist" in spoken or "do not drive" in spoken
    assert not any(t["meta"].get("consent_required") for t in hooks.turns)
    assert not any(t["meta"].get("vin_retry") for t in hooks.turns)
    assert "just to confirm, i heard" not in spoken
    assert "connecting you to a human specialist" not in spoken
    assert "didn't pass the check digit" not in spoken
    assert "is currently" not in spoken
    assert "all parties must consent" not in spoken


def _seed_case(case_id: str, pack_id: str) -> None:
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
                case_id, "int_preempt_seed", pack_id, now,
                "ENGINE", "status seed", now, "Medium",
                "rules", 2, "{}",
                None, None, 0,
                None, "open", None,
            ],
        )


async def test_emergency_preempts_voice_consent(orchestrator_factory):
    orch, hooks = orchestrator_factory()
    await orch.start()
    await orch.handle_customer_turn(_EMERGENCY, region="CA")
    _assert_escalated(orch, hooks)


async def test_emergency_preempts_spoken_human_request(orchestrator_factory):
    orch, hooks = orchestrator_factory()
    await orch.start()
    await orch.handle_customer_turn(
        "I need to talk to a human, the car is on fire"
    )
    _assert_escalated(orch, hooks)


async def test_emergency_preempts_invalid_vin_reask(orchestrator_factory):
    orch, hooks = orchestrator_factory()
    await orch.start()
    await orch.handle_customer_turn(
        "the car is on fire vin 1HGCR2F84HA000000"
    )
    _assert_escalated(orch, hooks)
    assert not orch.ctx.slots.get("vin")


async def test_emergency_preempts_low_asr_readback(orchestrator_factory):
    orch, hooks = orchestrator_factory()
    await orch.start()
    await orch.handle_customer_turn(
        _EMERGENCY, asr_confidence={"entity_1": 0.1},
    )
    _assert_escalated(orch, hooks)


async def test_emergency_preempts_case_status_lookup(orchestrator_factory):
    orch, hooks = orchestrator_factory()
    await orch.start()
    _seed_case("case_preempt_demo_1", orch.ctx.pack.id)
    await orch.handle_customer_turn(
        "the car is on fire, what is the status of case_preempt_demo_1"
    )
    _assert_escalated(orch, hooks)


async def test_intake_emergency_preempts_low_asr_and_case_status(orchestrator_factory):
    """IntakeAgent.run is the production entry for the two intake-owned gates."""
    orch, _ = orchestrator_factory()
    intake = IntakeAgent(orch.ctx)
    low = await intake.run(
        customer_turn=_EMERGENCY, asr_confidence={"entity_1": 0.05},
    )
    assert low.get("kill_switch") == "fire"
    assert not low.get("readback")

    orch2, _ = orchestrator_factory()
    _seed_case("case_preempt_demo_2", orch2.ctx.pack.id)
    status = await IntakeAgent(orch2.ctx).run(
        customer_turn="the car is on fire, status of case_preempt_demo_2",
    )
    assert status.get("kill_switch") == "fire"
    assert "case_status" not in status
    assert "is currently" not in (status.get("question") or "").lower()


async def test_hedged_lexicon_does_not_escalate_through_consent(orchestrator_factory):
    orch, hooks = orchestrator_factory()
    await orch.start()
    await orch.handle_customer_turn(
        "There was no fire, just a smell.", region="CA",
    )
    assert not orch.ctx.safety_flags.get("escalation")
    assert orch.ctx.state != DONE
    spoken = " ".join(hooks.agent_texts()).lower()
    assert "all parties must consent" in spoken
    assert "safety specialist" not in spoken


async def test_fire_drill_does_not_escalate_on_human_request(orchestrator_factory):
    orch, hooks = orchestrator_factory()
    await orch.start()
    await orch.handle_customer_turn(
        "I had a fire drill and I want to talk to a human"
    )
    assert not orch.ctx.safety_flags.get("escalation")
    assert orch.ctx.state == HANDOFF_PENDING
    spoken = " ".join(hooks.agent_texts()).lower()
    assert "connecting you to a human specialist" in spoken
    assert "safety specialist" not in spoken
