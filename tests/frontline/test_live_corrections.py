"""Correction state and diagnostic answers on the live contact path."""

from __future__ import annotations

import pytest

from src.agents.triage import score_severity
from src.frontline.elicitation import answers_for_interaction, register_question


def _skip_safety(orch) -> None:
    questions = orch.ctx.pack.manifest.safety.safety_questions or []
    orch.ctx.slots["__safety_questions_asked__"] = ",".join(
        str(i) for i in range(len(questions))
    )


@pytest.mark.asyncio
async def test_correction_rejection_and_diagnostic_answer(orchestrator_factory):
    """One confirmation flow keeps a rejection open and a diagnostic answer changes severity."""
    orch, hooks = orchestrator_factory()
    await orch.start()
    register_question(
        orch.ctx.pack.id,
        "Is the window fogged?",
        category="VISIBILITY",
    )
    smoke = register_question(
        orch.ctx.pack.id,
        "Did you see smoke or fire?",
        category="ENGINE",
    )
    orch.ctx.slots.update({
        "entity_1": "2019",
        "entity_2": "HONDA",
        "entity_3": "CR-V",
        "category": "SERVICE BRAKES",
        "description": "mild noise",
        "__hypotheses__": [
            {"category": "SERVICE BRAKES", "open": True},
            {"category": "ENGINE", "open": True},
        ],
    })
    orch.ctx.severity = "Low"
    orch.ctx.investigation_brief = {"brief": "stale"}
    orch.ctx.slots["__advisory__"] = {"id": "old"}
    orch.ctx.slots["__retrieval__"] = {"ids": ["old"]}
    orch.ctx.slots["__severity_cache__"] = "Low"
    _skip_safety(orch)

    before, _, _ = score_severity(orch.ctx)
    assert before != "Critical"

    await orch.handle_customer_turn("the brakes still make a mild noise")
    assert orch.ctx.slots.get("__diag_qid") == smoke["question_id"]
    assert orch.ctx.slots.get("__diag_prompt") == "Did you see smoke or fire?"
    assert "smoke or fire" in " ".join(hooks.agent_texts()).lower()
    assert orch.ctx.slots.get("__confirm_pending__") != "1"

    await orch.handle_customer_turn("yes")
    stored = answers_for_interaction(orch.ctx.interaction_id)
    assert stored
    row = stored[-1]
    assert row["polarity"] == "yes"
    assert row["provenance"] == "customer_turn"
    assert row["uncertainty"] == "stated"
    assert "fire" in (row["symptom"] or "")
    assert row["source_turn"] == "yes"
    after, _, reason = score_severity(orch.ctx)
    assert after == "Critical"
    assert "fire" in reason.lower()
    assert orch.ctx.slots.get("__downstream_stale__") == "1"

    await orch.handle_customer_turn("Actually it's a 2018, not 2019.")
    assert orch.ctx.slots.get("entity_1") == "2018"
    meta = orch.ctx.slots["__slot_meta__"]["entity_1"]
    assert meta["value"] == "2018"
    assert "2018" in meta["source_turn"]
    assert meta["confidence"] is not None
    assert meta["confirmation"] == "unconfirmed"
    assert int(meta["revision"]) >= 1
    assert orch.ctx.severity is None
    assert orch.ctx.investigation_brief is None
    assert "__advisory__" not in orch.ctx.slots
    assert "__retrieval__" not in orch.ctx.slots
    assert "__severity_cache__" not in orch.ctx.slots
    assert orch.ctx.slots.get("__confirmed__") != "1"

    await orch.handle_customer_turn("No, that's wrong.")
    assert orch.ctx.slots.get("__confirmed__") != "1"
    assert orch.ctx.slots["__slot_meta__"]["entity_1"]["confirmation"] == "unconfirmed"
    assert orch.ctx.state != "DONE"
