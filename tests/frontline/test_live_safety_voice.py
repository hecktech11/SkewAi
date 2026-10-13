"""Whole-clause safety is the reading intake and triage both use."""

from __future__ import annotations

from src.agents.base import InteractionContext
from src.agents.intake import (
    IntakeAgent,
    assess_safety,
    classify_answer,
    hazard_term_affirmed,
)
from src.agents.triage import _eval_condition, score_severity
from src.ml_runtime.registry import predict_severity

HURT_Q = "Is anyone hurt?"
SAFE_Q = "Are you in a safe location right now?"


def _ctx(pack, **slots) -> InteractionContext:
    ctx = InteractionContext(interaction_id="int_safety_voice", pack=pack)
    ctx.slots.update(slots)
    return ctx


def test_leading_no_with_injury_is_not_safe():
    text = "No, my son has a cut."
    assert classify_answer(HURT_Q, text) is None
    reading = assess_safety(text, question=HURT_Q)
    assert reading["contradictory"] is True
    assert reading["polarity"] is None
    assert reading["hazard"] == "cut"
    assert reading["who"] == "son"
    assert reading["spans"]
    assert reading["clarification"]


def test_leading_yes_unsafe_location_is_not_safe():
    text = "Yes, but I'm not in a safe location."
    assert classify_answer(SAFE_Q, text) is None
    reading = assess_safety(text, question=SAFE_Q)
    assert reading["contradictory"] is True
    assert reading["polarity"] is None
    assert "safe location" in (reading["clarification"] or "").lower()


def test_contradiction_asks_and_stays_unconfirmed(pack):
    ctx = _ctx(pack)
    ctx.slots["__safety_pending__"] = "0"
    hit = IntakeAgent(ctx)._evaluate_pending_safety_answer("No, my son has a cut.")
    assert hit is not None
    assert hit["kill_switch"] is None
    assert hit["safety_clarification"] is True
    assert "hurt" in hit["question"].lower()
    assert ctx.slots.get("__safety_pending__") == "0"
    assert not ctx.safety_flags.get("escalation")
    assert "0" not in str(ctx.slots.get("__safety_asked__") or "")

    ctx = _ctx(pack)
    ctx.slots["__safety_pending__"] = "1"
    hit = IntakeAgent(ctx)._evaluate_pending_safety_answer(
        "Yes, but I'm not in a safe location."
    )
    assert hit["kill_switch"] is None
    assert hit["safety_clarification"] is True
    assert ctx.slots.get("__safety_pending__") == "1"
    assert not ctx.safety_flags.get("escalation")


def test_negated_fire_does_not_take_fire_severity(pack):
    assert hazard_term_affirmed("no fire", "fire") is False
    assert hazard_term_affirmed("the car caught fire", "fire") is True
    assert hazard_term_affirmed(
        "there was no fire, but now the car is on fire", "fire"
    ) is True

    ctx = _ctx(pack, category="VISIBILITY", description="no fire")
    assert _eval_condition("description_matches_fire|smoke|crash|injury", ctx) is False
    severity, source, reason = score_severity(ctx)
    assert severity != "Critical"
    assert "fire" not in reason.lower() or "no_keyword" in reason.lower() or "fallback" in reason.lower()
    assert source in ("rules", "model")

    caught = _ctx(pack, category="VISIBILITY", description="the car caught fire")
    assert _eval_condition("description_matches_fire|smoke|crash|injury", caught) is True
    severity, _, reason = score_severity(caught)
    assert severity == "Critical"
    assert "fire" in reason.lower()


def test_registry_keyword_overlay_skips_negated_fire():
    severity, reason = predict_severity(
        "domains/automotive_nhtsa/models/severity_rules.json",
        {"description": "the car caught fire", "category": "VISIBILITY"},
    )
    assert severity == "Critical"
    assert reason == "json_rule:fire"
    try:
        predict_severity(
            "domains/automotive_nhtsa/models/severity_rules.json",
            {"description": "no fire", "category": "VISIBILITY"},
        )
    except LookupError as exc:
        assert "no_keyword" in str(exc)
    else:
        raise AssertionError("negated fire must not score as a fire keyword")
