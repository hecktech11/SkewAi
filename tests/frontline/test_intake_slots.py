"""Intake agent slot-extraction tests (per blueprint §15.2).

Covers:
  - Gazetteer extraction (year / make / model / category)
  - Synonym mapping (e.g. "brakes" → "SERVICE BRAKES")
  - Re-ask limits (slot is skipped after max_re_asks)
  - Kill-switch lexicon escalation
  - The automotive pack's slot frame specifically
"""

from __future__ import annotations

import pytest

from src.agents.base import InteractionContext
from src.agents.intake import IntakeAgent, _check_kill_switch, _extract_year, match_category_synonym


# ── Helper ─────────────────────────────────────────────────────────────────────


def _ctx(pack) -> InteractionContext:
    return InteractionContext(interaction_id="int_test_intake", pack=pack)


# ── Year extraction (year-range validation) ──────────────────────────────────


def test_extract_year_in_range(pack):
    """A 4-digit year inside the pack's year range is accepted."""
    slot = pack.slots_by_name()["entity_1"]
    assert _extract_year("I drive a 2019 Honda CR-V", slot.year_range) == "2019"
    assert _extract_year("It's a 2020 model", slot.year_range) == "2020"


def test_extract_year_out_of_range_rejected(pack):
    """Years outside the pack's range are rejected."""
    slot = pack.slots_by_name()["entity_1"]
    # 1899 is below the [1990, 2026] range.
    assert _extract_year("It's a 1899 model", slot.year_range) is None
    # 2099 is above.
    assert _extract_year("It's a 2099 model", slot.year_range) is None


def test_extract_year_no_match(pack):
    slot = pack.slots_by_name()["entity_1"]
    assert _extract_year("I have a vehicle", slot.year_range) is None


# ── Gazetteer extraction ──────────────────────────────────────────────────────


async def test_gazetteer_extracts_make_and_model(pack):
    """Honda + CR-V are both in the gazetteers and get canonicalised to uppercase."""
    ctx = _ctx(pack)
    agent = IntakeAgent(ctx)
    res = await agent.run(customer_turn="My 2019 Honda CR-V grinds when I brake.")
    extracted = res["extracted"]
    assert extracted["entity_2"] == "HONDA"
    assert extracted["entity_3"] == "CR-V"
    # Year slot is filled too (year-range validation).
    assert extracted["entity_1"] == "2019"


async def test_gazetteer_make_case_insensitive(pack):
    """Lowercase 'honda' is canonicalised to 'HONDA'."""
    ctx = _ctx(pack)
    agent = IntakeAgent(ctx)
    res = await agent.run(customer_turn="My honda civic has a problem.")
    assert res["extracted"].get("entity_2") == "HONDA"
    assert res["extracted"].get("entity_3") == "CIVIC"


async def test_category_synonym_brakes(pack):
    """'brakes' is mapped to the canonical NHTSA category 'SERVICE BRAKES'."""
    ctx = _ctx(pack)
    agent = IntakeAgent(ctx)
    res = await agent.run(customer_turn="My 2019 Honda CR-V has a brake problem.")
    assert res["extracted"].get("category") == "SERVICE BRAKES"


async def test_category_synonym_engine(pack):
    ctx = _ctx(pack)
    agent = IntakeAgent(ctx)
    res = await agent.run(customer_turn="My 2019 Honda CR-V has an engine misfire.")
    assert res["extracted"].get("category") == "ENGINE"


async def test_category_synonym_airbag(pack):
    ctx = _ctx(pack)
    agent = IntakeAgent(ctx)
    res = await agent.run(customer_turn="My 2017 Toyota Camry airbag warning is on.")
    assert res["extracted"].get("category") == "AIR BAGS"


def test_match_category_synonym_longest_match_and_speed_control_phrase():
    """Leftmost span after nested-longest; speed-control phrases outrank engine."""
    assert (
        match_category_synonym(
            "Tie rod snapped while driving my 2020 HONDA CR-V, wheel turned sideways and vehicle spun out."
        )
        == "STEERING"
    )
    assert (
        match_category_synonym(
            "Engine RPM spiked to redline and vehicle took off without input in my 2018 CHEVROLET SILVERADO."
        )
        == "VEHICLE SPEED CONTROL"
    )
    assert match_category_synonym("My 2019 Honda CR-V has an engine misfire.") == "ENGINE"
    # Non-overlapping later word must not steal the primary complaint.
    assert (
        match_category_synonym(
            "Seat belt pretensioner failed during impact and driver hit windshield, paramedics attended."
        )
        == "SEAT BELTS"
    )
    # This corpus codes parking-brake events as SERVICE BRAKES.
    assert (
        match_category_synonym(
            "Electronic parking brake engaged spontaneously at 50 mph."
        )
        == "SERVICE BRAKES"
    )
    assert (
        match_category_synonym(
            "Thick black smoke poured through the dashboard air vents of my 2019 TOYOTA CAMRY."
        )
        == "ELECTRICAL SYSTEM"
    )
    assert (
        match_category_synonym(
            "Vehicle lunged forward violently at stoplight in 2021 HYUNDAI TUCSON."
        )
        == "VEHICLE SPEED CONTROL"
    )
    # Maintenance / monitor collocations are not investigation classes.
    assert match_category_synonym("This check engine light is driving me crazy.") is None
    assert match_category_synonym("The battery is dead as a doornail and won't jump start.") is None
    assert match_category_synonym("No airbag deployment and no crash, but the light is on.") is None
    # Hypothetical, not a defect report.
    assert (
        match_category_synonym(
            "Wondering if the catalytic converter gets hot enough to cause a grass fire."
        )
        is None
    )


def test_extract_slot_category_longest_match_and_speed_control(pack):
    """Live extractor (gazetteer-filtered) uses the same longest-match rule."""
    ctx = _ctx(pack)
    agent = IntakeAgent(ctx)
    slot = pack.slots_by_name()["category"]
    assert (
        agent._extract_slot(
            slot,
            "Tie rod snapped while driving my 2020 HONDA CR-V, wheel turned sideways and vehicle spun out.",
        )
        == "STEERING"
    )
    assert (
        agent._extract_slot(
            slot,
            "Engine RPM spiked to redline and vehicle took off without input in my 2018 CHEVROLET SILVERADO.",
        )
        == "VEHICLE SPEED CONTROL"
    )


async def test_multisystem_voice_report_preserves_each_affected_system(pack):
    """Speech-to-text variants must not collapse a multi-fault report."""
    ctx = _ctx(pack)
    agent = IntakeAgent(ctx)
    await agent.run(
        customer_turn=(
            "The steering is not working, the breath is also not moving, "
            "gear is also stuck, and the dashboard music system is not playing."
        )
    )
    categories = {ctx.slots.get("category"), *(ctx.slots.get("secondary_categories") or [])}
    assert {
        "SERVICE BRAKES",
        "STEERING",
        "POWER TRAIN",
        "ELECTRICAL SYSTEM",
    } <= categories

    # Later issues remain visible after the primary routing category is set.
    await agent.run(customer_turn="The radio is also not working.")
    assert "ELECTRICAL SYSTEM" in set(ctx.slots.get("secondary_categories") or []) or \
        ctx.slots.get("category") == "ELECTRICAL SYSTEM"


async def test_description_is_free_text(pack):
    """The description slot captures the full customer turn verbatim."""
    ctx = _ctx(pack)
    agent = IntakeAgent(ctx)
    res = await agent.run(customer_turn="The steering wheel shakes at 60 mph.")
    # Description gets the full text (free-text validation).
    assert res["extracted"].get("description") == "The steering wheel shakes at 60 mph."


# ── Re-ask limits ─────────────────────────────────────────────────────────────


async def test_slot_skipped_after_max_re_asks(pack):
    """After max_re_asks attempts on a slot, the agent moves on to the next slot.

    The automotive pack sets max_re_asks=2 for entity_1/2/3 and max_re_asks=1
    for description. Safety questions are asked first (2 of them), then slot
    re-asks begin.
    """
    ctx = _ctx(pack)
    agent = IntakeAgent(ctx)

    # Safety questions first (asked, then answered — answers are bound).
    await agent.run(customer_turn="My car has a brake problem.")
    await agent.run(customer_turn="No, nobody is hurt.")
    # Answering the last safety question falls through to the first slot ask
    # on the same turn.
    await agent.run(customer_turn="Yes, I'm in a safe location.")
    a1 = ctx.slot_attempts.get("entity_1", 0)
    assert a1 == 1

    # Now slot re-asks continue. entity_1 (year) is first.
    await agent.run(customer_turn="It just doesn't stop.")
    a2 = ctx.slot_attempts.get("entity_1", 0)
    assert a2 == 2
    # Third slot turn — entity_1 has hit its max_re_asks (2), so the agent skips it
    # and asks for the next missing slot. entity_1 attempts did not grow.
    res = await agent.run(customer_turn="Still happening.")
    assert ctx.slot_attempts.get("entity_1", 0) == 2
    # The agent asked *some* question (still COLLECTING).
    assert res.get("question")


# ── Kill-switch lexicon ────────────────────────────────────────────────────────


def test_kill_switch_matches_fire(pack):
    ctx = _ctx(pack)
    assert _check_kill_switch("My car is on fire!", ctx) == "fire"


def test_kill_switch_matches_smoke(pack):
    ctx = _ctx(pack)
    assert _check_kill_switch("I smell smoke", ctx) == "smoke"


def test_kill_switch_matches_airbag_deployed(pack):
    ctx = _ctx(pack)
    # Use a phrase that contains 'airbag deployed' but no other lexicon term
    # (the lexicon is checked in order, and 'crash' would match first if present).
    assert _check_kill_switch("The airbag deployed during a minor collision", ctx) == "airbag deployed"


def test_kill_switch_no_match_on_neutral_text(pack):
    ctx = _ctx(pack)
    assert _check_kill_switch("My brake pedal feels soft", ctx) is None


def test_kill_switch_no_false_positive_on_substring(pack):
    """'fire' inside 'firewall' must not trigger the kill-switch (word boundary)."""
    ctx = _ctx(pack)
    # 'firewall' is one word; the lexicon uses `\bfire\b`, so it should not match.
    assert _check_kill_switch("There's a rattle near the firewall", ctx) is None


# ── Automotive pack slot frame ────────────────────────────────────────────────


def test_automotive_slot_frame_order(pack):
    """Slots are defined in fill order: year → make → model → category → description."""
    names = [s.name for s in pack.manifest.slot_frame]
    assert names == ["entity_1", "entity_2", "entity_3", "category", "description"]


def test_automotive_required_slots(pack):
    """All five slots are required in the automotive pack."""
    required = [s.name for s in pack.required_slots()]
    assert required == ["entity_1", "entity_2", "entity_3", "category", "description"]


def test_automotive_safety_questions(pack):
    """The automotive pack ships two safety questions asked before slot completion."""
    qs = pack.manifest.safety.safety_questions
    assert len(qs) == 2
    assert "hurt" in qs[0].lower()
    assert "safe location" in qs[1].lower()


def test_automotive_kill_switch_lexicon(pack):
    """The escalation lexicon includes fire/smoke/crash/injury/etc."""
    lexic = pack.manifest.safety.escalation_lexicon
    assert "fire" in lexic
    assert "smoke" in lexic
    assert "crash" in lexic
    assert "injury" in lexic
    assert "hurt" in lexic
    assert "injured" in lexic
    assert "bleeding" in lexic
