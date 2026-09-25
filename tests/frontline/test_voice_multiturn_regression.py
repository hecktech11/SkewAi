"""Regression: voice agent must survive past the first question.

Covers the three backend failure modes behind "after 1 question it does not
answer properly":
  1. filler STT ("hello hello hello", "yes") polluting the free-text
     description slot so the symptom question is skipped;
  2. a spoken 4-digit year ("2019") misclassified as DTMF keypad input;
  3. consecutive safety answers still advancing the dialogue.
"""

from __future__ import annotations

import pytest

from src.agents.base import InteractionContext
from src.agents.intake import IntakeAgent, _is_filler_description
from src.domains.loader import load_pack
from src.voice.policy import parse_dtmf


def _ctx(iid="reg-voice"):
    pack = load_pack("automotive_nhtsa")
    return InteractionContext(interaction_id=iid, pack=pack, channel="web_voice")


def test_filler_never_fills_description():
    assert _is_filler_description("hello hello hello hello hello hello")
    assert _is_filler_description("hello")
    assert _is_filler_description("yes")
    assert _is_filler_description("hi")
    assert not _is_filler_description("my brakes are grinding when I stop")
    assert not _is_filler_description("2019 Honda CR-V brakes grinding")


@pytest.mark.asyncio
async def test_hello_does_not_consume_description_slot():
    ctx = _ctx("reg-hello")
    res = await IntakeAgent(ctx).run(customer_turn="hello hello hello hello hello hello")
    assert "description" not in res.get("extracted", {})
    assert "description" not in ctx.slots
    # Safety script still leads — the contact is not stuck.
    assert res.get("question") == "Is anyone hurt?"


@pytest.mark.asyncio
async def test_yes_alone_does_not_become_description():
    ctx = _ctx("reg-yes")
    res = await IntakeAgent(ctx).run(customer_turn="yes")
    assert res.get("extracted", {}).get("description") is None
    assert ctx.slots.get("description") is None


def test_bare_year_is_not_dtmf():
    assert parse_dtmf("2019") is None
    assert parse_dtmf("1998") is None
    assert parse_dtmf("2023") is None
    # Real keypad presses still work.
    assert parse_dtmf("1") == "1"
    assert parse_dtmf("2") == "2"
    assert parse_dtmf("#") == "#"
    assert parse_dtmf("dtmf:1") == "1"


@pytest.mark.asyncio
async def test_year_2023_fills_slot_after_safety():
    from src.agents.base import InteractionContext
    from src.agents.intake import IntakeAgent
    from src.domains.loader import load_pack

    pack = load_pack("automotive_nhtsa")
    ctx = InteractionContext(interaction_id="reg-2023", pack=pack, channel="web_voice")
    ag = IntakeAgent(ctx)
    await ag.run(customer_turn="hello")
    await ag.run(customer_turn="no nobody is hurt")
    await ag.run(customer_turn="yes")
    res = await ag.run(customer_turn="2023")
    assert res.get("extracted", {}).get("entity_1") == "2023"
    assert "dtmf" not in res
    assert "make" in (res.get("question") or "").lower()


@pytest.mark.asyncio
async def test_dtmf_menu_never_loops():
    from src.agents.base import InteractionContext
    from src.agents.intake import IntakeAgent
    from src.domains.loader import load_pack

    pack = load_pack("automotive_nhtsa")

    async def _after_safety(iid):
        ctx = InteractionContext(interaction_id=iid, pack=pack, channel="web_voice")
        ag = IntakeAgent(ctx)
        await ag.run(customer_turn="hello")
        await ag.run(customer_turn="no nobody is hurt")
        await ag.run(customer_turn="yes")
        return ctx, ag

    # Bare "2" hands off immediately (one press).
    _, ag = await _after_safety("reg-dtmf-2")
    res = await ag.run(customer_turn="2")
    assert res.get("escalate_low_confidence") is True
    assert "specialist" in (res.get("question") or "").lower()

    # Bare "1" repeats the slot question instead of offering the menu again.
    ctx, ag = await _after_safety("reg-dtmf-1")
    res = await ag.run(customer_turn="1")
    assert "dtmf" not in res
    assert "__dtmf_offered__" not in ctx.slots
    assert "model year" in (res.get("question") or "").lower()

    # Repeat: pressing "1" five times must never return the keypad prompt.
    for _ in range(5):
        res = await ag.run(customer_turn="1")
        assert "press 1 for" not in (res.get("question") or "")


@pytest.mark.asyncio
async def test_spoken_year_fills_entity_not_dtmf_prompt():
    ctx = _ctx("reg-year")
    # Answer safety first so we reach slot collection.
    await IntakeAgent(ctx).run(customer_turn="filler hello")
    await IntakeAgent(ctx).run(customer_turn="no nobody is hurt")
    await IntakeAgent(ctx).run(customer_turn="yes safe location")
    res = await IntakeAgent(ctx).run(customer_turn="2019")
    assert "dtmf" not in res
    assert "press 1 for" not in (res.get("question") or "")
    assert res.get("extracted", {}).get("entity_1") == "2019" or ctx.slots.get("entity_1") == "2019"


@pytest.mark.asyncio
async def test_multiturn_safety_flow_still_escalates_on_yes():
    ctx = _ctx("reg-safety")
    await IntakeAgent(ctx).run(customer_turn="hello hello hello")
    res = await IntakeAgent(ctx).run(customer_turn="yes")
    assert res.get("kill_switch") == "safety_q_0"
    assert "safety specialist" in (res.get("question") or "").lower()


@pytest.mark.asyncio
async def test_safety_answer_does_not_pollute_description():
    ctx = _ctx("reg-safe-desc")
    await IntakeAgent(ctx).run(customer_turn="hello hello hello")
    res = await IntakeAgent(ctx).run(customer_turn="no nobody is hurt")
    assert ctx.slots.get("description") is None
    assert res.get("extracted", {}).get("description") is None


@pytest.mark.asyncio
async def test_bare_entity_does_not_fill_description():
    ctx = _ctx("reg-entity-desc")
    await IntakeAgent(ctx).run(customer_turn="hello")
    await IntakeAgent(ctx).run(customer_turn="no nobody is hurt")
    await IntakeAgent(ctx).run(customer_turn="yes")
    await IntakeAgent(ctx).run(customer_turn="2019")
    await IntakeAgent(ctx).run(customer_turn="Honda")
    res = await IntakeAgent(ctx).run(customer_turn="CR-V")
    assert res.get("extracted", {}).get("entity_3") == "CR-V"
    assert ctx.slots.get("description") is None


@pytest.mark.asyncio
async def test_full_voice_happy_path_asks_each_slot_in_order():
    ctx = _ctx("reg-happy")
    ag = IntakeAgent(ctx)
    r = await ag.run(customer_turn="hello hello hello")
    assert r["question"] == "Is anyone hurt?"
    r = await ag.run(customer_turn="no nobody is hurt")
    assert "safe location" in r["question"]
    r = await ag.run(customer_turn="yes")
    assert "model year" in r["question"]
    r = await ag.run(customer_turn="2019")
    assert "make" in r["question"].lower()
    r = await ag.run(customer_turn="Honda")
    assert "model" in r["question"].lower()
    r = await ag.run(customer_turn="CR-V")
    assert "system" in r["question"].lower()
    r = await ag.run(customer_turn="brakes are grinding and squealing when I stop")
    assert r["question"] == ""
    assert ctx.slots["entity_1"] == "2019"
    assert ctx.slots["description"] == "brakes are grinding and squealing when I stop"
