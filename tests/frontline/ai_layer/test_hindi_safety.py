"""Deterministic Hindi/Hinglish safety layer.  [AI-LAYER]  No key, no network.

Sentences here are SYNTHETIC test examples written for these tests.
"""

from __future__ import annotations

import pytest

from src.ai_layer import hindi_safety as hs


def test_analyse_devanagari_brake_failure_and_slots():
    a = hs.analyse("मेरी 2018 टोयोटा कैमरी के ब्रेक फेल हो गए")
    assert a["gloss"] == ["brakes failed"] and a["floor"] == "Critical"
    assert (a["category"], a["make"], a["model"], a["year"]) == ("SERVICE BRAKES", "TOYOTA", "CAMRY", "2018")


def test_negation_and_hedge_suppress_hazard():
    for t in ("कोई घायल नहीं हुआ", "koi chot nahi lagi", "धुआं नहीं है", "bina kisi accident ke", "शायद आग"):
        a = hs.analyse(t)
        assert a["gloss"] == [], t
        assert a["suppressed"], t


def test_intrinsic_negation_phrase_is_a_hazard():
    assert "could not stop" in hs.analyse("गाड़ी रुकी ही नहीं")["gloss"]
    assert "brakes failed" in hs.analyse("brake nahi lag rahe the")["gloss"]


def test_english_turn_untouched_and_disabled_flag(monkeypatch):
    class Ctx:
        interaction_id = "int_x"
        safety_flags: dict = {}
    import asyncio
    ires = {"question": "q"}
    assert asyncio.run(hs.maybe_apply_hindi_intake(Ctx(), "My brakes make noise", ires)) is ires
    monkeypatch.setenv("AI_HINDI_SAFETY", "0")
    assert asyncio.run(hs.maybe_apply_hindi_intake(Ctx(), "ब्रेक फेल", ires)) is ires


async def _run(pack, text, iid):
    from src.agents.base import InteractionContext
    from src.agents.intake import IntakeAgent
    from src.agents.triage import TriageAgent

    ctx = InteractionContext(interaction_id=iid, pack=pack, channel="web_text")
    ctx.record_turn("customer", text)
    ires = await IntakeAgent(ctx).run(customer_turn=text)
    ires = await hs.maybe_apply_hindi_intake(ctx, text, ires)
    tres = await TriageAgent(ctx).run()
    before = tres.get("severity")
    tres = await hs.maybe_raise_severity_hi(ctx, tres)
    return ctx, ires, tres, before


@pytest.mark.asyncio
async def test_devanagari_kill_switch_added_via_existing_lexicon(no_meta, pack, reset_ops_db):
    ctx, ires, _, _ = await _run(pack, "इंजन से धुआं निकला और आग लग गई", "int_hs_ks")
    assert ires["kill_switch"] in ("fire", "smoke") and ctx.safety_flags.get("escalation") is True
    assert no_meta == []


@pytest.mark.asyncio
async def test_devanagari_fills_empty_slots_and_triage_runs(no_meta, pack, reset_ops_db):
    ctx, ires, tres, _ = await _run(pack, "मेरी 2020 होंडा सिविक के ब्रेक से आवाज़ आती है", "int_hs_slots")
    assert not ires.get("kill_switch")
    assert ctx.slots.get("category") == "SERVICE BRAKES" and ctx.slots.get("entity_2") == "HONDA"
    assert ctx.slots.get("description")
    assert not tres.get("skipped") and tres.get("severity") in ("Medium", "Critical")


@pytest.mark.asyncio
async def test_never_overwrites_and_never_lowers(no_meta, pack, reset_ops_db):
    from src.agents.base import InteractionContext

    ctx = InteractionContext(interaction_id="int_hs_low", pack=pack, channel="web_text")
    ctx.slots["category"] = "ENGINE"
    ctx.severity = "Critical"
    hs._remember("int_hs_low", {"floor": "Medium", "gloss": []})
    tres = {"severity": "Critical", "priority": 1}
    assert (await hs.maybe_raise_severity_hi(ctx, tres)) is tres
    ctx2 = InteractionContext(interaction_id="int_hs_keep", pack=pack, channel="web_text")
    ctx2.slots["category"] = "ENGINE"
    await hs.maybe_apply_hindi_intake(ctx2, "मेरी गाड़ी के ब्रेक से तेज़ आवाज़ आती है", {})
    assert ctx2.slots["category"] == "ENGINE"  # existing value kept
    assert ctx2.slots.get("description")


@pytest.mark.asyncio
async def test_floor_raises_low_to_critical(no_meta, pack, reset_ops_db):
    from src.agents.base import InteractionContext

    ctx = InteractionContext(interaction_id="int_hs_raise", pack=pack, channel="web_text")
    ctx.severity = "Low"
    hs._remember("int_hs_raise", {"floor": "Critical", "gloss": ["smoke"]})
    out = await hs.maybe_raise_severity_hi(ctx, {"severity": "Low", "priority": 3})
    assert out["severity"] == "Critical" and ctx.severity == "Critical"
    assert "smoke" in out["ai_layer_hindi"]["reason"]


def test_eval_harness_summary_counts():
    from eval.hindi_fresh import run_hindi_eval as rhe

    calls = [{"id": "a", "lang": "hi", "kill_switch": True, "expect_category": None, "text": "x"},
             {"id": "b", "lang": "hi", "kill_switch": False, "expect_category": "ENGINE", "text": "y"}]
    res = [{"severity": None, "triage_skipped": True, "kill_switch": True, "category": None, "error": None},
           {"severity": "Medium", "triage_skipped": False, "kill_switch": True, "category": "ENGINE", "error": None}]
    s = rhe.summarise(calls, res)["all"]
    assert (s["kill_switch_hits"], s["kill_switch_false_alarms"], s["category_correct"]) == (1, 1, 1)
    assert rhe.lowers([{"severity": "Critical"}], [{"severity": "Low"}]) == 1
