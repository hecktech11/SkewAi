"""Enhanced intake + Hindi/Hinglish + sentiment raise-only.  [AI-LAYER]

All Meta calls are faked. The deterministic result is computed first by the
real IntakeAgent; the enhancer may only fill still-empty slots.
"""

from __future__ import annotations

import pytest

from src.ai_layer.intake_enhancer import maybe_enhance_intake
from src.ai_layer.language import accept_ai_normalisation, detect

HINDI_TURN = "मेरी होंडा सिविक 2019 के ब्रेक से बहुत आवाज़ आ रही है"
HINDI_EN = "My Honda Civic 2019 brakes are making a lot of noise"


def _slots_reply(normalized=HINDI_EN, slots=None, frustration=0.2, conf=0.95, language="hi"):
    return lambda p: {
        "language": language,
        "normalized_en": normalized,
        "normalization_confidence": conf,
        "slots": slots or [],
        "frustration": frustration,
        "frustration_reason": "caller sounds worried",
    }


async def _det_turn(orch, text):
    from src.agents.intake import IntakeAgent

    orch.ctx.record_turn("customer", text)
    return await IntakeAgent(orch.ctx).run(customer_turn=text)


def test_detect_languages():
    assert detect(HINDI_TURN)["lang"] == "hi"
    assert detect("meri gaadi ka brake kaam nahi kar raha")["lang"] == "hinglish"
    assert detect("The brakes on my Honda make noise")["lang"] == "en"
    # English words alone that look like weak markers do not flip the language
    assert detect("I parked the car to the side")["lang"] == "en"


def test_normalisation_sanity_checks():
    assert accept_ai_normalisation(HINDI_TURN, HINDI_EN, 0.9, 0.7)
    assert not accept_ai_normalisation(HINDI_TURN, HINDI_EN, 0.5, 0.7)          # low confidence
    assert not accept_ai_normalisation(HINDI_TURN, "My 2021 Honda", 0.9, 0.7)    # invented number
    assert not accept_ai_normalisation(HINDI_TURN, "होंडा", 0.9, 0.7)           # not English
    assert accept_ai_normalisation("२०१९ होंडा", "2019 Honda", 0.9, 0.7)       # Devanagari digits ok


@pytest.mark.asyncio
async def test_no_key_returns_same_object_and_no_io(no_meta, orchestrator_factory):
    orch, _ = orchestrator_factory()
    await orch.start()
    ires = await _det_turn(orch, HINDI_TURN)
    before = dict(orch.ctx.slots)
    out = await maybe_enhance_intake(orch.ctx, HINDI_TURN, ires)
    assert out is ires
    assert orch.ctx.slots == before
    assert no_meta == []


@pytest.mark.asyncio
async def test_hindi_fills_empty_slots_via_deterministic_on_translation(fake_meta, orchestrator_factory):
    orch, _ = orchestrator_factory()
    await orch.start()
    ires = await _det_turn(orch, HINDI_TURN)
    det_year = orch.ctx.slots.get("entity_1")
    assert det_year == "2019"  # deterministic got the digits
    assert not orch.ctx.slots.get("entity_2")  # but not the Devanagari make
    fake_meta.chat["intake_slots"] = _slots_reply()
    out = await maybe_enhance_intake(orch.ctx, HINDI_TURN, ires)
    assert orch.ctx.slots["entity_2"] == "HONDA"
    assert orch.ctx.slots["entity_3"].upper() == "CIVIC"
    assert orch.ctx.slots["category"] == "SERVICE BRAKES"
    assert orch.ctx.slots["entity_1"] == det_year  # deterministic value kept
    meta = orch.ctx.slots["__slot_meta__"]
    assert meta["entity_2"]["source"] == "deterministic_on_ai_normalized"
    assert out["ai_layer"]["language"] == "hi"
    assert set(out["ai_layer"]["filled"]) >= {"entity_2", "entity_3", "category"}
    # Question recomputed: never re-asks a slot that is now filled
    assert "make" not in (out["question"] or "").lower()
    from src.ai_layer.store import list_events

    ev = [e for e in list_events(orch.ctx.interaction_id) if e["kind"] == "intake"]
    assert ev and ev[0]["deterministic"]["slots"]["entity_2"] is None
    assert ev[0]["accepted"]["language"]["original"]  # original stored
    assert ev[0]["accepted"]["language"]["normalized_en"] == HINDI_EN


@pytest.mark.asyncio
async def test_never_overwrites_deterministic_values(fake_meta, orchestrator_factory):
    orch, _ = orchestrator_factory()
    await orch.start()
    text = "My 2019 Honda Accord brakes squeal"
    ires = await _det_turn(orch, text)
    before = {k: orch.ctx.slots.get(k) for k in ("entity_1", "entity_2", "entity_3", "category")}
    fake_meta.chat["intake_slots"] = _slots_reply(
        normalized=text, language="en",
        slots=[{"slot": "make", "value": "TOYOTA", "confidence": 0.99, "evidence": "Honda"},
               {"slot": "model", "value": "CAMRY", "confidence": 0.99, "evidence": "Accord"}],
    )
    await maybe_enhance_intake(orch.ctx, text, ires)
    after = {k: orch.ctx.slots.get(k) for k in before}
    assert after == before


@pytest.mark.asyncio
async def test_model_proposals_need_grounding_confidence_and_pack_validation(fake_meta, orchestrator_factory):
    orch, _ = orchestrator_factory()
    await orch.start()
    text = "meri gaadi bahut awaaz kar rahi hai"
    ires = await _det_turn(orch, text)
    fake_meta.chat["intake_slots"] = _slots_reply(
        normalized="my car is making a lot of noise", language="hinglish",
        slots=[
            {"slot": "make", "value": "HONDA", "confidence": 0.99, "evidence": "Honda"},          # not said
            {"slot": "model", "value": "AMAZE", "confidence": 0.99, "evidence": "gaadi"},       # not in gazetteer
            {"slot": "year", "value": "2018", "confidence": 0.4, "evidence": "gaadi"},          # low confidence
        ],
    )
    out = await maybe_enhance_intake(orch.ctx, text, ires)
    assert not orch.ctx.slots.get("entity_2")
    assert not orch.ctx.slots.get("entity_3")
    assert not orch.ctx.slots.get("entity_1")
    rej = out["ai_layer"]["rejected"]
    assert rej["entity_2"] == "evidence_not_in_utterance"
    assert rej["entity_3"] == "failed_pack_validation"
    assert rej["entity_1"] == "low_confidence"


@pytest.mark.asyncio
async def test_grounded_model_proposal_is_accepted(fake_meta, orchestrator_factory):
    orch, _ = orchestrator_factory()
    await orch.start()
    text = "meri gaadi ka steering bahut bhaari hai, gaadi Civic hai"
    ires = await _det_turn(orch, text)
    filled_before = orch.ctx.slots.get("entity_3")
    fake_meta.chat["intake_slots"] = _slots_reply(
        normalized="my car's steering is very heavy, the car is a Civic", language="hinglish",
        slots=[{"slot": "make", "value": "Honda", "confidence": 0.9, "evidence": "Civic"}],
    )
    await maybe_enhance_intake(orch.ctx, text, ires)
    # 'Civic' is in the utterance and HONDA is a canonical make, so the
    # deterministic-on-translation pass or the grounded proposal fills make.
    assert orch.ctx.slots.get("entity_2") == "HONDA"
    if filled_before:
        assert orch.ctx.slots.get("entity_3") == filled_before


@pytest.mark.asyncio
async def test_hindi_brake_failure_raises_kill_switch_via_pack_lexicon(fake_meta, orchestrator_factory):
    orch, _ = orchestrator_factory()
    await orch.start()
    text = "गाड़ी चलाते समय ब्रेक पूरी तरह फेल हो गए"
    ires = await _det_turn(orch, text)
    assert not ires.get("kill_switch")  # deterministic lexicon is English-only
    fake_meta.chat["intake_slots"] = _slots_reply(normalized="While driving the brakes failed completely")
    out = await maybe_enhance_intake(orch.ctx, text, ires)
    assert out["kill_switch"] == "brakes failed"
    assert orch.ctx.safety_flags.get("escalation") is True
    assert out["question"] == orch.ctx.pack.manifest.safety.escalation_script


@pytest.mark.asyncio
async def test_ai_cannot_deescalate(fake_meta, orchestrator_factory):
    orch, _ = orchestrator_factory()
    await orch.start()
    text = "my brakes failed and there was smoke"
    ires = await _det_turn(orch, text)
    assert ires.get("kill_switch")
    fake_meta.chat["intake_slots"] = _slots_reply(normalized="everything is fine", language="en")
    out = await maybe_enhance_intake(orch.ctx, text, ires)
    assert out is ires
    assert fake_meta.calls == []  # never even asked


@pytest.mark.asyncio
async def test_transport_failure_falls_back_silently(fake_meta, orchestrator_factory):
    orch, _ = orchestrator_factory()
    await orch.start()
    ires = await _det_turn(orch, HINDI_TURN)
    before = dict(orch.ctx.slots)
    fake_meta.raise_exc = OSError("down")
    out = await maybe_enhance_intake(orch.ctx, HINDI_TURN, ires)
    assert out is ires
    assert orch.ctx.slots == before


@pytest.mark.asyncio
async def test_invalid_model_output_falls_back(fake_meta, orchestrator_factory):
    orch, _ = orchestrator_factory()
    await orch.start()
    ires = await _det_turn(orch, HINDI_TURN)
    before = dict(orch.ctx.slots)
    fake_meta.chat["intake_slots"] = lambda p: {"language": "hi"}  # missing required keys
    out = await maybe_enhance_intake(orch.ctx, HINDI_TURN, ires)
    assert out is ires and orch.ctx.slots == before


@pytest.mark.asyncio
async def test_sentiment_is_max_of_rule_and_ai(fake_meta, orchestrator_factory):
    orch, _ = orchestrator_factory()
    await orch.start()
    text = "this is ridiculous, my Honda brakes squeal"
    orch.ctx.record_turn("customer", text)
    from src.agents.intake import IntakeAgent
    from src.agents.sentiment import SentimentAgent

    await SentimentAgent(orch.ctx).run(customer_turn=text)
    rule = orch.ctx.turns[-1]["frustration_score"]
    assert rule > 0.5
    ires = await IntakeAgent(orch.ctx).run(customer_turn=text)
    fake_meta.chat["intake_slots"] = _slots_reply(normalized=text, language="en", frustration=0.1)
    out = await maybe_enhance_intake(orch.ctx, text, ires)
    assert orch.ctx.turns[-1]["frustration_score"] == rule  # AI lower -> rule kept
    assert out["ai_layer"]["sentiment"]["final"] == rule
    fake_meta.chat["intake_slots"] = _slots_reply(normalized=text, language="en", frustration=0.95)
    out = await maybe_enhance_intake(orch.ctx, text, ires)
    assert orch.ctx.turns[-1]["frustration_score"] == 0.95
    assert orch.ctx.peak_frustration >= 0.95
    assert out["ai_layer"]["sentiment"]["reason"]


@pytest.mark.asyncio
async def test_full_orchestrated_hindi_call_with_fake_meta(fake_meta, orchestrator_factory):
    orch, hooks = orchestrator_factory()
    fake_meta.chat["intake_slots"] = _slots_reply()
    fake_meta.chat["severity"] = lambda p: {"severity": "Low", "confidence": 0.9, "reason": "noise only",
                                             "evidence": "noise"}
    await orch.start()
    await orch.handle_customer_turn(HINDI_TURN)
    assert orch.ctx.slots.get("entity_2") == "HONDA"
    from src.data.warehouse import ops_con

    with ops_con() as con:
        rows = con.execute(
            "SELECT agent, action_type FROM agent_actions WHERE interaction_id = ? AND agent = 'ai_layer'",
            [orch.ctx.interaction_id],
        ).fetchall()
    assert ("ai_layer", "slot_extracted") in rows
