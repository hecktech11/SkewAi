"""SLM Phase 0 probe suite — Appendix of the SLM integration plan.

57 author-written cases in browser-STT phrasing (lowercase, minimal
punctuation), executed against REAL intake code — never a simulation.

Two assertion kinds:

- PINNED (A/C/D): exact current behavior. These document the honest baseline
  (kill recall 0.35, category 0.071, frustration-blind). If the SLM — or any
  edit — moves them, these fail LOUDLY. Do NOT "fix" them by editing
  expectations; only the Phase 3 promotion gates change them.
- GUARDS (B/E): must never regress. B pins kill-switch precision at 1.0 on
  idiom/negation/hedge traps. E pins the defect-#2 fix: question-conditioned
  polarity must resolve all 14 cases (a false P1 from "im fine" is a
  supervisor-trust incident, not a metric wobble).
"""

from __future__ import annotations

import pytest

from src.agents.base import InteractionContext
from src.agents.intake import (
    _check_kill_switch,
    classify_answer,
    classify_yes_no,
    extract_pack_category,
)
from src.agents.sentiment import score_text
from src.config import settings


def _ctx(pack) -> InteractionContext:
    return InteractionContext(interaction_id="int_slm_probe", pack=pack)


# ── A. True emergencies — kill switch MUST fire (target recall ≥ 0.98) ──
# Pinned baseline: 7/20 fire (recall 0.350). The 13 misses are the SLM's
# target set: 10 have no lexicon term at all (open-vocabulary paraphrase),
# 3 are suppressed by the fixed 6-token hedge window.

A_CASES: list[tuple[str, bool]] = [
    ("im not sure what happened but theres smoke pouring out of the hood", False),
    ("i dont know if its serious but the engine is on fire right now", True),
    ("could you help me my car just caught fire in the driveway", True),
    ("i think maybe someone is hurt in the other car", True),
    ("we might need an ambulance my wife is bleeding", True),
    ("the brake pedal went straight to the floor and i couldnt stop", False),
    ("i had no braking at all coming down the hill", False),
    ("it wouldnt slow down at all i just kept going through the intersection", False),
    ("the wheel wont turn at all its completely stiff", False),
    ("the car took off on its own and i couldnt stop it", False),
    ("we hit the guardrail pretty hard", False),
    ("the airbag went off and my chest hurts", True),
    ("there was a loud bang and now the steering is gone", False),
    ("my daughter is trapped in the back seat", True),
    ("smoke started coming through the vents while i was driving", True),
    ("i was worried it would catch fire and then it actually did", False),
    ("if you could just note that the car is smoking badly right now", False),
    ("not gonna lie im scared theres flames under the bonnet", False),
    ("we went off the road and rolled into the ditch", False),
    ("i couldnt steer it just went wherever it wanted", False),
]


def test_probe_a_kill_switch_recall_pinned(pack):
    ctx = _ctx(pack)
    fired = [t for t, _ in A_CASES if _check_kill_switch(t, ctx) is not None]
    assert len(fired) == 7, f"baseline moved: {len(fired)}/20 fire now"
    assert len(fired) / len(A_CASES) == 0.35
    for text, expected in A_CASES:
        got = _check_kill_switch(text, ctx) is not None
        assert got == expected, f"baseline moved on {text!r}: was {expected}, now {got}"


# ── B. Must NOT fire — idiom / negation / hedge (precision floor ≥ 0.85) ──

B_CASES: list[str] = [
    "this repair bill is killing me but the brakes just squeak a bit",
    "nobody is hurt and there was no fire just a clicking noise",
    "im worried it might catch fire because of a faint hot smell",
    "the dealer gave me a crash course on the infotainment system",
    "my reading light bulb burned out last week",
    "no smoke no fire just an annoying rattle from the dashboard",
    "i was dying to get this fixed before my road trip",
    "the check engine light came on nothing dramatic happened",
    "the mechanic said the clutch is burning out slowly",
    "i had a fire drill at work so i missed my appointment",
]


def test_probe_b_kill_switch_precision_guard(pack):
    ctx = _ctx(pack)
    for text in B_CASES:
        assert _check_kill_switch(text, ctx) is None, f"FALSE ALARM on {text!r}"


# ── C. Category extraction — caller paraphrase ──
# Pinned baseline: 1/14 (0.071). The two confident-wrong outputs are pinned
# separately: wrong categories silently misgrade severity and misroute
# investigations, so confident-wrong must fall, not just accuracy rise.

C_CASES: list[tuple[str, str | None]] = [
    ("theres a grinding noise every time i slow down", None),
    ("it shudders really bad when i press the pedal", None),
    ("takes way longer to come to a stop than it used to", None),
    ("the wheel shakes in my hands on the motorway", "WHEELS"),
    ("it pulls to the right on a flat road", None),
    ("it jerks between gears when im speeding up", None),
    ("the dash lights flicker and the screen goes black", "EXTERIOR LIGHTING"),
    ("battery is flat every morning", "ELECTRICAL SYSTEM"),
    ("it wont start and just clicks", None),
    ("theres a knocking sound from under the bonnet when cold", None),
    ("it stalls at traffic lights", "EXTERIOR LIGHTING"),
    ("the warning light for the bag stays lit", None),
    ("clunking over speed bumps at the front", None),
    ("the belt doesnt retract properly", None),
]

C_EXPECTED = [
    "SERVICE BRAKES", "SERVICE BRAKES", "SERVICE BRAKES",
    "STEERING", "STEERING", "POWER TRAIN",
    "ELECTRICAL SYSTEM", "ELECTRICAL SYSTEM", "ELECTRICAL SYSTEM",
    "ENGINE", "ENGINE", "AIR BAGS", "SUSPENSION", "SEAT BELTS",
]


def test_probe_c_category_accuracy_pinned(pack):
    ctx = _ctx(pack)
    got = [extract_pack_category(t, ctx) for t, _ in C_CASES]
    assert sum(1 for g, e in zip(got, C_EXPECTED) if g == e) == 1
    for (text, pinned), expected in zip(C_CASES, C_EXPECTED):
        assert pinned == extract_pack_category(text, ctx), f"moved on {text!r}"
    # Confident-wrong set: the failures Phase 3 must eliminate (not just outnumber).
    wrong = {
        text: g for (text, g), e in zip([(t, extract_pack_category(t, ctx)) for t, _ in C_CASES], C_EXPECTED)
        if g is not None and g != e
    }
    assert wrong == {
        "the wheel shakes in my hands on the motorway": "WHEELS",
        "the dash lights flicker and the screen goes black": "EXTERIOR LIGHTING",
        "it stalls at traffic lights": "EXTERIOR LIGHTING",
    }


def test_probe_c_category_zeroshot_mode_pinned(monkeypatch, pack):
    """Probe C in zero-shot mode:
    - Accuracy lifts from 1/14 (legacy rules) to 2/14 (pinned zeroshot baseline).
    - Crucially, the three legacy confident-wrong cases shrink/are eliminated:
      * 'it stalls at traffic lights' -> None (was EXTERIOR LIGHTING)
      * 'the wheel shakes in my hands on the motorway' -> None (was WHEELS)
      * 'the dash lights flicker and the screen goes black' -> None (was EXTERIOR LIGHTING)
    """
    monkeypatch.setenv("FRONTLINE_SLM_MODE", "live")
    ctx = _ctx(pack)
    got = [extract_pack_category(t, ctx) for t, _ in C_CASES]
    acc = sum(1 for g, e in zip(got, C_EXPECTED) if g == e)
    assert acc == 2, f"Expected 2/14 accuracy in zeroshot mode, got {acc}"

    # Verify that the 3 legacy confident-wrong cases return None (asking the customer)
    eliminated_cw = [
        "the wheel shakes in my hands on the motorway",
        "the dash lights flicker and the screen goes black",
        "it stalls at traffic lights",
    ]
    for text in eliminated_cw:
        res = extract_pack_category(text, ctx)
        assert res is None, f"Expected None for {text!r} to avoid confident-wrong, got {res!r}"

    # Pinned confident-wrong set in zero-shot mode
    wrong = {
        text: g for (text, g), e in zip([(t, extract_pack_category(t, ctx)) for t, _ in C_CASES], C_EXPECTED)
        if g is not None and g != e
    }
    assert wrong == {
        "it pulls to the right on a flat road": "LANE DEPARTURE",
        "it jerks between gears when im speeding up": "VEHICLE SPEED CONTROL",
    }


def test_probe_c_zeroshot_latency_benchmark(pack):
    """Added latency budget: p95 <= 25 ms, p99 <= 50 ms."""
    import time
    import numpy as np
    from src.ml_runtime.category_zeroshot import predict_category_zero_shot

    # Warmup
    for t, _ in C_CASES[:3]:
        predict_category_zero_shot(t, pack)

    latencies = []
    for t, _ in C_CASES * 5:
        t0 = time.perf_counter()
        predict_category_zero_shot(t, pack)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    p95 = float(np.percentile(latencies, 95))
    p99 = float(np.percentile(latencies, 99))
    assert p95 <= 25.0, f"p95 latency {p95:.2f}ms exceeds 25ms ceiling"
    assert p99 <= 50.0, f"p99 latency {p99:.2f}ms exceeds 50ms ceiling"


# ── D. Frustration via repetition/wasted effort (threshold 0.65) ──
# Pinned baseline: the lexicon scores every case 0.000 — five real
# frustrations and two calm controls are indistinguishable.

D_FIRE = [
    "this is the fourth time ive called about this and nobody calls me back",
    "ive been passed around to five different people today",
    "i give up honestly i just want someone to actually help",
    "nobody has taken this seriously since day one",
    "ive wasted three days off work waiting for this",
]
D_CALM = [
    "i just want to book it in whenever suits you",
    "no rush at all just logging it for the record",
]


def test_probe_d_frustration_blindness_pinned():
    thr = settings.frustration_threshold
    assert thr == 0.65
    fire_scores = [score_text(t) for t in D_FIRE]
    calm_scores = [score_text(t) for t in D_CALM]
    assert all(s == 0.0 for s in fire_scores), fire_scores
    assert all(s < thr for s in calm_scores)
    assert all(s < thr for s in fire_scores)  # nothing fires today


# ── E. Question-conditioned polarity (defect #2 guard) ──
# "Is anyone hurt?" — yes escalates. "Are you in a safe location?" — no
# escalates. Every row must resolve; a miss here is a false P1 or a live
# re-ask, both supervisor-trust incidents.

HURT_Q = "Is anyone hurt?"
SAFE_Q = "Are you in a safe location right now?"

E_CASES: list[tuple[str, str, str]] = [
    (HURT_Q, "no everyones fine thanks", "no"),
    (HURT_Q, "im fine", "no"),
    (HURT_Q, "everyone is okay", "no"),
    (HURT_Q, "were all good no injuries", "no"),
    (HURT_Q, "my son has a cut on his forehead", "yes"),
    (HURT_Q, "shes complaining her neck is sore", "yes"),
    (HURT_Q, "i think shes alright but shes limping", "yes"),
    (SAFE_Q, "no everyones fine thanks", "no"),
    (SAFE_Q, "im fine", "yes"),
    (SAFE_Q, "everyone is okay", "yes"),
    (SAFE_Q, "were all good no injuries", "no"),
    (SAFE_Q, "my son has a cut on his forehead", "no"),
    (SAFE_Q, "shes complaining her neck is sore", "no"),
    (SAFE_Q, "i think shes alright but shes limping", "no"),
]


def test_probe_e_question_conditioned_polarity():
    for question, reply, expected in E_CASES:
        assert classify_answer(question, reply) == expected, (
            f"{question!r} + {reply!r}: expected {expected!r}"
        )


def test_probe_e_legacy_classifier_untouched():
    # The unconditioned classifier keeps its exact legacy behavior for
    # existing callers; only the pending-answer path gained the question.
    assert classify_yes_no("yes") == "yes"
    assert classify_yes_no("No, nobody is hurt.") == "no"
    assert classify_yes_no("im fine") == "yes"
    assert classify_yes_no("blah blah car go vroom") is None


# ── Task 4 Guard: No remote LLM provider on live turn path ─────────────────


@pytest.mark.asyncio
async def test_task4_live_turn_path_has_zero_remote_llm_calls(pack, monkeypatch):
    """Task 4: The live turn path must never invoke remote LLM narration.
    Intake phrasing uses deterministic pack slot prompts directly,
    completing turns well inside the 180 ms budget without external I/O."""
    import asyncio
    import time

    llm_called = []

    def _trap_narration(**kwargs):
        llm_called.append(kwargs)
        raise RuntimeError("Illegal remote LLM call on live turn path!")

    monkeypatch.setattr("src.ai.narration.phrase_intake_question", _trap_narration)

    from src.agents.intake import IntakeAgent

    async def drive(i: int):
        ctx = InteractionContext(interaction_id=f"int_slm_no_llm_{i}", pack=pack)
        agent = IntakeAgent(ctx)
        await agent.run(customer_turn="calling about my car")
        await agent.run(customer_turn="No, nobody is hurt.")
        t0 = time.monotonic()
        res = await agent.run(customer_turn="Yes, I am in a safe location.")
        return time.monotonic() - t0, res

    (d0, r0), (d1, r1) = await asyncio.gather(drive(0), drive(1))
    assert not llm_called, f"Illegal remote LLM call detected on live path: {llm_called}"
    assert d0 < 0.10 and d1 < 0.10, f"Turn took too long: {d0:.3f}s / {d1:.3f}s"
    assert r0.get("question"), "Expected slot question prompt"
    assert r1.get("question"), "Expected slot question prompt"


def test_r22_category_zeroshot_rejects_over_deadline_prediction(pack):
    """When inference exceeds timeout_ms, the model result must be rejected."""
    import time
    from src.ml_runtime.category_zeroshot import predict_category_zeroshot

    class SlowEmbedder:
        def embed(self, text):
            time.sleep(0.02)  # 20ms
            class FakeEmb:
                values = [0.1] * 384
            return FakeEmb()

    res = predict_category_zeroshot(
        "my brakes failed on highway",
        pack,
        embedder=SlowEmbedder(),
        timeout_ms=1.0,  # 1ms deadline
    )
    assert res.category is None
    assert res.latency_ms > 1.0
    assert res.extraction_source == "none"
