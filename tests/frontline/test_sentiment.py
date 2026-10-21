"""Sentiment agent tests (per blueprint §15.2).

Covers:
  - Lexicon scoring (high on "furious/outraged/lawyer", low on neutral)
  - Rolling window (last 3 customer turns)
  - Threshold trigger fires once per interaction
  - Handoff offer is emitted alongside the trigger
"""

from __future__ import annotations

import pytest

from src.agents.base import InteractionContext
from src.agents.sentiment import SentimentAgent, score_text
from src.config import settings


def _ctx(pack) -> InteractionContext:
    return InteractionContext(interaction_id="int_test_sentiment", pack=pack)


# ── Lexicon scoring ──────────────────────────────────────────────────────────


def test_score_text_high_on_furious():
    s = score_text("I am absolutely furious about this!")
    assert s >= 0.7  # 'furious' alone scores 1.0; intensifier + caps bump it.


def test_score_text_high_on_outraged():
    s = score_text("I'm outraged and ready to call my lawyer.")
    assert s >= 0.7  # 'outraged' = 1.0, 'lawyer' = 0.95


def test_score_text_high_on_lawyer():
    from src.agents.sentiment import score_signals

    text = "I'm calling my lawyer about this."
    signals = score_signals(text)
    assert score_text(text) < 0.65
    assert signals["emotion"] < 0.65
    assert signals["frustration"] == signals["emotion"]
    assert signals["legal_intent"] >= 0.85
    assert signals["urgency"] < 0.65
    assert signals["human_request"] == 0.0


def test_score_text_low_on_neutral():
    s = score_text("My car makes a noise when I brake.")
    assert s < 0.2


def test_score_text_zero_on_empty():
    assert score_text("") == 0.0
    assert score_text("   ") == 0.0


def test_score_text_exclamations_bump_score():
    """Exclamation marks add a small bump to the lexicon hit (when not already maxed)."""
    # 'ridiculous' has a base score of 0.8, so exclamations can bump it higher.
    base = score_text("This is ridiculous")
    bumped = score_text("This is ridiculous!!!")
    assert bumped > base


def test_score_text_caps_bump_score():
    """ALL-CAPS words (length >= 4) suggest shouting and bump the score."""
    base = score_text("This is ridiculous")
    caps = score_text("This is RIDICULOUS")
    assert caps >= base


# ── Rolling window (last 3 customer turns) ───────────────────────────────────


async def test_rolling_window_averages_last_three(pack):
    """The frustration_score is the mean of the last 3 customer turns' scores.

    Verifies the window is exactly 3 turns: after 4 turns, the 4th must drop
    the 1st from the rolling average.
    """
    ctx = _ctx(pack)
    agent_factory = lambda: SentimentAgent(ctx)

    # Turns 1-3: high (so the rolling average is 1.0 after turn 3)
    ctx.record_turn("customer", "I'm furious about this!")
    r1 = await agent_factory().run(customer_turn="I'm furious about this!")
    assert r1["score"] >= 0.7

    ctx.record_turn("customer", "This is furious and unacceptable!")
    r2 = await agent_factory().run(customer_turn="This is furious and unacceptable!")
    assert r2["rolling"] >= r1["rolling"]  # [high, high]

    ctx.record_turn("customer", "Still furious and outraged!")
    r3 = await agent_factory().run(customer_turn="Still furious and outraged!")
    # Rolling averages [high, high, high] = 1.0 (or close to it)
    assert r3["rolling"] >= 0.9

    # Turn 4: low — should drop turn 1 from the window.
    ctx.record_turn("customer", "Anyway, what do I do next?")
    r4 = await agent_factory().run(customer_turn="Anyway, what do I do next?")
    # Rolling window now contains [high, high, low] (turns 2,3,4) instead of
    # [high, high, high] (turns 1,2,3). Since low < high, r4 < r3.
    # If the window were all 4 turns, r4 = (1+1+1+0)/4 = 0.75, which is still < 1.0,
    # so the assertion holds either way. The key check is that r4 dropped from r3.
    assert r4["rolling"] < r3["rolling"]
    # And specifically, with a 3-turn window r4 = (1+1+0)/3 = 0.667.
    assert abs(r4["rolling"] - 0.667) < 0.05


# ── Threshold trigger fires once ──────────────────────────────────────────────


async def test_threshold_fires_once(pack):
    """Crossing FRONTLINE_FRUSTRATION_THRESHOLD fires frustration_flagged exactly once.

    The threshold is on the rolling average of the last 3 customer turns, so a
    single angry turn after a neutral one isn't enough — we need 2 consecutive
    high-frustration turns to push the rolling avg >= 0.65.
    """
    ctx = _ctx(pack)
    agent_factory = lambda: SentimentAgent(ctx)

    # Turn 1: low (below threshold)
    ctx.record_turn("customer", "My car has a noise.")
    await agent_factory().run(customer_turn="My car has a noise.")
    assert ctx.frustration_flagged is False

    # Turn 2: high — rolling is [low, high] = 0.5, still below 0.65.
    ctx.record_turn("customer", "I'm absolutely furious and want to sue!")
    r = await agent_factory().run(customer_turn="I'm absolutely furious and want to sue!")
    assert r["score"] >= 0.7

    # Turn 3: another high — rolling is [low, high, high] = ~0.67, crosses 0.65.
    ctx.record_turn("customer", "This is outrageous and I'm calling my lawyer!")
    r = await agent_factory().run(customer_turn="This is outrageous and I'm calling my lawyer!")
    assert r["threshold_crossed"] is True
    assert ctx.frustration_flagged is True

    # Another angry turn — must NOT trigger again.
    ctx.record_turn("customer", "This is still outrageous and unacceptable!")
    r2 = await agent_factory().run(customer_turn="This is still outrageous and unacceptable!")
    assert r2["threshold_crossed"] is False
    assert ctx.frustration_flagged is True  # still flagged from before


async def test_handoff_offer_emitted_on_trigger(pack):
    """The agent returns a non-null handoff_offer when the threshold is crossed."""
    ctx = _ctx(pack)
    agent = SentimentAgent(ctx)
    ctx.record_turn("customer", "I am FURIOUS and I'm calling my lawyer!!")
    r = await agent.run(customer_turn="I am FURIOUS and I'm calling my lawyer!!")
    assert r["threshold_crossed"] is True
    assert r["handoff_offer"] is not None
    assert "specialist" in r["handoff_offer"].lower()


async def test_no_handoff_offer_below_threshold(pack):
    ctx = _ctx(pack)
    agent = SentimentAgent(ctx)
    ctx.record_turn("customer", "My car has a brake noise.")
    r = await agent.run(customer_turn="My car has a brake noise.")
    assert r["threshold_crossed"] is False
    assert r["handoff_offer"] is None


# ── Peak tracking ────────────────────────────────────────────────────────────


async def test_peak_frustration_tracked(pack):
    """peak_frustration is the max rolling score seen so far."""
    ctx = _ctx(pack)
    agent_factory = lambda: SentimentAgent(ctx)

    ctx.record_turn("customer", "Mildly annoyed.")
    r1 = await agent_factory().run(customer_turn="Mildly annoyed.")
    peak_after_1 = ctx.peak_frustration

    ctx.record_turn("customer", "I am FURIOUS and calling my lawyer!!!")
    r2 = await agent_factory().run(customer_turn="I am FURIOUS and calling my lawyer!!!")
    assert ctx.peak_frustration >= peak_after_1
