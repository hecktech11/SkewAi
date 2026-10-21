"""Sentiment Monitor — lexicon-based frustration scoring.

Scores every customer turn with a deterministic VADER-style scorer (no LLM,
no audio ML — text only, so it works identically for voice and text channels).
Stores frustration_score per turn; rolling average of the last 3 customer
turns is the live frustration level; peak stored on the interaction.

Crossing FRONTLINE_FRUSTRATION_THRESHOLD (default 0.65) once per interaction:
Intake delivers a fast-path empathy + handoff offer, the console gets a
frustration_update highlight, and the interaction is marked for supervisor
attention.
"""

from __future__ import annotations

from typing import Any

from src.agents.base import Agent, InteractionContext
from src.config import settings
from src.ledger import record_action


# ── Mini friction lexicon (heuristic, not VADER/ML) ─────────────────────────
# Tuned for support-call frustration. Scores in [0,1] where 1 = maximally
# frustrated. Thresholds live in settings.frustration_threshold.
# NOTE: intensifiers below are MULTIPLIERS ONLY (never standalone hits).

_FRUSTRATION_LEXICON: dict[str, float] = {
    # anger / frustration only. Legal intent and urgency are separate signals.
    "angry": 0.9, "furious": 1.0, "outraged": 1.0, "outrageous": 1.0, "mad": 0.8,
    "frustrated": 0.85, "frustrating": 0.85, "annoyed": 0.7, "annoying": 0.7,
    "ridiculous": 0.8, "unacceptable": 0.85, "fed up": 0.9, "sick of": 0.9,
    "tired of": 0.7, "done with": 0.8, "enough": 0.6,
    "news": 0.7, "media": 0.7, "twitter": 0.6, "facebook": 0.6,
    # negation dampeners
    "not happy": 0.7, "not satisfied": 0.75, "not impressed": 0.7,
    # exclamation / caps handled separately
}

_LEGAL_LEXICON: dict[str, float] = {
    "lawyer": 0.95, "lawsuit": 0.95, "sue": 0.95, "attorney": 0.9,
    "complaint": 0.7, "better business": 0.8, "bbb": 0.8,
    "regulator": 0.85, "regulatory": 0.85, "fca": 0.85, "nhtsa": 0.85,
    "cfpb": 0.85, "consumer affairs": 0.8,
}

_URGENCY_LEXICON: dict[str, float] = {
    "dangerous": 0.9, "unsafe": 0.9, "scared": 0.8, "terrified": 0.95,
    "could have died": 1.0, "almost crashed": 0.95, "could have been killed": 1.0,
}

_HUMAN_REQUEST_PHRASES = (
    "talk to a human",
    "speak to a person",
    "speak to a human",
    "talk to a person",
    "real person",
    "live agent",
)

_INTENSIFIERS: dict[str, float] = {
    "really": 1.15, "very": 1.15, "extremely": 1.2, "absolutely": 1.2,
    "completely": 1.15, "totally": 1.1, "incredibly": 1.2,
}
# Back-compat alias (tests/imports may reference _LEXICON)
_LEXICON = _FRUSTRATION_LEXICON

# Punctuation / caps multipliers
_EXCLAM_CAPS_BUMP = 0.15
_ALLCAPS_BUMP = 0.1


def score_text(text: str) -> float:
    """Return a frustration score in [0, 1] for a customer utterance.

    Deterministic. No LLM.
    """
    if not text or not text.strip():
        return 0.0
    lower = text.lower()

    # ── Lexicon hit scoring (whole-word always; intensifiers excluded) ────
    hits: list[float] = []
    for term, score in _FRUSTRATION_LEXICON.items():
        import re
        count = len(re.findall(rf"\b{re.escape(term)}\b", lower))
        if count:
            hits.append(min(1.0, score) if count == 1 else min(1.0, score + 0.05 * (count - 1)))

    if not hits:
        base = 0.0
    else:
        # Take the max hit and apply a small bonus for multiple distinct hits.
        base = max(hits)
        if len(hits) > 1:
            base = min(1.0, base + 0.05 * (len(hits) - 1))
        # Intensifier multiplier ONLY when co-occurring with a real hit
        for term, mult in _INTENSIFIERS.items():
            import re
            if re.search(rf"\b{re.escape(term)}\b", lower):
                base = min(1.0, base * mult)
                break

        # Punctuation / caps amplify an emotion hit. They are not anger by themselves,
        # so "CFPB" or "lawyer" alone cannot cross the frustration threshold.
        excl = text.count("!")
        if excl >= 1:
            base = min(1.0, base + _EXCLAM_CAPS_BUMP * min(excl, 3) / 3)
        import re
        caps_words = re.findall(r"\b[A-Z]{4,}\b", text)
        if caps_words:
            base = min(1.0, base + _ALLCAPS_BUMP * min(len(caps_words), 3) / 3)

    # ── Profanity proxy ──────────────────────────────────────────────────
    # Crude check: 2+ consecutive masking chars (@ or #) suggest redacted
    # profanity. A single @ (emails) or single # (case #123) must NOT
    # trigger — the {2,} consecutive-symbol rule is the whole discriminator.
    import re as _re2

    if _re2.search(r"[@#]{2,}", text):
        base = min(1.0, base + 0.1)

    return round(base, 3)


def _max_lexicon(text: str, lexicon: dict[str, float]) -> float:
    if not text or not text.strip():
        return 0.0
    import re

    lower = text.lower()
    best = 0.0
    for term, weight in lexicon.items():
        if re.search(rf"\b{re.escape(term)}\b", lower):
            best = max(best, float(weight))
    return round(min(1.0, best), 3)


def _human_request_score(text: str) -> float:
    """1 when the caller asks for a person. The bare word agent is not a request."""
    if not text or not text.strip():
        return 0.0
    import re

    lower = text.lower()
    if any(phrase in lower for phrase in _HUMAN_REQUEST_PHRASES):
        return 1.0
    if re.search(r"\b(?:representative|supervisor)\b", lower):
        return 1.0
    return 0.0


def score_signals(text: str) -> dict[str, float]:
    """Separate emotion, urgency, legal intent, and a request for a human.

    ``frustration`` stays the emotion score. Complaint, CFPB, and lawyer alone
    are legal intent, not anger.
    """
    emotion = score_text(text)
    return {
        "emotion": emotion,
        "urgency": _max_lexicon(text, _URGENCY_LEXICON),
        "legal_intent": _max_lexicon(text, _LEGAL_LEXICON),
        "human_request": _human_request_score(text),
        "frustration": emotion,
    }


# ── Sentiment agent ────────────────────────────────────────────────────────


class SentimentAgent(Agent):
    """Scores customer turns and tracks rolling frustration.

    Trigger policy (documented for item 28): the handoff fires on the
    ROLLING 3-turn average crossing the threshold, while ``peak`` records
    the max single-turn score. A single 1.0 turn therefore raises ``peak``
    immediately (visible on the console) but pages a human only when
    frustration is sustained across the window — one angry word is not a
    page. This preserves the existing rolling-only compatibility behavior
    deliberately: paging on every spike caused alert fatigue in pilots.
    """

    name = "sentiment"

    async def run(self, customer_turn: str, **kwargs: Any) -> dict[str, Any]:
        score = score_text(customer_turn)

        # Record on the context (for the rolling window)
        ctx = self.ctx
        # Attach to the most recent customer turn
        for turn in reversed(ctx.turns):
            if turn["speaker"] == "customer":
                turn["frustration_score"] = score
                break

        # Rolling avg of last 3 customer turns + true single-turn peak
        recent_scores = [
            t.get("frustration_score", 0.0)
            for t in ctx.turns
            if t["speaker"] == "customer" and "frustration_score" in t
        ][-3:]
        rolling = sum(recent_scores) / len(recent_scores) if recent_scores else 0.0
        ctx.frustration_score = round(rolling, 3)
        # peak = max single-turn score ever (not rolling avg — a single 1.0
        # must not be diluted to 0.33 by two prior 0.0s).
        if score > ctx.peak_frustration:
            ctx.peak_frustration = round(score, 3)
        if rolling > ctx.peak_frustration:
            ctx.peak_frustration = round(rolling, 3)

        # Threshold crossing — fires once per interaction on rolling avg.
        # (Single-turn spikes raise peak immediately above but only trip the
        # handoff when sustained across the 3-turn window — see test_threshold.)
        triggered = (
            not ctx.frustration_flagged
            and rolling >= settings.frustration_threshold
        )
        if triggered:
            ctx.frustration_flagged = True
            record_action(self._action(
                action_type="frustration_flagged",
                input_summary=f"rolling frustration {rolling:.2f} >= threshold {settings.frustration_threshold}",
                output_summary="handoff offer will be emitted by orchestrator",
                evidence_ids=[],
            ))
        else:
            # Only ledger threshold-relevant scoring to avoid noise
            record_action(self._action(
                action_type="turn_scored",
                input_summary=f"customer turn scored",
                output_summary=f"score={score}, rolling={rolling:.2f}, peak={ctx.peak_frustration:.2f}",
                evidence_ids=[],
            ))

        return {
            "score": score,
            "rolling": round(rolling, 3),
            "peak": ctx.peak_frustration,
            "threshold_crossed": triggered,
            "handoff_offer": (
                "I can flag this for a human specialist right away — meanwhile let me make sure I have the details right."
                if triggered else None
            ),
        }
