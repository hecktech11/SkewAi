"""Voice turn policy — closes the 3/10 telephony gap to pilot-grade.

Covers the five items from the 10/10 review §1 that were missing:
  - ASR confidence per token/entity → low-confidence must trigger readback
  - barge-in, silence timeout, DTMF fallback
  - TTS latency budget per turn (>1.5s perceived as dead air)
  - telephony handoff (warm transfer with context payload)
  - two-party recording consent by jurisdiction (voice = biometric in some regimes)

Hermetic + deterministic: no audio DSP, no carrier SDK. The policy layer
operates on text turns + confidence metadata supplied by any STT
(Web Speech, Whisper, Twilio Media Streams). Carrier wiring stays in
``src/channels/twilio_media.py``; this module decides WHAT to do.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any


# ── Tunables (env-overridable, safe defaults) ─────────────────────────────

def asr_confidence_threshold() -> float:
    try:
        return float(os.getenv("VOICE_ASR_CONF_THRESHOLD", "0.72"))
    except ValueError:
        return 0.72


def tts_latency_budget_ms() -> int:
    try:
        return int(os.getenv("VOICE_TTS_BUDGET_MS", "1500"))
    except ValueError:
        return 1500


def silence_timeout_ms() -> int:
    try:
        return int(os.getenv("VOICE_SILENCE_TIMEOUT_MS", "6000"))
    except ValueError:
        return 6000


def barge_in_grace_ms() -> int:
    try:
        return int(os.getenv("VOICE_BARGEIN_GRACE_MS", "800"))
    except ValueError:
        return 800


# ── Consent by jurisdiction ───────────────────────────────────────────────
# Two-party / all-party consent states + biometric-voice regimes. Minimal
# embedded table so pilots fail closed without a legal API call.

_TWO_PARTY_STATES = frozenset({
    "CA", "CT", "DE", "FL", "IL", "MD", "MA", "MT", "NV", "NH", "PA", "WA",
    "CALIFORNIA", "FLORIDA", "ILLINOIS", "WASHINGTON", "MASSACHUSETTS",
})

# Regimes where voiceprint = biometric data (BIPA Illinois, GDPR Art.9, etc.)
_BIOMETRIC_REGIMES = frozenset({"IL", "BIPA", "EU", "GDPR", "UK"})


def consent_requirement(region: str | None) -> dict[str, Any]:
    """Return consent obligations for a region code."""
    r = (region or "").strip().upper()
    two_party = r in _TWO_PARTY_STATES
    biometric = r in _BIOMETRIC_REGIMES or r in {"IL", "EU"}
    return {
        "region": r or "UNKNOWN",
        "two_party": two_party,
        "voice_is_biometric": biometric,
        "requires_explicit_voice_consent": bool(two_party or biometric),
        "disclosure_required": True,  # always disclose recording in pilot
        "script": (
            "This call may be recorded for quality and safety. "
            + ("All parties must consent to recording. Say 'I consent' to continue. "
               if two_party else "By continuing you consent to recording. ")
            + ("Your voice may be processed as biometric data; you may opt out and use text instead. "
               if biometric else "")
        ).strip(),
    }


# ── ASR confidence → readback ─────────────────────────────────────────────

@dataclass
class EntityHypothesis:
    slot: str
    value: str
    confidence: float = 1.0
    token_confidences: list[float] = field(default_factory=list)


def low_confidence_entities(
    hyps: list[EntityHypothesis], *, threshold: float | None = None
) -> list[EntityHypothesis]:
    """Entities that must trigger readback, not silent acceptance."""
    thr = asr_confidence_threshold() if threshold is None else threshold
    out = []
    for h in hyps:
        conf = h.confidence
        if h.token_confidences:
            conf = min([h.confidence] + list(h.token_confidences))
        if conf < thr:
            out.append(h)
    return out


def readback_prompt(slot: str, value: str, *, lang: str = "en") -> str:
    """Deterministic readback question for a low-confidence entity."""
    templates = {
        "en": "Just to confirm, I heard '{v}' for {s} — is that right? Say yes, or tell me the correction.",
        "es": "Solo para confirmar, escuché '{v}' para {s} — ¿es correcto? Diga sí o corríjame.",
    }
    t = templates.get(lang, templates["en"])
    return t.format(v=value, s=slot)


def parse_readback_answer(text: str) -> str | None:
    """Map a readback reply to confirmed / denied. None = unclear → re-ask."""
    t = (text or "").strip().lower()
    if not t:
        return None
    if re.match(r"^(yes|yeah|yep|yup|sí|si\b|correct|right|confirm)", t):
        return "confirmed"
    if re.match(r"^(no|nope|nah|wrong|incorrect|no es)", t):
        return "denied"
    # bare correction ("actually 2019") counts as denied + new value path
    if len(t.split()) <= 6:
        return "denied"
    return None


# ── Barge-in / silence / DTMF ─────────────────────────────────────────────

@dataclass
class TurnTiming:
    tts_started_ms: int = 0
    tts_ended_ms: int = 0
    customer_speech_ms: int | None = None
    silence_ms: int = 0


def barge_in_may_cancel(t: TurnTiming, *, now_ms: int) -> bool:
    """Barge-in may cancel TTS only after the grace window (speaker bleed ≠ barge-in)."""
    if t.tts_started_ms <= 0:
        return False
    if t.tts_ended_ms and now_ms >= t.tts_ended_ms:
        return False  # TTS already done — normal turn, not barge-in
    return (now_ms - t.tts_started_ms) >= barge_in_grace_ms()


def silence_action(silence_ms: int, *, reprompt_count: int = 0) -> str:
    """reprompt | escalate_to_human | hangup_offer."""
    if silence_ms < silence_timeout_ms():
        return "reprompt"
    if reprompt_count < 1:
        return "reprompt"
    if reprompt_count < 2:
        return "escalate_to_human"
    return "hangup_offer"


_DTMF_RE = re.compile(r"^(?:dtmf:)?([0-9*#]+)$", re.I)


def parse_dtmf(text: str) -> str | None:
    """DTMF fallback: 'press 1 if…' — accept '1', 'DTMF:1', '#', '*'.

    A bare multi-digit utterance such as "2019" is a vehicle year, not a
    keypad selection — only treat it as DTMF when it carries an explicit
    ``dtmf:`` prefix. Single keypad chars (0-9, *, #) match bare.
    """
    raw = (text or "").strip()
    if not raw:
        return None
    m = _DTMF_RE.match(raw)
    if not m:
        return None
    digits = m.group(1)
    # Explicit prefix is always honoured (telephony path).
    if raw.lower().startswith("dtmf:"):
        return digits
    # Bare input: single keypad press only. Anything longer (years like
    # "2019", case numbers, phone fragments) must go through slot extraction.
    if len(digits) == 1 and digits in "0123456789*#":
        return digits
    return None


def dtmf_fallback_prompt(options: list[str]) -> str:
    lines = [f"press {i + 1} for {opt}" for i, opt in enumerate(options[:4])]
    return "I didn't catch that. Using keys: " + ", ".join(lines) + "."


# ── TTS latency budget ────────────────────────────────────────────────────

def latency_verdict(elapsed_ms: int, *, budget_ms: int | None = None) -> dict[str, Any]:
    budget = tts_latency_budget_ms() if budget_ms is None else budget_ms
    over = elapsed_ms > budget
    return {
        "elapsed_ms": elapsed_ms,
        "budget_ms": budget,
        "over_budget": over,
        "action": "fill_with_acknowledgement" if over else "speak_directly",
        # filler avoids dead air while synthesis finishes
        "filler": "One moment — pulling that up." if over else "",
    }


# ── Warm handoff with context ─────────────────────────────────────────────

def warm_handoff_payload(
    *, interaction_id: str, slots: dict[str, str], severity: str,
    summary: str, transcript_tail: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Context bundle a supervisor / carrier transfer must receive."""
    return {
        "type": "warm_handoff",
        "interaction_id": interaction_id,
        "slots": {k: v for k, v in (slots or {}).items() if not k.startswith("__")},
        "severity": severity,
        "summary": (summary or "")[:1000],
        "transcript_tail": (transcript_tail or [])[-6:],
        "requires_ack": True,  # cold transfer forbidden for safety escalations
    }


__all__ = [
    "asr_confidence_threshold", "tts_latency_budget_ms", "silence_timeout_ms",
    "barge_in_grace_ms", "consent_requirement", "EntityHypothesis",
    "low_confidence_entities", "readback_prompt", "parse_readback_answer",
    "TurnTiming", "barge_in_may_cancel", "silence_action", "parse_dtmf",
    "dtmf_fallback_prompt", "latency_verdict", "warm_handoff_payload",
]
