"""Intake Agent — the conversation driver.

Runs the conversation; fills the pack's slot frame. The LLM never owns state.

Extraction (deterministic first):
  - gazetteer match (whole-word substring) for entity slots + category
  - year-range validation for entity_1 (when pack uses year-range)
  - regex fallback for slot-specific patterns
  - LLM structured-extraction ONLY when regex finds nothing; output validated
    against the gazetteer before acceptance.

Dialogue policy (deterministic):
  - One question per turn for the highest-priority missing required slot.
  - Per-slot re-ask limits from the manifest.
  - Global cap FRONTLINE_MAX_TURNS then wrap up.

LLM used for: empathetic phrasing of questions/acknowledgments (low temp,
~80-token cap). Fast-path (no LLM): greeting, safety questions, advisory
notice, handoff offer, goodbye, case-number readback.

Kill-switch: utterance matches the pack's escalation lexicon → set flag, skip
remaining slots, notify Sentinel, read the pack's static escalation script.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from src.agents.base import Agent, InteractionContext
from src.config import settings
from src.ledger import action_timer, record_action


# ── Extraction helpers ─────────────────────────────────────────────────────

_YEAR_RE = re.compile(r"\b(19[89]\d|20[0-2]\d)\b")


# ── Category synonyms (deterministic mapping of common words → gazetteer values) ──
# Customers say "brakes" / "engine"; the gazetteer stores canonical NHTSA
# category names like "SERVICE BRAKES". This map bridges the gap without an LLM.
# It is intentionally small and pack-specific in real deployments — the
# automotive pack ships this in code rather than the gazetteer for v2.
_CATEGORY_SYNONYMS: dict[str, str] = {
    # ── Automotive (NHTSA) ──
    "brakes": "SERVICE BRAKES",
    "brake": "SERVICE BRAKES",
    "brake pedal": "SERVICE BRAKES",
    # Common Web Speech substitutions.  These are intentionally contextual:
    # a bare "break"/"broke" must not be reclassified as a brake fault.
    "brake is not moving": "SERVICE BRAKES",
    "brake is also not moving": "SERVICE BRAKES",
    "break is not moving": "SERVICE BRAKES",
    "break is also not moving": "SERVICE BRAKES",
    "breath is not moving": "SERVICE BRAKES",
    "breath is also not moving": "SERVICE BRAKES",
    "abs": "SERVICE BRAKES",
    "engine": "ENGINE",
    "motor": "ENGINE",
    "transmission": "POWER TRAIN",
    "gearbox": "POWER TRAIN",
    "gear": "POWER TRAIN",
    "gear stuck": "POWER TRAIN",
    "gear is stuck": "POWER TRAIN",
    "gear is also stuck": "POWER TRAIN",
    "power train": "POWER TRAIN",
    "fuel": "FUEL SYSTEM",
    "gas tank": "FUEL SYSTEM",
    "steering": "STEERING",
    "suspension": "SUSPENSION",
    "shocks": "SUSPENSION",
    "struts": "SUSPENSION",
    "electrical": "ELECTRICAL SYSTEM",
    "wiring": "ELECTRICAL SYSTEM",
    "battery": "ELECTRICAL SYSTEM",
    "dashboard": "ELECTRICAL SYSTEM",
    "instrument cluster": "ELECTRICAL SYSTEM",
    "instrument panel": "ELECTRICAL SYSTEM",
    "music system": "ELECTRICAL SYSTEM",
    "infotainment": "ELECTRICAL SYSTEM",
    "stereo": "ELECTRICAL SYSTEM",
    "radio": "ELECTRICAL SYSTEM",
    "lights": "EXTERIOR LIGHTING",
    "headlights": "EXTERIOR LIGHTING",
    "taillights": "EXTERIOR LIGHTING",
    "airbag": "AIR BAGS",
    "air bag": "AIR BAGS",
    "airbags": "AIR BAGS",
    "seatbelt": "SEAT BELTS",
    "seat belt": "SEAT BELTS",
    "seatbelts": "SEAT BELTS",
    "tires": "TIRES",
    "tire": "TIRES",
    "wheels": "WHEELS",
    "wheel": "WHEELS",
    "windshield": "VISIBILITY",
    "wipers": "VISIBILITY",
    "visibility": "VISIBILITY",
    "child seat": "CHILD SEAT",
    "car seat": "CHILD SEAT",
    "structure": "STRUCTURE",
    "frame": "STRUCTURE",
    "speed control": "VEHICLE SPEED CONTROL",
    "cruise control": "VEHICLE SPEED CONTROL",
    "accelerator": "VEHICLE SPEED CONTROL",
    "throttle": "VEHICLE SPEED CONTROL",
    "unintended acceleration": "VEHICLE SPEED CONTROL",
    "sudden acceleration": "VEHICLE SPEED CONTROL",
    "surged forward": "VEHICLE SPEED CONTROL",
    "full throttle": "VEHICLE SPEED CONTROL",
    "took off without input": "VEHICLE SPEED CONTROL",
    "spiked to redline": "VEHICLE SPEED CONTROL",
    "accelerated uncontrollably": "VEHICLE SPEED CONTROL",
    "lunged forward": "VEHICLE SPEED CONTROL",
    "tie rod": "STEERING",
    "power steering": "STEERING",
    "air vents": "ELECTRICAL SYSTEM",
    "switch panel": "ELECTRICAL SYSTEM",
    "catalytic converter": "ENGINE",
    "rollover": "STRUCTURE",
    "seat collapse": "SEATS",
    "forward collision": "FORWARD COLLISION AVOIDANCE",
    "lane departure": "LANE DEPARTURE",
    # This corpus codes electronic parking-brake events as SERVICE BRAKES.
    "parking brake": "SERVICE BRAKES",
    "esc": "ELECTRONIC STABILITY CONTROL",
    "stability control": "ELECTRONIC STABILITY CONTROL",
    "traction control": "TRACTION CONTROL SYSTEM",
    "hybrid": "HYBRID PROPULSION SYSTEM",
    # ── Finance (CFPB) ──
    "double charged": "Incorrect charges",
    "double charge": "Incorrect charges",
    "charged twice": "Incorrect charges",
    "extra fee": "Fees or interest",
    "hidden fee": "Fees or interest",
    "late fee": "Late fees",
    "overdraft fee": "Fees or interest",
    "annual fee": "Fees or interest",
    "interest rate": "Fees or interest",
    "unauthorized": "Unauthorized transactions",
    "unauthorized transaction": "Unauthorized transactions",
    "unauthorized charge": "Unauthorized transactions",
    "identity theft": "Identity theft",
    "stolen identity": "Identity theft",
    "someone opened": "Identity theft",
    "fraud": "Fraud or scam",
    "scam": "Fraud or scam",
    "scammed": "Fraud or scam",
    "phishing": "Fraud or scam",
    "billing dispute": "Billing disputes",
    "disputed charge": "Billing disputes",
    "wrong charge": "Billing disputes",
    "closed my account": "Closing an account",
    "account closed": "Closing an account",
    "can't close": "Closing an account",
    "opened an account": "Opening an account",
    "account opening": "Opening an account",
    "credit report": "Problem with credit reporting",
    "credit reporting": "Problem with credit reporting",
    "wrong information": "Incorrect information on your report",
    "customer service": "Customer service",
    "can't reach": "Customer service",
    "no response": "Communication tactics",
    "calls": "Communication tactics",
    "card declined": "Trouble using card",
    "card not working": "Trouble using card",
    "transfer failed": "Transaction issues",
    "transaction failed": "Transaction issues",
}

# Speed-control phrases outrank a leading "engine"/"motor" mention.
_SPEED_CONTROL_PHRASES: frozenset[str] = frozenset(
    {
        "took off without input",
        "spiked to redline",
        "accelerated uncontrollably",
        "lunged forward",
    }
)

# Longer collocations that contain a system synonym but are not an NHTSA
# investigation class: monitors, maintenance, control-location, price talk.
_CATEGORY_SKIP_PHRASES: tuple[str, ...] = (
    "check engine light",
    "engine coolant",
    "tire pressure",
    "tire gauge",
    "windshield wiper",
    "windshield washer",
    "washer fluid",
    "wiper fluid",
    "fuel economy",
    "price of gas",
    "fan motor",
    "blower fan",
    "cruise control button",
    "battery is dead",
    "jump start",
    "turning steering wheel",
    "on steering wheel",
)


def _skip_spans(lower: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for ph in _CATEGORY_SKIP_PHRASES:
        for m in re.finditer(rf"\b{re.escape(ph)}\b", lower):
            spans.append((m.start(), m.end()))
    return spans


def _overlaps_span(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    return any(not (end <= s or start >= e) for s, e in spans)


def _category_hit_suppressed(text: str, start: int, end: int, syn: str) -> bool:
    """Drop maintenance/location collocations, negated hits, and 'wondering if'."""
    lower = text.lower()
    if _overlaps_span(start, end, _skip_spans(lower)):
        return True
    # Negation belongs to the current clause.  Looking across commas meant
    # "steering is not working, the brakes are not moving" suppressed the
    # brake fault because it inherited the first clause's "not".
    clause_start = max(
        lower.rfind(",", 0, start),
        lower.rfind(";", 0, start),
        lower.rfind(".", 0, start),
        lower.rfind("!", 0, start),
        lower.rfind("?", 0, start),
    ) + 1
    clause_prefix = lower[clause_start:start]
    clause_tokens = re.findall(r"[a-z0-9'-]+", clause_prefix)
    if any(token in _NEGATORS for token in clause_tokens[-_HEDGE_WINDOW_TOKENS:]):
        return True
    prefix = clause_prefix
    tail = prefix[-48:]
    if re.search(r"\bwondering if\b", tail):
        return True
    return False


def _category_synonym_hits(
    text: str,
    *,
    allowed: Any | None = None,
) -> list[tuple[int, int, str, str]]:
    """Non-contained synonym spans, leftmost-first.

    Nested shorter hits are dropped. Skip-phrases and negated mentions
    never become the primary class.
    """
    lower = text.lower()
    hits: list[tuple[int, int, str, str]] = []
    for syn, canonical in _CATEGORY_SYNONYMS.items():
        if allowed is not None and not allowed(canonical):
            continue
        for m in re.finditer(rf"\b{re.escape(syn)}\b", lower):
            if _category_hit_suppressed(text, m.start(), m.end(), syn):
                continue
            hits.append((m.start(), m.end(), syn, canonical))
    keep: list[tuple[int, int, str, str]] = []
    for h in hits:
        h_len = h[1] - h[0]
        contained = False
        for o in hits:
            if o[0] <= h[0] and o[1] >= h[1] and (o[1] - o[0]) > h_len:
                contained = True
                break
        if not contained:
            keep.append(h)
    keep.sort(key=lambda span: (span[0], -(span[1] - span[0])))
    return keep


def match_category_synonym(
    text: str,
    *,
    allowed: Any | None = None,
) -> str | None:
    """Primary category: speed-control phrase, else leftmost remaining span.

    ``allowed`` is optional ``canonical -> truthy`` (gazetteer lookup) so a
    finance synonym cannot fire on an automotive pack.
    """
    hits = _category_synonym_hits(text, allowed=allowed)
    if not hits:
        return None
    for _start, _end, syn, canonical in hits:
        if syn in _SPEED_CONTROL_PHRASES:
            return canonical
    return hits[0][3]


def extract_pack_category(
    text: str,
    ctx: InteractionContext | None = None,
    *,
    rules_only: bool = False,
) -> str | None:
    """Shared category extractor for live intake and shadow scoring."""
    allowed = None
    gaz_val = None
    if ctx is not None:
        gaz = ctx.pack.gazetteer_for_slot("category")
        allowed = gaz.lookup if gaz is not None else None
        gaz_val = _extract_via_gazetteer(text, ctx, "category")
    syn = match_category_synonym(text, allowed=allowed)
    if syn:
        rules_cat = syn
    elif not gaz_val:
        rules_cat = None
    else:
        lower = text.lower()
        key = gaz_val.lower()
        m = re.search(rf"\b{re.escape(key)}\b", lower)
        if m and _category_hit_suppressed(text, m.start(), m.end(), key):
            rules_cat = None
        else:
            rules_cat = gaz_val

    if not rules_only and ctx is not None:
        try:
            from src.ml_runtime.category_zeroshot import extract_category_with_mode
            cat, _source, _meta = extract_category_with_mode(text, ctx, rules_cat)
            return cat
        except Exception:
            return rules_cat
    return rules_cat


def _extract_year(text: str, year_range: tuple[int, int] | None) -> str | None:
    """Find a 4-digit year in text that falls within `year_range`."""
    for m in _YEAR_RE.finditer(text):
        y = int(m.group(1))
        if year_range is None or year_range[0] <= y <= year_range[1]:
            return str(y)
    return None


def _extract_via_gazetteer(text: str, ctx: InteractionContext, slot_name: str) -> str | None:
    """Find a gazetteer value that appears in `text` (whole-word match)."""
    gaz = ctx.pack.gazetteer_for_slot(slot_name)
    if gaz is None:
        return None
    return gaz.match_substring(text)


# Filler utterances that must never fill the free-text description slot.
# Voice STT often delivers bare greetings ("hello hello hello"), single-word
# acknowledgements ("yes", "okay"), or mic-check loops. Accepting those as the
# symptom description pollutes the slot frame: the real "what's happening?"
# question is then skipped and the contact closes with junk evidence.
_FILLER_TOKENS = frozenset({
    "hello", "hi", "hey", "hii", "helo",
    "yes", "yeah", "yep", "yup", "yea", "no", "nope", "nah",
    "ok", "okay", "k", "thanks", "thank", "you", "please",
    "good", "great", "fine", "well", "sure", "right",
    "testing", "test", "mic", "mike", "check", "hello?",
})


def _is_filler_description(text: str) -> bool:
    """True when `text` carries no symptom content (greeting/ack/mic-check)."""
    stripped = (text or "").strip()
    if not stripped:
        return True
    # Strip punctuation, collapse whitespace — "hello hello hello" -> tokens.
    cleaned = re.sub(r"[^a-z0-9\s]", " ", stripped.lower())
    tokens = [t for t in cleaned.split() if t]
    if not tokens:
        return True
    # Single repeated token ("hello hello hello hello") is mic-check, not a symptom.
    if len(set(tokens)) == 1:
        return True
    # Very short utterances composed only of filler vocabulary.
    if len(tokens) <= 3 and all(t in _FILLER_TOKENS for t in tokens):
        return True
    # Bare acknowledgement with no content ("yes", "okay thanks").
    if len(" ".join(tokens)) < 8 and all(t in _FILLER_TOKENS for t in tokens):
        return True
    return False


def _extract_description(text: str) -> str | None:
    """Free-text description that rejects filler so the symptom question survives."""
    stripped = (text or "").strip()
    if not stripped:
        return None
    if _is_filler_description(stripped):
        return None
    return stripped


# ── Kill-switch ─────────────────────────────────────────────────────────────

_SAFETY_ASKED_KEY = "__safety_questions_asked__"
_SAFETY_PENDING_KEY = "__safety_pending__"

# Spoken yes/no safety is a live-call protocol. Ticket ingest is not a caller
# and must not be fed canned "nobody is hurt" answers (or force-enriched while
# those questions are still pending).
_NON_INTERACTIVE_CHANNELS = frozenset({"webhook", "ticket", "email_batch"})


def _asks_spoken_safety(ctx: InteractionContext) -> bool:
    ch = (getattr(ctx, "channel", None) or "web_text").strip().lower()
    return ch not in _NON_INTERACTIVE_CHANNELS

# Injury-family lexicon terms that are also the vocabulary of "is anyone hurt?"
# Negated forms ("nobody is hurt") must not fire the kill-switch.
_NEGATABLE_LEXICON = frozenset({
    "hurt", "hurts", "injured", "injury", "bleeding", "burned", "trapped",
})

# Negation/hedge window (audit 2.3): a negator or hedge within N tokens
# BEFORE any kill-switch term downgrades the hit to a safety question
# instead of an escalation. False P1s burn supervisor trust and pollute the
# safety base rate; a question still keeps the contact safe.
_NEGATORS = frozenset({
    "no", "not", "nobody", "none", "never", "n't", "without",
    "hardly", "barely", "scarcely", "didn't", "didnt", "did not",
    "wasn't", "wasnt", "was not", "weren't", "werent", "were not",
    "isn't", "isnt", "is not", "aren't", "arent", "are not",
    "haven't", "havent", "hasn't", "hasnt", "no-one", "nothing",
})
_HEDGES = frozenset({
    "might", "may", "could", "worried", "afraid", "concerned", "wondering",
    "if", "whether", "almost", "nearly", "possible", "possibly",
})
_HEDGE_WINDOW_TOKENS = 6


def classify_yes_no(text: str) -> str | None:
    """Map a free-text safety reply to yes / no. None = unclear."""
    t = (text or "").strip().lower()
    if not t:
        return None
    if re.search(r"\b(nobody|no one|no-one|none)\b.{0,32}\b(hurt|hurts|injured|harmed|bleeding)\b", t):
        return "no"
    if re.search(r"\b(not|n't|isnt|isn't)\s+(hurt|injured|bleeding|safe)\b", t):
        return "no"
    if re.search(r"\b(unsafe|in danger|not safe)\b", t):
        return "no"
    if re.match(r"^(no|nope|nah|negative)\b", t):
        return "no"
    if re.match(r"^(yes|yeah|yep|yup|yea|affirmative)\b", t):
        return "yes"
    if re.search(r"\b(hurt|hurts|injured|bleeding|ambulance|hospital|burned|trapped)\b", t):
        return "yes"
    if re.search(r"\b(safe|i'm fine|im fine|okay|ok)\b", t):
        return "yes"
    return None


def escalate_on_for_prompt(prompt: str) -> str | None:
    """Which polarity of a yes/no answer should escalate this safety question.

    ``yes`` — "Is anyone hurt?" / "Has money been taken?"
    ``no``  — "Are you in a safe location?"
    ``None`` — informational (e.g. "Have you contacted the fraud department?").
    """
    p = (prompt or "").strip().lower()
    if "safe location" in p or re.search(r"\bare you (safe|in a safe)\b", p):
        return "no"
    if any(w in p for w in ("hurt", "injur", "danger", "taken", "stolen", "emergency")):
        return "yes"
    return None


# Injury-state vocabulary for question-conditioned polarity (defect #2 fix).
# Scoped ONLY to interpreting replies to safety questions — never a
# kill-switch input, never a category signal, never added to the pack
# lexicons. A stopgap until the SLM (§4 of the SLM plan); do not extend as a
# substitute for open-vocabulary understanding (§7).
_INJURY_STATE_TERMS = frozenset({
    "hurt", "hurts", "injured", "injury", "injuries",
    "bleeding", "bleed", "cut", "wound", "wounded",
    "sore", "limping", "limp", "pain", "painful",
    "dizzy", "unconscious", "trapped", "ambulance", "hospital",
    "broke", "broken", "fracture", "fractured", "swollen", "bruise", "bruised",
})

# Speaker-wellbeing assertions ("we're fine") — polarity is the opposite of
# whatever would escalate the question being answered.
_WELLBEING_TERMS = frozenset({
    "fine", "okay", "ok", "good", "safe", "alright", "well", "great",
    "all good", "no injuries", "no injury",
})


def _has_term(text: str, terms: frozenset[str]) -> bool:
    for term in terms:
        if " " in term:
            if term in text:
                return True
        elif re.search(rf"\b{re.escape(term)}\b", text):
            return True
    return False


def classify_answer(question: str, text: str) -> str | None:
    """Polarity of a reply CONDITIONED on the safety question asked.

    The legacy ``classify_yes_no`` never sees the question, so *"im fine"*
    answering *"Is anyone hurt?"* resolved to ``yes`` (false P1), and injury
    reports without yes/no tokens (*"my son has a cut"*) resolved to ``None``
    (re-asked on a live call). Rules, in order:

    1. explicit leading yes/no token → literal, question-independent;
    2. negated-injury phrasing → ``no``;
    3. injury-state assertion → the polarity that escalates *this* question
       (injury reported against any safety question stays on the safe side);
    4. speaker-wellbeing assertion → the opposite of the escalate polarity
       (``im fine`` = ``no`` to "hurt?", ``yes`` to "safe location?");
    5. anything else → legacy ``classify_yes_no`` (unchanged behavior).
    """
    t = (text or "").strip().lower()
    if not t:
        return None
    if re.match(r"^(no|nope|nah|negative)\b", t):
        return "no"
    if re.match(r"^(yes|yeah|yep|yup|yea|affirmative)\b", t):
        return "yes"
    if re.search(r"\b(nobody|no one|no-one|none)\b.{0,32}\b(hurt|hurts|injured|harmed|bleeding)\b", t):
        return "no"
    if re.search(r"\b(not|n't|isnt|isn't)\s+(hurt|injured|bleeding|safe)\b", t):
        return "no"
    if re.search(r"\b(unsafe|in danger|not safe)\b", t):
        return "no"
    if re.search(r"\bno\s+(injur\w*|hurt|harmed|bleeding|blood)\b", t):
        return "no"
    if _has_term(t, _INJURY_STATE_TERMS):
        return escalate_on_for_prompt(question) or "yes"
    if _has_term(t, _WELLBEING_TERMS):
        esc = escalate_on_for_prompt(question)
        if esc == "yes":
            return "no"
        if esc == "no":
            return "yes"
    return classify_yes_no(text)


def _parse_asked_indices(raw: Any) -> set[int]:
    out: set[int] = set()
    for part in str(raw or "").split(","):
        part = part.strip()
        if part.isdigit():
            out.add(int(part))
    return out


def _store_asked_indices(ctx: InteractionContext, asked: set[int]) -> None:
    ctx.slots[_SAFETY_ASKED_KEY] = ",".join(str(i) for i in sorted(asked))


def _lexicon_term_negated(cleaned: str, term: str) -> bool:
    """True when *term* appears only inside a negated injury phrase."""
    if term.lower() not in _NEGATABLE_LEXICON:
        return False
    esc = re.escape(term.lower())
    if re.search(rf"\b(no|not|nobody|no one|no-one|none|never|n't)\b.{{0,32}}\b{esc}\b", cleaned):
        return True
    return False


def _term_hedged_or_negated(text: str, term: str, *, hedges: bool = True) -> bool:
    """True when a negator/hedge sits within the token window before *term*.

    Covers every lexicon term (not just the injury family): "no fire, just
    a smell", "I'm worried it might catch fire", "without any smoke", "I didn't crash",
    "nobody went to the hospital". Category extraction passes hedges=False so
    "worried my brakes failed" still yields SERVICE BRAKES.
    """
    blockers = _NEGATORS | _HEDGES if hedges else _NEGATORS
    t_lower = (text or "").lower()
    esc = re.escape(term.lower())
    m = re.search(rf"\b{esc}\b", t_lower)
    if m:
        prefix = t_lower[:m.start()]
        toks = re.findall(r"[a-z0-9'-]+", prefix)
        if toks:
            window = toks[-_HEDGE_WINDOW_TOKENS:]
            if any(w in blockers for w in window):
                return True
            win_str = " ".join(window)
            if any(ph in win_str for ph in ("no one", "no fire", "not on fire", "didn't", "did not", "no injuries", "no accident")):
                return True
        return False

    toks = re.findall(r"[a-z0-9']+", t_lower)
    targets = set(term.lower().split())
    for i, tok in enumerate(toks):
        if tok not in targets:
            continue
        window = toks[max(0, i - _HEDGE_WINDOW_TOKENS):i]
        if any(w in blockers for w in window):
            return True
        win_str = " ".join(window)
        if any(ph in win_str for ph in ("no one", "no fire", "not on fire", "didn't", "did not")):
            return True
    return False


def _check_kill_switch(text: str, ctx: InteractionContext) -> str | None:
    """If `text` matches the pack's escalation lexicon, return the matched term.

    Excludes common safe contexts (e.g. 'fraud department', 'fraud alert') so
    that answering safety questions doesn't false-trigger the kill-switch.
    Injury-family terms are skipped when the utterance is a negation
    ("nobody is hurt") — those answers are bound by the safety-question path.
    """
    text_lower = text.lower()
    # Common safe phrases and metaphorical idioms that contain kill-switch terms
    # but aren't actual safety emergencies (prevents supervisor alert burnout).
    safe_phrases = [
        "fraud department", "fraud alert", "fraud division", "fraud team",
        "fraud prevention", "fraud protection",
        "killing me", "killing my", "killing us", "killing the",
        "dying to", "dying of", "heart attack", "scared to death",
        "worried to death", "bored to death", "sick to death",
        "smoke and mirrors", "fire drill", "crash course",
        "burn through", "burning through", "burning a hole",
        "spitting fire", "burned out", "burning out", "burn out",
        "grass fire", "reading light burned", "bulb burned",
    ]
    cleaned = text_lower
    for safe in safe_phrases:
        cleaned = cleaned.replace(safe, "")
    for term in ctx.pack.manifest.safety.escalation_lexicon:
        # whole-word-ish match on the cleaned text
        if re.search(rf"\b{re.escape(term.lower())}\b", cleaned):
            if _term_hedged_or_negated(cleaned, term) or _lexicon_term_negated(cleaned, term):
                continue
            return term
    return None


# ── Intake agent ──────────────────────────────────────────────────────────


class IntakeAgent(Agent):
    """Drives the conversation. Fills the slot frame deterministically."""

    name = "intake"

    async def run(self, customer_turn: str = "", **kwargs: Any) -> dict[str, Any]:
        """Process one customer turn. Returns:
            {
              "extracted": {slot: value, ...},
              "question": str,                # next question to ask
              "fast_path": bool,             # True if no LLM was used
              "kill_switch": str | None,     # matched term if escalation
              "slots_filled_now": bool,
            }
        """
        ctx = self.ctx
        extracted: dict[str, str] = {}

        # ── 10/10: ASR confidence → readback + DTMF + identity (pilot voice) ─
        # kwargs: asr_confidence={slot: 0..1}, dtmf="1", identity={"verified": bool}
        # Low-confidence entities never silently accept: force readback turn.
        try:
            from src.voice.policy import (
                asr_confidence_threshold, dtmf_fallback_prompt, parse_dtmf,
                readback_prompt,
            )
            _conf = kwargs.get("asr_confidence") or {}
            if isinstance(_conf, dict) and _conf:
                _thr = asr_confidence_threshold()
                _low = [(s, float(c)) for s, c in _conf.items()
                        if isinstance(c, (int, float)) and float(c) < _thr]
                if _low and customer_turn:
                    if ctx.count_turn() >= settings.max_turns and not ctx.has_required_slots():
                        return self._max_turns_response()
                    _slot, _c = _low[0]
                    _q = readback_prompt(_slot, customer_turn.strip()[:120])
                    record_action(self._action(
                        action_type="question_asked",
                        input_summary=f"low asr confidence slot='{_slot}' conf={_c:.2f}<{_thr:.2f}",
                        output_summary=_q[:500],
                    ))
                    return {"extracted": {}, "question": _q, "fast_path": True,
                            "kill_switch": None, "slots_filled_now": False,
                            "readback": True, "readback_slot": _slot,
                            "asr_confidence": _c}
            # DTMF menu is stateful: offering the menu sets a flag; the NEXT
            # digit is a selection, not another menu offer (fixes infinite
            # "press 1 ... press 1 ..." loop). Year-like input never reaches
            # here: parse_dtmf rejects bare multi-digit, and we double-guard
            # with an explicit year check so "2023" always fills entity_1.
            _raw_turn = (customer_turn or "").strip()
            _year_first = _extract_year(_raw_turn, self._year_range_for_entity_1())
            _dtmf = parse_dtmf(_raw_turn)
            if _year_first and _dtmf and len(_raw_turn) > 1:
                _dtmf = None  # year wins over keypad on any multi-char input
            if _dtmf and ctx.slots.get("__dtmf_offered__"):
                ctx.slots.pop("__dtmf_offered__", None)
                if _dtmf == "1":
                    # "continue": fall through to normal slot extraction below
                    # so the current question is re-asked with progress kept.
                    record_action(self._action(
                        action_type="slot_extracted",
                        input_summary=f"dtmf selection: '{_dtmf}' (continue)",
                        output_summary="dtmf menu accepted; resuming slot collection",
                    ))
                    _dtmf = None
                elif _dtmf == "2":
                    record_action(self._action(
                        action_type="handoff_offer_emitted",
                        input_summary=f"dtmf selection: '{_dtmf}' (speak to someone)",
                        output_summary="escalate_to_human",
                    ))
                    try:
                        from src.frontline.validation_queue import enqueue_insight

                        enqueue_insight(
                            kind="dtmf_handoff",
                            interaction_id=ctx.interaction_id,
                            summary="caller pressed 2 for a specialist",
                        )
                    except Exception:
                        pass
                    return {"extracted": {}, "question": (
                        "Connecting you to a human specialist now — thanks for holding."
                    ), "fast_path": True, "kill_switch": None,
                        "slots_filled_now": False,
                        "escalate_low_confidence": True,
                        "low_confidence_slot": "dtmf_menu",
                        "dtmf": _dtmf}
                else:
                    # Unknown key: re-offer once, then fall through.
                    record_action(self._action(
                        action_type="slot_extracted",
                        input_summary=f"dtmf selection: '{_dtmf}' (unknown key)",
                        output_summary="re-offering dtmf menu once",
                    ))
                    return {"extracted": {}, "question": dtmf_fallback_prompt(["continue", "speak to someone"]),
                            "fast_path": True, "kill_switch": None,
                            "slots_filled_now": False, "dtmf": _dtmf}
            # No menu pending: a bare "2" is an explicit human request (one
            # press, immediate handoff). A bare "1" just falls through to
            # normal collection so the current question repeats naturally —
            # never an infinite menu loop.
            if _dtmf == "2" and not ctx.has_required_slots():
                record_action(self._action(
                    action_type="handoff_offer_emitted",
                    input_summary="dtmf: bare '2' (speak to someone)",
                    output_summary="escalate_to_human",
                ))
                return {"extracted": {}, "question": (
                    "Connecting you to a human specialist now — thanks for holding."
                ), "fast_path": True, "kill_switch": None,
                    "slots_filled_now": False,
                    "escalate_low_confidence": True,
                    "low_confidence_slot": "dtmf_menu",
                    "dtmf": _dtmf}
            if _dtmf == "1":
                record_action(self._action(
                    action_type="slot_extracted",
                    input_summary="dtmf: bare '1' (continue)",
                    output_summary="ignoring bare key; resuming slot collection",
                ))
                _dtmf = None
            if _dtmf and not ctx.has_required_slots() and not _year_first:
                ctx.slots["__dtmf_offered__"] = "1"
                record_action(self._action(
                    action_type="slot_extracted",
                    input_summary=f"dtmf fallback: '{_dtmf}'",
                    output_summary="dtmf digit captured; mapped by orchestrator menu",
                ))
                return {"extracted": {}, "question": dtmf_fallback_prompt(["continue", "speak to someone"]),
                        "fast_path": True, "kill_switch": None,
                        "slots_filled_now": False, "dtmf": _dtmf}
        except Exception:
            pass

        # ── Case-status lookup (returning caller with case number) ────────
        if customer_turn:
            try:
                from src.agents.case_status import try_case_status_from_utterance

                status_hit = try_case_status_from_utterance(customer_turn)
                if status_hit and status_hit.get("found"):
                    record_action(self._action(
                        action_type="question_asked",
                        input_summary="case_status_lookup",
                        output_summary=status_hit["reply"][:500],
                    ))
                    return {
                        "extracted": {},
                        "question": status_hit["reply"],
                        "fast_path": True,
                        "kill_switch": None,
                        "slots_filled_now": False,
                        "case_status": status_hit,
                    }
            except Exception:
                pass

        # ── Prefill from entity memory (dynamic slot skip) ───────────────
        self._prefill_from_memory()

        # ── Bind the pending safety-question answer BEFORE the lexicon ───
        # The pack asks "Is anyone hurt?" then used to ignore the reply.
        # Affirmative/negative answers are interpreted against escalate_on.
        safety_answered = False
        if customer_turn and not ctx.safety_flags.get("escalation"):
            pending_before = ctx.slots.get(_SAFETY_PENDING_KEY)
            asked_before = ctx.slots.get(_SAFETY_ASKED_KEY)
            safety_hit = self._evaluate_pending_safety_answer(customer_turn)
            if safety_hit:
                return safety_hit
            # A non-escalating safety reply ("nobody is hurt") was consumed:
            # it must not double as the symptom description.
            if pending_before is not None and str(pending_before).strip() != "":
                pending_after = ctx.slots.get(_SAFETY_PENDING_KEY)
                asked_after = ctx.slots.get(_SAFETY_ASKED_KEY)
                if pending_after != pending_before or asked_after != asked_before:
                    safety_answered = True

        # ── Kill-switch check FIRST ──────────────────────────────────────
        kill_term = _check_kill_switch(customer_turn, ctx) if customer_turn else None
        if kill_term and customer_turn and _term_hedged_or_negated(customer_turn, kill_term):
            # Downgrade (audit 2.3): negated/hedged mention ("no fire, just a
            # smell", "worried it might catch fire") becomes a safety
            # QUESTION, not a Critical/P1 escalation — still ledgered, still
            # safe, without burning supervisor trust.
            record_action(self._action(
                action_type="safety_flag_raised",
                input_summary=f"hedged/negated lexicon mention: '{kill_term}'",
                output_summary="downgraded to safety_question (no escalation flag)",
                evidence_ids=[],
                ok=True,
            ))
            safety_q = self._next_safety_question() or (
                "Are you in a safe location right now?"
            )
            return {
                "extracted": {},
                "question": safety_q,
                "fast_path": True,
                "kill_switch": None,
                "slots_filled_now": False,
                "safety_question": True,
                "safety_downgraded": kill_term,
            }
        if kill_term:
            ctx.safety_flags["escalation"] = True
            ctx.safety_flags[kill_term] = True
            record_action(self._action(
                action_type="safety_flag_raised",
                input_summary=f"customer utterance matched lexicon: '{kill_term}'",
                output_summary=f"safety_flag[{kill_term}]=True; intake will skip remaining slots",
                evidence_ids=[],
            ))
            return {
                "extracted": {},
                "question": ctx.pack.manifest.safety.escalation_script,
                "fast_path": True,
                "kill_switch": kill_term,
                "slots_filled_now": False,
            }

        # ── Slot extraction from the customer turn ────────────────────────
        if customer_turn:
            # First pass: structured slots (year/gazetteer/regex). The
            # free-text description runs second so a one-word entity answer
            # ("CR-V", "Honda", "2019") cannot also become the symptom.
            for slot in ctx.pack.manifest.slot_frame:
                if slot.name == "description":
                    continue
                if ctx.slots.get(slot.name):
                    # A caller can report more than one affected system. Keep
                    # the first category as the routing primary, while adding
                    # every later category to the live case context.
                    if slot.name == "category":
                        self._merge_secondary_categories(
                            customer_turn, str(ctx.slots.get("category") or "")
                        )
                    continue  # already filled
                value = self._extract_slot(slot, customer_turn)
                if value:
                    extracted[slot.name] = value
                    ctx.slots[slot.name] = value
            # Second pass: description needs real symptom content — not a
            # safety reply, not filler, and not a bare entity echo.
            if not safety_answered and not ctx.slots.get("description"):
                desc = self._extract_slot(
                    next(s for s in ctx.pack.manifest.slot_frame if s.name == "description"),
                    customer_turn,
                )
                if desc:
                    # Bare entity answers ("CR-V") must not double as symptoms.
                    # Require symptom-length content unless every other
                    # required slot is already filled (then any text is the
                    # answer to "what's happening?").
                    others_pending = any(
                        s.name != "description" and not ctx.slots.get(s.name)
                        for s in ctx.pack.manifest.slot_frame
                    )
                    toks = [t for t in re.sub(r"[^a-z0-9\s]", " ", desc.lower()).split() if t]
                    if not others_pending or (len(desc.strip()) >= 8 and len(toks) >= 2):
                        extracted["description"] = desc
                        ctx.slots["description"] = desc

            if extracted:
                claims = []
                if "category" in extracted and ctx.slots.get("__category_source__") == "slm":
                    span_info = ctx.slots.get("__category_span__") or {}
                    if span_info.get("span_text"):
                        claims.append({
                            "claim_text": span_info["span_text"],
                            "evidence_id": "turn_customer",
                            "span_start": span_info.get("span_start", 0),
                            "span_end": span_info.get("span_end", 0),
                        })
                record_action(self._action(
                    action_type="slot_extracted",
                    input_summary=f"customer turn: '{customer_turn[:200]}'",
                    output_summary=f"extracted: {extracted}",
                    evidence_ids=[],
                    claims=claims,
                ))

        # ── Pick the next question ────────────────────────────────────────
        # Always ask safety questions first if pack defines them AND no
        # escalation flag has been raised. The global turn cap still wins:
        # unanswered safety questions must not loop past FRONTLINE_MAX_TURNS.
        customer_turn_count = ctx.count_turn()
        if customer_turn_count < settings.max_turns:
            safety_q = self._next_safety_question()
            if safety_q:
                record_action(self._action(
                    action_type="question_asked",
                    input_summary="safety question required by pack",
                    output_summary=safety_q,
                ))
                return {
                    "extracted": extracted,
                    "question": safety_q,
                    "fast_path": True,
                    "kill_switch": None,
                    "slots_filled_now": bool(extracted),
                    "safety_question": True,
                }
        else:
            ctx.slots.pop(_SAFETY_PENDING_KEY, None)

        # Global turn cap: stop collecting and signal orchestrator to wrap up.
        if customer_turn_count >= settings.max_turns and not ctx.has_required_slots():
            out = self._max_turns_response()
            out["extracted"] = extracted
            out["slots_filled_now"] = bool(extracted)
            return out

        # Otherwise pick highest-priority missing required slot.
        next_slot = self._next_required_slot()
        if next_slot is None and not ctx.has_required_slots():
            # All remaining slots exhausted re-asks → confidence escalate
            remaining = ctx.required_slots_remaining()
            if remaining:
                record_action(self._action(
                    action_type="handoff_offer_emitted",
                    input_summary=f"low confidence; exhausted re-asks for {remaining}",
                    output_summary="escalate_to_human",
                ))
                try:
                    from src.frontline.validation_queue import enqueue_insight

                    enqueue_insight(
                        kind="low_confidence",
                        interaction_id=ctx.interaction_id,
                        summary=f"low confidence; exhausted re-asks for {remaining}",
                    )
                except Exception:
                    pass
                return {
                    "extracted": extracted,
                    "question": (
                        "I'm having trouble capturing that clearly. "
                        "Let me connect you with a specialist who can help."
                    ),
                    "fast_path": True,
                    "kill_switch": None,
                    "slots_filled_now": bool(extracted),
                    "escalate_low_confidence": True,
                    "low_confidence_slot": remaining[0],
                }
        if next_slot:
            ctx.slot_attempts.setdefault(next_slot.name, 0)
            ctx.slot_attempts[next_slot.name] += 1
            attempts = ctx.slot_attempts[next_slot.name]
            max_re = getattr(next_slot, "max_re_asks", 2) or 2
            # Confidence-gated escalation: too many re-asks without a solid fill → handoff.
            # Count a re-ask as failed when this turn did not extract *this* slot.
            filled_this = next_slot.name in extracted
            if attempts > max_re and not filled_this:
                record_action(self._action(
                    action_type="handoff_offer_emitted",
                    input_summary=f"low confidence on slot '{next_slot.name}' after {attempts} attempts",
                    output_summary="escalate_to_human",
                ))
                try:
                    from src.frontline.validation_queue import enqueue_insight

                    enqueue_insight(
                        kind="low_confidence",
                        interaction_id=ctx.interaction_id,
                        summary=f"low confidence on slot '{next_slot.name}' after {attempts} attempts",
                    )
                except Exception:
                    pass
                return {
                    "extracted": extracted,
                    "question": (
                        "I'm having trouble capturing that clearly. "
                        "Let me connect you with a specialist who can help."
                    ),
                    "fast_path": True,
                    "kill_switch": None,
                    "slots_filled_now": bool(extracted),
                    "escalate_low_confidence": True,
                    "low_confidence_slot": next_slot.name,
                }
            # Task 4: Retired LLM phrasing site on live turn path. Use deterministic pack prompt directly.
            q_text = next_slot.prompt
            record_action(self._action(
                action_type="question_asked",
                input_summary=f"missing required slot '{next_slot.name}' (attempt {attempts})",
                output_summary=q_text[:500],
            ))
            return {
                "extracted": extracted,
                "question": q_text,
                "fast_path": True,                # LLM optional; fallback is pack prompt
                "kill_switch": None,
                "slots_filled_now": bool(extracted),
            }

        # All required slots filled.
        record_action(self._action(
            action_type="intake_completed",
            input_summary="all required slots filled",
            output_summary=f"slots: {ctx.slots}",
        ))
        return {
            "extracted": extracted,
            "question": "",
            "fast_path": True,
            "kill_switch": None,
            "slots_filled_now": bool(extracted),
        }

    def _prefill_from_memory(self) -> None:
        """Skip slots already known from contact_memory (feature #22)."""
        ctx = self.ctx
        if getattr(ctx, "_memory_prefill_done", False):
            return
        ctx._memory_prefill_done = True  # type: ignore[attr-defined]
        e1 = ctx.slots.get("entity_1")
        e2 = ctx.slots.get("entity_2")
        e3 = ctx.slots.get("entity_3")
        # Also try loading memory when only entity_2 is known
        try:
            from src.enterprise.memory import lookup_memory

            mem = lookup_memory(
                ctx.pack.id,
                entity_1=e1,
                entity_2=e2,
                entity_3=e3,
            )
            if not mem:
                return
            filled = []
            for slot_name, mem_key in (
                ("entity_1", "entity_1"),
                ("entity_2", "entity_2"),
                ("entity_3", "entity_3"),
                ("category", "last_category"),
            ):
                if ctx.slots.get(slot_name):
                    continue
                val = mem.get(mem_key)
                if val:
                    ctx.slots[slot_name] = str(val)
                    filled.append(slot_name)
            if filled:
                record_action(self._action(
                    action_type="slot_extracted",
                    input_summary="prefill_from_entity_memory",
                    output_summary=f"filled: {filled}",
                ))
        except Exception:
            return

    def _year_range_for_entity_1(self):
        try:
            for s in self.ctx.pack.required_slots():
                if s.name == "entity_1" and getattr(s, "validation", "") == "year-range":
                    return getattr(s, "year_range", None)
        except Exception:
            pass
        return (1990, 2026)

    def _merge_secondary_categories(self, text: str, primary: str | None) -> None:
        """Preserve every affected system without replacing the primary."""
        gaz = self.ctx.pack.gazetteer_for_slot("category")
        allowed = gaz.lookup if gaz is not None else None
        observed = [canonical for _s, _e, _syn, canonical in
                    _category_synonym_hits(text, allowed=allowed)]
        existing = self.ctx.slots.get("secondary_categories") or []
        if not isinstance(existing, list):
            existing = [str(existing)]
        merged: list[str] = []
        for candidate in [*existing, *observed]:
            if candidate and candidate != primary and candidate not in merged:
                merged.append(candidate)
        if merged:
            self.ctx.slots["secondary_categories"] = merged

    # ── Slot extraction ──────────────────────────────────────────────
    def _extract_slot(self, slot, text: str) -> str | None:
        if slot.validation == "year-range":
            return _extract_year(text, slot.year_range)
        if slot.validation == "gazetteer":
            # 2. For category slots, try the synonym map and capture secondary issues.
            if slot.name == "category":
                gaz = self.ctx.pack.gazetteer_for_slot(slot.name)
                allowed = gaz.lookup if gaz is not None else None
                matched_categories: list[str] = []
                try:
                    from src.ml_runtime.category_zeroshot import extract_category_with_mode

                    syn = match_category_synonym(text, allowed=allowed)
                    if syn:
                        rules_cat = syn
                    else:
                        gaz_val = _extract_via_gazetteer(text, self.ctx, "category")
                        if gaz_val:
                            lower = text.lower()
                            key = gaz_val.lower()
                            m = re.search(rf"\b{re.escape(key)}\b", lower)
                            if m and _category_hit_suppressed(text, m.start(), m.end(), key):
                                rules_cat = None
                            else:
                                rules_cat = gaz_val
                        else:
                            rules_cat = None

                    primary, source, meta = extract_category_with_mode(text, self.ctx, rules_cat)
                    self.ctx.slots["__category_source__"] = source
                except Exception:
                    primary = extract_pack_category(text, self.ctx)
                    self.ctx.slots["__category_source__"] = "rules" if primary else "none"
                if primary:
                    matched_categories.append(primary)
                for _s, _e, _syn, canonical in _category_synonym_hits(text, allowed=allowed):
                    if canonical not in matched_categories:
                        matched_categories.append(canonical)
                if matched_categories:
                    self._merge_secondary_categories(text, matched_categories[0])
                    return matched_categories[0]
                return None
            return _extract_via_gazetteer(text, self.ctx, slot.name)
        if slot.validation == "regex" and slot.regex:
            m = re.search(slot.regex, text)
            if m:
                return m.group(1) if m.groups() else m.group(0)
            return None
        if slot.validation == "free-text":
            return _extract_description(text)
        return None

    # ── Dialogue policy ──────────────────────────────────────────────────
    def _evaluate_pending_safety_answer(self, text: str) -> dict[str, Any] | None:
        """Interpret `text` as the reply to the pending safety question.

        Returns an intake result dict when the answer escalates; otherwise
        records the answer and returns None so the rest of the turn proceeds.
        Unclear replies leave the question pending (it will be re-asked).
        """
        ctx = self.ctx
        raw_pending = ctx.slots.get(_SAFETY_PENDING_KEY)
        if raw_pending is None or str(raw_pending).strip() == "":
            return None
        try:
            idx = int(str(raw_pending).strip())
        except ValueError:
            ctx.slots.pop(_SAFETY_PENDING_KEY, None)
            return None
        questions = ctx.pack.manifest.safety.safety_questions or []
        if idx < 0 or idx >= len(questions):
            ctx.slots.pop(_SAFETY_PENDING_KEY, None)
            return None
        prompt = questions[idx]
        # Defect #2 fix: polarity is conditioned on the question asked —
        # legacy classify_yes_no never saw it ("im fine" → false P1).
        polarity = classify_answer(prompt, text)
        escalate_on = escalate_on_for_prompt(prompt)
        if polarity is None:
            # Keep pending; _next_safety_question will re-ask this index.
            return None
        asked = _parse_asked_indices(ctx.slots.get(_SAFETY_ASKED_KEY))
        asked.add(idx)
        _store_asked_indices(ctx, asked)
        ctx.slots.pop(_SAFETY_PENDING_KEY, None)
        if escalate_on and polarity == escalate_on:
            flag = f"safety_q_{idx}"
            ctx.safety_flags["escalation"] = True
            ctx.safety_flags[flag] = True
            record_action(self._action(
                action_type="safety_flag_raised",
                input_summary=f"safety question {idx} ({prompt!r}) answered {polarity!r}",
                output_summary=f"escalate_on={escalate_on}; intake will skip remaining slots",
                evidence_ids=[],
            ))
            return {
                "extracted": {},
                "question": ctx.pack.manifest.safety.escalation_script,
                "fast_path": True,
                "kill_switch": flag,
                "slots_filled_now": False,
            }
        return None

    def _next_safety_question(self) -> str | None:
        """Return the next unanswered safety question, if any.

        Asked-question state lives in ``ctx.slots`` so a rebuilt orchestrator
        (process restart / resume) does not replay the full safety script.
        A question is marked asked only after its answer is bound.
        """
        if not _asks_spoken_safety(self.ctx):
            return None
        questions = self.ctx.pack.manifest.safety.safety_questions or []
        if not questions:
            return None
        raw_pending = self.ctx.slots.get(_SAFETY_PENDING_KEY)
        if raw_pending is not None and str(raw_pending).strip() != "":
            try:
                i = int(str(raw_pending).strip())
            except ValueError:
                i = -1
            if 0 <= i < len(questions):
                return questions[i]
        asked = _parse_asked_indices(self.ctx.slots.get(_SAFETY_ASKED_KEY))
        # Honour leftover in-memory attr from older contacts (best-effort).
        legacy = getattr(self.ctx, "_safety_asked", None)
        if isinstance(legacy, set):
            asked |= {int(x) for x in legacy if str(x).isdigit() or isinstance(x, int)}
        for i, q in enumerate(questions):
            if i not in asked:
                self.ctx.slots[_SAFETY_PENDING_KEY] = str(i)
                return q
        return None

    def _max_turns_response(self) -> dict[str, Any]:
        """Deterministic wrap-up payload when FRONTLINE_MAX_TURNS is exhausted."""
        record_action(self._action(
            action_type="intake_completed",
            input_summary=f"max_turns={settings.max_turns} reached with incomplete slots",
            output_summary=f"slots: {self.ctx.slots}; force wrap-up",
        ))
        return {
            "extracted": {},
            "question": "",
            "fast_path": True,
            "kill_switch": None,
            "slots_filled_now": False,
            "max_turns_reached": True,
        }

    def _next_required_slot(self):
        """Pick the highest-priority missing required slot that hasn't
        exceeded its re-ask limit."""
        # Count customer turns (NOT slot attempts) for the global cap.
        customer_turn_count = self.ctx.count_turn()
        for slot in self.ctx.pack.required_slots():
            if self.ctx.slots.get(slot.name):
                continue
            attempts = self.ctx.slot_attempts.get(slot.name, 0)
            if attempts >= slot.max_re_asks:
                continue
            # Hard global cap on customer turns — stop asking; orchestrator will wrap up.
            if customer_turn_count >= settings.max_turns:
                return None
            return slot
        return None

    def reextract_overwrite(self, text: str) -> dict[str, str]:
        """Correction policy (board #4): re-extract ALLOWING overwrites.

        Normal extraction skips filled slots; a correction ("actually it's
        a 2018") must replace them. Only slots the text actually resolves
        are touched; free-text slots (description) are NEVER overwritten by
        a one-line correction ("yes, that's right" is not a description).
        Every change is ledgered as a correction.
        """
        changed: dict[str, str] = {}
        for slot in self.ctx.pack.required_slots():
            if getattr(slot, "validation", "") == "free-text":
                continue
            try:
                value = self._extract_slot(slot, text or "")
            except Exception:
                continue
            if value and value != (self.ctx.slots.get(slot.name) or ""):
                self.ctx.slots[slot.name] = value
                changed[slot.name] = value
        if changed:
            record_action(self._action(
                action_type="slot_extracted",
                input_summary=f"correction turn: '{(text or '')[:200]}'",
                output_summary=f"corrected: {changed}",
                evidence_ids=[],
            ))
        return changed
