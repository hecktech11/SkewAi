"""Deterministic Hindi (Devanagari) / Hinglish safety + slot layer.  [AI-LAYER]

Runs alongside the existing deterministic ``IntakeAgent`` (no API key needed,
no network, no model). It exists because the existing extractors are English
only: a Devanagari turn fills almost no slots and never reaches the pack's
kill-switch lexicon.

What it does, for a turn that ``language.detect`` labels ``hi`` or ``hinglish``:

1. Matches a small vocabulary of Hindi / romanised-Hindi vehicle terms and
   hazard phrases (written from general vocabulary, NOT from any eval file's
   expected labels) and turns them into an English *gloss* such as
   ``"brakes failed"`` or ``"smoke"``.
2. Safety: runs the pack's EXISTING kill-switch check (``_check_kill_switch``,
   unchanged) on that gloss. A hit can only ADD an escalation, exactly like
   the Meta intake enhancer does on its translation.
3. Slots: fills ONLY slots that are still empty (never overwrites): category
   (pack gazetteer validated), make / model from Devanagari transliterations
   (gazetteer validated), year from Devanagari digits, and, for Devanagari
   turns, the description = the caller's own words.
4. Severity: ``maybe_raise_severity_hi`` runs after the existing triage and
   can only RAISE (final = max(rule, floor)); a hazard signal gives a floor
   of Critical, a safety-critical component a floor of Medium.

Negation ("कोई घायल नहीं हुआ", "koi chot nahi", "bina accident ke") and
hedges ("शायद") suppress a hazard term in that clause; a suppressed term
never raises anything, and nothing here ever lowers severity or clears a
flag. ``AI_HINDI_SAFETY=0`` turns the whole layer off.
"""

from __future__ import annotations

import os
import re
from collections import OrderedDict
from typing import Any

ORDER = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}

_DEV = "\u0900-\u097F"
_L = rf"(?<![\w{_DEV}])"  # left word boundary that also treats Devanagari as word chars
_R = rf"(?![\w{_DEV}])"
_GAP = rf"(?:[\w{_DEV}-]+\s+){{0,3}}?"  # up to 3 words in between


def enabled() -> bool:
    return os.environ.get("AI_HINDI_SAFETY", "1").strip().lower() not in ("0", "false", "no", "off")


def normalise(text: str) -> str:
    """Lower-case; fold nukta and chandrabindu so spelling variants match."""
    t = (text or "").lower()
    t = t.replace("\u093c", "")          # nukta: ज़ -> ज, ड़ -> ड, फ़ -> फ
    t = t.replace("\u0901", "\u0902")    # chandrabindu -> anusvara: धुआँ -> धुआं
    t = t.translate(str.maketrans("०१२३४५६७८९", "0123456789"))
    return re.sub(r"\s+", " ", t)


# ── Hazard phrases whose meaning already contains a negation ──────────────
# ("brakes did not apply", "car would not stop"). Matched first; never negatable.
_NEG = r"(?:नहीं|नही|nahi|nahin|nhi|nai)"
INTRINSIC: list[tuple[str, str]] = [
    (rf"{_L}ब्रेक(?:्स)?\s+{_NEG}\s+लग", "brakes failed"),
    (rf"{_L}ब्रेक(?:्स)?\s+(?:ने\s+)?काम\s+{_NEG}", "brakes failed"),
    (rf"{_L}(?:रुकी|रुका|रुके)\s+(?:ही\s+)?{_NEG}{_R}", "could not stop"),
    (rf"{_L}रोक\s+{_NEG}\s+पा", "could not stop"),
    (rf"{_L}स्टी?य?े?रिंग\s+{_GAP}{_NEG}\s+(?:घूम|मुड|चल)", "loss of steering"),
    (rf"{_L}(?:कंट्रोल|नियंत्रण)\s+(?:में\s+)?{_NEG}{_R}", "loss of control"),
    (rf"{_L}breaks?\s+{_NEG}\s+lag", "brakes failed"),
    (rf"{_L}brakes?\s+{_NEG}\s+lag", "brakes failed"),
    (rf"{_L}brakes?\s+(?:ne\s+)?kaam\s+{_NEG}", "brakes failed"),
    (rf"{_L}(?:ruki|ruka|ruke)\s+(?:hi\s+)?{_NEG}{_R}", "could not stop"),
    (rf"{_L}rok\s+{_NEG}\s+pa", "could not stop"),
    (rf"{_L}steering\s+{_GAP}{_NEG}\s+(?:ghoom|ghum|mud|chal)", "loss of steering"),
    (rf"{_L}control\s+(?:mein\s+|me\s+)?{_NEG}{_R}", "loss of control"),
]

# ── Hazard terms (negatable) -> English gloss that the pack lexicon knows ──
HAZARDS: list[tuple[str, str]] = [
    # Devanagari
    (rf"{_L}ब्रेक(?:्स)?\s+{_GAP}फेल", "brakes failed"),
    (rf"{_L}आग{_R}", "fire"),
    (rf"{_L}लपट", "flames"),
    (rf"{_L}(?:धुआं|धुंआ|धुंवा|धुआ){_R}", "smoke"),
    (rf"{_L}(?:टक्कर|टकरा)", "collision"),
    (rf"{_L}(?:दुर्घटना|हादसा|हादसे|एक्सीडेंट|ऐक्सीडेंट|एक्सिडेंट)", "crash"),
    (rf"{_L}पलट\s+(?:गई|गयी|गया|गए)", "rollover"),
    (rf"{_L}चोट", "injury"),
    (rf"{_L}(?:घायल|जख्मी|जख़्मी)", "injured"),
    (rf"{_L}खून", "bleeding"),
    (rf"{_L}अस्पताल", "hospital"),
    (rf"{_L}(?:एम्बुलेंस|एंबुलेंस)", "ambulance"),
    (rf"{_L}जल\s+(?:गया|गए|गई|गयी|गये){_R}", "burned"),
    (rf"{_L}स्टी?य?े?रिंग\s+{_GAP}(?:जाम|लॉक|अटक|फंस)", "steering locked"),
    (rf"{_L}स्टी?य?े?रिंग\s+{_GAP}फेल", "loss of steering"),
    (rf"{_L}(?:एयरबैग|एयर\s+बैग)\s+{_GAP}खुल\s+(?:गया|गए|गई|गयी|गये)", "airbag deployed"),
    (rf"{_L}(?:एक्सीलेटर|एक्सेलरेटर|एक्सिलरेटर)\s+{_GAP}(?:अटक|फंस|जाम)", "accelerator stuck"),
    (rf"{_L}अपने\s+आप\s+{_GAP}(?:तेज|भाग|स्पीड|आगे)", "unintended acceleration"),
    (rf"{_L}(?:कंट्रोल|नियंत्रण)\s+खो", "loss of control"),
    (rf"{_L}बेकाबू", "loss of control"),
    (rf"{_L}सीट\s*बेल्ट\s+{_GAP}टूट", "seatbelt failed"),
    # Romanised Hindi (English words such as "fire"/"smoke" are already handled by the existing lexicon)
    (rf"{_L}(?:brakes?|breaks?)\s+{_GAP}fail", "brakes failed"),
    (rf"{_L}aag{_R}", "fire"),
    (rf"{_L}(?:lapat|lapte|lapten){_R}", "flames"),
    (rf"{_L}(?:dhuan|dhuaan|dhuaa|dhuwan|dhuwa|dhua|dhuan){_R}", "smoke"),
    (rf"{_L}(?:takkar|takar|takra)", "collision"),
    (rf"{_L}(?:durghatna|hadsa|hadse|accident){_R}", "crash"),
    (rf"{_L}palat\s+(?:gayi|gaya|gai|gaye){_R}", "rollover"),
    (rf"{_L}(?:chot|chott){_R}", "injury"),
    (rf"{_L}(?:ghayal|zakhmi|jakhmi){_R}", "injured"),
    (rf"{_L}khoon{_R}", "bleeding"),
    (rf"{_L}(?:aspatal|aspataal|haspatal){_R}", "hospital"),
    (rf"{_L}jal\s+(?:gaya|gaye|gayi|gai){_R}", "burned"),
    (rf"{_L}steering\s+{_GAP}(?:jam|lock|atak|fans|phans)", "steering locked"),
    (rf"{_L}steering\s+{_GAP}fail", "loss of steering"),
    (rf"{_L}air\s*bag\s+{_GAP}khul", "airbag deployed"),
    (rf"{_L}accelerator\s+{_GAP}(?:atak|fans|phans|jam)", "accelerator stuck"),
    (rf"{_L}apne\s+aap\s+{_GAP}(?:speed|tez|tezi|bhaag|bhag|aage)", "unintended acceleration"),
    (rf"{_L}control\s+(?:se\s+bahar|kho)", "loss of control"),
    (rf"{_L}(?:bekabu|bekaabu){_R}", "loss of control"),
]

# ── Components -> NHTSA pack category (validated against the gazetteer) ───
COMPONENTS: list[tuple[str, str]] = [
    (rf"{_L}(?:हैंडब्रेक|हैंड\s+ब्रेक|handbrake|hand\s+brake)", "PARKING BRAKE"),
    (rf"{_L}(?:ब्रेक|brake|brakes|break|breaks){_R}|{_L}ब्रेक", "SERVICE BRAKES"),
    (rf"{_L}(?:स्टीयरिंग|स्टेयरिंग|स्टियरिंग|steering)", "STEERING"),
    (rf"{_L}(?:एयरबैग|एयर\s+बैग|airbag|air\s+bag)", "AIR BAGS"),
    (rf"{_L}(?:सीट\s*बेल्ट|seat\s*belt)", "SEAT BELTS"),
    (rf"{_L}(?:एक्सीलेटर|एक्सेलरेटर|एक्सिलरेटर|accelerator|क्रूज|cruise\s+control)", "VEHICLE SPEED CONTROL"),
    (rf"{_L}(?:इंजन|इंजिन|engine|injan){_R}", "ENGINE"),
    (rf"{_L}(?:गियरबॉक्स|गियर|क्लच|ट्रांसमिशन|gearbox|gear|clutch|transmission){_R}", "POWER TRAIN"),
    (rf"{_L}(?:पेट्रोल|डीजल|ईंधन|फ्यूल|टंकी|petrol|diesel|fuel|tanki){_R}", "FUEL SYSTEM"),
    (rf"{_L}(?:बैटरी|वायरिंग|फ्यूज|battery|wiring|fuse){_R}", "ELECTRICAL SYSTEM"),
    (rf"{_L}(?:हेडलाइट|headlight|headlights){_R}", "EXTERIOR LIGHTING"),
    (rf"{_L}(?:टायर|पंचर|tyre|tyres|tire|tires|puncture){_R}", "TIRES"),
    (rf"{_L}(?:पहिया|पहिये|पहिए|pahiya|pahiye){_R}", "WHEELS"),
    (rf"{_L}(?:सस्पेंशन|शॉकर|suspension|shocker|shockers){_R}", "SUSPENSION"),
    (rf"{_L}(?:वाइपर|wiper|wipers){_R}", "VISIBILITY"),
]
SAFETY_COMPONENTS = frozenset({"SERVICE BRAKES", "PARKING BRAKE", "STEERING", "AIR BAGS", "SEAT BELTS",
                               "FUEL SYSTEM", "VEHICLE SPEED CONTROL", "TIRES", "WHEELS"})

# Devanagari transliterations of common makes / models (validated against the pack gazetteer).
MAKES = {
    "होंडा": "HONDA", "टोयोटा": "TOYOTA", "फोर्ड": "FORD", "हुंडई": "HYUNDAI", "ह्युंडई": "HYUNDAI",
    "हुंडाई": "HYUNDAI", "ह्यूंदै": "HYUNDAI", "किआ": "KIA", "कीया": "KIA", "निसान": "NISSAN",
    "शेवरले": "CHEVROLET", "शेवरलेट": "CHEVROLET", "जीप": "JEEP", "बीएमडब्ल्यू": "BMW", "ऑडी": "AUDI",
    "मर्सिडीज": "MERCEDES-BENZ", "माज्दा": "MAZDA", "मज्दा": "MAZDA", "सुबारू": "SUBARU",
    "टेस्ला": "TESLA", "फॉक्सवैगन": "VOLKSWAGEN", "वोक्सवैगन": "VOLKSWAGEN", "डॉज": "DODGE",
    "लेक्सस": "LEXUS", "मित्सुबिशी": "MITSUBISHI", "वोल्वो": "VOLVO", "क्रिसलर": "CHRYSLER",
    "जीएमसी": "GMC", "एक्यूरा": "ACURA", "इनफिनिटी": "INFINITI", "पोर्श": "PORSCHE", "रैम": "RAM",
    "लिंकन": "LINCOLN", "कैडिलैक": "CADILLAC", "ब्यूक": "BUICK",
}
MODELS = {
    "कैमरी": "CAMRY", "कोरोला": "COROLLA", "सिविक": "CIVIC", "अकॉर्ड": "ACCORD", "एकॉर्ड": "ACCORD",
    "एलांट्रा": "ELANTRA", "इलांट्रा": "ELANTRA", "सोनाटा": "SONATA", "टक्सन": "TUCSON", "ट्यूसॉन": "TUCSON",
    "अल्टिमा": "ALTIMA", "एस्केप": "ESCAPE", "एक्सप्लोरर": "EXPLORER", "सोरेंटो": "SORENTO",
    "ऑप्टिमा": "OPTIMA", "रैंगलर": "WRANGLER", "चेरोकी": "CHEROKEE", "सिल्वराडो": "SILVERADO",
    "मालिबू": "MALIBU", "इक्विनॉक्स": "EQUINOX", "फ्यूजन": "FUSION", "पायलट": "PILOT", "ओडिसी": "ODYSSEY",
    "हाइलैंडर": "HIGHLANDER", "प्रियस": "PRIUS", "टैकोमा": "TACOMA", "सेंट्रा": "SENTRA", "रोग": "ROGUE",
    "आउटबैक": "OUTBACK", "फॉरेस्टर": "FORESTER", "जेट्टा": "JETTA", "सीआर-वी": "CR-V", "सीआरवी": "CR-V",
    "आरएवी4": "RAV4",
}

_CLAUSE_SPLIT = re.compile(r"[,.;!?।|]+|\s(?:लेकिन|मगर|परंतु|पर|और|फिर|but|lekin|magar|aur|and|phir|par)\s")
_NEG_AFTER = re.compile(rf"{_L}(?:नहीं|नही|ना|न|मत|nahi|nahin|nhi|nai|na|mat|not|no){_R}")
_NEG_BEFORE = re.compile(rf"(?:{_L}(?:बिना|bina|without|koi|कोई|no){_R})\s*(?:[\w{_DEV}]+\s+){{0,1}}$")
_HEDGE = re.compile(rf"{_L}(?:शायद|shayad|sayad|maybe|perhaps){_R}")


def _clauses(text: str) -> list[str]:
    return [c.strip() for c in _CLAUSE_SPLIT.split(f" {text} ") if c and c.strip()]


def _negated(clause: str, start: int, end: int) -> bool:
    if _HEDGE.search(clause):
        return True
    after = clause[end:].split()
    if any(_NEG_AFTER.fullmatch(w) for w in after[:4]):
        return True
    return bool(_NEG_BEFORE.search(clause[:start]))


def analyse(text: str) -> dict[str, Any]:
    """Pure function: hazards (gloss), suppressed hazards, components, make/model/year."""
    t = normalise(text)
    hazards: list[dict[str, str]] = []
    suppressed: list[dict[str, str]] = []
    for clause in _clauses(t):
        rest = clause
        for pat, gloss in INTRINSIC:
            m = re.search(pat, rest)
            if m:
                hazards.append({"gloss": gloss, "match": m.group(0)})
                rest = rest[:m.start()] + " " * (m.end() - m.start()) + rest[m.end():]
        for pat, gloss in HAZARDS:
            for m in re.finditer(pat, rest):
                item = {"gloss": gloss, "match": m.group(0).strip()}
                (suppressed if _negated(rest, m.start(), m.end()) else hazards).append(item)
    comps: list[tuple[int, str]] = []
    for pat, cat in COMPONENTS:
        m = re.search(pat, t)
        if m:
            comps.append((m.start(), cat))
    comps.sort()
    seen: list[str] = []
    for _, c in comps:
        if c not in seen:
            seen.append(c)
    words = re.findall(rf"[\w{_DEV}\-]+", t)
    make = next((MAKES[w] for w in words if w in MAKES), None)
    model = next((MODELS[w] for w in words if w in MODELS), None)
    ym = re.search(r"(?<!\d)(19[9]\d|20[0-2]\d)(?!\d)", t)
    gloss = sorted({h["gloss"] for h in hazards})
    return {
        "hazards": hazards, "suppressed": suppressed, "gloss": gloss,
        "components": seen, "category": seen[0] if seen else None,
        "make": make, "model": model, "year": ym.group(1) if ym else None,
        "floor": "Critical" if gloss else ("Medium" if any(c in SAFETY_COMPONENTS for c in seen) else None),
    }


# Per-interaction signals for the post-triage raise (bounded).
_STATE: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_STATE_MAX = 512


def _remember(iid: str, info: dict[str, Any]) -> None:
    st = _STATE.setdefault(iid, {"floor": None, "gloss": []})
    f = info.get("floor")
    if f and (st["floor"] is None or ORDER[f] > ORDER[st["floor"]]):
        st["floor"] = f
    st["gloss"] = sorted(set(st["gloss"]) | set(info.get("gloss") or []))
    _STATE.move_to_end(iid)
    while len(_STATE) > _STATE_MAX:
        _STATE.popitem(last=False)


def signals(iid: str) -> dict[str, Any]:
    return dict(_STATE.get(iid) or {"floor": None, "gloss": []})


_SPECIAL_KEYS = ("kill_switch", "readback", "dtmf", "case_status", "max_turns_reached",
                 "escalate_low_confidence", "handoff_requested", "safety_downgraded")


def _description_ok(text: str) -> bool:
    toks = re.findall(rf"[\w{_DEV}]+", text or "")
    return len((text or "").strip()) >= 8 and len(toks) >= 3


async def maybe_apply_hindi_intake(ctx, text: str, ires: dict[str, Any]) -> dict[str, Any]:
    """Return ``ires`` (possibly with extra fills / an ADDED escalation). Never raises."""
    if not enabled() or not text or not isinstance(ires, dict):
        return ires
    try:
        from src.ai_layer.language import detect

        lang = detect(text)["lang"]
        if lang not in ("hi", "hinglish"):
            return ires
        if any(ires.get(k) for k in _SPECIAL_KEYS) or (getattr(ctx, "safety_flags", None) or {}).get("escalation"):
            return ires
        return _apply(ctx, text, ires, lang)
    except Exception as e:  # never break the live call
        try:
            from src.ai_layer.store import record_event

            record_event(kind="hindi_intake_error", interaction_id=getattr(ctx, "interaction_id", None),
                         mode="deterministic", reason=type(e).__name__)
        except Exception:
            pass
        return ires


def _apply(ctx, text: str, ires: dict[str, Any], lang: str) -> dict[str, Any]:
    from src.agents.intake import IntakeAgent, _check_kill_switch
    from src.ai_layer.store import record_event
    from src.ledger import record_action
    from src.ledger.writer import AgentAction

    iid = ctx.interaction_id
    info = analyse(text)
    _remember(iid, info)
    gloss_text = ". ".join(info["gloss"])
    term = _check_kill_switch(gloss_text, ctx) if gloss_text else None
    if term:
        ctx.safety_flags["escalation"] = True
        ctx.safety_flags[term] = True
        record_action(AgentAction(
            interaction_id=iid, agent="ai_layer", action_type="safety_flag_raised",
            input_summary=f"{lang} turn: Hindi safety vocabulary gloss '{gloss_text[:120]}'",
            output_summary=f"safety_flag[{term}]=True via existing pack lexicon on the deterministic gloss",
        ))
        record_event(kind="hindi_safety_raise", interaction_id=iid, mode="deterministic",
                     deterministic={"lang": lang, "hazards": info["hazards"], "suppressed": info["suppressed"]},
                     accepted={"kill_switch": term}, reason="lexicon_hit_on_hindi_gloss")
        out = dict(ires)
        out.update({
            "question": ctx.pack.manifest.safety.escalation_script,
            "kill_switch": term, "fast_path": True,
            "slots_filled_now": bool(ires.get("extracted")),
            "ai_layer": {"safety_raised": term, "language": lang, "source": "hindi_safety"},
        })
        return out

    filled: dict[str, str] = {}
    intake = IntakeAgent(ctx)

    def _fill(name: str, value: str | None) -> None:
        if not value or ctx.slots.get(name):
            return
        gaz = ctx.pack.gazetteer_for_slot(name) if name != "description" and name != "entity_1" else None
        if gaz is not None:
            value = gaz.lookup(value)
            if not value:
                return
        intake._stamp_slot(name, value, source_turn=text, confidence=0.7)
        meta = dict(ctx.slots.get("__slot_meta__") or {})
        if isinstance(meta.get(name), dict):
            meta[name] = {**meta[name], "source": "hindi_safety"}
            ctx.slots["__slot_meta__"] = meta
        filled[name] = value

    frame = {s.name: s for s in ctx.pack.manifest.slot_frame}
    if "category" in frame:
        _fill("category", info["category"])
    if "entity_2" in frame:
        _fill("entity_2", info["make"])
    if "entity_3" in frame:
        _fill("entity_3", info["model"])
    if "entity_1" in frame and info["year"]:
        yr = getattr(frame["entity_1"], "year_range", None)
        if not yr or int(yr[0]) <= int(info["year"]) <= int(yr[1]):
            _fill("entity_1", info["year"])
    if "description" in frame and lang == "hi" and _description_ok(text):
        _fill("description", text.strip()[:500])
    if filled:
        record_action(AgentAction(
            interaction_id=iid, agent="ai_layer", action_type="slot_extracted",
            input_summary=f"hindi_safety deterministic pass on customer turn ({lang})",
            output_summary=f"filled empty slots: {filled}; existing values kept",
        ))
    record_event(kind="hindi_intake", interaction_id=iid, mode="deterministic",
                 deterministic={"lang": lang, "hazards": info["hazards"], "suppressed": info["suppressed"],
                                "components": info["components"]},
                 accepted={"filled": filled}, reason="filled" if filled else "nothing_to_fill")
    if filled:
        out = dict(ires)
        out["ai_layer_hindi"] = {"filled": filled, "language": lang}
        return out
    return ires


async def maybe_raise_severity_hi(ctx, tres: dict[str, Any]) -> dict[str, Any]:
    """Post-triage, raise-only: final = max(rule, floor). Never lowers. Never raises."""
    if not enabled() or not isinstance(tres, dict) or tres.get("skipped"):
        return tres
    try:
        from src.ai_layer.severity import _allowed_levels, combine

        sig = signals(getattr(ctx, "interaction_id", ""))
        floor = sig.get("floor")
        rule = str(getattr(ctx, "severity", None) or tres.get("severity") or "Low")
        final = combine(rule, floor, allowed=_allowed_levels(ctx))
        if final == rule:
            return tres
        from src.agents.triage import _priority_from_matrix
        from src.ai_layer.store import record_event
        from src.ledger import record_action
        from src.ledger.writer import AgentAction

        ctx.severity = final
        ctx.priority = _priority_from_matrix(final, ctx)
        why = f"Hindi/Hinglish safety vocabulary: {', '.join(sig.get('gloss') or []) or 'safety-critical component'}"
        record_action(AgentAction(
            interaction_id=ctx.interaction_id, agent="ai_layer", action_type="severity_scored",
            input_summary=f"rule severity={rule}; hindi_safety floor={floor}",
            output_summary=f"severity raised {rule}->{final} (max(rule, floor)); reason={why[:160]}",
        ))
        record_event(kind="hindi_severity", interaction_id=ctx.interaction_id, mode="deterministic",
                     deterministic={"severity": rule}, accepted={"severity": final}, reason="raised_by_floor")
        new = dict(tres)
        new.update({"severity": final, "priority": ctx.priority,
                    "ai_layer_hindi": {"rule": rule, "floor": floor, "reason": why[:200]}})
        return new
    except Exception:
        return tres


__all__ = ["analyse", "enabled", "maybe_apply_hindi_intake", "maybe_raise_severity_hi", "normalise", "signals"]
