"""Enhanced intake: Muse Spark slot proposals + Hindi/Hinglish + sentiment.  [AI-LAYER]

Runs AFTER the deterministic ``IntakeAgent.run`` on the same customer turn,
and only when a Meta key is configured. Without a key ``maybe_enhance_intake``
returns the deterministic result object unchanged, with no I/O.

Rules enforced here (see HACKATHON.md):

* Deterministic values are never overwritten. Only slots that are still empty
  after deterministic extraction may be filled.
* For a non-English turn, the deterministic extractors are first re-run on the
  AI-normalised English text (rules decide, AI only translated). Only when
  that finds nothing is the model's own proposal considered.
* A model proposal is accepted only if it is schema-valid, at or above
  ``META_MIN_CONFIDENCE``, quotes an evidence span that literally occurs in the
  caller's words, and validates against the pack (gazetteer canonical value,
  year range, non-filler description).
* Safety: the pack's existing kill-switch lexicon is re-run on the normalised
  English text. A hit can only ADD an escalation. AI judgement alone never
  escalates or de-escalates.
* Sentiment: AI frustration can only raise the turn score (max(rule, AI)).
* Both deterministic and AI values are logged to ``ai_layer_events`` and the
  accepted fills are ledgered through the existing ``record_action``.
"""

from __future__ import annotations

import asyncio
import re
from collections import OrderedDict
from typing import Any

from src.ai_layer import config
from src.ai_layer.language import accept_ai_normalisation, deterministic_normalise, detect
from src.ai_layer.meta_client import MetaError, chat_json
from src.ai_layer.store import record_event

# Per-interaction AI context (normalised text, last hints), bounded.
_STATE: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_STATE_MAX = 512


def interaction_state(interaction_id: str) -> dict[str, Any]:
    st = _STATE.get(interaction_id)
    if st is None:
        st = {"turns": []}
        _STATE[interaction_id] = st
        while len(_STATE) > _STATE_MAX:
            _STATE.popitem(last=False)
    else:
        _STATE.move_to_end(interaction_id)
    return st


# Results the enhancer must never rewrite (deterministic gates already chose).
_SPECIAL_KEYS = (
    "kill_switch", "readback", "dtmf", "case_status", "max_turns_reached",
    "escalate_low_confidence", "handoff_requested", "safety_downgraded",
)

SLOT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "language": {"type": "string", "enum": ["en", "hi", "hinglish", "other"]},
        "normalized_en": {"type": "string", "maxLength": 1200},
        "normalization_confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "slots": {
            "type": "array",
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "slot": {"type": "string", "maxLength": 40},
                    "value": {"type": "string", "maxLength": 300},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "evidence": {"type": "string", "maxLength": 300},
                },
                "required": ["slot", "value", "confidence", "evidence"],
            },
        },
        "frustration": {"type": "number", "minimum": 0, "maximum": 1},
        "frustration_reason": {"type": "string", "maxLength": 200},
    },
    "required": ["language", "normalized_en", "normalization_confidence", "slots", "frustration", "frustration_reason"],
}

_SLOT_ALIASES = {
    # make/model/year/component/symptom vocabulary -> pack slot names
    "year": "entity_1", "model_year": "entity_1",
    "make": "entity_2", "brand": "entity_2",
    "model": "entity_3",
    "component": "category", "system": "category",
    "symptom": "description", "description": "description",
}


def _skip(ires: dict[str, Any]) -> bool:
    return any(ires.get(k) for k in _SPECIAL_KEYS)


def _slot_hint(ctx, slot) -> str:
    label = getattr(slot, "label", slot.name)
    hint = f"- {slot.name} ({label}; validation={slot.validation})"
    if slot.validation == "year-range" and getattr(slot, "year_range", None):
        hint += f" range {slot.year_range[0]}-{slot.year_range[1]}"
    gaz = ctx.pack.gazetteer_for_slot(slot.name)
    if gaz is not None and slot.name == "category":
        hint += " one of: " + "; ".join(gaz.values[:60])
    elif gaz is not None:
        hint += " examples: " + ", ".join(gaz.values[:25])
    return hint


def build_prompt(ctx, text: str, empty_slots: list) -> tuple[str, str]:
    system = (
        "You assist a vehicle/product complaint intake agent. Return JSON only. "
        "1) Detect the caller's language (en, hi = Devanagari Hindi, hinglish = romanised Hindi mixed with English). "
        "2) Translate the utterance faithfully to plain English in normalized_en; keep every number exactly; add nothing. "
        "3) For the listed EMPTY slots only, propose a value if the caller actually said it. "
        "For every proposal copy the exact caller words that support it into evidence (verbatim from the utterance). "
        "Use only values from the allowed lists where a list is given. Never guess; omit a slot you are unsure about. "
        "4) Rate caller frustration 0-1 with a one-line reason. Do not invent IDs, recalls, or facts."
    )
    hints = "\n".join(_slot_hint(ctx, s) for s in empty_slots) or "(none)"
    user = f"EMPTY SLOTS:\n{hints}\n\nCALLER UTTERANCE:\n{text[:1000]}"
    return system, user


def _validate_value(ctx, slot, value: str) -> str | None:
    """Pack validation for an AI-proposed value. Returns the canonical value."""
    from src.agents.intake import _is_filler_description

    v = (value or "").strip()
    if not v:
        return None
    if slot.validation == "year-range":
        if not re.fullmatch(r"(19|20)\d{2}", v):
            return None
        yr = getattr(slot, "year_range", None)
        if yr and not (int(yr[0]) <= int(v) <= int(yr[1])):
            return None
        return v
    if slot.validation == "gazetteer":
        gaz = ctx.pack.gazetteer_for_slot(slot.name)
        return gaz.lookup(v) if gaz is not None else None
    if slot.validation == "free-text":
        toks = [t for t in re.sub(r"[^a-z0-9\s]", " ", v.lower()).split() if t]
        if len(v) < 8 or len(toks) < 2 or _is_filler_description(v):
            return None
        return v[:500]
    return None


def _evidence_grounded(evidence: str, original: str, normalized: str) -> bool:
    ev = re.sub(r"\s+", " ", (evidence or "").strip().lower())
    if len(ev) < 2:
        return False
    for hay in (original, normalized):
        if ev in re.sub(r"\s+", " ", (hay or "").lower()):
            return True
    return False


def _deterministic_on_normalized(ctx, slot, normalized: str) -> str | None:
    """Pure (side-effect free) deterministic extractors on the English text."""
    from src.agents.intake import (
        _extract_description,
        _extract_via_gazetteer,
        _extract_year,
        match_category_synonym,
    )

    if slot.validation == "year-range":
        return _extract_year(normalized, getattr(slot, "year_range", None))
    if slot.validation == "gazetteer":
        if slot.name == "category":
            gaz = ctx.pack.gazetteer_for_slot("category")
            allowed = gaz.lookup if gaz is not None else None
            return match_category_synonym(normalized, allowed=allowed) or _extract_via_gazetteer(normalized, ctx, "category")
        return _extract_via_gazetteer(normalized, ctx, slot.name)
    if slot.validation == "free-text":
        return _extract_description(normalized)
    return None


async def _call_model(system: str, user: str) -> dict[str, Any]:
    timeout = config.intake_timeout_s()
    return await asyncio.wait_for(
        asyncio.to_thread(
            chat_json, system=system, user=user, schema=SLOT_SCHEMA,
            schema_name="intake_slots", timeout=timeout,
        ),
        timeout=timeout + 0.5,
    )


async def maybe_enhance_intake(ctx, text: str, ires: dict[str, Any]) -> dict[str, Any]:
    """Return ``ires`` (possibly with extra fills / a raised escalation)."""
    if not config.meta_enabled() or not text or not isinstance(ires, dict):
        return ires
    if _skip(ires) or (getattr(ctx, "safety_flags", None) or {}).get("escalation"):
        return ires
    try:
        return await _enhance(ctx, text, ires)
    except Exception as e:  # never break the live call
        record_event(kind="intake_error", interaction_id=getattr(ctx, "interaction_id", None),
                     reason=f"{type(e).__name__}")
        return ires


async def _enhance(ctx, text: str, ires: dict[str, Any]) -> dict[str, Any]:
    from src.agents.intake import IntakeAgent, _check_kill_switch
    from src.ledger import record_action
    from src.ledger.writer import AgentAction

    iid = ctx.interaction_id
    det_lang = detect(text)
    det_norm = deterministic_normalise(text, det_lang["lang"])
    frame = list(ctx.pack.manifest.slot_frame)
    empty = [s for s in frame if not ctx.slots.get(s.name)]
    deterministic_snapshot = {s.name: ctx.slots.get(s.name) for s in frame}

    system, user = build_prompt(ctx, text, empty)
    try:
        out = await _call_model(system, user)
    except (MetaError, asyncio.TimeoutError, TimeoutError) as e:
        reason = getattr(e, "reason", type(e).__name__)
        record_event(kind="intake", interaction_id=iid, deterministic={"slots": deterministic_snapshot, "language": det_norm},
                     ai_proposal=None, accepted=None, reason=f"fallback:{reason}")
        st = interaction_state(iid)
        st["turns"].append({"original": text, "normalized_en": det_norm["normalized_en"], "lang": det_lang["lang"], "method": det_norm["method"]})
        return ires

    min_conf = config.min_confidence()
    # ── Language: store original + normalised ──────────────────────────────
    ai_norm = str(out.get("normalized_en") or "")
    norm_ok = det_lang["lang"] in ("hi", "hinglish") and accept_ai_normalisation(
        text, ai_norm, float(out.get("normalization_confidence") or 0), min_conf
    )
    if norm_ok:
        language = {"original": text, "normalized_en": ai_norm.strip(), "lang": det_lang["lang"], "method": "meta_muse_spark"}
    else:
        language = dict(det_norm)
        language["lang"] = det_lang["lang"]
    st = interaction_state(iid)
    st["turns"].append(language)
    st["turns"] = st["turns"][-20:]
    normalized = language["normalized_en"]

    # ── Safety: deterministic lexicon on the normalised text (raise only) ──
    if norm_ok and normalized != text:
        term = _check_kill_switch(normalized, ctx)
        if term:
            ctx.safety_flags["escalation"] = True
            ctx.safety_flags[term] = True
            record_action(AgentAction(
                interaction_id=iid, agent="ai_layer", action_type="safety_flag_raised",
                input_summary=f"normalised {language['lang']} turn matched lexicon: '{term}'",
                output_summary=f"safety_flag[{term}]=True via deterministic lexicon on Muse Spark translation",
            ))
            record_event(kind="safety_raise", interaction_id=iid,
                         deterministic={"kill_switch": None, "language": det_norm},
                         ai_proposal={"normalized_en": normalized}, accepted={"kill_switch": term},
                         reason="lexicon_hit_on_normalized_text", model_id=config.model_text())
            out_res = dict(ires)
            out_res.update({
                "question": ctx.pack.manifest.safety.escalation_script,
                "kill_switch": term,
                "fast_path": True,
                "slots_filled_now": bool(ires.get("extracted")),
                "ai_layer": {"safety_raised": term, "language": language["lang"]},
            })
            return out_res

    # ── Slots: fill only what deterministic left empty ─────────────────────
    proposals: dict[str, dict[str, Any]] = {}
    for p in out.get("slots") or []:
        name = _SLOT_ALIASES.get(str(p.get("slot") or "").strip().lower(), str(p.get("slot") or "").strip())
        if name and name not in proposals:
            proposals[name] = p
    filled: dict[str, dict[str, Any]] = {}
    rejected: dict[str, str] = {}
    intake = IntakeAgent(ctx)
    for slot in empty:
        if ctx.slots.get(slot.name):
            continue
        value = None
        source = ""
        conf = 0.0
        if norm_ok and normalized != text:
            value = _deterministic_on_normalized(ctx, slot, normalized)
            source, conf = "deterministic_on_ai_normalized", float(out.get("normalization_confidence") or 0)
        if not value and slot.name in proposals:
            p = proposals[slot.name]
            conf = float(p.get("confidence") or 0)
            if conf < min_conf:
                rejected[slot.name] = "low_confidence"
            elif not _evidence_grounded(str(p.get("evidence") or ""), text, normalized):
                rejected[slot.name] = "evidence_not_in_utterance"
            else:
                value = _validate_value(ctx, slot, str(p.get("value") or ""))
                source = "meta_muse_spark"
                if not value:
                    rejected[slot.name] = "failed_pack_validation"
        if value:
            intake._stamp_slot(slot.name, value, source_turn=text, confidence=round(conf, 3))
            meta = dict(ctx.slots.get("__slot_meta__") or {})
            if isinstance(meta.get(slot.name), dict):
                meta[slot.name] = {**meta[slot.name], "source": source}
                ctx.slots["__slot_meta__"] = meta
            filled[slot.name] = {"value": value, "source": source, "confidence": round(conf, 3)}

    if filled:
        record_action(AgentAction(
            interaction_id=iid, agent="ai_layer", action_type="slot_extracted",
            input_summary=f"ai layer on customer turn ({language['lang']})",
            output_summary=f"filled empty slots: { {k: v['value'] for k, v in filled.items()} }; deterministic values kept",
        ))

    # ── Sentiment (raise only) ─────────────────────────────────────────────
    sentiment = None
    try:
        ai_f = float(out.get("frustration") or 0.0)
        for turn in reversed(getattr(ctx, "turns", []) or []):
            if turn.get("speaker") == "customer":
                rule_f = float(turn.get("frustration_score") or 0.0)
                final = max(rule_f, round(ai_f, 3))
                turn["frustration_score"] = final
                if final > float(getattr(ctx, "peak_frustration", 0.0) or 0.0):
                    ctx.peak_frustration = final
                sentiment = {"rule": rule_f, "ai": round(ai_f, 3), "final": final,
                             "reason": str(out.get("frustration_reason") or "")[:200]}
                break
    except (TypeError, ValueError):
        sentiment = None

    record_event(
        kind="intake", interaction_id=iid, model_id=config.model_text(),
        deterministic={"slots": deterministic_snapshot, "language": det_norm},
        ai_proposal={"slots": out.get("slots"), "normalized_en": ai_norm, "language": out.get("language"),
                     "frustration": out.get("frustration")},
        accepted={"filled": filled, "rejected": rejected, "language": language, "sentiment": sentiment},
        reason="ok",
    )

    result = dict(ires)
    result["ai_layer"] = {"filled": filled, "rejected": rejected, "language": language["lang"], "sentiment": sentiment}
    if filled:
        extracted = dict(ires.get("extracted") or {})
        extracted.update({k: v["value"] for k, v in filled.items()})
        result["extracted"] = extracted
        result["slots_filled_now"] = True
        # The deterministic question may target a slot we just filled. Only a
        # plain slot question is recomputed; safety questions always stand.
        if not ires.get("safety_question"):
            nxt = intake._next_required_slot()
            if nxt is not None:
                result["question"] = nxt.prompt
            elif ctx.has_required_slots():
                result["question"] = ""
                record_action(AgentAction(
                    interaction_id=iid, agent="ai_layer", action_type="intake_completed",
                    input_summary="all required slots filled (ai layer)",
                    output_summary="slots complete after ai-layer fill",
                ))
    return result


__all__ = ["SLOT_SCHEMA", "build_prompt", "interaction_state", "maybe_enhance_intake"]
