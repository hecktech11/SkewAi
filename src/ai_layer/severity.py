"""Enhanced severity with a one-line reason: final = max(rule, AI).  [AI-LAYER]

Runs after the existing TriageAgent (rules / ML artifact) during enrichment.
The AI may only RAISE severity. It can never lower it, never clear a safety
flag, and the Qubot severity-sanity check (safety flag => Critical) still runs
afterwards and has the final say. Without a key this is a no-op.
"""

from __future__ import annotations

import asyncio
from typing import Any

from src.ai_layer import config
from src.ai_layer.meta_client import MetaError, chat_json
from src.ai_layer.store import record_event

ORDER = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}

SEVERITY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "severity": {"type": "string", "enum": ["Low", "Medium", "High", "Critical"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string", "maxLength": 200},
        "evidence": {"type": "string", "maxLength": 300},
    },
    "required": ["severity", "confidence", "reason", "evidence"],
}


def combine(rule: str, ai: str | None, *, allowed: set[str]) -> str:
    """max(rule, ai) on the Low<Medium<High<Critical order; unknown -> rule."""
    if not ai or ai not in ORDER or ai not in allowed:
        return rule
    if rule not in ORDER:
        return rule
    return ai if ORDER[ai] > ORDER[rule] else rule


def _allowed_levels(ctx) -> set[str]:
    try:
        matrix = ctx.pack.manifest.severity.priority_matrix or {}
        levels = {k for k in matrix if k in ORDER}
        return levels or {"Low", "Medium", "Critical"}
    except Exception:
        return {"Low", "Medium", "Critical"}


async def maybe_raise_severity(ctx, tres: dict[str, Any]) -> dict[str, Any]:
    if not config.meta_enabled() or not isinstance(tres, dict) or tres.get("skipped"):
        return tres
    try:
        return await _raise(ctx, tres)
    except Exception as e:
        record_event(kind="severity_error", interaction_id=getattr(ctx, "interaction_id", None), reason=type(e).__name__)
        return tres


async def _raise(ctx, tres: dict[str, Any]) -> dict[str, Any]:
    from src.agents.triage import _priority_from_matrix
    from src.ai_layer.intake_enhancer import interaction_state
    from src.ledger import record_action
    from src.ledger.writer import AgentAction

    rule = str(ctx.severity or tres.get("severity") or "Low")
    if rule == "Critical":
        return tres  # already the ceiling; nothing to raise
    allowed = _allowed_levels(ctx)
    turns = [t.get("text", "") for t in (getattr(ctx, "turns", []) or []) if t.get("speaker") == "customer"]
    normalized = [t.get("normalized_en", "") for t in interaction_state(ctx.interaction_id).get("turns", [])]
    transcript = "\n".join(turns[-8:])
    english = "\n".join(n for n in normalized[-8:] if n)
    system = (
        "You rate the safety severity of a customer complaint for a quality team. Return JSON only. "
        f"Allowed severities: {', '.join(sorted(allowed, key=ORDER.get))}. "
        "Critical = risk of injury, fire, crash, loss of vehicle control. "
        "Give a one-line reason and copy the exact caller words that justify it into evidence. "
        "Do not invent facts."
    )
    user = (
        f"category={ctx.slots.get('category')} make={ctx.slots.get('entity_2')} model={ctx.slots.get('entity_3')}\n"
        f"description={ctx.slots.get('description')}\n"
        f"rule_severity={rule}\nCALLER TRANSCRIPT:\n{transcript[:2000]}\n"
        + (f"ENGLISH NORMALISATION:\n{english[:2000]}" if english else "")
    )
    try:
        out = await asyncio.wait_for(
            asyncio.to_thread(chat_json, system=system, user=user, schema=SEVERITY_SCHEMA,
                              schema_name="severity", timeout=config.timeout_s()),
            timeout=config.timeout_s() + 0.5,
        )
    except (MetaError, asyncio.TimeoutError, TimeoutError) as e:
        record_event(kind="severity", interaction_id=ctx.interaction_id, deterministic={"severity": rule},
                     reason=f"fallback:{getattr(e, 'reason', type(e).__name__)}")
        return tres
    ai = str(out.get("severity"))
    conf = float(out.get("confidence") or 0)
    evidence = str(out.get("evidence") or "").strip().lower()
    haystack = (transcript + "\n" + english + "\n" + str(ctx.slots.get("description") or "")).lower()
    grounded = len(evidence) >= 3 and evidence in haystack
    final = rule
    reason = "kept_rule"
    if conf < config.min_confidence():
        reason = "low_confidence"
    elif not grounded:
        reason = "evidence_not_in_transcript"
    else:
        final = combine(rule, ai, allowed=allowed)
        reason = "raised" if final != rule else "ai_not_higher"
    record_event(kind="severity", interaction_id=ctx.interaction_id, model_id=config.model_text(),
                 deterministic={"severity": rule, "source": tres.get("severity_source")},
                 ai_proposal=out, accepted={"severity": final}, reason=reason)
    if final == rule:
        return tres
    ctx.severity = final
    ctx.priority = _priority_from_matrix(final, ctx)
    line = str(out.get("reason") or "")[:200]
    record_action(AgentAction(
        interaction_id=ctx.interaction_id, agent="ai_layer", action_type="severity_scored",
        input_summary=f"rule severity={rule}; ai severity={ai} conf={conf:.2f}",
        output_summary=f"severity raised {rule}->{final} (max(rule, ai)); reason={line}",
    ))
    new = dict(tres)
    new.update({"severity": final, "priority": ctx.priority, "ai_layer": {"rule": rule, "ai": ai, "reason": line}})
    return new


__all__ = ["ORDER", "SEVERITY_SCHEMA", "combine", "maybe_raise_severity"]
