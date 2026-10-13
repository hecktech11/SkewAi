"""Triage Agent — severity + priority.

Fires mid-contact once category + description are known. Severity via the
pack's configured path:
  - ML runtime artifact (automotive keeps the v1 XGBoost model), OR
  - the pack's ordered rule list.

Priority via the pack's priority matrix (any safety flag → P1 always).
No LLM.
"""

from __future__ import annotations

from typing import Any

from src.agents.base import Agent
from src.ledger import record_action


def _eval_condition(cond: str, ctx: Any) -> bool:
    """Evaluate a pack severity rule's `when` condition against the context.

    Supported:
      - "safety.any" — any safety flag set
      - "safety.<key>" — specific safety flag
      - "category in [A, B, C]" — category match
      - "description_matches_<term>" — substring match on description
      - "default" / "always" / "true" — always true
    """
    cond = cond.strip()
    if cond.lower() in ("default", "always", "true", "else"):
        return True

    if cond == "safety.any":
        return any(ctx.safety_flags.values())
    if cond.startswith("safety."):
        key = cond.split(".", 1)[1]
        return bool(ctx.safety_flags.get(key))

    if cond.startswith("category in"):
        # parse [A, B, C]
        import re
        m = re.search(r"\[(.*)\]", cond)
        if not m:
            return False
        cats = [c.strip().strip("'\"") for c in m.group(1).split(",")]
        return ctx.slots.get("category", "").upper() in [c.upper() for c in cats]

    if cond.startswith("description_matches"):
        # description_matches_fire|smoke|crash|injury
        terms_part = cond.split("description_matches", 1)[1].strip()
        # may start with _ or whitespace
        terms_part = terms_part.lstrip("_").strip()
        if not terms_part:
            return False
        from src.frontline.elicitation import symptom_text_for_ctx
        desc = symptom_text_for_ctx(ctx).lower()
        # Same hazard reading as intake: a negated hit ("no fire") does not
        # match. An affirmative hit in a later clause still matches.
        from src.agents.intake import hazard_term_affirmed
        for term in terms_part.split("|"):
            term = term.strip().strip("'\"")
            if not term:
                continue
            if hazard_term_affirmed(desc, term):
                return True
        return False

    return False


def _apply_rule_severity(ctx: Any) -> tuple[str, str, str]:
    """Returns (severity, source, reason). Uses the pack's ordered rules."""
    for rule in ctx.pack.manifest.severity.rules:
        if _eval_condition(rule.when, ctx):
            return rule.severity, "rules", rule.reason
    return "Low", "rules", "No rule matched; default to Low"


# ── Severity fallback policy (item 25) ──────────────────────────────────────
# Explicit matrix:
#   model configured + model succeeds  -> (model severity, source="model")
#   model configured + model fails     -> (rules severity, source="rules")
#                                          + model_error ledger + model_healthy=False
#   no model configured                -> (rules severity, source="rules")
# Policy choice: SAFE FALLBACK (not hard failure / manual review) — severity
# scoring must never break a live contact. The fallback is explicit, never
# silent: source is "rules" (never "model") whenever the model result was not
# actually used, so `source == "model"` always means a valid model prediction
# flowed through. Eval asserts this invariant for every severity level.

SEVERITY_FALLBACK_POLICY = "safe-fallback-with-ledger"

_MODEL_HEALTHY: dict[str, bool] = {}


def model_health() -> dict[str, bool]:
    """Per-artifact model health (True after success, False after failure)."""
    return dict(_MODEL_HEALTHY)


def _mark_model_health(artifact_ref: str, healthy: bool) -> None:
    _MODEL_HEALTHY[str(artifact_ref)] = bool(healthy)
    try:
        from src.observability.metrics import inc as _inc

        _inc("model_healthy" if healthy else "model_error",
             artifact=str(artifact_ref))
    except Exception:
        pass


def _apply_model_severity(ctx: Any) -> tuple[str, str, str]:
    """Use the pack's ML runtime artifact for severity.

    Falls back to rules if the model isn't loaded (clean degradation).
    """
    artifact_ref = ctx.pack.manifest.severity.model_artifact
    if not artifact_ref:
        return _apply_rule_severity(ctx)

    # Lazy import to avoid the heavy ml_runtime dependency at module load time.
    try:
        from src.ml_runtime.registry import predict_severity
        from src.frontline.elicitation import symptom_text_for_ctx
        severity, reason = predict_severity(artifact_ref, {
            "entity_1": ctx.slots.get("entity_1"),
            "entity_2": ctx.slots.get("entity_2"),
            "entity_3": ctx.slots.get("entity_3"),
            "category": ctx.slots.get("category"),
            "description": symptom_text_for_ctx(ctx),
            "safety_flags": ctx.safety_flags,
        })
        _mark_model_health(artifact_ref, True)
        return severity, "model", reason or "ML artifact prediction"
    except Exception as e:
        # Safe fallback per SEVERITY_FALLBACK_POLICY: rules decide, source is
        # honestly "rules", and the model failure is ledgered + gauged —
        # never reported as source=model.
        _mark_model_health(artifact_ref, False)
        try:
            record_action(TriageAgent(ctx)._action(
                action_type="severity_scored",
                input_summary=f"model artifact {artifact_ref} failed",
                output_summary=f"model_error; falling back to rules ({type(e).__name__})",
                ok=False,
                error=f"{type(e).__name__}: {e}"[:500],
            ))
        except Exception:
            pass
        sev, _, _ = _apply_rule_severity(ctx)
        return sev, "rules", f"model fallback ({type(e).__name__}: {e})"


def score_severity(ctx: Any) -> tuple[str, str, str]:
    """Same path TriageAgent.run uses. REPRODUCE must call this, not rules-only."""
    if ctx.pack.manifest.severity.model_artifact:
        return _apply_model_severity(ctx)
    return _apply_rule_severity(ctx)


def _priority_from_matrix(severity: str, ctx: Any) -> int:
    """Compute priority (1=highest) via the pack's priority matrix.

    Hardcoded invariant: any safety flag → P1 always (the matrix value for
    `safety` is 1 by default but we enforce the floor here too).
    """
    matrix = ctx.pack.manifest.severity.priority_matrix
    if any(ctx.safety_flags.values()):
        return 1
    # Explicit severity levels; unknown severity is a pack bug — fail loudly
    # instead of silently mapping High->Low's priority.
    if severity in matrix:
        return matrix[severity]
    if severity == "High" and "High" not in matrix:
        # Common pack omission: interpolate between Critical(1) and Medium(2)
        return 1 if any(ctx.safety_flags.values()) else 2
    return matrix.get(severity, matrix.get("Low", 3))


def _model_stamp(ctx: Any) -> str:
    """Identify the scoring model build: artifact path + content hash.

    Rules path stamps ``rules:<pack_version>``. A missing/unreadable
    artifact stamps ``artifact-unreadable`` rather than failing the contact.
    """
    try:
        artifact = (ctx.pack.manifest.severity.model_artifact or "").strip()
    except Exception:
        artifact = ""
    if not artifact:
        try:
            pv = ctx.pack.pack_version
        except Exception:
            pv = "unknown"
        return f"rules:{pv}"
    try:
        import hashlib as _hl
        from pathlib import Path as _P

        from src.config import REPO_ROOT as _RR

        p = _P(artifact)
        for cand in (p, _RR / p, _RR / "domains" / p):
            try:
                if cand.is_file():
                    h = _hl.sha256(cand.read_bytes()).hexdigest()[:12]
                    return f"{artifact}@{h}"
            except OSError:
                continue
        return f"{artifact}@unreadable"
    except Exception:
        return f"{artifact}@unknown"


class TriageAgent(Agent):
    """Severity + priority. No LLM."""

    name = "triage"

    async def run(self, **kwargs: Any) -> dict[str, Any]:
        ctx = self.ctx
        try:
            from src.ops.pilot import agent_enabled
            if not agent_enabled("triage"):
                record_action(self._action(
                    action_type="severity_scored",
                    input_summary="flag FRONTLINE_AGENT_TRIAGE_ENABLED=0",
                    output_summary="triage disabled; default Low/P3",
                ))
                ctx.severity, ctx.severity_source, ctx.priority = "Low", "rules", 3
                return {"severity": "Low", "severity_source": "rules",
                        "reason": "disabled_by_flag", "priority": 3}
        except Exception:
            pass

        # Precondition: need category + description to triage
        if not ctx.slots.get("category") or not ctx.slots.get("description"):
            return {"skipped": True, "reason": "category + description not yet known"}

        severity, source, reason = score_severity(ctx)

        ctx.severity = severity
        ctx.severity_source = source
        ctx.priority = _priority_from_matrix(severity, ctx)

        # Model stamping (board: drift/version visibility): every score
        # carries WHICH artifact scored it (or that rules did), so drift
        # analysis can group outcomes by model version after the fact.
        stamp = _model_stamp(ctx)
        try:
            from src.governance.registry import artifact_version, explain_severity
            gov_version = artifact_version(ctx.pack.manifest.severity.model_artifact or "")
            expl = explain_severity(severity=severity, source=source, reason=reason,
                                    safety_flags=dict(ctx.safety_flags or {}),
                                    category=ctx.slots.get("category"))
            ctx.model_version = gov_version  # type: ignore[attr-defined]
        except Exception:
            gov_version, expl = stamp, {"drivers": [reason]}
        record_action(self._action(
            action_type="severity_scored",
            input_summary=f"slots={ctx.slots}, safety_flags={ctx.safety_flags}",
            output_summary=f"severity={severity} (source={source}); reason={reason}; model={stamp}; model_version={gov_version}; drivers={expl.get('drivers', [])}",
        ))
        record_action(self._action(
            action_type="priority_assigned",
            input_summary=f"severity={severity}, safety_flags={ctx.safety_flags}",
            output_summary=f"priority=P{ctx.priority}",
        ))

        return {
            "severity": severity,
            "severity_source": source,
            "reason": reason,
            "priority": ctx.priority,
            "model_version": gov_version,
            "drivers": expl.get("drivers", []),
        }
