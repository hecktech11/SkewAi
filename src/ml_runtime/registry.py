"""ML runtime registry — severity prediction with honest rules fallback.

No binary model ships in the pilot fixture. When a pack declares
``model_artifact``, we try a local JSON rules overlay if present; otherwise
raise so triage falls back to pack severity rules and sets severity_source=rules.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config import REPO_ROOT


def predict_severity(
    artifact_ref: str,
    features: dict[str, Any],
) -> tuple[str, str]:
    """Return (severity, reason). Raises if artifact cannot be loaded."""
    path = Path(artifact_ref)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        # Also try pack models dir relative paths
        alt = REPO_ROOT / "domains" / artifact_ref
        if alt.exists():
            path = alt
        else:
            raise FileNotFoundError(f"severity artifact not found: {artifact_ref}")

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        # Simple keyword → severity overlay (honest, inspectable, not XGBoost).
        text = " ".join(
            str(features.get(k) or "")
            for k in ("description", "category", "entity_2", "entity_3")
        ).lower()
        flags = features.get("safety_flags") or {}
        if any(flags.values()) if isinstance(flags, dict) else False:
            return "Critical", "safety_flag_override"
        from src.agents.intake import hazard_term_affirmed

        for rule in data.get("rules") or []:
            kw = (rule.get("keyword") or "").lower().strip()
            if not kw:
                continue
            # Same affirmed-hazard check intake uses. "no fire" is not fire.
            # "misfires" still does not hit "fire" (word boundary inside it).
            if hazard_term_affirmed(text, kw):
                return str(rule.get("severity") or "Medium"), f"json_rule:{kw}"
        # No keyword hit → raise so triage uses pack YAML severity rules
        # (honest: JSON overlay is additive, not a silent default that masks pack rules).
        raise LookupError("json_severity_no_keyword_match")

    raise RuntimeError(
        f"unsupported severity artifact type {path.suffix} "
        f"(XGBoost runtime not shipped; use .json rules overlay)"
    )


__all__ = ["predict_severity"]
