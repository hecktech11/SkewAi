"""Stand-alone Hindi/Hinglish -> English normalisation.  [AI-LAYER]

Used by ``POST /api/ai/language/normalize`` and the offline eval. The live
call path uses the combined per-turn call in ``intake_enhancer`` instead.
"""

from __future__ import annotations

from typing import Any

from src.ai_layer import config
from src.ai_layer.language import accept_ai_normalisation, deterministic_normalise, detect
from src.ai_layer.meta_client import MetaError, chat_json

NORMALIZE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "normalized_en": {"type": "string", "maxLength": 2000},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["normalized_en", "confidence"],
}


def normalize_text(text: str) -> dict[str, Any]:
    det = detect(text)
    lang = det["lang"]
    base = deterministic_normalise(text, lang)
    base["lang"] = lang
    base["mode"] = config.ai_mode()
    if lang not in ("hi", "hinglish") or not config.meta_enabled():
        return base
    try:
        out = chat_json(
            system=(
                "Translate the customer's Hindi or Hinglish complaint into plain English. Return JSON only. "
                "Keep every number exactly as said. Add nothing that was not said."
            ),
            user=text[:2000],
            schema=NORMALIZE_SCHEMA,
            schema_name="normalize",
        )
    except MetaError as e:
        base["fallback_reason"] = e.reason
        return base
    if accept_ai_normalisation(text, str(out.get("normalized_en") or ""), float(out.get("confidence") or 0),
                               config.min_confidence()):
        return {"original": text, "normalized_en": str(out["normalized_en"]).strip(), "lang": lang,
                "method": "meta_muse_spark", "mode": "enhanced"}
    base["fallback_reason"] = "rejected_by_sanity_checks"
    return base


__all__ = ["NORMALIZE_SCHEMA", "normalize_text"]
