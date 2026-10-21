"""Optional Meta AI layer for SkewAi.  [AI-LAYER]

Everything in this package is additive. The deterministic regex / gazetteer /
rules path stays the default and the fallback:

* With no ``META_API_KEY`` (or ``MODEL_API_KEY``) set, every entry point here
  returns immediately without network I/O and leaves the caller's data
  untouched, so behaviour is byte-for-byte what it was before this package.
* With a key, Meta Model API calls (Muse Spark, SAM 3.1, Muse Image) may *add*
  information: fill empty intake slots, normalise Hindi/Hinglish, raise (never
  lower) severity, describe photo evidence, and write grounded briefs whose
  every claim cites real record ids that pass the Qubot verifiers.
* Any error, timeout, schema-invalid output or low-confidence answer falls back
  silently to the deterministic result.

See ``docs/ai_layer.md`` and ``HACKATHON.md`` for what is verified and what is
not about the Meta API surface.
"""

from src.ai_layer.config import ai_mode, ai_status, meta_enabled

__all__ = ["ai_mode", "ai_status", "meta_enabled"]
