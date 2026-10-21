"""Environment-driven configuration for the optional Meta AI layer.  [AI-LAYER]

Every endpoint, model id and limit is env-configurable. Defaults follow the
public Meta Model API docs at https://dev.meta.ai/docs (read 2026-10-21) but
have NOT been exercised against the live API from this repo: no key exists in
this project. See HACKATHON.md "What is unverified".

Env vars (all optional):

  META_API_KEY / MODEL_API_KEY   bearer token. Absent -> mode "deterministic".
  FRONTLINE_AI_LAYER             "auto" (default) | "off". "off" forces the
                                 deterministic path even when a key is set.
  META_API_BASE                  default https://api.meta.ai/v1
  META_MODEL_TEXT                default muse-spark-1.3   (chat completions)
  META_MODEL_VISION              default muse-spark-1.3   (image understanding)
  META_MODEL_SAM                 default sam-3.1          (Responses API)
  META_MODEL_IMAGE               default muse-image-1.0   (images/generations)
  META_CHAT_PATH                 default /chat/completions
  META_SAM_ENDPOINT              default <META_API_BASE>/responses
  META_IMAGE_ENDPOINT            default <META_API_BASE>/images/generations
  META_ALLOWED_HOSTS             extra comma-separated hosts that may receive
                                 the key (e.g. a self-hosted Muse Spark Open /
                                 Muse Glimmer vLLM gateway). api.meta.ai is
                                 always allowed.
  META_TIMEOUT_S                 default 8 (per HTTP attempt)
  META_INTAKE_TIMEOUT_S          default 4 (live-call turn budget)
  META_RETRIES                   default 1 (extra attempts on 429/5xx/network)
  META_MIN_CONFIDENCE            default 0.7 (AI slot/category floor)
  META_DAILY_CALL_CAP            default 2000 (process-local guard)
  AI_EVIDENCE_MAX_BYTES          default 8 MiB
  AI_EVIDENCE_DIR                default <repo>/data/ai_evidence
"""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse

DEFAULT_API_BASE = "https://api.meta.ai/v1"
OFFICIAL_HOST = "api.meta.ai"
DEFAULT_MODEL_TEXT = "muse-spark-1.3"
DEFAULT_MODEL_VISION = "muse-spark-1.3"
DEFAULT_MODEL_SAM = "sam-3.1"
DEFAULT_MODEL_IMAGE = "muse-image-1.0"


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def meta_api_key() -> str:
    """Bearer token. ``META_API_KEY`` wins; ``MODEL_API_KEY`` is the name used
    by the official docs and is accepted as an alias. Never logged."""
    return _env("META_API_KEY") or _env("MODEL_API_KEY")


def layer_switch() -> str:
    raw = _env("FRONTLINE_AI_LAYER", "auto").lower()
    if raw in {"0", "off", "false", "no", "deterministic"}:
        return "off"
    return "auto"


def meta_enabled() -> bool:
    """True only when a key is configured and the layer is not switched off."""
    return layer_switch() != "off" and bool(meta_api_key())


def ai_mode() -> str:
    return "enhanced" if meta_enabled() else "deterministic"


def api_base() -> str:
    return _env("META_API_BASE", DEFAULT_API_BASE).rstrip("/")


def model_text() -> str:
    return _env("META_MODEL_TEXT", DEFAULT_MODEL_TEXT)


def model_vision() -> str:
    return _env("META_MODEL_VISION", DEFAULT_MODEL_VISION)


def model_sam() -> str:
    return _env("META_MODEL_SAM", DEFAULT_MODEL_SAM)


def model_image() -> str:
    return _env("META_MODEL_IMAGE", DEFAULT_MODEL_IMAGE)


def chat_url() -> str:
    path = _env("META_CHAT_PATH", "/chat/completions")
    if not path.startswith("/"):
        path = "/" + path
    return api_base() + path


def sam_url() -> str:
    return _env("META_SAM_ENDPOINT") or (api_base() + "/responses")


def image_url() -> str:
    return _env("META_IMAGE_ENDPOINT") or (api_base() + "/images/generations")


def allowed_hosts() -> set[str]:
    hosts = {OFFICIAL_HOST}
    for raw in _env("META_ALLOWED_HOSTS").split(","):
        host = raw.strip().lower()
        if host:
            hosts.add(host)
    return hosts


def timeout_s() -> float:
    return max(0.5, min(60.0, _env_float("META_TIMEOUT_S", 8.0)))


def intake_timeout_s() -> float:
    return max(0.5, min(30.0, _env_float("META_INTAKE_TIMEOUT_S", 4.0)))


def retries() -> int:
    return max(0, min(3, _env_int("META_RETRIES", 1)))


def min_confidence() -> float:
    return max(0.0, min(1.0, _env_float("META_MIN_CONFIDENCE", 0.7)))


def daily_call_cap() -> int:
    return max(1, _env_int("META_DAILY_CALL_CAP", 2000))


def evidence_max_bytes() -> int:
    return max(1024, _env_int("AI_EVIDENCE_MAX_BYTES", 8 * 1024 * 1024))


def _host(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


def ai_status() -> dict[str, Any]:
    """Public, secret-free status for GET /api/ai/status and the UI badge."""
    enabled = meta_enabled()
    base_host = _host(api_base())
    return {
        "mode": ai_mode(),
        "provider": "meta" if enabled else None,
        "key_configured": bool(meta_api_key()),
        "layer_switch": layer_switch(),
        "api_host": base_host,
        "api_host_allowed": base_host in allowed_hosts(),
        "models": {
            "text": model_text(),
            "vision": model_vision(),
            "segmentation": model_sam(),
            "image": model_image(),
        },
        "features": {
            "enhanced_intake": enabled,
            "hindi_hinglish_normalisation": enabled,
            "grounded_brief": enabled,
            "severity_sentiment_raise_only": enabled,
            "photo_evidence_upload": True,
            "photo_evidence_analysis": enabled,
        },
        "fallback": "deterministic regex/gazetteer/rules path (always on)",
        "verified_against_live_api": False,
    }


__all__ = [
    "ai_mode",
    "ai_status",
    "allowed_hosts",
    "api_base",
    "chat_url",
    "daily_call_cap",
    "evidence_max_bytes",
    "image_url",
    "intake_timeout_s",
    "meta_api_key",
    "meta_enabled",
    "min_confidence",
    "model_image",
    "model_sam",
    "model_text",
    "model_vision",
    "retries",
    "sam_url",
    "timeout_s",
]
