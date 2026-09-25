"""LLM provider interface — optional OpenAI-compatible HTTP + deterministic fallback.

When no key, spend/turn caps hit, or the HTTP call fails, callers receive
``ok=False`` and must use the deterministic template path. Env keys alone do
not change agent behavior until ``narrate()`` is explicitly invoked and succeeds.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from src.config import settings
from src.data.timeutil import utc_now
from src.data.warehouse import ops_con


@dataclass
class NarrationResult:
    text: str
    ok: bool
    model_id: str = "deterministic"
    prompt_hash: str = ""
    used_llm: bool = False
    reason: str = ""


def _prompt_hash(system: str, user: str) -> str:
    return hashlib.sha256(f"{system}\n---\n{user}".encode("utf-8")).hexdigest()[:16]


def _day_key() -> str:
    return utc_now().strftime("%Y-%m-%d")


def get_spend(day: str | None = None) -> dict[str, Any]:
    d = day or _day_key()
    with ops_con(read_only=True) as con:
        try:
            row = con.execute(
                "SELECT day_key, turn_count, cost_units FROM llm_spend WHERE day_key = ?",
                [d],
            ).fetchone()
        except Exception:
            return {"day_key": d, "turn_count": 0, "cost_units": 0.0}
    if not row:
        return {"day_key": d, "turn_count": 0, "cost_units": 0.0}
    return {"day_key": row[0], "turn_count": int(row[1]), "cost_units": float(row[2])}


def _record_spend(cost_units: float = 0.01) -> None:
    d = _day_key()
    now = utc_now()
    with ops_con() as con:
        exists = con.execute(
            "SELECT turn_count, cost_units FROM llm_spend WHERE day_key = ?", [d]
        ).fetchone()
        if exists:
            con.execute(
                """
                UPDATE llm_spend
                SET turn_count = turn_count + 1,
                    cost_units = cost_units + ?,
                    updated_at = ?
                WHERE day_key = ?
                """,
                [cost_units, now, d],
            )
        else:
            con.execute(
                """
                INSERT INTO llm_spend (day_key, turn_count, cost_units, updated_at)
                VALUES (?, 1, ?, ?)
                """,
                [d, cost_units, now],
            )


def _openai_key() -> str:
    return (os.getenv("OPENAI_API_KEY") or settings.openai_api_key or "").strip()


def _claude_key() -> str:
    return (os.getenv("CLAUDE_API_KEY") or settings.claude_api_key or "").strip()


def llm_provider() -> str:
    """Active provider name: ``openai``, ``anthropic``, or empty.

    OpenAI wins when both keys are set. A Claude-only install must never
    send ``CLAUDE_API_KEY`` to ``api.openai.com``.
    """
    if _openai_key():
        return "openai"
    if _claude_key():
        return "anthropic"
    return ""


def llm_enabled() -> bool:
    """True only when a provider key is set AND FRONTLINE_LLM_ENABLED=1."""
    raw = os.getenv("FRONTLINE_LLM_ENABLED", "").strip().lower()
    if raw not in {"1", "true", "yes", "on"}:
        return False
    return bool(llm_provider())


def can_spend() -> tuple[bool, str]:
    if not llm_enabled():
        return False, "llm_disabled_or_no_key"
    spend = get_spend()
    if spend["turn_count"] >= max(1, settings.llm_turn_cap):
        return False, "turn_cap"
    if spend["cost_units"] >= float(settings.max_daily_claude_cost):
        return False, "daily_cost_cap"
    return True, "ok"


def _http_chat(system: str, user: str, model: str) -> str:
    """Dispatch to the provider that actually owns the configured key."""
    provider = llm_provider()
    if provider == "anthropic":
        return _http_chat_anthropic(system, user, model)
    if provider == "openai":
        return _http_chat_openai(system, user, model)
    raise ValueError("no_provider_key")


# Hosts that may receive a provider API key. `*_BASE_URL` is an ambient env
# var — on a developer box it is routinely pointed at a proxy or gateway — so
# the SSRF guard alone is not enough: it clears any *public* host, which would
# silently ship the key to a third party. Redirecting credentialed traffic is
# therefore an explicit operator decision, not a default.
_OFFICIAL_LLM_HOSTS = {
    "openai": "api.openai.com",
    "anthropic": "api.anthropic.com",
}


def _allowed_llm_hosts(provider: str) -> set[str]:
    """Official host for `provider`, plus any FRONTLINE_LLM_ALLOWED_HOSTS entries."""
    hosts = {_OFFICIAL_LLM_HOSTS[provider]}
    for extra in (os.getenv("FRONTLINE_LLM_ALLOWED_HOSTS") or "").split(","):
        host = extra.strip().lower()
        if host:
            hosts.add(host)
    return hosts


def _guard_llm_url(url: str, provider: str) -> str:
    """Confirm the host may receive the key, then SSRF-check it.

    Allowlist first: a disallowed host should not even reach the resolver, and
    failing before DNS keeps the reason deterministic offline. The SSRF check
    still runs for approved hosts — an allowlisted gateway can resolve to a
    private or metadata address.
    """
    from src.security.url_guard import validate_outbound_url

    host = (urlparse(url).hostname or "").lower()
    if host not in _allowed_llm_hosts(provider):
        # "blocked" keeps the reason greppable alongside the url_guard reasons.
        raise ValueError(f"llm_base_url_blocked:{host}")
    validate_outbound_url(url)
    return url


def _http_chat_openai(system: str, user: str, model: str) -> str:
    """Minimal OpenAI-compatible chat completions call (stdlib only)."""
    api_key = _openai_key()
    if not api_key:
        raise ValueError("openai_key_missing")
    base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    url = f"{base}/chat/completions"

    _guard_llm_url(url, "openai")
    body = {
        "model": model or os.getenv("FRONTLINE_LLM_MODEL", "gpt-4o-mini"),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.3,
        "max_tokens": 120,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=8.0) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return (data["choices"][0]["message"]["content"] or "").strip()


def _anthropic_model(model: str) -> str:
    cand = (model or "").strip()
    if cand.startswith("claude"):
        return cand
    env = os.getenv("FRONTLINE_LLM_MODEL", "").strip()
    if env.startswith("claude"):
        return env
    return "claude-haiku-4-5-20251001"


def _http_chat_anthropic(system: str, user: str, model: str) -> str:
    """Anthropic Messages API (stdlib only). Never sends the key to OpenAI."""
    api_key = _claude_key()
    if not api_key:
        raise ValueError("claude_key_missing")
    base = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
    url = f"{base}/v1/messages"

    _guard_llm_url(url, "anthropic")
    body = {
        "model": _anthropic_model(model),
        "max_tokens": 120,
        "temperature": 0.3,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=8.0) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    blocks = data.get("content") or []
    texts = [
        (b.get("text") or "")
        for b in blocks
        if isinstance(b, dict) and b.get("type", "text") == "text"
    ]
    return "".join(texts).strip()


def narrate(
    *,
    site: str,
    system: str,
    user: str,
    fallback: str,
    model: str | None = None,
) -> NarrationResult:
    """Try LLM narration; always return usable text via fallback on any miss.

    ``site`` is one of: intake_phrasing | investigator_brief | followup_draft.
    """
    ph = _prompt_hash(system, user)
    ok_spend, reason = can_spend()
    if not ok_spend:
        return NarrationResult(
            text=fallback,
            ok=False,
            model_id="deterministic",
            prompt_hash=ph,
            used_llm=False,
            reason=reason,
        )
    try:
        text = _http_chat(system, user, model or "")
        if not text:
            raise ValueError("empty_llm_response")
        _record_spend(0.01)
        provider = llm_provider()
        default_model = (
            _anthropic_model(model or "")
            if provider == "anthropic"
            else (model or os.getenv("FRONTLINE_LLM_MODEL", "gpt-4o-mini"))
        )
        return NarrationResult(
            text=text[:500],
            ok=True,
            model_id=default_model,
            prompt_hash=ph,
            used_llm=True,
            reason="ok",
        )
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError, ValueError, OSError) as e:
        try:
            from src.observability.degradation import step_down

            step_down("llm_narration", reason=f"llm_error:{type(e).__name__}")
        except Exception:
            pass
        return NarrationResult(
            text=fallback,
            ok=False,
            model_id="deterministic",
            prompt_hash=ph,
            used_llm=False,
            reason=f"llm_error:{type(e).__name__}",
        )


__all__ = [
    "NarrationResult",
    "narrate",
    "llm_enabled",
    "llm_provider",
    "can_spend",
    "get_spend",
]
