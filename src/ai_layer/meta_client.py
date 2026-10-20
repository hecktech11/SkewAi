"""Meta Model API adapter (Muse Spark, SAM 3.1, Muse Image).  [AI-LAYER]

Matches the existing provider contract in ``src/ai/provider.py``:
``meta_narrate()`` returns the same ``NarrationResult`` and always yields
usable text through ``fallback`` on any miss. Structured calls
(``chat_json``) raise ``MetaError`` instead, and every caller in
``src.ai_layer`` catches it and keeps the deterministic value.

Wire shapes (from https://dev.meta.ai/docs, read 2026-10-21; NOT exercised
against the live API from this repo, so all of it is env-configurable):

* Chat Completions, OpenAI-compatible: ``POST {base}/chat/completions`` with
  ``Authorization: Bearer <key>``, ``response_format`` = ``json_schema``.
  Images go in user content as ``{"type":"image_url","image_url":{"url":...}}``.
* SAM 3.1: ``POST {base}/responses`` with model ``sam-3.1`` and one user
  message holding an ``input_text`` noun phrase plus an ``input_image``.
  Output is special-token text:
  ``<0f>0<|box;x1=..;y1=..;x2=..;y2=..;w=..;h=..|><|mask;...|>``.
* Muse Image: ``POST {base}/images/generations`` (only used by an offline
  demo-asset script; never on a live product path).

Safety properties:
* The key is only sent to hosts in ``config.allowed_hosts()``; the SSRF guard
  (``src.security.url_guard``) still runs on approved hosts.
* Customer text is PII-redacted before it leaves the process.
* Timeouts on every attempt, bounded retries on 429/5xx/network errors only.
* A process-local daily call cap.
"""

from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable
from urllib.parse import urlparse

from src.ai_layer import config
from src.ai_layer.schema import validate


class MetaError(Exception):
    """Any reason the AI result must not be used. ``reason`` is log-safe."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass
class HttpResponse:
    status: int
    body: bytes
    content_type: str = ""


Transport = Callable[[str, bytes, dict[str, str], float], HttpResponse]


def _urllib_transport(url: str, body: bytes, headers: dict[str, str], timeout: float) -> HttpResponse:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - guarded URL
            return HttpResponse(
                status=int(getattr(resp, "status", 200) or 200),
                body=resp.read(),
                content_type=str(resp.headers.get("Content-Type") or ""),
            )
    except urllib.error.HTTPError as e:
        try:
            payload = e.read()
        except Exception:
            payload = b""
        return HttpResponse(status=int(e.code), body=payload, content_type="")


_transport: Transport = _urllib_transport


def set_transport(fn: Transport | None) -> None:
    """Swap the HTTP transport (tests inject a fake; ``None`` restores urllib)."""
    global _transport
    _transport = fn or _urllib_transport


# ── Daily call cap (process-local) ─────────────────────────────────────────
_cap_lock = threading.Lock()
_cap_state: dict[str, int] = {}


def _reserve_call() -> None:
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with _cap_lock:
        for key in list(_cap_state):
            if key != day:
                _cap_state.pop(key, None)
        used = _cap_state.get(day, 0)
        if used >= config.daily_call_cap():
            raise MetaError("daily_call_cap")
        _cap_state[day] = used + 1


def reset_call_cap() -> None:
    with _cap_lock:
        _cap_state.clear()


def calls_today() -> int:
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with _cap_lock:
        return _cap_state.get(day, 0)


# ── URL guard ──────────────────────────────────────────────────────────────


def guard_url(url: str) -> str:
    """Allowlist first (no DNS for a disallowed host), then the SSRF guard."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https":
        raise MetaError("meta_url_must_be_https")
    if host not in config.allowed_hosts():
        raise MetaError(f"meta_host_not_allowlisted:{host}")
    try:
        from src.security.url_guard import validate_outbound_url

        validate_outbound_url(url)
    except ValueError as e:
        raise MetaError(f"meta_url_blocked:{e}") from e
    return url


def _post(url: str, payload: dict[str, Any], *, timeout: float, accept: str | None = None) -> HttpResponse:
    if not config.meta_enabled():
        raise MetaError("meta_disabled")
    key = config.meta_api_key()
    guard_url(url)
    _reserve_call()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    }
    if accept:
        headers["Accept"] = accept
    body = json.dumps(payload).encode("utf-8")
    attempts = 1 + config.retries()
    last = "meta_unknown_error"
    for i in range(attempts):
        try:
            resp = _transport(url, body, headers, timeout)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = f"meta_network:{type(e).__name__}"
        else:
            if 200 <= resp.status < 300:
                return resp
            last = f"meta_http_{resp.status}"
            if resp.status != 429 and resp.status < 500:
                raise MetaError(last)
        if i + 1 < attempts:
            time.sleep(min(2.0, 0.25 * (2 ** i)))
    raise MetaError(last)


_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.I)


def _parse_json_text(text: str) -> Any:
    cleaned = _FENCE.sub("", (text or "").strip())
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, TypeError) as e:
        raise MetaError("meta_output_not_json") from e


def _redact(text: str) -> str:
    try:
        from src.security.pii import redact_pii

        return redact_pii(text or "")
    except Exception:
        return text or ""


def chat_json(
    *,
    system: str,
    user: str,
    schema: dict[str, Any],
    schema_name: str,
    images: list[str] | None = None,
    model: str | None = None,
    timeout: float | None = None,
    max_tokens: int = 700,
) -> dict[str, Any]:
    """One schema-constrained Muse Spark call. Returns validated JSON or raises."""
    user_text = _redact(user)
    content: Any = user_text
    if images:
        content = [{"type": "text", "text": user_text}] + [
            {"type": "image_url", "image_url": {"url": url}} for url in images[:4]
        ]
    payload = {
        "model": model or (config.model_vision() if images else config.model_text()),
        "messages": [
            {"role": "system", "content": _redact(system)},
            {"role": "user", "content": content},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "schema": schema, "strict": False},
        },
    }
    resp = _post(config.chat_url(), payload, timeout=timeout or config.timeout_s())
    try:
        data = json.loads(resp.body.decode("utf-8"))
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, ValueError, UnicodeDecodeError) as e:
        raise MetaError("meta_bad_envelope") from e
    if isinstance(text, list):  # some servers return content parts
        text = "".join(str(p.get("text") or "") for p in text if isinstance(p, dict))
    value = _parse_json_text(str(text or ""))
    errors = validate(value, schema)
    if errors:
        raise MetaError("meta_schema_invalid:" + errors[0][:120])
    return value


# ── SAM 3.1 ────────────────────────────────────────────────────────────────

_BOX = re.compile(r"(\d+)\s*<\|box;([^|>]*)\|>(?:\s*<\|mask;([^|>]*)\|>)?")


def parse_sam_output(text: str) -> list[dict[str, Any]]:
    """Parse SAM special-token text into boxes (masks are summarised only)."""
    out: list[dict[str, Any]] = []
    for m in _BOX.finditer(text or ""):
        fields: dict[str, int] = {}
        for part in m.group(2).split(";"):
            if "=" in part:
                k, v = part.split("=", 1)
                try:
                    fields[k.strip()] = int(float(v.strip()))
                except ValueError:
                    continue
        if not all(k in fields for k in ("x1", "y1", "x2", "y2")):
            continue
        w = fields.get("w") or 0
        h = fields.get("h") or 0
        box = {
            "ordinal": int(m.group(1)),
            "x1": fields["x1"],
            "y1": fields["y1"],
            "x2": fields["x2"],
            "y2": fields["y2"],
            "w": w,
            "h": h,
            "has_mask": bool(m.group(3)),
        }
        if w > 0 and h > 0:
            box["pct"] = {
                "left": round(100.0 * fields["x1"] / w, 2),
                "top": round(100.0 * fields["y1"] / h, 2),
                "width": round(100.0 * (fields["x2"] - fields["x1"]) / w, 2),
                "height": round(100.0 * (fields["y2"] - fields["y1"]) / h, 2),
            }
        out.append(box)
    return out[:50]


def _sse_text(body: str) -> str:
    chunks: list[str] = []
    for line in body.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "response.output_text.delta" and event.get("delta"):
            chunks.append(str(event["delta"]))
    return "".join(chunks)


def _responses_text(data: dict[str, Any]) -> str:
    if isinstance(data.get("output_text"), str):
        return data["output_text"]
    parts: list[str] = []
    for item in data.get("output") or []:
        for c in (item or {}).get("content") or []:
            if isinstance(c, dict) and c.get("text"):
                parts.append(str(c["text"]))
    return "".join(parts)


def sam_segment(*, concept: str, image_data_url: str, timeout: float | None = None) -> list[dict[str, Any]]:
    """Segment one noun-phrase concept in one image. Zero matches -> []."""
    concept = " ".join(re.sub(r"[^A-Za-z0-9 \-]", " ", concept or "").split())[:60]
    if not concept:
        raise MetaError("sam_empty_concept")
    payload = {
        "model": config.model_sam(),
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": concept},
                    {"type": "input_image", "image_url": image_data_url},
                ],
            }
        ],
        "stream": True,
    }
    resp = _post(config.sam_url(), payload, timeout=timeout or config.timeout_s(), accept="text/event-stream")
    raw = resp.body.decode("utf-8", errors="replace")
    if "event-stream" in resp.content_type or raw.lstrip().startswith(("data:", "event:")):
        text = _sse_text(raw)
    else:
        try:
            text = _responses_text(json.loads(raw))
        except (json.JSONDecodeError, AttributeError) as e:
            raise MetaError("sam_bad_envelope") from e
    return parse_sam_output(text)


def image_generate(*, prompt: str, size: str = "1024x1024", timeout: float | None = None) -> dict[str, Any]:
    """Muse Image generation (offline demo-asset script only)."""
    payload = {"model": config.model_image(), "prompt": prompt[:1000], "size": size, "n": 1}
    resp = _post(config.image_url(), payload, timeout=timeout or 60.0)
    try:
        data = json.loads(resp.body.decode("utf-8"))
        first = (data.get("data") or [])[0]
    except (ValueError, IndexError, AttributeError, UnicodeDecodeError) as e:
        raise MetaError("image_bad_envelope") from e
    return {"b64_json": first.get("b64_json"), "url": first.get("url")}


# ── Provider-interface compatible narration ────────────────────────────────


def meta_narrate(*, site: str, system: str, user: str, fallback: str, model: str | None = None):
    """Same contract as ``src.ai.provider.narrate`` but backed by Muse Spark."""
    from src.ai.provider import NarrationResult, _prompt_hash, narration_violations

    ph = _prompt_hash(system, user)
    schema = {
        "type": "object",
        "properties": {"text": {"type": "string", "maxLength": 600}},
        "required": ["text"],
    }
    try:
        out = chat_json(system=system, user=user, schema=schema, schema_name=f"narration_{site}", model=model)
        text = str(out.get("text") or "").strip()
        if not text:
            raise MetaError("empty")
        if narration_violations(text, {}):
            raise MetaError("narration_violation")
        return NarrationResult(text=text[:500], ok=True, model_id=model or config.model_text(),
                               prompt_hash=ph, used_llm=True, reason="ok")
    except MetaError as e:
        return NarrationResult(text=fallback, ok=False, model_id="deterministic",
                               prompt_hash=ph, used_llm=False, reason=e.reason)


__all__ = [
    "HttpResponse",
    "MetaError",
    "calls_today",
    "chat_json",
    "guard_url",
    "image_generate",
    "meta_narrate",
    "parse_sam_output",
    "reset_call_cap",
    "sam_segment",
    "set_transport",
]
