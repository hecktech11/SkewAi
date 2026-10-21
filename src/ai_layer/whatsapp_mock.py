"""WhatsApp intake — MOCK.  [AI-LAYER]

NO WhatsApp / Meta Cloud API is called and nothing leaves the box. This module
accepts a webhook body shaped like the WhatsApp Cloud API ``messages`` event,
feeds each text message into the EXISTING intake orchestrator
(``create_interaction`` + ``handle_customer_turn``) and returns the agent's
replies in the response so the mock chat UI can render them. Kill-switch,
safety assessment, Qubot and audit run exactly as for any other contact.

Sessions are held in process memory keyed by the sender id; the sender id is
stored only as a sha256 ``customer_ref`` (never raw).
"""

from __future__ import annotations

import asyncio
import hashlib
from collections import OrderedDict
from typing import Any

MOCK_LABEL = "MOCK — simulated WhatsApp channel; no WhatsApp/Meta API is called and no message is sent"
CHANNEL = "whatsapp_mock"
_TERMINAL = ("DONE", "ABANDONED")
_MAX_TEXT = 2000

_sessions: dict[str, dict[str, Any]] = {}
_locks: dict[str, asyncio.Lock] = {}
_seen: "OrderedDict[str, None]" = OrderedDict()


def reset_sessions() -> None:
    _sessions.clear(); _locks.clear(); _seen.clear()


def parse_webhook(body: Any) -> list[dict[str, Any]]:
    """Extract inbound messages from a Cloud-API-shaped body. Unknown shapes -> []."""
    out: list[dict[str, Any]] = []
    if not isinstance(body, dict):
        return out
    for entry in body.get("entry") or []:
        for change in (entry or {}).get("changes") or []:
            value = (change or {}).get("value") or {}
            names = {c.get("wa_id"): ((c.get("profile") or {}).get("name")) for c in value.get("contacts") or [] if isinstance(c, dict)}
            for m in value.get("messages") or []:
                if not isinstance(m, dict) or not m.get("from"):
                    continue
                text = ((m.get("text") or {}).get("body")) if m.get("type") == "text" else None
                out.append({"from": str(m["from"])[:32], "id": str(m.get("id") or ""), "type": m.get("type"),
                            "text": (text or "")[:_MAX_TEXT] if text is not None else None, "name": names.get(m.get("from"))})
    return out


def _ref(wa_id: str) -> str:
    return hashlib.sha256(f"whatsapp:{wa_id}".encode()).hexdigest()


def _reply(wa_id: str, text: str) -> dict[str, Any]:
    return {"messaging_product": "whatsapp", "to": wa_id, "type": "text", "text": {"body": text}, "status": "mock_not_sent"}


async def handle_message(msg: dict[str, Any], *, pack_id: str | None = None) -> dict[str, Any]:
    from src.agents.orchestrator import OrchestratorHooks, create_interaction

    wa_id = msg["from"]
    if msg.get("id"):
        if msg["id"] in _seen:
            return {"from": wa_id, "duplicate": True, "replies": []}
        _seen[msg["id"]] = None
        while len(_seen) > 5000:
            _seen.popitem(last=False)
    if msg.get("text") is None:
        return {"from": wa_id, "replies": [_reply(wa_id, "This mock only accepts text messages. Please describe the problem in words.")]}
    lock = _locks.setdefault(wa_id, asyncio.Lock())
    async with lock:
        s = _sessions.get(wa_id)
        if s is None or s["orch"].ctx.state in _TERMINAL:
            outbox: list[str] = []

            async def _said(text: str, meta: dict[str, Any], _o=outbox) -> None:
                if text:
                    _o.append(str(text))

            orch, greeting = await create_interaction(channel=CHANNEL, pack_id=pack_id, customer_ref=_ref(wa_id),
                                                      hooks=OrchestratorHooks(emit_customer_turn=_said))
            s = {"orch": orch, "outbox": outbox}
            _sessions[wa_id] = s
            if greeting and greeting not in outbox:
                outbox.append(str(greeting))
        start = 0 if not s.get("started") else len(s["outbox"])
        s["started"] = True
        await s["orch"].handle_customer_turn(msg["text"])
        ctx = s["orch"].ctx
        new = s["outbox"][start:]
        if ctx.state in _TERMINAL:
            _sessions.pop(wa_id, None)
        return {"from": wa_id, "interaction_id": ctx.interaction_id, "state": ctx.state, "case_id": ctx.case_id,
                "severity": ctx.severity, "safety_flags": sorted(k for k, v in (ctx.safety_flags or {}).items() if v),
                "replies": [_reply(wa_id, t) for t in new]}


async def handle_webhook(body: Any, *, pack_id: str | None = None) -> dict[str, Any]:
    msgs = parse_webhook(body)
    results = [await handle_message(m, pack_id=pack_id) for m in msgs]
    return {"mock": True, "label": MOCK_LABEL, "received": len(msgs), "results": results}


__all__ = ["CHANNEL", "MOCK_LABEL", "handle_message", "handle_webhook", "parse_webhook", "reset_sessions"]
