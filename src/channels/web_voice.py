"""Web voice channel — the browser Call Widget.

The browser does STT (Web Speech API) and TTS (`speechSynthesis`); the server
only sees text over a WebSocket. This adapter serializes orchestrator output
into the `/ws/interaction/{id}` JSON contract (see docs §8.2).

The adapter wraps a FastAPI WebSocket. It does NOT drive the orchestrator —
``src/api/routes/interactions.py`` (WS handlers on the app root) reads inbound
frames and calls the orchestrator.
"""

from __future__ import annotations

from typing import Any

from src.channels.base import ChannelAdapter, ChannelCapabilities


class WebVoiceChannel(ChannelAdapter):
    """Serialize orchestrator output to the widget's WebSocket JSON contract."""

    channel_name = "web_voice"

    def __init__(self, websocket: Any, *, text_only: bool = False) -> None:
        # `websocket` is a fastapi.WebSocket (or any object with an async
        # `send_json(dict)` method — keeps this testable without FastAPI).
        self._ws = websocket
        self._text_only = text_only
        if text_only:
            self.channel_name = "web_text"

    def capabilities(self) -> ChannelCapabilities:
        from src.voice.policy import silence_timeout_ms, tts_latency_budget_ms
        return ChannelCapabilities(
            voice=not self._text_only,
            text=True,
            barge_in=not self._text_only,
            server_tts=False,               # browser does TTS
            supervisor_takeover=True,
            dtmf=True,                      # widget offers keypad fallback
            asr_confidence=False,           # Web Speech gives no confidence → readback on re-ask
            silence_timeout_ms=silence_timeout_ms(),
            tts_budget_ms=tts_latency_budget_ms(),
        )

    async def _send(self, payload: dict[str, Any]) -> None:
        await self._ws.send_json(payload)

    async def send_control(self, payload: dict[str, Any]) -> None:
        await self._send(payload)

    async def send_turn(self, text: str, *, speaker: str = "agent", meta: dict[str, Any] | None = None) -> None:
        meta = meta or {}
        await self._send({
            "type": "agent_turn",
            "text": text,
            "turn_id": meta.get("turn_id"),
            "speak": not self._text_only,
            "speaker": speaker,
        })

    async def send_activity(self, payload: dict[str, Any]) -> None:
        await self._send({
            "type": "agent_activity",
            "agent": payload.get("agent"),
            "action_type": payload.get("action_type"),
            "summary": payload.get("summary", ""),
            "evidence_ids": payload.get("evidence_ids", []),
            "ok": payload.get("ok", True),
        })

    async def send_slots_update(self, slots: dict[str, str]) -> None:
        clean = {k: v for k, v in slots.items() if not k.startswith("__")}
        await self._send({"type": "slots_update", "slots": clean})

    async def send_handoff_offer(self) -> None:
        await self._send({"type": "handoff_offer"})

    async def send_interaction_ended(self, payload: dict[str, Any]) -> None:
        await self._send({
            "type": "interaction_ended",
            "case_id": payload.get("case_id"),
            "investigation_id": payload.get("investigation_id"),
            "investigation_opened": payload.get("investigation_opened", False),
            "audit_pending": payload.get("audit_pending", True),
        })

    async def hangup(self) -> None:
        # The gateway owns socket close; nothing extra to send.
        return None
