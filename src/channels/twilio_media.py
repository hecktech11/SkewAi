"""Twilio Media Streams adapter — pilot-working path without live carrier.

When ``TWILIO_ACCOUNT_SID`` + ``TWILIO_AUTH_TOKEN`` are unset, the adapter
runs in **text pilot mode**: it accepts the same ChannelAdapter contract as
WebVoice, records outbound turns, and never raises. When credentials exist,
``stream_url()`` and ``media_frame_handlers`` are ready for Media Streams
wiring; STT/TTS are pluggable via ``src.channels.stt_tts``.

This replaces the old NotImplementedError stub so feature #8 is shippable
at pilot depth (no live PSTN required for CI).
"""

from __future__ import annotations

import os
from typing import Any, Callable, Awaitable

from src.channels.base import ChannelAdapter, ChannelCapabilities
from src.channels.stt_tts import ServerSpeechStack


def carrier_audio_ready() -> bool:
    """The registered Twilio route is not a completed live carrier conversation."""
    return False


class TwilioMediaChannel(ChannelAdapter):
    """Telephony channel adapter with deterministic pilot fallback."""

    channel_name = "twilio_media"
    # Outbound turn payloads retain transcript text. Drop the oldest past this bound.
    OUTBOUND_LIMIT = 64

    def __init__(
        self,
        send_json: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
        *,
        account_sid: str | None = None,
        auth_token: str | None = None,
        speech: ServerSpeechStack | None = None,
    ) -> None:
        self._send = send_json
        self._account_sid = (account_sid or os.getenv("TWILIO_ACCOUNT_SID", "")).strip()
        self._auth_token = (auth_token or os.getenv("TWILIO_AUTH_TOKEN", "")).strip()
        self._speech = speech or ServerSpeechStack()
        self.outbound: list[dict[str, Any]] = []
        self._closed = False

    @property
    def live_credentials(self) -> bool:
        return bool(self._account_sid and self._auth_token)

    def capabilities(self) -> ChannelCapabilities:
        from src.voice.policy import silence_timeout_ms, tts_latency_budget_ms
        return ChannelCapabilities(
            voice=True,
            text=True,
            barge_in=True,
            server_tts=self._speech.playable_tts_available(),
            supervisor_takeover=True,
            dtmf=True,
            asr_confidence=True,
            silence_timeout_ms=silence_timeout_ms(),
            tts_budget_ms=tts_latency_budget_ms(),
        )

    def stream_url(self, public_base: str, interaction_id: str) -> str:
        """Twilio Media Streams WSS target (document for ops; not dialed in CI)."""
        base = public_base.rstrip("/")
        return f"{base.replace('https://', 'wss://').replace('http://', 'ws://')}/ws/twilio/{interaction_id}"

    async def _emit(self, payload: dict[str, Any]) -> None:
        self.outbound.append(payload)
        overflow = len(self.outbound) - self.OUTBOUND_LIMIT
        if overflow > 0:
            del self.outbound[:overflow]
        if self._send is not None:
            await self._send(payload)

    async def send_turn(
        self, text: str, *, speaker: str = "agent", meta: dict[str, Any] | None = None
    ) -> None:
        meta = meta or {}
        playable = self._speech.playable_tts_available()
        audio_meta = self._speech.synthesize_meta(text) if playable else {}
        await self._emit(
            {
                "type": "agent_turn",
                "channel": self.channel_name,
                "text": text,
                "speaker": speaker,
                "turn_id": meta.get("turn_id"),
                "server_tts": playable,
                "tts": audio_meta,
                "live_twilio": self.live_credentials,
            }
        )

    async def send_activity(self, payload: dict[str, Any]) -> None:
        await self._emit({"type": "agent_activity", "channel": self.channel_name, **payload})

    async def send_slots_update(self, slots: dict[str, str]) -> None:
        clean = {k: v for k, v in slots.items() if not k.startswith("__")}
        await self._emit({"type": "slots_update", "slots": clean})

    async def send_handoff_offer(self) -> None:
        await self._emit({"type": "handoff_offer"})

    async def send_interaction_ended(self, payload: dict[str, Any]) -> None:
        await self._emit({"type": "interaction_ended", **payload})

    async def send_clear_stream(self, stream_sid: str) -> None:
        """Twilio Media Streams: cancel buffered audio packets on barge-in (audit 2.4)."""
        await self._emit({"event": "clear", "streamSid": stream_sid})

    def generate_transfer_twiml(
        self,
        target: str,
        *,
        method: str = "sip",
        caller_id: str | None = None,
    ) -> str:
        """Generate compliant TwiML for warm transfer via SIP REFER or Conference (audit 2.4)."""
        from xml.sax.saxutils import escape as _esc, quoteattr as _qa

        safe_target = _esc(str(target or ""), {'"': "&quot;", "'": "&apos;"})
        cid_attr = f" callerId={_qa(str(caller_id))}" if caller_id else ""
        if method.lower() == "conference":
            return (
                '<?xml version="1.0" encoding="UTF-8"?>'
                f'<Response><Dial{cid_attr}><Conference>{safe_target}</Conference></Dial></Response>'
            )
        elif method.lower() == "sip":
            sip_uri = safe_target if safe_target.startswith("sip:") else f"sip:{safe_target}"
            return (
                '<?xml version="1.0" encoding="UTF-8"?>'
                f'<Response><Dial{cid_attr}><Sip>{sip_uri}</Sip></Dial></Response>'
            )
        else:
            return (
                '<?xml version="1.0" encoding="UTF-8"?>'
                f'<Response><Dial{cid_attr}><Number>{safe_target}</Number></Dial></Response>'
            )

    async def hangup(self) -> None:
        self._closed = True
        await self._emit({"type": "hangup", "channel": self.channel_name})

    def ingest_media_frame(self, frame: dict[str, Any]) -> str | None:
        """Convert a media/STT frame into customer text (None if partial).

        Audio media frames are not a conversation unless operational STT,
        playable TTS, and interruption handling are all connected. The text
        pilot path (``type=text``) is unchanged.
        """
        if frame.get("event") == "media" and frame.get("media", {}).get("payload"):
            from src.voice.telephony_bridge import (
                TelephonyServiceUnavailable,
                streaming_services_ready,
            )

            if not streaming_services_ready(self._speech, interruption_connected=False):
                raise TelephonyServiceUnavailable()
            return self._speech.transcribe_pcm(b"")
        if frame.get("event") == "mark" or frame.get("type") == "text":
            return str(frame.get("text") or frame.get("transcript") or "") or None
        return None


# Back-compat alias used by older imports
TwilioChannel = TwilioMediaChannel
