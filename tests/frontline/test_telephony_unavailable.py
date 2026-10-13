"""Twilio streaming conversation fails closed when speech service is down.

Drives the production bridge the ``/ws/twilio`` route runs. Decoded audio
must be a rejection, not a transcript, playable audio, or a swallowed no-op.
"""

from __future__ import annotations

import base64
import json

import pytest

from src.channels.twilio_media import TwilioMediaChannel
from src.voice.telephony_bridge import (
    TelephonyServiceUnavailable,
    pcm16k_to_ulaw8k,
)
from src.voice.wiring import telephony_bridge_for


class _ScriptedSocket:
    def __init__(self, frames: list[str]) -> None:
        self._frames = list(frames)
        self.sent: list[str] = []

    async def receive_text(self) -> str:
        if not self._frames:
            raise RuntimeError("socket closed")
        return self._frames.pop(0)

    async def send_text(self, text: str) -> None:
        self.sent.append(text)


@pytest.mark.asyncio
async def test_streaming_conversation_rejects_unavailable_service(monkeypatch):
    monkeypatch.delenv("STT_API_URL", raising=False)
    monkeypatch.delenv("FASTER_WHISPER_MODEL", raising=False)
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)

    pcm = b"\x10\x00" * 320
    mulaw = pcm16k_to_ulaw8k(pcm)
    assert mulaw, "decoded carrier audio must be non-empty for this turn"
    payload = base64.b64encode(mulaw).decode("ascii")
    heard: list[bytes] = []

    async def on_speech(frame: bytes) -> None:
        heard.append(frame)

    ws = _ScriptedSocket([
        json.dumps({"event": "start", "start": {"streamSid": "MZ_REJECT"}}),
        json.dumps({"event": "media", "media": {"payload": payload}}),
        json.dumps({"event": "stop"}),
    ])
    bridge = telephony_bridge_for(ws, "int_stream_reject", on_speech, lambda: None)
    await bridge.run()

    assert heard == []
    assert bridge.rejections
    body = json.loads(ws.sent[-1])
    assert body["type"] == "error"
    assert body["code"] == "service_unavailable"
    assert body["transcript"] is None
    assert body["audio"] is None
    assert "unavailable" in body["detail"].lower()
    assert "stt" in body["detail"].lower()

    channel = TwilioMediaChannel()
    with pytest.raises(TelephonyServiceUnavailable):
        channel.ingest_media_frame({"event": "media", "media": {"payload": payload}})
    assert channel.ingest_media_frame(
        {"type": "text", "text": "customer said brakes"}
    ) == "customer said brakes"
