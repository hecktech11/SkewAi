"""Twilio / SIP WebSocket telephony media stream bridge."""

from __future__ import annotations

import asyncio
import base64
import json
import struct
from typing import Any, Callable, Optional

from src.security.logging import get_logger

logger = get_logger("voice.telephony_bridge")

SERVICE_UNAVAILABLE = "service_unavailable"
SERVICE_UNAVAILABLE_DETAIL = (
    "operational STT or playable TTS including interruption handling is unavailable"
)


class TelephonyServiceUnavailable(RuntimeError):
    """Streaming conversation refused because STT, TTS, or barge-in is not operational."""

    code = SERVICE_UNAVAILABLE

    def __init__(self, detail: str = SERVICE_UNAVAILABLE_DETAIL) -> None:
        super().__init__(detail)
        self.detail = detail


def streaming_services_ready(
    speech: Any = None,
    *,
    interruption_connected: bool = False,
) -> bool:
    """True only when audio STT, playable TTS, and barge-in playback are all live.

    Decoded mu-law is not a conversation. The text-pilot STT shortcut does not
    count, and a ``tts://`` metadata ref is not playable audio.
    """
    if speech is None:
        from src.channels.stt_tts import ServerSpeechStack

        speech = ServerSpeechStack()
    return bool(
        speech.operational_audio_stt()
        and speech.playable_tts_available()
        and interruption_connected
    )


def service_unavailable_payload() -> dict[str, Any]:
    return {
        "type": "error",
        "code": SERVICE_UNAVAILABLE,
        "detail": SERVICE_UNAVAILABLE_DETAIL,
        "transcript": None,
        "audio": None,
    }

# Try audioop (Python <= 3.12 or audioop-lts), else table-driven G.711 codec
try:
    import audioop  # type: ignore
    _HAS_AUDIOOP = True
except ImportError:
    _HAS_AUDIOOP = False

# ITU-T G.711 mu-law lookup tables
_ULAW_TO_LIN_TABLE: list[int] = []
for i in range(256):
    u = ~i & 0xFF
    sign = u & 0x80
    exponent = (u >> 4) & 0x07
    mantissa = u & 0x0F
    sample = ((mantissa << 3) + 0x84) << exponent
    sample -= 0x84
    sample = -sample if sign else sample
    _ULAW_TO_LIN_TABLE.append(sample)


def _lin2ulaw_sample(sample: int) -> int:
    BIAS = 0x84
    CLIP = 32635
    sign = 0x80 if sample < 0 else 0x00
    if sample < 0:
        sample = -sample
    if sample > CLIP:
        sample = CLIP
    sample += BIAS
    exponent = 7
    for exp in range(7):
        if sample < (0x100 << exp):
            exponent = exp
            break
    mantissa = (sample >> (exponent + 3)) & 0x0F
    return ~(sign | (exponent << 4) | mantissa) & 0xFF


_LIN_TO_ULAW_TABLE: list[int] = [_lin2ulaw_sample(s - 32768) for s in range(65536)]


def ulaw8k_to_pcm16k(payload_mulaw: bytes) -> bytes:
    """Convert 8kHz G.711 mu-law bytes to 16kHz Linear PCM (16-bit little-endian)."""
    if not payload_mulaw:
        return b""
    if _HAS_AUDIOOP:
        pcm_8k = audioop.ulaw2lin(payload_mulaw, 2)
        pcm_16k, _ = audioop.ratecv(pcm_8k, 2, 1, 8000, 16000, None)
        return pcm_16k

    pcm_8k_samples = [_ULAW_TO_LIN_TABLE[b] for b in payload_mulaw]
    pcm_16k_samples: list[int] = []
    for s in pcm_8k_samples:
        pcm_16k_samples.extend((s, s))
    return struct.pack(f"<{len(pcm_16k_samples)}h", *pcm_16k_samples)


def ulaw8k_to_pcm16k_carrying(payload_mulaw: bytes, state: Any = None) -> tuple[bytes, Any]:
    """Convert one mu-law chunk and return the resampler state for the next chunk.

    Carrying ``state`` keeps chunk boundaries aligned with one whole-buffer
    conversion. ``ulaw8k_to_pcm16k`` still starts from a fresh state.
    """
    if not payload_mulaw:
        return b"", state
    if _HAS_AUDIOOP:
        pcm_8k = audioop.ulaw2lin(payload_mulaw, 2)
        pcm_16k, new_state = audioop.ratecv(pcm_8k, 2, 1, 8000, 16000, state)
        return pcm_16k, new_state
    return ulaw8k_to_pcm16k(payload_mulaw), None


def pcm16k_to_ulaw8k(pcm_16k_chunk: bytes) -> bytes:
    """Convert 16kHz Linear PCM (16-bit little-endian) to 8kHz G.711 mu-law bytes."""
    if not pcm_16k_chunk:
        return b""
    if _HAS_AUDIOOP:
        pcm_8k, _ = audioop.ratecv(pcm_16k_chunk, 2, 1, 16000, 8000, None)
        return audioop.lin2ulaw(pcm_8k, 2)

    n_samples = len(pcm_16k_chunk) // 2
    if n_samples == 0:
        return b""
    samples = struct.unpack(f"<{n_samples}h", pcm_16k_chunk[:n_samples * 2])
    samples_8k = samples[0::2]
    return bytes(_LIN_TO_ULAW_TABLE[s + 32768] for s in samples_8k)


def pcm16k_to_ulaw8k_carrying(pcm_16k_chunk: bytes, state: Any = None) -> tuple[bytes, Any]:
    """Downsample one PCM chunk and return the resampler state for the next chunk."""
    if not pcm_16k_chunk:
        return b"", state
    if _HAS_AUDIOOP:
        pcm_8k, new_state = audioop.ratecv(pcm_16k_chunk, 2, 1, 16000, 8000, state)
        return audioop.lin2ulaw(pcm_8k, 2), new_state
    return pcm16k_to_ulaw8k(pcm_16k_chunk), None


class TelephonyMediaBridge:
    """
    Bi-directional bridge between Twilio WebSocket Media Streams (8kHz mu-law)
    and internal high-fidelity audio pipelines (16kHz Linear PCM).
    """

    # Rejection payloads retain error text. Drop the oldest past this bound.
    REJECTION_LIMIT = 32

    def __init__(
        self,
        websocket: Any,
        interaction_id: str,
        on_customer_speech_frame: Callable[[bytes], Any],
        on_barge_in_detected: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.ws = websocket
        self.interaction_id = interaction_id
        self.on_speech_frame = on_customer_speech_frame
        self.on_barge_in = on_barge_in_detected
        self.stream_sid: Optional[str] = None
        self._is_active = True
        self._outbound_seq = 0
        # Playback is not connected to this loop, so barge-in cannot cut audio.
        self._interruption_connected = False
        self._inbound_rate_state: Any = None
        self._outbound_rate_state: Any = None
        self.rejections: list[dict[str, Any]] = []

    async def run(self) -> None:
        try:
            while self._is_active:
                message_raw = await self.ws.receive_text()
                data = json.loads(message_raw)
                event_type = data.get("event")

                if event_type == "start":
                    self.stream_sid = data.get("start", {}).get("streamSid")
                    logger.info("twilio_stream_started", stream_sid=self.stream_sid)

                elif event_type == "media":
                    payload_mulaw = base64.b64decode(data.get("media", {}).get("payload", ""))
                    pcm_16k, self._inbound_rate_state = ulaw8k_to_pcm16k_carrying(
                        payload_mulaw, self._inbound_rate_state
                    )
                    if not pcm_16k:
                        continue
                    # Fail closed. Do not hand decoded audio to a text stub,
                    # and do not call the speech callback.
                    if not streaming_services_ready(
                        interruption_connected=self._interruption_connected,
                    ):
                        await self._reject_service_unavailable()
                        continue
                    res = self.on_speech_frame(pcm_16k)
                    if asyncio.iscoroutine(res):
                        await res

                elif event_type == "stop":
                    logger.info("twilio_stream_stopped", stream_sid=self.stream_sid)
                    self._is_active = False

        except Exception as e:
            logger.warn("twilio_ws_disconnected", interaction_id=self.interaction_id, error=str(e))
            self._is_active = False

    async def _reject_service_unavailable(self) -> None:
        payload = service_unavailable_payload()
        self.rejections.append(payload)
        overflow = len(self.rejections) - self.REJECTION_LIMIT
        if overflow > 0:
            del self.rejections[:overflow]
        raw = json.dumps(payload)
        send = getattr(self.ws, "send_text", None)
        if send is None:
            raise TelephonyServiceUnavailable(payload["detail"])
        await send(raw)

    async def emit_audio_chunk(self, pcm_16k_chunk: bytes) -> None:
        """
        Converts agent synthesized 16kHz PCM to 8kHz mu-law and transmits to Twilio.
        """
        if not self.stream_sid or not self._is_active:
            return

        mulaw_payload, self._outbound_rate_state = pcm16k_to_ulaw8k_carrying(
            pcm_16k_chunk, self._outbound_rate_state
        )
        base64_audio = base64.b64encode(mulaw_payload).decode("utf-8")

        payload = {
            "event": "media",
            "streamSid": self.stream_sid,
            "media": {"payload": base64_audio},
        }
        await self.ws.send_text(json.dumps(payload))
        self._outbound_seq += 1

    async def clear_playback_buffer(self) -> None:
        """
        Immediate Twilio buffer clearance to execute hard interruption/barge-in.
        """
        if not self.stream_sid:
            return
        clear_message = {
            "event": "clear",
            "streamSid": self.stream_sid,
        }
        await self.ws.send_text(json.dumps(clear_message))
        logger.info("twilio_playback_buffer_flushed", stream_sid=self.stream_sid)
