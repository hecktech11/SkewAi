"""Server-side STT/TTS option (feature #9).

Browser Web Speech is Chrome-only. This stack provides a hermetic text pilot:
- A short ASCII payload is returned as text. It is not audio transcription.
- Decoded PCM is transcribed only when a PCM transcriber is attached.
- ``synthesize_audio`` returns playable bytes only when a synthesizer is
  attached. Metadata alone is not playable TTS.

``STT_API_URL`` and ``FASTER_WHISPER_MODEL`` do not by themselves make
audio STT or playable TTS available.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any


class ServerSpeechStack:
    """Pluggable STT/TTS with deterministic offline behavior."""

    # Partial audio/text fragments. Drop the oldest past this bound.
    PARTIAL_LIMIT = 32

    def __init__(
        self,
        *,
        stt_api_url: str | None = None,
        tts_voice: str | None = None,
    ) -> None:
        self.stt_api_url = (stt_api_url or os.getenv("STT_API_URL", "")).strip()
        self.tts_voice = (tts_voice or os.getenv("TTS_VOICE", "pilot-neutral")).strip()
        self._partial: list[str] = []

    def _remember_partial(self, raw: str) -> None:
        self._partial.append(raw)
        overflow = len(self._partial) - self.PARTIAL_LIMIT
        if overflow > 0:
            del self._partial[:overflow]

    @property
    def server_stt_available(self) -> bool:
        """True only when decoded PCM is handed to a real transcriber."""
        return self.operational_audio_stt()

    def transcribe_chunk(self, payload_b64_or_text: str) -> str | None:
        """Return finalized text or None if still accumulating.

        Pilot rule: if payload looks like plain text (not base64-ish long audio),
        return it immediately. Otherwise accumulate and return a hash-tag stub
        only when a terminal marker is present.
        """
        raw = payload_b64_or_text or ""
        if not raw:
            return None
        # Short ASCII → treat as already-transcribed (test / pilot)
        if len(raw) < 500 and all(ord(c) < 128 for c in raw[:80]):
            if raw.startswith("AUDIO:"):
                self._remember_partial(raw)
                return None
            return raw.strip()
        self._remember_partial(raw[:64])
        return None

    def finalize_partial(self) -> str:
        joined = "".join(self._partial)
        self._partial.clear()
        if not joined:
            return ""
        digest = hashlib.sha256(joined.encode()).hexdigest()[:12]
        return f"[stt_pending:{digest}]"

    def operational_audio_stt(self) -> bool:
        """True only when decoded PCM is handed to a real transcriber.

        ``STT_API_URL`` / ``FASTER_WHISPER_MODEL`` and the short-ASCII text
        pilot are not operational audio STT. No carrier call is made here.
        """
        return callable(getattr(self, "_pcm_transcriber", None))

    def transcribe_pcm(self, pcm: bytes) -> str | None:
        """Transcribe decoded PCM. Never invents text from audio bytes."""
        if not pcm or not self.operational_audio_stt():
            return None
        text = self._pcm_transcriber(pcm)  # type: ignore[attr-defined]
        if not isinstance(text, str) or not text.strip():
            return None
        return text.strip()

    def synthesize_audio(self, text: str) -> bytes | None:
        """Playable audio bytes, or None when TTS is metadata only."""
        return None

    def playable_tts_available(self) -> bool:
        """True only when synthesis returns audio a caller can play."""
        sample = self.synthesize_audio(" ")
        return isinstance(sample, (bytes, bytearray)) and len(sample) > 0

    def synthesize_meta(self, text: str) -> dict[str, Any]:
        """Return TTS metadata. ``audio_ref`` is set only when audio is playable."""
        playable = self.playable_tts_available()
        digest = hashlib.sha256((text or "").encode()).hexdigest()[:16]
        return {
            "provider": "pilot" if not os.getenv("ELEVENLABS_API_KEY") else "elevenlabs",
            "voice": self.tts_voice,
            "text_chars": len(text or ""),
            "audio_ref": f"tts://{digest}" if playable else None,
            "server_side": playable,
            "playable": playable,
        }

    def capabilities(self) -> dict[str, Any]:
        playable = self.playable_tts_available()
        return {
            "server_stt": self.operational_audio_stt(),
            "server_tts": playable,
            "offline_pilot": not self.operational_audio_stt(),
        }
