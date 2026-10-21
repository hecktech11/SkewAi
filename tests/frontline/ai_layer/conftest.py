"""Fixtures for the optional Meta AI layer tests.  [AI-LAYER]

Every network call is faked: ``FakeMeta`` replaces the adapter's HTTP
transport. No test talks to a real endpoint and no real key exists; the key
used here is an obviously fake placeholder.
"""

from __future__ import annotations

import json
from typing import Any, Callable

import pytest

FAKE_KEY = "test-placeholder-not-a-real-key"


class FakeMeta:
    """Routes fake responses by json_schema name (chat) or by model (SAM)."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.chat: dict[str, Callable[[dict[str, Any]], Any]] = {}
        self.sam: Callable[[dict[str, Any]], str] | None = None
        self.status_queue: list[int] = []
        self.raise_exc: Exception | None = None

    def __call__(self, url: str, body: bytes, headers: dict[str, str], timeout: float):
        from src.ai_layer.meta_client import HttpResponse

        payload = json.loads(body.decode("utf-8"))
        self.calls.append({"url": url, "payload": payload, "headers": dict(headers), "timeout": timeout})
        if self.raise_exc is not None:
            raise self.raise_exc
        if self.status_queue:
            status = self.status_queue.pop(0)
            if status != 200:
                return HttpResponse(status=status, body=b'{"error":"fake"}')
        if url.endswith("/responses"):
            text = self.sam(payload) if self.sam else ""
            events = [
                {"type": "response.output_text.delta", "delta": text},
                {"type": "response.completed"},
            ]
            raw = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
            return HttpResponse(status=200, body=raw.encode(), content_type="text/event-stream")
        name = (payload.get("response_format") or {}).get("json_schema", {}).get("name", "")
        handler = self.chat.get(name)
        if handler is None:
            return HttpResponse(status=500, body=b"{}")
        content = handler(payload)
        if not isinstance(content, str):
            content = json.dumps(content)
        env = {"choices": [{"index": 0, "message": {"role": "assistant", "content": content}}]}
        return HttpResponse(status=200, body=json.dumps(env).encode(), content_type="application/json")

    def chat_calls(self, name: str | None = None) -> list[dict[str, Any]]:
        out = [c for c in self.calls if not c["url"].endswith("/responses")]
        if name:
            out = [c for c in out if c["payload"]["response_format"]["json_schema"]["name"] == name]
        return out


@pytest.fixture
def no_meta(monkeypatch):
    """Deterministic mode, and any attempted network call fails the test."""
    from src.ai_layer import meta_client

    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    calls: list[str] = []

    def _boom(url, body, headers, timeout):
        calls.append(url)
        raise AssertionError("network call attempted in deterministic mode")

    meta_client.set_transport(_boom)
    yield calls
    meta_client.set_transport(None)


@pytest.fixture
def fake_meta(monkeypatch):
    from src.ai_layer import meta_client

    monkeypatch.setenv("META_API_KEY", FAKE_KEY)
    monkeypatch.delenv("FRONTLINE_AI_LAYER", raising=False)
    monkeypatch.setenv("META_RETRIES", "1")
    meta_client.reset_call_cap()
    monkeypatch.setattr("src.ai_layer.meta_client.time.sleep", lambda s: None)
    fake = FakeMeta()
    meta_client.set_transport(fake)
    yield fake
    meta_client.set_transport(None)
    meta_client.reset_call_cap()


def tiny_png(width: int = 4, height: int = 3) -> bytes:
    import struct
    import zlib

    raw = b"".join(b"\x00" + b"\xff\x00\x00" * width for _ in range(height))
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def tiny_jpeg(width: int = 16, height: int = 8) -> bytes:
    import struct

    sof = b"\xff\xc0" + struct.pack(">HBHHB", 11, 8, height, width, 1) + b"\x01\x11\x00"
    return b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00" + sof + b"\xff\xd9"
