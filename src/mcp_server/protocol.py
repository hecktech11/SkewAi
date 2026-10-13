"""Minimal MCP stdio server (Content-Length JSON-RPC, read tools only)."""

from __future__ import annotations

import json
import os
from typing import Any, BinaryIO

from src.mcp_server.tools import call_tool, tool_result_text, tool_specs
from src.security.identifiers import InvalidIdentifier

CLOSED = (
    "Skew MCP is closed. For a local demo leave FRONTLINE_API_KEY empty "
    "or set FRONTLINE_OPEN_MODE=1. When auth is required, set FRONTLINE_API_KEY "
    "in this process."
)


def mcp_open() -> bool:
    """Same gate as the API: open mode, or this process holds the API key."""
    from src.api.auth import is_open_mode

    if is_open_mode():
        return True
    return bool(os.getenv("FRONTLINE_API_KEY", "").strip())


def _error(msg_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def _tool_error(msg_id: Any, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": msg_id,
        "result": {
            "content": [{"type": "text", "text": message}],
            "isError": True,
        },
    }


def handle_message(msg: Any) -> dict[str, Any] | None:
    """One JSON-RPC message. Notifications return None (no response)."""
    if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0" or "method" not in msg:
        mid = msg.get("id") if isinstance(msg, dict) else None
        return _error(mid, -32600, "invalid request")
    method = str(msg.get("method") or "")
    mid = msg.get("id")
    note = "id" not in msg
    if method == "notifications/initialized":
        return None
    if method == "ping":
        return {"jsonrpc": "2.0", "id": mid, "result": {}}
    if method == "initialize":
        params = msg.get("params") if isinstance(msg.get("params"), dict) else {}
        version = str(params.get("protocolVersion") or "2024-11-05")
        return {
            "jsonrpc": "2.0",
            "id": mid,
            "result": {
                "protocolVersion": version,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "skew-ai", "version": "0.1.0"},
            },
        }
    if method in {"tools/list", "tools/call"} and not mcp_open():
        return _error(mid, -32001, CLOSED)
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": mid, "result": {"tools": tool_specs()}}
    if method == "tools/call":
        params = msg.get("params") if isinstance(msg.get("params"), dict) else {}
        name = str(params.get("name") or "")
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            return _error(mid, -32602, "arguments must be an object")
        try:
            payload = call_tool(name, arguments)
        except KeyError:
            return _tool_error(mid, f"unknown tool {name}")
        except (InvalidIdentifier, LookupError, ValueError, FileNotFoundError) as e:
            return _tool_error(mid, str(e))
        return {
            "jsonrpc": "2.0",
            "id": mid,
            "result": {
                "content": [{"type": "text", "text": tool_result_text(payload)}],
                "isError": False,
            },
        }
    if note:
        return None
    return _error(mid, -32601, f"method not found: {method}")


def read_message(stream: BinaryIO) -> dict[str, Any] | None:
    headers: dict[str, str] = {}
    while True:
        line = stream.readline()
        if not line:
            return None
        if line in (b"\r\n", b"\n"):
            break
        if b":" not in line:
            continue
        key, value = line.split(b":", 1)
        headers[key.decode("ascii", "replace").strip().lower()] = value.decode("ascii", "replace").strip()
    try:
        length = int(headers.get("content-length") or "0")
    except ValueError:
        return None
    if length <= 0:
        return None
    body = stream.read(length)
    if len(body) < length:
        return None
    return json.loads(body.decode("utf-8"))


def write_message(stream: BinaryIO, payload: dict[str, Any]) -> None:
    raw = json.dumps(payload).encode("utf-8")
    stream.write(f"Content-Length: {len(raw)}\r\n\r\n".encode("ascii"))
    stream.write(raw)
    stream.flush()


def serve(stdin: BinaryIO, stdout: BinaryIO) -> None:
    while True:
        try:
            msg = read_message(stdin)
        except json.JSONDecodeError:
            write_message(stdout, _error(None, -32700, "parse error"))
            continue
        if msg is None:
            return
        reply = handle_message(msg)
        if reply is not None:
            write_message(stdout, reply)


__all__ = ["CLOSED", "handle_message", "mcp_open", "read_message", "serve", "write_message"]
