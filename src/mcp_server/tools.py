"""MCP tool handlers. They call the same functions the HTTP API uses."""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

from src.config import REPO_ROOT
from src.security.identifiers import InvalidIdentifier, safe_pack_id, safe_token_id

_COUNTS = re.compile(
    r"Pack `([^`]+)`\. Hits (\d+)\. Misses (\d+)\. Unconfirmed flags (\d+)\."
)
_TOOLS = (
    "list_packs",
    "list_spikes",
    "list_investigations",
    "get_investigation_brief",
    "get_recall_lead_summary",
)


def tool_specs() -> list[dict[str, Any]]:
    return [
        {
            "name": "list_packs",
            "description": "List domain packs installed in this checkout.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "list_spikes",
            "description": "Current early-warning spikes for one pack. Empty means none were computed.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "pack_id": {
                        "type": "string",
                        "description": "Domain pack. Defaults to automotive_nhtsa.",
                    },
                },
            },
        },
        {
            "name": "list_investigations",
            "description": "Open and recent investigations for one pack.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "pack_id": {"type": "string"},
                    "status": {"type": "string"},
                    "limit": {"type": "integer"},
                    "window_days": {"type": "integer"},
                },
                "required": ["pack_id"],
            },
        },
        {
            "name": "get_investigation_brief",
            "description": "Investigation brief: slice, sources, and matched recall when one exists.",
            "inputSchema": {
                "type": "object",
                "properties": {"investigation_id": {"type": "string"}},
                "required": ["investigation_id"],
            },
        },
        {
            "name": "get_recall_lead_summary",
            "description": "Hits, misses, and unconfirmed flags from the last written lead-time report.",
            "inputSchema": {
                "type": "object",
                "properties": {"pack_id": {"type": "string"}},
                "required": ["pack_id"],
            },
        },
    ]


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def _bounded_int(value: Any, default: int, lo: int, hi: int, name: str) -> int:
    if value in (None, ""):
        return default
    try:
        n = int(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name} must be an integer") from e
    if n < lo or n > hi:
        raise ValueError(f"{name} out of range")
    return n


def recall_lead_summary(pack_id: str, report_dir: str | Path | None = None) -> dict[str, Any]:
    """Read a written markdown report. Does not rescore the warehouse."""
    pack = str(safe_pack_id(pack_id))
    root = Path(report_dir) if report_dir else (REPO_ROOT / "reports" / "nhtsa")
    root = root.resolve()
    empty = {
        "pack_id": pack,
        "report_path": None,
        "hits": None,
        "misses": None,
        "unconfirmed": None,
        "note": "no lead-time report for this pack",
    }
    if not root.is_dir():
        return empty
    matches: list[tuple[Path, re.Match[str]]] = []
    for path in sorted(root.glob("*.md")):
        resolved = path.resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        text = resolved.read_text(encoding="utf-8", errors="replace")
        found = _COUNTS.search(text)
        if found and found.group(1) == pack:
            matches.append((resolved, found))
    if not matches:
        return empty
    preferred = [item for item in matches if item[0].name == "recall_lead_time.md"]
    chosen, found = preferred[0] if preferred else matches[-1]
    return {
        "pack_id": pack,
        "report_path": str(chosen),
        "hits": int(found.group(2)),
        "misses": int(found.group(3)),
        "unconfirmed": int(found.group(4)),
    }


def call_tool(name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run one read tool. Raises LookupError, ValueError, or InvalidIdentifier."""
    if name not in _TOOLS:
        raise KeyError(name)
    args = arguments or {}
    if not isinstance(args, dict):
        raise ValueError("arguments must be an object")
    if name == "list_packs":
        from src.domains.loader import list_packs

        return {"packs": list(list_packs())}
    if name == "list_spikes":
        from src.ml_runtime.anomalies import current_spikes

        pack = str(safe_pack_id(str(args.get("pack_id") or "automotive_nhtsa")))
        return _plain(current_spikes(pack))
    if name == "list_investigations":
        from src.qubot.retrievers import investigation_status

        pack = str(safe_pack_id(str(args.get("pack_id") or "")))
        window = _bounded_int(args.get("window_days"), 90, 1, 3650, "window_days")
        limit = _bounded_int(args.get("limit"), 50, 1, 200, "limit")
        status = str(args.get("status") or "").strip()
        rows = []
        for row in investigation_status(window_days=window):
            if str(row.get("pack_id") or "") != pack:
                continue
            if status and str(row.get("status") or "") != status:
                continue
            rows.append(_plain(row))
            if len(rows) >= limit:
                break
        return {"pack_id": pack, "investigations": rows}
    if name == "get_investigation_brief":
        from src.frontline.investigation_brief import build_investigation_brief

        inv_id = safe_token_id(str(args.get("investigation_id") or ""), kind="investigation_id")
        return _plain(build_investigation_brief(inv_id))
    pack = str(safe_pack_id(str(args.get("pack_id") or "")))
    return recall_lead_summary(pack)


def tool_result_text(payload: dict[str, Any]) -> str:
    return json.dumps(payload, default=str)


__all__ = [
    "InvalidIdentifier",
    "call_tool",
    "recall_lead_summary",
    "tool_result_text",
    "tool_specs",
]
