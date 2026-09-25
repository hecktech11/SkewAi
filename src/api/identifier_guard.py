"""ASGI gate: reject ids that cannot be a path or DuckDB filename."""

from __future__ import annotations

import re
from urllib.parse import parse_qs

from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from src.security.identifiers import (
    InvalidIdentifier,
    safe_cluster_id,
    safe_pack_id,
    safe_token_id,
)

_PACK_PATH = re.compile(r"/packs/([^/]+)")
_WS_INTERACTION = re.compile(r"^/ws/interaction/([^/]+)")
_HTTP_INTERACTION = re.compile(r"/interactions/([^/]+)")
_INVESTIGATION = re.compile(r"/investigations/([^/]+)")
_CLUSTER = re.compile(r"/clusters/([^/]+)")
_AUDIT = re.compile(r"/audits/([^/?#]+)")
_CASE = re.compile(r"/cases/([^/?#]+)")
_CASE_STATUS = re.compile(r"/cases/status/([^/?#]+)")
_DSR = re.compile(r"/dsr/([^/?#]+)")


class IdentifierGuardASGI:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        path = scope.get("path") or ""
        try:
            qs = (scope.get("query_string") or b"").decode("latin-1")
            params = parse_qs(qs, keep_blank_values=True)
            if "pack_id" in params and params["pack_id"]:
                raw = params["pack_id"][0]
                if raw:
                    safe_pack_id(raw)
            m = _PACK_PATH.search(path)
            if m and m.group(1) not in {"active", ""}:
                safe_pack_id(m.group(1))
            inv = _INVESTIGATION.search(path)
            if inv:
                safe_token_id(inv.group(1), kind="investigation_id")
            aud = _AUDIT.search(path)
            if aud and aud.group(1) not in {"export", ""}:
                safe_token_id(aud.group(1), kind="interaction_id")
            case = _CASE.search(path)
            if case and case.group(1) not in {"export", "status", ""}:
                safe_token_id(case.group(1), kind="case_id")
            # /cases/status/{case_id} — first segment is "status", validate trailing id
            _m_status = _CASE_STATUS.search(path)
            if _m_status:
                safe_token_id(_m_status.group(1), kind="case_id")
            dsr = _DSR.search(path)
            if dsr:
                safe_token_id(dsr.group(1), kind="interaction_id")
            cl = _CLUSTER.search(path)
            if cl and cl.group(1) not in {"rebuild", ""}:
                safe_cluster_id(cl.group(1))
            if scope["type"] == "websocket":
                wm = _WS_INTERACTION.match(path)
                if wm:
                    safe_token_id(wm.group(1), kind="interaction_id")
            else:
                im = _HTTP_INTERACTION.search(path)
                if im and im.group(1) not in {"", "start", "ingest"}:
                    safe_token_id(im.group(1), kind="interaction_id")
        except InvalidIdentifier as e:
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1008})
                return
            response = JSONResponse({"detail": str(e)}, status_code=400)
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)
