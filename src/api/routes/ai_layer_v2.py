"""AI-layer v2 routes: Meta call health + supervisor review of AI raises.  [AI-LAYER]

  GET  /api/ai/call-health                    -> per call type calls / ok / failed / fallback / latency
  GET  /api/ai/status/overview                -> /api/ai/status + call health + review stats (status page)
  GET  /api/ai/raises                         -> AI severity raises with their verdicts (case:read)
  GET  /api/ai/raises/stats                   -> agreement rate (no rate when n < 20)
  POST /api/ai/raises/{event_id}/review       -> supervisor-only accept / reject (+ note)
  GET  /api/ai/raises/audit/verify            -> verify the review audit hash chain (audit:read)

Mounted from ``src/api/routes/ai_layer.py`` by a one-line ``[AI-LAYER]``
include. A verdict never changes case severity.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.api.auth import require_api_key
from src.api.rbac import get_actor, get_role, require_perm_dep
from src.data.warehouse import ops_in_thread

router = APIRouter(tags=["ai-layer"], dependencies=[Depends(require_api_key)])

REVIEWER_ROLES = frozenset({"supervisor"})


class ReviewBody(BaseModel):
    verdict: str = Field(..., description="accept | reject")
    note: str | None = Field(default=None, max_length=2000)


@router.get("/api/ai/call-health")
async def call_health(
    _role: str = Depends(require_perm_dep("case:read", open_mode_ok=True)),
) -> dict[str, Any]:
    from src.ai_layer.call_health import summary

    return await ops_in_thread(summary)


@router.get("/api/ai/status/overview")
async def status_overview(
    _role: str = Depends(require_perm_dep("case:read", open_mode_ok=True)),
) -> dict[str, Any]:
    from src.ai_layer.call_health import summary
    from src.ai_layer.config import ai_status
    from src.ai_layer.meta_client import calls_today
    from src.ai_layer.raise_review import stats

    status = ai_status()
    status["calls_today"] = calls_today()
    return {
        "status": status,
        "call_health": await ops_in_thread(summary),
        "raise_reviews": await ops_in_thread(stats),
    }


@router.get("/api/ai/raises")
async def list_ai_raises(
    limit: int = 100,
    _role: str = Depends(require_perm_dep("case:read", open_mode_ok=True)),
) -> dict[str, Any]:
    from src.ai_layer.raise_review import list_raises

    rows = await ops_in_thread(list_raises, max(1, min(int(limit), 500)))
    return {"raises": rows, "count": len(rows)}


@router.get("/api/ai/raises/stats")
async def raise_stats(
    _role: str = Depends(require_perm_dep("case:read", open_mode_ok=True)),
) -> dict[str, Any]:
    from src.ai_layer.raise_review import stats

    return await ops_in_thread(stats)


@router.post("/api/ai/raises/{event_id}/review")
async def review_raise(
    event_id: str,
    body: ReviewBody,
    role: str = Depends(get_role),
    actor: str = Depends(get_actor),
) -> dict[str, Any]:
    from src.ai_layer.raise_review import ReviewError, record_denied, submit_review
    from src.security.identifiers import safe_token_id

    try:
        safe_token_id(event_id, kind="event_id")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    # Supervisor only; open mode never bypasses this (unsigned callers are 'agent').
    if role not in REVIEWER_ROLES:
        await ops_in_thread(record_denied, event_id=event_id, actor=str(actor or "unknown"), actor_role=str(role))
        raise HTTPException(status_code=403, detail=f"role {role} cannot review AI raises (supervisor only)")
    try:
        return await ops_in_thread(submit_review, event_id=event_id, verdict=body.verdict, note=body.note,
                                   reviewer=str(actor or "supervisor"), reviewer_role=role)
    except ReviewError as e:
        raise HTTPException(status_code=e.status, detail=e.detail) from e


@router.get("/api/ai/raises/audit/verify")
async def verify_review_audit(
    _role: str = Depends(require_perm_dep("audit:read")),
) -> dict[str, Any]:
    from src.ai_layer.raise_review import verify_audit_chain

    return await ops_in_thread(verify_audit_chain)
