"""Enterprise capability APIs — timeline, root-cause, copilot, risk, graph, memory, scenarios, decision flow."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.auth import require_api_key
from src.api.jsonutil import json_safe
from src.api.rbac import require_perm_dep

router = APIRouter(
    prefix="/api/frontline/enterprise",
    tags=["enterprise"],
    dependencies=[Depends(require_api_key)],
)


# ── 0. Recent interactions (picker for timeline / RCA / decision) ────────────


@router.get("/interactions/recent")
async def recent_interactions(
    limit: int = Query(default=25, ge=1, le=100),
    status: str | None = None,
) -> dict[str, Any]:
    """Newest interactions for Enterprise Ops pickers (additive catalog)."""
    from src.enterprise.catalog import list_recent_interactions

    rows = list_recent_interactions(limit=limit, status=status)
    return {"interactions": rows, "count": len(rows)}


# ── 1. Incident Timeline ─────────────────────────────────────────────────────


@router.get("/timeline/{interaction_id}")
async def get_timeline(
    interaction_id: str,
    scrub_pii: bool = Query(default=True),
    _role: str = Depends(require_perm_dep("ledger:read", open_mode_ok=True)),
) -> dict[str, Any]:
    from src.api.rbac import require_perm
    from src.enterprise.timeline import build_incident_timeline
    from src.security.pii import redact_dict

    try:
        data = build_incident_timeline(interaction_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    if not scrub_pii:
        require_perm(_role, "dsr:export")
        return json_safe(data)
    return json_safe(redact_dict(data))


# ── 2. Root-cause explorer ───────────────────────────────────────────────────


@router.get("/root-cause/{interaction_id}")
async def get_root_cause(interaction_id: str) -> dict[str, Any]:
    from src.enterprise.root_cause import analyze_root_cause

    try:
        return json_safe(analyze_root_cause(interaction_id))
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.get("/root-cause")
async def list_root_causes(
    limit: int = Query(default=25, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    cursor: str | None = None,
) -> dict[str, Any]:
    from src.api.pagination import clamp_limit, page_meta, resolve_offset
    from src.enterprise.root_cause import list_failure_postmortems

    lim = clamp_limit(limit, default=25, max_limit=100)
    off = resolve_offset(offset=offset, cursor=cursor)
    # Fetch enough for offset+page, then slice (postmortems are already bounded).
    rows = list_failure_postmortems(limit=min(off + lim + 1, 100))
    page = rows[off : off + lim]
    has_extra = len(rows) > off + lim
    total = None if has_extra else len(rows)
    return {
        "postmortems": page,
        "count": len(page),
        "pagination": page_meta(limit=lim, offset=off, returned=len(page), total=total),
    }


# ── 3. Supervisor Copilot ────────────────────────────────────────────────────


@router.post("/copilot")
async def copilot_query(body: dict[str, Any]) -> dict[str, Any]:
    from src.enterprise.copilot import answer_supervisor_query

    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON object required")
    q = str(body.get("query") or body.get("q") or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="query required")
    limit = int(body.get("limit") or 10)
    return json_safe(answer_supervisor_query(q, limit=limit))


# ── 4. Predictive risk ───────────────────────────────────────────────────────


@router.get("/risk/active")
async def risk_active(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    cursor: str | None = None,
    persist: bool = False,
) -> dict[str, Any]:
    from src.api.pagination import paginate_list
    from src.enterprise.risk import score_active_risks

    rows = score_active_risks(limit=200, persist=persist)
    page, meta = paginate_list(rows, limit=limit, offset=offset, cursor=cursor)
    return {"risks": page, "count": len(page), "pagination": meta, "persisted": persist}


@router.get("/risk/{interaction_id}/history")
async def risk_history(
    interaction_id: str,
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, Any]:
    from src.enterprise.risk import list_risk_history

    try:
        rows = list_risk_history(interaction_id, limit=limit)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return {
        "interaction_id": interaction_id,
        "snapshots": rows,
        "count": len(rows),
    }


@router.get("/risk/{interaction_id}")
async def risk_one(
    interaction_id: str,
    persist: bool = False,
) -> dict[str, Any]:
    from src.enterprise.risk import score_interaction_risk

    try:
        return json_safe(score_interaction_risk(interaction_id, persist=persist))
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


# ── 5. Knowledge graph ───────────────────────────────────────────────────────


@router.get("/graph")
async def ops_graph(
    pack_id: str | None = None,
    limit_cases: int = 40,
) -> dict[str, Any]:
    from src.enterprise.graph import build_ops_graph

    return json_safe(build_ops_graph(pack_id=pack_id, limit_cases=limit_cases))


# ── 6. Cross-contact memory ──────────────────────────────────────────────────


@router.get("/memory")
async def memory_list(
    pack_id: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    cursor: str | None = None,
) -> dict[str, Any]:
    from src.api.pagination import paginate_list
    from src.enterprise.memory import list_memories

    rows = list_memories(pack_id=pack_id, limit=200)
    page, meta = paginate_list(rows, limit=limit, offset=offset, cursor=cursor)
    return {"memories": page, "count": len(page), "pagination": meta}


@router.get("/memory/lookup")
async def memory_lookup(
    pack_id: str = Query(...),
    entity_1: str | None = None,
    entity_2: str | None = None,
    entity_3: str | None = None,
) -> dict[str, Any]:
    from src.enterprise.memory import lookup_memory

    row = lookup_memory(pack_id, entity_1=entity_1, entity_2=entity_2, entity_3=entity_3)
    if not row:
        return {"found": False, "memory": None}
    return {"found": True, "memory": row}


@router.post("/memory/upsert/{interaction_id}")
async def memory_upsert(interaction_id: str) -> dict[str, Any]:
    from src.enterprise.memory import upsert_memory_from_interaction

    row = upsert_memory_from_interaction(interaction_id)
    if not row:
        raise HTTPException(
            status_code=400,
            detail="no entity slots on interaction or interaction missing",
        )
    return {"ok": True, "memory": row}


# ── 7. Scenario playbooks ────────────────────────────────────────────────────


@router.get("/scenarios")
async def scenarios_list(
    pack_id: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    cursor: str | None = None,
) -> dict[str, Any]:
    from src.api.pagination import paginate_list
    from src.enterprise.scenarios import list_scenarios

    rows = list_scenarios(pack_id=pack_id, limit=200)
    page, meta = paginate_list(rows, limit=limit, offset=offset, cursor=cursor)
    return {"scenarios": page, "count": len(page), "pagination": meta}


@router.post("/scenarios/validate")
async def scenarios_validate(body: dict[str, Any]) -> dict[str, Any]:
    """Validate playbook steps without saving (pure; no orchestrator)."""
    from src.enterprise.scenarios import validate_scenario_steps

    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON object required")
    result = validate_scenario_steps(
        list(body.get("steps") or []),
        name=body.get("name"),
        pack_id=body.get("pack_id"),
    )
    return result


@router.post("/scenarios")
async def scenarios_create(body: dict[str, Any]) -> dict[str, Any]:
    from src.enterprise.scenarios import create_scenario

    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON object required")
    try:
        return create_scenario(
            pack_id=str(body.get("pack_id") or "automotive_nhtsa"),
            name=str(body.get("name") or ""),
            steps=list(body.get("steps") or []),
            description=str(body.get("description") or ""),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

@router.get("/scenarios/{scenario_id}")
async def scenarios_get(scenario_id: str) -> dict[str, Any]:
    from src.enterprise.scenarios import get_scenario

    scn = get_scenario(scenario_id)
    if not scn:
        raise HTTPException(status_code=404, detail="scenario not found")
    return scn


@router.delete("/scenarios/{scenario_id}")
async def scenarios_delete(scenario_id: str) -> dict[str, Any]:
    from src.enterprise.scenarios import delete_scenario

    ok = delete_scenario(scenario_id)
    if not ok:
        raise HTTPException(status_code=404, detail="scenario not found")
    return {"ok": True, "scenario_id": scenario_id}


@router.post("/scenarios/{scenario_id}/run")
async def scenarios_run(scenario_id: str) -> dict[str, Any]:
    """Run a scenario off the event loop (same spirit as /simulate + to_thread)."""
    import asyncio

    from src.enterprise.scenarios import run_scenario

    try:
        result = await asyncio.to_thread(
            lambda: asyncio.run(run_scenario(scenario_id))
        )
        return json_safe(result)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}") from e


# ── 8. Decision flow ─────────────────────────────────────────────────────────


@router.get("/decision-flow/{interaction_id}")
async def decision_flow(interaction_id: str) -> dict[str, Any]:
    from src.enterprise.decision_flow import build_decision_flow

    try:
        return json_safe(build_decision_flow(interaction_id))
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


__all__ = ["router"]
