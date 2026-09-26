"""Frontline v3 OS APIs — learning, experiments, governance."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.auth import require_api_key
from src.api.jsonutil import json_safe
from src.api.rbac import get_actor, get_role, require_perm

router = APIRouter(
    prefix="/api/v3",
    tags=["v3-platform"],
    dependencies=[Depends(require_api_key)],
)


# ── Learning ─────────────────────────────────────────────────────────────────


@router.post("/learning/run")
async def learning_run(limit: int = Query(default=25, ge=1, le=100)) -> dict[str, Any]:
    """Scan recent failures and create improvement proposals."""
    from src.v3.learning import generate_proposals_from_failures

    return generate_proposals_from_failures(limit=limit)


@router.get("/learning/proposals")
async def learning_list(
    status: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, Any]:
    from src.v3.learning import list_proposals

    rows = list_proposals(status=status, limit=limit)
    return {"proposals": rows, "count": len(rows)}


@router.get("/learning/proposals/{proposal_id}")
async def learning_get(proposal_id: str) -> dict[str, Any]:
    from src.v3.learning import get_proposal

    row = get_proposal(proposal_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"proposal not found: {proposal_id}")
    return row


@router.post("/learning/proposals/{proposal_id}/review")
async def learning_review(
    proposal_id: str,
    body: dict[str, Any],
    role: str = Depends(get_role),
    actor: str = Depends(get_actor),
) -> dict[str, Any]:
    from src.v3.learning import review_proposal

    require_perm(role, "approval:decide")
    if not isinstance(body, dict) or "status" not in body:
        raise HTTPException(status_code=400, detail="body.status required")
    try:
        return review_proposal(
            proposal_id,
            status=str(body["status"]),
            reviewed_by=actor,
            review_note=str(body.get("review_note") or ""),
        )
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/learning/trends")
async def learning_trends_api(window_days: int = 30) -> dict[str, Any]:
    from src.v3.learning import learning_trends

    return learning_trends(window_days=window_days)


# ── Experiments ──────────────────────────────────────────────────────────────


@router.post("/artifacts")
async def artifacts_create(body: dict[str, Any]) -> dict[str, Any]:
    from src.v3.experiments import register_artifact

    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON object required")
    try:
        return register_artifact(
            kind=str(body.get("kind") or ""),
            name=str(body.get("name") or ""),
            version=str(body.get("version") or ""),
            body=dict(body.get("body") or {}),
            notes=str(body.get("notes") or ""),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/artifacts")
async def artifacts_list(
    kind: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, Any]:
    from src.v3.experiments import list_artifacts

    rows = list_artifacts(kind=kind, limit=limit)
    return {"artifacts": rows, "count": len(rows)}


@router.post("/experiments")
async def experiments_create(body: dict[str, Any]) -> dict[str, Any]:
    from src.v3.experiments import create_experiment

    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON object required")
    try:
        return create_experiment(
            name=str(body.get("name") or ""),
            control_artifact_id=str(body.get("control_artifact_id") or ""),
            candidate_artifact_id=str(body.get("candidate_artifact_id") or ""),
            mode=str(body.get("mode") or "ab"),
            description=str(body.get("description") or ""),
        )
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/experiments")
async def experiments_list(
    status: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, Any]:
    from src.v3.experiments import list_experiments

    rows = list_experiments(status=status, limit=limit)
    return {"experiments": rows, "count": len(rows)}


@router.get("/experiments/{experiment_id}")
async def experiments_get(experiment_id: str) -> dict[str, Any]:
    from src.v3.experiments import get_experiment

    row = get_experiment(experiment_id)
    if not row:
        raise HTTPException(status_code=404, detail="experiment not found")
    return json_safe(row)


@router.post("/experiments/{experiment_id}/trials")
async def experiments_trial(experiment_id: str, body: dict[str, Any]) -> dict[str, Any]:
    from src.v3.experiments import record_trial

    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON object required")
    try:
        return record_trial(
            experiment_id,
            arm=str(body.get("arm") or ""),
            interaction_id=body.get("interaction_id"),
            resolution_ok=body.get("resolution_ok"),
            escalated=body.get("escalated"),
            latency_ms=body.get("latency_ms"),
            groundedness_ok=body.get("groundedness_ok"),
            hallucination_flag=body.get("hallucination_flag"),
            cost_units=body.get("cost_units"),
        )
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/experiments/{experiment_id}/complete")
async def experiments_complete(experiment_id: str) -> dict[str, Any]:
    from src.v3.experiments import complete_experiment_report

    try:
        return json_safe(complete_experiment_report(experiment_id))
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


# ── Governance ───────────────────────────────────────────────────────────────


@router.post("/governance/deployments")
async def deploy_create(body: dict[str, Any], role: str = Depends(get_role)) -> dict[str, Any]:
    from src.v3.governance import create_deployment

    require_perm(role, "deploy:create", open_mode_ok=True)
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON object required")
    try:
        return create_deployment(
            label=str(body.get("label") or ""),
            artifact_versions=dict(body.get("artifact_versions") or {}),
            pack_version=body.get("pack_version"),
            note=str(body.get("note") or ""),
            activated_by=str(body.get("activated_by") or "operator"),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/governance/deployments")
async def deploy_list(limit: int = Query(default=50, ge=1, le=200)) -> dict[str, Any]:
    from src.v3.governance import get_active_deployment, list_deployments

    rows = list_deployments(limit=limit)
    return {
        "deployments": rows,
        "count": len(rows),
        "active": get_active_deployment(),
    }


@router.post("/governance/deployments/{deployment_id}/activate")
async def deploy_activate(
    deployment_id: str,
    body: dict[str, Any] | None = None,
    role: str = Depends(get_role),
) -> dict[str, Any]:
    from src.v3.governance import activate_deployment

    require_perm(role, "deploy:activate", open_mode_ok=True)
    body = body or {}
    try:
        return activate_deployment(
            deployment_id,
            activated_by=str(body.get("activated_by") or "operator"),
        )
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.post("/governance/rollback")
async def deploy_rollback(
    body: dict[str, Any] | None = None,
    role: str = Depends(get_role),
) -> dict[str, Any]:
    from src.v3.governance import rollback_deployment

    require_perm(role, "deploy:activate", open_mode_ok=True)
    body = body or {}
    try:
        return rollback_deployment(
            to_deployment_id=body.get("to_deployment_id"),
            activated_by=str(body.get("activated_by") or "operator"),
        )
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.get("/governance/stamps/{interaction_id}")
async def stamp_get(interaction_id: str) -> dict[str, Any]:
    from src.v3.governance import get_stamp, stamp_interaction

    row = get_stamp(interaction_id)
    if not row:
        # Lazy stamp if interaction exists
        row = stamp_interaction(interaction_id)
    if not row:
        raise HTTPException(status_code=404, detail="interaction or stamp not found")
    return row


@router.post("/governance/stamps/{interaction_id}")
async def stamp_post(interaction_id: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    from src.v3.governance import stamp_interaction

    body = body or {}
    row = stamp_interaction(
        interaction_id,
        experiment_id=body.get("experiment_id"),
    )
    if not row:
        raise HTTPException(status_code=404, detail="interaction not found")
    return row


__all__ = ["router"]
