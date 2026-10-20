"""REST routes for the optional Meta AI layer.  [AI-LAYER]

  GET  /api/ai/status                                  -> mode badge (enhanced | deterministic)
  GET  /api/ai/events                                  -> deterministic-vs-AI decision log
  GET  /api/ai/investigations/{investigation_id}/grounded-brief
  GET  /api/ai/early-warning/safety                    -> investigations ordered by safety risk
  GET  /api/ai/investigations/{investigation_id}/evidence
  GET  /api/ai/clusters/{cluster_id}/evidence?pack_id=
  POST /api/ai/language/normalize                      -> detect + normalise (Hindi/Hinglish)
  POST /api/cases/{case_id}/evidence                   -> upload photo evidence (multipart "file")
  GET  /api/cases/{case_id}/evidence
  POST /api/interactions/{interaction_id}/evidence     -> mid-call upload (attaches to the case later)
  GET  /api/interactions/{interaction_id}/evidence
  GET  /api/evidence/{evidence_id}
  GET  /api/evidence/{evidence_id}/file
  POST /api/evidence/{evidence_id}/analyze

Auth mirrors the existing routers: ``require_api_key`` on the router plus the
existing RBAC permissions (``case:read`` / ``case:write`` / ``contact:write``).
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from src.api.auth import require_api_key
from src.api.limiter import limiter
from src.api.rbac import require_perm_dep
from src.data.warehouse import ops_in_thread

router = APIRouter(tags=["ai-layer"], dependencies=[Depends(require_api_key)])


def _evidence_http(exc: Exception) -> HTTPException:
    from src.ai_layer.evidence import EvidenceError
    from src.security.identifiers import InvalidIdentifier

    if isinstance(exc, EvidenceError):
        return HTTPException(status_code=exc.status, detail=exc.detail)
    if isinstance(exc, InvalidIdentifier):
        return HTTPException(status_code=400, detail=str(exc))
    raise exc


@router.get("/api/ai/status")
async def ai_status() -> dict[str, Any]:
    from src.ai_layer.config import ai_status as _status
    from src.ai_layer.meta_client import calls_today

    out = _status()
    out["calls_today"] = calls_today()
    return out


@router.get("/api/ai/events")
async def ai_events(
    interaction_id: str | None = None,
    limit: int = 100,
    _role: str = Depends(require_perm_dep("case:read", open_mode_ok=True)),
) -> dict[str, Any]:
    from src.ai_layer.store import list_events
    from src.security.identifiers import safe_token_id

    if interaction_id:
        try:
            safe_token_id(interaction_id, kind="interaction_id")
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
    rows = await ops_in_thread(list_events, interaction_id, limit)
    return {"events": rows, "count": len(rows)}


@router.get("/api/ai/investigations/{investigation_id}/grounded-brief")
async def grounded_brief(
    investigation_id: str,
    _role: str = Depends(require_perm_dep("case:read", open_mode_ok=True)),
) -> dict[str, Any]:
    from src.ai_layer.grounded_brief import build_grounded_brief

    try:
        return await asyncio.to_thread(build_grounded_brief, investigation_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.get("/api/ai/early-warning/safety")
async def early_warning_safety(
    limit: int = 50,
    _role: str = Depends(require_perm_dep("case:read", open_mode_ok=True)),
) -> dict[str, Any]:
    from src.ai_layer.safety_board import safety_board

    return await ops_in_thread(safety_board, max(1, min(int(limit), 200)))


@router.get("/api/ai/investigations/{investigation_id}/evidence")
async def investigation_evidence(
    investigation_id: str,
    _role: str = Depends(require_perm_dep("case:read", open_mode_ok=True)),
) -> dict[str, Any]:
    from src.ai_layer.evidence import list_for_investigation

    rows = await ops_in_thread(list_for_investigation, investigation_id)
    return {"investigation_id": investigation_id, "evidence": rows}


@router.get("/api/ai/clusters/{cluster_id}/evidence")
async def cluster_evidence(
    cluster_id: int,
    pack_id: str = "automotive_nhtsa",
    _role: str = Depends(require_perm_dep("case:read", open_mode_ok=True)),
) -> dict[str, Any]:
    from src.ai_layer.evidence import list_for_cluster

    rows = await ops_in_thread(list_for_cluster, pack_id, cluster_id)
    return {"pack_id": pack_id, "cluster_id": cluster_id, "evidence": rows}


@router.post("/api/ai/language/normalize")
@limiter.limit("60 per minute")
async def normalize_language(
    request: Request,
    body: dict[str, Any],
    _role: str = Depends(require_perm_dep("contact:write", open_mode_ok=True)),
) -> dict[str, Any]:
    from src.ai_layer.normalize import normalize_text

    text = str((body or {}).get("text") or "")[:2000]
    return await asyncio.to_thread(normalize_text, text)


async def _read_upload(file: UploadFile) -> bytes:
    from src.ai_layer.config import evidence_max_bytes

    limit = evidence_max_bytes()
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(status_code=413, detail=f"file too large (max {limit} bytes)")
    return data


def _schedule_analysis(evidence: dict[str, Any]) -> None:
    from src.ai_layer.config import meta_enabled
    from src.ai_layer.evidence import analyze_evidence

    if not meta_enabled() or evidence.get("analysis_status") != "pending":
        return

    async def _run() -> None:
        try:
            await asyncio.to_thread(analyze_evidence, evidence["evidence_id"])
        except Exception:
            pass

    try:
        asyncio.get_running_loop().create_task(_run())
    except RuntimeError:
        pass


@router.post("/api/cases/{case_id}/evidence", status_code=201)
@limiter.limit("30 per minute")
async def upload_case_evidence(
    request: Request,
    case_id: str,
    file: UploadFile = File(...),
    _role: str = Depends(require_perm_dep("case:write", open_mode_ok=True)),
) -> dict[str, Any]:
    from src.ai_layer.evidence import store_evidence

    data = await _read_upload(file)
    try:
        ev = await ops_in_thread(
            lambda: store_evidence(data=data, filename=file.filename, declared_mime=file.content_type, case_id=case_id)
        )
    except Exception as e:
        raise _evidence_http(e) from e
    _schedule_analysis(ev)
    return ev


@router.get("/api/cases/{case_id}/evidence")
async def list_case_evidence(
    case_id: str,
    _role: str = Depends(require_perm_dep("case:read", open_mode_ok=True)),
) -> dict[str, Any]:
    from src.ai_layer.evidence import list_for_case

    try:
        rows = await ops_in_thread(list_for_case, case_id)
    except Exception as e:
        raise _evidence_http(e) from e
    return {"case_id": case_id, "evidence": rows}


@router.post("/api/interactions/{interaction_id}/evidence", status_code=201)
@limiter.limit("30 per minute")
async def upload_interaction_evidence(
    request: Request,
    interaction_id: str,
    file: UploadFile = File(...),
    _role: str = Depends(require_perm_dep("contact:write", open_mode_ok=True)),
) -> dict[str, Any]:
    from src.ai_layer.evidence import store_evidence

    data = await _read_upload(file)
    try:
        ev = await ops_in_thread(
            lambda: store_evidence(data=data, filename=file.filename, declared_mime=file.content_type,
                                   interaction_id=interaction_id)
        )
    except Exception as e:
        raise _evidence_http(e) from e
    _schedule_analysis(ev)
    return ev


@router.get("/api/interactions/{interaction_id}/evidence")
async def list_interaction_evidence(
    interaction_id: str,
    _role: str = Depends(require_perm_dep("case:read", open_mode_ok=True)),
) -> dict[str, Any]:
    from src.ai_layer.evidence import list_for_interaction

    rows = await ops_in_thread(list_for_interaction, interaction_id)
    return {"interaction_id": interaction_id, "evidence": rows}


def _check_evidence_id(evidence_id: str) -> None:
    from src.security.identifiers import safe_token_id

    try:
        safe_token_id(evidence_id, kind="evidence_id")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/api/evidence/{evidence_id}")
async def get_evidence(
    evidence_id: str,
    _role: str = Depends(require_perm_dep("case:read", open_mode_ok=True)),
) -> dict[str, Any]:
    from src.ai_layer.evidence import get_evidence as _get

    _check_evidence_id(evidence_id)
    try:
        return await ops_in_thread(_get, evidence_id)
    except Exception as e:
        raise _evidence_http(e) from e


@router.get("/api/evidence/{evidence_id}/file")
async def get_evidence_file(
    evidence_id: str,
    _role: str = Depends(require_perm_dep("case:read", open_mode_ok=True)),
):
    from pathlib import Path

    from src.ai_layer.evidence import evidence_dir, get_evidence as _get

    _check_evidence_id(evidence_id)
    try:
        ev = await ops_in_thread(lambda: _get(evidence_id, include_path=True))
    except Exception as e:
        raise _evidence_http(e) from e
    path = Path(ev["storage_path"]).resolve()
    root = evidence_dir()
    if root not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail="evidence file missing")
    return FileResponse(str(path), media_type=ev["mime"], headers={"Cache-Control": "private, no-store"})


@router.post("/api/evidence/{evidence_id}/analyze")
@limiter.limit("20 per minute")
async def analyze(
    request: Request,
    evidence_id: str,
    _role: str = Depends(require_perm_dep("case:write", open_mode_ok=True)),
) -> dict[str, Any]:
    from src.ai_layer.evidence import analyze_evidence

    _check_evidence_id(evidence_id)
    try:
        return await asyncio.to_thread(analyze_evidence, evidence_id)
    except Exception as e:
        raise _evidence_http(e) from e
