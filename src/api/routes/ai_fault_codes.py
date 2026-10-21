"""OBD-II fault-code (DTC) routes.  [AI-LAYER]

  POST /api/ai/fault-codes/lookup                    -> validate + generic meaning (no storage)
  POST /api/ai/interactions/{interaction_id}/fault-codes -> attach to a live intake (raise happens at triage)
  POST /api/ai/cases/{case_id}/fault-codes           -> attach to a case (raise-only severity, needs-review flag)
  GET  /api/ai/cases/{case_id}/fault-codes           -> codes shown as evidence on the case view

Body: ``{"fault_codes": ["C0035", "P0300"]}`` or ``{"fault_codes": "C0035 P0300"}``. Each code must
match ``^[PCBU][0-3][0-9A-F]{3}$`` (422 otherwise). Unknown codes are stored as 'unknown code' with no effect.
Mounted from ``src/api/routes/ai_layer_v2.py``.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.api.rbac import require_perm_dep
from src.data.warehouse import ops_in_thread

router = APIRouter(tags=["ai-layer"])


class FaultCodesBody(BaseModel):
    fault_codes: list[str] | str = Field(..., description="OBD-II DTCs, e.g. ['C0035','P0300'] or 'C0035 P0300'")


def _validate(body: FaultCodesBody) -> list[str]:
    from src.ai_layer.fault_codes import DTC_RE, MAX_CODES, split_codes

    valid, invalid = split_codes(body.fault_codes)
    if invalid:
        raise HTTPException(status_code=422, detail=f"invalid fault code(s) {invalid}; expected format {DTC_RE.pattern}")
    if not valid:
        raise HTTPException(status_code=422, detail="no fault codes given")
    return valid[:MAX_CODES]


@router.post("/api/ai/fault-codes/lookup")
async def lookup_codes(
    body: FaultCodesBody,
    _role: str = Depends(require_perm_dep("case:read", open_mode_ok=True)),
) -> dict[str, Any]:
    from src.ai_layer.fault_codes import assess

    return assess(_validate(body), None)


@router.post("/api/ai/interactions/{interaction_id}/fault-codes", status_code=201)
async def attach_interaction_codes(
    interaction_id: str,
    body: FaultCodesBody,
    _role: str = Depends(require_perm_dep("contact:write", open_mode_ok=True)),
) -> dict[str, Any]:
    from src.ai_layer.fault_codes import store

    codes = _validate(body)
    out = await ops_in_thread(lambda: store(codes, interaction_id=interaction_id, source="api_intake"))
    out["interaction_id"] = interaction_id
    out["note"] = "stored; a safety-system code can raise severity at triage (raise-only)"
    return out


@router.post("/api/ai/cases/{case_id}/fault-codes", status_code=201)
async def attach_case_codes(
    case_id: str,
    body: FaultCodesBody,
    _role: str = Depends(require_perm_dep("case:write", open_mode_ok=True)),
) -> dict[str, Any]:
    from src.ai_layer.fault_codes import attach_to_case

    codes = _validate(body)
    out = await ops_in_thread(attach_to_case, case_id, codes)
    if out.get("error") == "not_found":
        raise HTTPException(status_code=404, detail="case not found")
    if out.get("error"):
        raise HTTPException(status_code=422, detail=str(out))
    return out


@router.get("/api/ai/cases/{case_id}/fault-codes")
async def case_codes(
    case_id: str,
    _role: str = Depends(require_perm_dep("case:read", open_mode_ok=True)),
) -> dict[str, Any]:
    from src.ai_layer.fault_codes import for_case

    return await ops_in_thread(for_case, case_id)
