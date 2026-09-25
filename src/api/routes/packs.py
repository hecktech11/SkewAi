"""REST routes for packs."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from src.api.auth import require_api_key
from src.api.rbac import get_role, require_perm
from src.domains.active_pack import resolve_active_pack_id, set_active_pack_id
from src.domains.loader import lint_pack, list_packs, load_pack

# FIND-R04: all pack surfaces require the same API-key gate as other ops APIs
# (no-op in open mode; enforced when FRONTLINE_API_KEY / AUTH_REQUIRED set).
router = APIRouter(
    prefix="/api/packs",
    tags=["packs"],
    dependencies=[Depends(require_api_key)],
)


@router.get("")
async def list_all_packs() -> dict[str, Any]:
    """List available packs + the active pack."""
    active = resolve_active_pack_id()
    packs = []
    from src.security.identifiers import PACK_ID_RE

    for pid in list_packs():
        if not PACK_ID_RE.match(pid):
            continue
        try:
            pack = load_pack(pid)
            errors = lint_pack(pid)
            packs.append({
                "id": pack.id,
                "display_name": pack.display_name,
                "pack_version": pack.pack_version,
                "slots": [s.name for s in pack.manifest.slot_frame],
                "entity_labels": pack.entity_labels,
                "locale": getattr(pack, "locale", "en-US") or "en-US",
                "lint_errors": errors,
                "is_active": pid == active,
            })
        except Exception as e:
            packs.append({"id": pid, "error": str(e), "is_active": False})
    return {"packs": packs, "active": active}


@router.put("/active")
async def set_active_pack(
    pack_id: str,
    role: str = Depends(get_role),
) -> dict[str, Any]:
    """Switch the active pack (admin ``pack:activate`` when auth required)."""
    from src.api.auth import is_open_mode
    from src.security.audit_log import security_event

    # Open local demos may still switch packs; hardened requires admin.
    if not is_open_mode():
        require_perm(role, "pack:activate")

    if pack_id not in list_packs():
        raise HTTPException(status_code=404, detail=f"pack not found: {pack_id}")
    errors = lint_pack(pack_id)
    if errors:
        raise HTTPException(status_code=400, detail=f"pack has lint errors: {errors}")
    set_active_pack_id(pack_id)
    pack = load_pack(pack_id)
    security_event(
        "pack.activate",
        outcome="success",
        role=role,
        resource=pack_id,
        detail={"pack_version": pack.pack_version},
    )
    return {
        "active": pack_id,
        "display_name": pack.display_name,
        "pack_version": pack.pack_version,
    }


@router.get("/{pack_id}")
async def get_pack(pack_id: str) -> dict[str, Any]:
    """Get details for a single pack (auth-gated; full manifest not public)."""
    try:
        pack = load_pack(pack_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"pack not found: {pack_id}")
    return {
        "id": pack.id,
        "display_name": pack.display_name,
        "pack_version": pack.pack_version,
        "manifest": pack.manifest.model_dump(),
        "entity_labels": pack.entity_labels,
        "gazetteers": sorted(pack.gazetteers),
    }


__all__ = ["router"]
