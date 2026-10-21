"""Additional AI-layer routes (hackathon extensions).  [AI-LAYER]

Mounted inside ``src.api.routes.ai_layer.router`` (so it inherits
``require_api_key``); no change to ``src/api/main.py`` is needed.

  GET  /api/ai/vehicle-risk?make=&model=&year=&mileage=   per-vehicle fault risk (real counts)
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from src.api.limiter import limiter
from src.api.rbac import require_perm_dep

router = APIRouter(tags=["ai-layer"])


@router.get("/api/ai/vehicle-risk")
@limiter.limit("60 per minute")
async def vehicle_risk_route(
    request: Request,
    make: str,
    model: str,
    year: str,
    mileage: int | None = None,
    _role: str = Depends(require_perm_dep("case:read", open_mode_ok=True)),
) -> dict[str, Any]:
    from src.ai_layer.vehicle_risk import vehicle_risk

    if not (make.strip() and model.strip() and year.strip()) or len(make) > 40 or len(model) > 60 or not year.strip().isdigit():
        raise HTTPException(status_code=400, detail="make, model and a numeric year are required")
    if mileage is not None and not (0 <= mileage <= 2_000_000):
        raise HTTPException(status_code=400, detail="mileage out of range")
    return await asyncio.to_thread(vehicle_risk, make, model, year, mileage)
