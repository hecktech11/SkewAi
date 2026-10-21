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


# ── Manufacturer alert feed (no external sending) ──────────────────────────
@router.post("/api/ai/alerts/scan")
@limiter.limit("20 per minute")
async def alerts_scan(
    request: Request,
    _role: str = Depends(require_perm_dep("case:write", open_mode_ok=True)),
) -> dict[str, Any]:
    from src.ai_layer.alerts import scan_alerts

    return await asyncio.to_thread(scan_alerts)


@router.get("/api/ai/alerts")
async def alerts_list(
    status: str | None = None,
    limit: int = 100,
    _role: str = Depends(require_perm_dep("case:read", open_mode_ok=True)),
) -> dict[str, Any]:
    from src.ai_layer.alerts import list_alerts, thresholds

    if status not in (None, "open", "acknowledged"):
        raise HTTPException(status_code=400, detail="status must be open or acknowledged")
    alerts = await asyncio.to_thread(list_alerts, status, max(1, min(int(limit), 500)))
    return {"alerts": alerts, "count": len(alerts), "thresholds": thresholds(),
            "delivery": "pull feed only; nothing is sent externally"}


@router.get("/api/ai/alerts/feed.json")
async def alerts_feed(
    request: Request,
    _role: str = Depends(require_perm_dep("case:read", open_mode_ok=True)),
) -> dict[str, Any]:
    from src.ai_layer.alerts import json_feed

    return await asyncio.to_thread(json_feed, str(request.base_url).rstrip("/"))


@router.post("/api/ai/alerts/{alert_id}/ack")
async def alerts_ack(
    alert_id: str,
    _role: str = Depends(require_perm_dep("case:write", open_mode_ok=True)),
) -> dict[str, Any]:
    from src.ai_layer.alerts import ack_alert

    if not alert_id.startswith("alrt_") or len(alert_id) > 40 or not alert_id[5:].isalnum():
        raise HTTPException(status_code=400, detail="bad alert id")
    if not await asyncio.to_thread(ack_alert, alert_id):
        raise HTTPException(status_code=404, detail="alert not found")
    return {"alert_id": alert_id, "status": "acknowledged"}


# ── Service-centre inspection queue ─────────────────────────────────────────
@router.get("/api/ai/service-centre/queue")
async def service_centre_queue(
    limit: int = 50,
    include_simulated: bool = True,
    _role: str = Depends(require_perm_dep("case:read", open_mode_ok=True)),
) -> dict[str, Any]:
    from src.ai_layer.service_centre import service_queue

    return await asyncio.to_thread(service_queue, max(1, min(int(limit), 500)), include_simulated)


# ── Fault spread by region / time ───────────────────────────────────────────
@router.get("/api/ai/spread")
async def ai_fault_spread(
    make: str | None = None,
    model: str | None = None,
    component: str | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    months: int = 60,
    _role: str = Depends(require_perm_dep("case:read", open_mode_ok=True)),
) -> dict[str, Any]:
    from src.ai_layer.spread import fault_spread

    for v in (make, model, component):
        if v is not None and len(v) > 80:
            raise HTTPException(status_code=400, detail="filter too long")
    try:
        return await asyncio.to_thread(fault_spread, make, model, component, year_min, year_max,
                                       months=max(1, min(int(months), 240)))
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="domain warehouse not built") from None


# ── WhatsApp intake MOCK (no WhatsApp/Meta API is called) ───────────────────
@router.post("/api/ai/whatsapp-mock/webhook")
async def whatsapp_mock_webhook(
    request: Request,
    _role: str = Depends(require_perm_dep("contact:write", open_mode_ok=True)),
) -> dict[str, Any]:
    from src.ai_layer.whatsapp_mock import handle_webhook
    from src.ops.drain import ServiceDrainingError

    try:
        body = await request.json()
    except ValueError:
        raise HTTPException(status_code=400, detail="JSON body required") from None
    try:
        return await handle_webhook(body)
    except ServiceDrainingError:
        raise HTTPException(status_code=503, detail="service_draining") from None
