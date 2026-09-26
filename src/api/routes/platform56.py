"""API surface for features added in the 56-feature completion pass."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request

from src.api.auth import require_api_key, require_api_key_strict
from src.api.limiter import limiter
from src.api.rbac import get_actor, get_role, require_perm, require_perm_dep

router = APIRouter(
    prefix="/api/frontline",
    tags=["frontline-platform"],
    dependencies=[Depends(require_api_key)],
)


@router.get("/analytics/forecast")
async def analytics_forecast(
    pack_id: str | None = None,
    window_days: int = 28,
    threshold: int = 20,
) -> dict[str, Any]:
    from src.frontline.analytics import forecast_cluster_volume

    return forecast_cluster_volume(pack_id=pack_id, window_days=window_days, threshold=threshold)


@router.get("/analytics/cross-pack")
async def analytics_cross_pack() -> dict[str, Any]:
    from src.frontline.analytics import cross_pack_patterns

    return cross_pack_patterns()


@router.get("/analytics/hotspots")
async def analytics_hotspots(pack_id: str | None = None, window_days: int = 90) -> dict[str, Any]:
    from src.frontline.analytics import geographic_hotspots

    return geographic_hotspots(pack_id=pack_id, window_days=window_days)


@router.get("/analytics/severity-drift")
async def analytics_severity_drift(pack_id: str | None = None) -> dict[str, Any]:
    from src.frontline.analytics import severity_drift

    return severity_drift(pack_id=pack_id)


@router.get("/analytics/cohorts")
async def analytics_cohorts(pack_id: str | None = None, group_field: str = "entity_3") -> dict[str, Any]:
    from src.frontline.analytics import cohort_analysis
    from src.security.sql_ident import safe_column

    try:
        field = safe_column(group_field)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return cohort_analysis(pack_id=pack_id, group_field=field)


@router.get("/analytics/regulator-watch")
async def analytics_regulator(pack_id: str | None = None) -> dict[str, Any]:
    from src.frontline.analytics import regulator_filing_watch

    return regulator_filing_watch(pack_id=pack_id)


@router.get("/analytics/financial-impact")
async def analytics_financial(pack_id: str | None = None) -> dict[str, Any]:
    from src.frontline.analytics import financial_impact

    return financial_impact(pack_id=pack_id)


@router.get("/analytics/fairness")
async def analytics_fairness(pack_id: str | None = None) -> dict[str, Any]:
    from src.frontline.analytics import bias_fairness_report

    return bias_fairness_report(pack_id=pack_id)


@router.post("/booking/offer")
async def booking_offer(body: dict[str, Any]) -> dict[str, Any]:
    from src.frontline.booking import offer_for_advisory

    return offer_for_advisory(
        advisory_id=body.get("advisory_id"),
        case_id=body.get("case_id"),
        pack_id=body.get("pack_id") or "automotive_nhtsa",
        interaction_id=body.get("interaction_id"),
    )


@router.post("/booking/book")
async def booking_book(body: dict[str, Any]) -> dict[str, Any]:
    from src.frontline.booking import book_appointment

    if not body.get("slot_start"):
        raise HTTPException(400, "slot_start required")
    return book_appointment(
        case_id=body.get("case_id"),
        interaction_id=body.get("interaction_id"),
        pack_id=body.get("pack_id") or "automotive_nhtsa",
        slot_start=str(body["slot_start"]),
        location=str(body.get("location") or "authorized_service"),
    )


@router.post("/callbacks")
async def callbacks_create(body: dict[str, Any]) -> dict[str, Any]:
    from src.frontline.callback import schedule_callback

    return schedule_callback(
        interaction_id=body.get("interaction_id"),
        pack_id=body.get("pack_id") or "automotive_nhtsa",
        phone_or_channel=str(body.get("phone_or_channel") or "unknown"),
        preferred_window=str(body.get("preferred_window") or "next_business_day"),
        case_id=body.get("case_id"),
        reason=str(body.get("reason") or "queue_or_supervisor"),
    )


@router.get("/callbacks")
async def callbacks_list(status: str | None = "scheduled") -> dict[str, Any]:
    from src.frontline.callback import list_callbacks

    rows = list_callbacks(status=status)
    return {"callbacks": rows, "count": len(rows)}


@router.post("/approvals")
async def approvals_request(
    request: Request,
    body: dict[str, Any],
    actor: str = Depends(get_actor),
) -> dict[str, Any]:
    from src.frontline.four_eyes import request_approval
    from src.api.rbac import session_token_from_cookies

    has_auth = bool(
        request.headers.get("x-frontline-session")
        or session_token_from_cookies(request.cookies)
        or request.headers.get("x-api-key")
        or request.headers.get("authorization")
    )
    if not has_auth or not actor or actor in {"anonymous", "operator"}:
        raise HTTPException(status_code=401, detail="Authentication required to request approvals")
    return request_approval(
        str(body.get("action_type") or ""),
        resource_id=str(body.get("resource_id") or ""),
        requested_by=actor,
        reason=str(body.get("reason") or ""),
        payload=body.get("payload") if isinstance(body.get("payload"), dict) else None,
        interaction_id=body.get("interaction_id"),
    )


@router.post("/approvals/{approval_id}/decide")
async def approvals_decide(
    request: Request,
    approval_id: str,
    body: dict[str, Any],
    actor: str = Depends(get_actor),
    _role: str = Depends(require_perm_dep("approval:decide", open_mode_ok=False)),
) -> dict[str, Any]:
    from src.frontline.four_eyes import decide_approval
    from src.api.rbac import session_token_from_cookies

    has_auth = bool(
        request.headers.get("x-frontline-session")
        or session_token_from_cookies(request.cookies)
        or request.headers.get("x-api-key")
        or request.headers.get("authorization")
    )
    if not has_auth or not actor or actor in {"anonymous", "operator"}:
        raise HTTPException(status_code=401, detail="Authentication required to decide approvals")

    try:
        return decide_approval(
            approval_id,
            reviewer=actor,
            approve=bool(body.get("approve", True)),
            interaction_id=body.get("interaction_id"),
        )
    except KeyError:
        raise HTTPException(404, "approval not found")
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/marketplace/packs")
async def marketplace_list() -> dict[str, Any]:
    from src.domains.marketplace import list_marketplace

    packs = list_marketplace()
    return {"packs": packs, "count": len(packs)}


@router.post("/marketplace/install")
@limiter.limit("20 per minute")
async def marketplace_install(
    request: Request,
    body: dict[str, Any],
    role: str = Depends(get_role),
) -> dict[str, Any]:
    from src.api.rbac import require_perm
    from src.domains.marketplace import pack_install
    from src.security.audit_log import security_event

    require_perm(role, "marketplace:install", open_mode_ok=not bool(body.get("source_path")))
    pid = body.get("pack_id")
    if not pid:
        raise HTTPException(400, "pack_id required")
    try:
        out = pack_install(str(pid), source_path=body.get("source_path") or None)
        security_event(
            "marketplace.install",
            outcome="success",
            role=role,
            resource=str(pid),
            detail={"has_source_path": bool(body.get("source_path"))},
            ip=request.client.host if request.client else None,
        )
        return out
    except ValueError as e:
        security_event(
            "marketplace.install",
            outcome="denied",
            role=role,
            resource=str(pid),
            detail={"error": str(e)[:200]},
            ip=request.client.host if request.client else None,
        )
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.post(
    "/packs/{pack_id}/edit",
    dependencies=[Depends(require_api_key_strict)],
)
async def pack_edit(
    pack_id: str,
    body: dict[str, Any],
    role: str = Depends(get_role),
) -> dict[str, Any]:
    """Lint/preview pack edits. Disk write requires admin ``pack:edit`` (H2)."""
    from src.domains.editor import apply_pack_edits
    from src.security.audit_log import security_event

    dry_run = bool(body.get("dry_run", False))
    # Default write_disk off — must opt in explicitly (and hold pack:edit).
    write_disk = bool(body.get("write_disk", False))
    if write_disk and not dry_run:
        require_perm(role, "pack:edit")
    edits = body.get("edits") if isinstance(body.get("edits"), dict) else body
    try:
        out = apply_pack_edits(
            pack_id,
            {k: v for k, v in edits.items() if k not in {"edits", "dry_run", "write_disk"}},
            dry_run=dry_run or not write_disk,
            write_disk=write_disk and not dry_run,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if write_disk and not dry_run:
        security_event(
            "pack.edit",
            outcome="success" if out.get("applied") else "failure",
            role=role,
            resource=pack_id,
            detail={"changed_keys": out.get("changed_keys"), "rolled_back": out.get("rolled_back")},
        )
    return out


@router.post("/packs/{pack_id}/reload")
async def pack_reload(pack_id: str) -> dict[str, Any]:
    from src.domains.editor import hot_reload

    return hot_reload(pack_id)


@router.get("/jobs")
async def jobs_list(status: str | None = None) -> dict[str, Any]:
    from src.jobs.queue import list_jobs

    rows = list_jobs(status=status)
    return {"jobs": rows, "count": len(rows)}


@router.post(
    "/jobs",
    dependencies=[Depends(require_api_key_strict)],
)
async def jobs_enqueue(
    body: dict[str, Any],
    role: str = Depends(get_role),
    actor: str = Depends(get_actor),
) -> dict[str, Any]:
    """Queue a job — gated by the job type's own permission (R01).

    Submitting a job is submitting the operation: a worker executes the
    payload with full process authority, so the API key alone is not
    sufficient authorization.
    """
    from src.jobs.queue import enqueue
    from src.security.audit_log import security_event

    jt = body.get("job_type")
    if not jt:
        raise HTTPException(400, "job_type required")
    try:
        result = enqueue(
            str(jt),
            body.get("payload") if isinstance(body.get("payload"), dict) else {},
            role=role,
            principal=actor,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    security_event(
        "jobs.enqueue",
        outcome="success",
        role=role,
        actor=actor,
        detail={"job_type": str(jt), "job_id": result.get("job_id")},
    )
    return result


@router.post(
    "/jobs/run-next",
    dependencies=[Depends(require_api_key_strict)],
)
async def jobs_run_next(role: str = Depends(get_role)) -> dict[str, Any]:
    """Execute next queued job — admin only (H4)."""
    from src.jobs.queue import run_next
    from src.security.audit_log import security_event

    require_perm(role, "jobs:run")
    result = run_next()
    security_event(
        "jobs.run_next",
        outcome="success",
        role=role,
        detail={"status": (result or {}).get("status", "empty")},
    )
    return result or {"status": "empty"}


@router.get("/metrics/prometheus")
async def metrics_prometheus() -> Any:
    from fastapi.responses import PlainTextResponse
    from src.observability.metrics import prometheus_text

    return PlainTextResponse(prometheus_text(), media_type="text/plain; version=0.0.4")


@router.get("/ops/backend")
async def ops_backend() -> dict[str, Any]:
    from src.data.postgres_backend import backend_status

    return backend_status()


@router.get("/ops/drain")
async def ops_drain_status() -> dict[str, Any]:
    from src.ops.drain import DRAIN

    return DRAIN.status()


@router.post(
    "/ops/drain",
    dependencies=[Depends(require_api_key_strict)],
)
async def ops_drain_begin(
    request: Request,
    role: str = Depends(get_role),
) -> dict[str, Any]:
    """Begin call drain — admin only (H3)."""
    from src.ops.drain import DRAIN
    from src.security.audit_log import security_event

    require_perm(role, "ops:drain")
    out = DRAIN.begin_drain()
    security_event(
        "ops.drain",
        outcome="success",
        role=role,
        detail=out if isinstance(out, dict) else {},
        ip=request.client.host if request.client else None,
    )
    return out


@router.post(
    "/ops/drain/cancel",
    dependencies=[Depends(require_api_key_strict)],
)
async def ops_drain_cancel(
    request: Request,
    role: str = Depends(get_role),
) -> dict[str, Any]:
    """Clear drain so the process accepts new contacts again."""
    from src.ops.drain import DRAIN
    from src.security.audit_log import security_event

    require_perm(role, "ops:drain")
    DRAIN.reset()
    out = DRAIN.status()
    security_event(
        "ops.drain_cancel",
        outcome="success",
        role=role,
        detail=out if isinstance(out, dict) else {},
        ip=request.client.host if request.client else None,
    )
    return out


@router.get("/auth/oidc")
async def auth_oidc() -> dict[str, Any]:
    from src.api.rbac import oidc_discovery

    return oidc_discovery()


@router.get("/auth/verify")
async def auth_verify(role: str = Depends(get_role)) -> dict[str, Any]:
    """Validate the caller credential for the dashboard route guard.

    The router already enforces ``require_api_key`` (401 when auth is required
    and the key is missing/invalid). Returns the resolved principal role so
    the UI can tell a key-based operator apart from an anonymous visitor.
    In open pilot mode this passes for everyone — callers must also check
    ``auth_required`` and only trust an API key when it is true.
    """
    from src.api.auth import auth_required

    return {"ok": True, "auth_required": auth_required(), "role": role}


@router.get("/auth/me")
async def auth_me(
    request: Request,
    x_frontline_session: str | None = Header(default=None),
) -> dict[str, Any]:
    """Identity for the signed-in web session (httpOnly cookie preferred)."""
    from fastapi import HTTPException as _HTTP
    from src.api.rbac import session_token_from_cookies, verify_session

    token = (
        x_frontline_session or session_token_from_cookies(request.cookies) or ""
    ).strip()
    if not token:
        return {"signed_in": False}
    try:
        body = verify_session(token)
    except _HTTP:
        return {"signed_in": False, "expired": True}
    return {
        "signed_in": True,
        "subject": body.get("sub"),
        "role": body.get("role"),
        "exp": body.get("exp"),
    }


@router.post("/auth/session")
@limiter.limit("20 per minute")
async def auth_session(
    request: Request,
    body: dict[str, Any],
    role: str = Depends(get_role),
) -> dict[str, Any]:
    """JSON-body session mint is closed. Use Google / OIDC."""
    from src.api.auth import is_open_mode
    from src.security.audit_log import security_event

    security_event(
        "auth.session_mint",
        outcome="denied",
        actor=str(body.get("subject") or ""),
        detail={"reason": "idp_required"},
        ip=request.client.host if request.client else None,
    )
    # Enforcement site for session:mint (JSON mint stays closed either way).
    if not is_open_mode():
        require_perm(role, "session:mint")
    raise HTTPException(
        status_code=403,
        detail="session mint requires Google / OIDC sign-in",
    )


@router.post("/coach/suggest")
async def coach_suggest(body: dict[str, Any]) -> dict[str, Any]:
    from src.frontline.coach import suggest_replies

    return {
        "suggestions": suggest_replies(
            slots=body.get("slots") if isinstance(body.get("slots"), dict) else {},
            last_customer_text=str(body.get("last_customer_text") or ""),
            severity=body.get("severity"),
        )
    }


@router.post("/coach/whisper")
async def coach_whisper(body: dict[str, Any]) -> dict[str, Any]:
    from src.frontline.coach import whisper

    iid = body.get("interaction_id")
    text = body.get("text")
    if not iid or not text:
        raise HTTPException(400, "interaction_id and text required")
    return whisper(str(iid), str(text), from_role=str(body.get("from_role") or "coach"))


@router.get("/subscriptions")
async def subscriptions_list() -> dict[str, Any]:
    from src.frontline.subscriptions import list_subscriptions

    rows = list_subscriptions()
    return {"subscriptions": rows, "count": len(rows)}


@router.post("/subscriptions")
async def subscriptions_create(body: dict[str, Any]) -> dict[str, Any]:
    from src.frontline.subscriptions import subscribe

    try:
        return subscribe(
            channel=str(body.get("channel") or "email"),
            target=str(body.get("target") or ""),
            report_type=str(body.get("report_type") or "daily_digest"),
            cron_hint=str(body.get("cron_hint") or "0 8 * * *"),
        )
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/subscriptions/tick")
async def subscriptions_tick(force: bool = True) -> dict[str, Any]:
    from src.frontline.subscriptions import tick_due

    sent = tick_due(force=force)
    return {"sent": sent, "count": len(sent)}


@router.get("/usage")
async def usage_get() -> dict[str, Any]:
    """Usage + plan/seats for the process tenant (client tenant_id ignored — M8).

    Merges metering totals with billing plan fields so TrustPipeline can
    render plan/seats. The duplicate GET /usage on pipeline.py is renamed.
    """
    from src.data.timeutil import utc_now
    from src.frontline.billing import usage_dashboard
    from src.ops.tenant import get_tenant

    out = usage_dashboard(tenant_id=get_tenant())
    out.setdefault("as_of", utc_now().isoformat())
    return out


@router.post("/usage")
async def usage_record(body: dict[str, Any]) -> dict[str, Any]:
    from src.frontline.metering import record_usage
    from src.ops.tenant import get_tenant

    return record_usage(
        str(body.get("metric") or "contacts"),
        float(body.get("quantity") or 1),
        tenant_id=get_tenant(),
        meta=body.get("meta") if isinstance(body.get("meta"), dict) else None,
    )


@router.post(
    "/channels/email/ingest",
    dependencies=[Depends(require_api_key_strict)],
)
@limiter.limit("60 per minute")
async def email_ingest(
    request: Request,
    body: dict[str, Any],
    role: str = Depends(get_role),
) -> dict[str, Any]:
    """Email intake — always authenticated (H4); needs channel:ingest."""
    from src.channels.email_intake import EmailIntakeChannel

    require_perm(role, "channel:ingest")
    ch = EmailIntakeChannel()
    return ch.process_message(
        subject=str(body.get("subject") or ""),
        body=str(body.get("body") or ""),
        from_addr=str(body.get("from") or ""),
    )


@router.post(
    "/biometrics/match",
    dependencies=[Depends(require_api_key_strict)],
)
@limiter.limit("60 per minute")
async def biometrics_match(
    request: Request,
    body: dict[str, Any],
    role: str = Depends(get_role),
) -> dict[str, Any]:
    """Biometrics match — always authenticated (H4); pilot integrity only."""
    from src.frontline.biometrics import fingerprint_from_audio_features, match_caller

    require_perm(role, "biometrics:match")
    fp = body.get("fingerprint") or fingerprint_from_audio_features(
        body.get("features") or body.get("ani") or "unknown"
    )
    return match_caller(
        str(body.get("pack_id") or "automotive_nhtsa"),
        str(fp),
        entity_1=body.get("entity_1"),
        entity_2=body.get("entity_2"),
        entity_3=body.get("entity_3"),
    )


@router.post("/i18n/normalize")
async def i18n_normalize(body: dict[str, Any]) -> dict[str, Any]:
    from src.frontline.i18n import normalize_to_english

    return normalize_to_english(str(body.get("text") or ""), lang=body.get("lang"))


@router.post("/slots/dynamic-skip")
async def slots_dynamic_skip(body: dict[str, Any]) -> dict[str, Any]:
    from src.frontline.dynamic_slots import apply_dynamic_skip

    return apply_dynamic_skip(
        str(body.get("pack_id") or "automotive_nhtsa"),
        list(body.get("required_slots") or ["entity_1", "entity_2", "category", "description"]),
        body.get("current") if isinstance(body.get("current"), dict) else {},
    )
