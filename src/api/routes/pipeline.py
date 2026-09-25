"""API for data-trust, provenance, queue, commercial, onboarding, analyst/ops."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from src.api.auth import require_api_key
from src.api.rbac import get_actor, get_role
from src.config import REPO_ROOT

router = APIRouter(
    prefix="/api/frontline",
    tags=["pipeline"],
    dependencies=[Depends(require_api_key)],
)


# ── Pack-builder CSV jail (arbitrary local file read fix) ────────────────────
# csv_path must live under data/uploads/ or data/builder_packs/ (or the
# isolated test DOMAIN_DB_PATH equivalent). Absolute paths outside the jail,
# symlink escapes, and missing files are rejected with 400/404 — never read.
def _resolve_builder_csv(csv_path: str) -> "Path":
    from pathlib import Path

    raw = (csv_path or "").strip()
    if not raw:
        raise HTTPException(400, "csv_path required")
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    else:
        # resolve() follows symlinks so symlink escapes are caught below
        try:
            p = p.resolve()
        except OSError:
            raise HTTPException(400, "invalid csv_path")
    allowed_roots: list["Path"] = []
    for cand in (REPO_ROOT / "data" / "uploads", REPO_ROOT / "data" / "builder_packs"):
        try:
            allowed_roots.append(cand.resolve())
        except OSError:
            allowed_roots.append(cand)
    # Hermetic tests may point DOMAIN_DB_PATH elsewhere — allow the
    # configured uploads dir if it exists outside the repo (test isolation).
    try:
        import os as _os

        alt = _os.getenv("BUILDER_UPLOAD_DIR", "").strip()
        if alt:
            allowed_roots.append(Path(alt).expanduser().resolve())
    except Exception:
        pass
    try:
        is_allowed = any(p.is_relative_to(r) for r in allowed_roots)
    except AttributeError:
        # py<3.9 fallback
        is_allowed = any(str(p).startswith(str(r) + "/") or p == r for r in allowed_roots)
    if not is_allowed:
        raise HTTPException(400, "csv_path outside uploads jail")
    if p.suffix.lower() != ".csv":
        raise HTTPException(400, "csv_path must be a .csv file")
    if not p.is_file():
        raise HTTPException(404, "csv file not found")
    # 10 MB cap on insight reads (uploads are capped tighter at write time)
    try:
        if p.stat().st_size > 10 * 1024 * 1024:
            raise HTTPException(400, "csv too large (max 10 MB)")
    except HTTPException:
        raise
    except OSError:
        raise HTTPException(404, "csv file not found")
    return p


MAX_BUILDER_UPLOAD_BYTES = 5 * 1024 * 1024


@router.get("/trust/kpis")
async def trust_kpis() -> dict[str, Any]:
    from src.frontline.provenance import list_kpis

    return {"kpis": list_kpis()}


@router.get("/provenance/kpis")
async def provenance_kpis() -> dict[str, Any]:
    from src.frontline.provenance import list_kpis

    return {"kpis": list_kpis()}


@router.get("/provenance/kpis/{kpi_id}")
async def provenance_kpi(kpi_id: str, record_id: str | None = None, pack_id: str | None = None) -> dict[str, Any]:
    from src.frontline.provenance import figure_lineage

    try:
        return figure_lineage(kpi_id, record_id=record_id, pack_id=pack_id)
    except KeyError as e:
        raise HTTPException(404, f"unknown kpi {e}") from e


@router.get("/validation-queue")
async def validation_queue() -> dict[str, Any]:
    from src.frontline.validation_queue import list_queue_all, list_reviews

    items = list_queue_all()
    return {"items": items, "count": len(items),
            "reviews": list_reviews(status="open")}


@router.post("/reviews/{review_id}/assign")
async def review_assign(
    review_id: str,
    body: dict[str, Any],
    role: str = Depends(get_role),
    actor: str = Depends(get_actor),
) -> dict[str, Any]:
    """Claim a review-queue row (board #10): open → assigned with an owner."""
    from src.api.rbac import require_perm
    from src.frontline.validation_queue import assign_review

    require_perm(role, "approval:decide")
    if not isinstance(body, dict):
        raise HTTPException(400, "JSON object required")
    try:
        return assign_review(review_id, str(body.get("owner") or actor or ""))
    except LookupError as e:
        raise HTTPException(404, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@router.post("/reviews/{review_id}/resolve")
async def review_resolve(
    review_id: str,
    body: dict[str, Any],
    role: str = Depends(get_role),
    actor: str = Depends(get_actor),
) -> dict[str, Any]:
    """Close a review with verdict ai_wrong|data_drift|false_alarm (board #9)."""
    from src.api.rbac import require_perm
    from src.frontline.validation_queue import resolve_review

    require_perm(role, "approval:decide")
    if not isinstance(body, dict):
        raise HTTPException(400, "JSON object required")
    try:
        return resolve_review(
            review_id,
            str(body.get("verdict") or ""),
            actor=str(actor or ""),
        )
    except LookupError as e:
        raise HTTPException(404, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@router.post("/sandbox/boot")
async def sandbox_boot(body: dict[str, Any] | None = None) -> dict[str, Any]:
    from src.frontline.sandbox import boot_sandbox

    body = body or {}
    return boot_sandbox(pack_id=str(body.get("pack_id") or "automotive_nhtsa"))


@router.post("/pack-builder/profile")
async def pack_builder_profile(file: UploadFile = File(...)) -> dict[str, Any]:
    """Upload a CSV: save it, propose column mapping, lint."""
    import secrets as _secrets
    from pathlib import Path
    from src.domains.builder.pack_builder import lint_mapping, profile_csv

    dest_dir = REPO_ROOT / "data" / "uploads"
    dest_dir.mkdir(parents=True, exist_ok=True)
    safe = Path(file.filename or "upload.csv").name or "upload.csv"
    # Enforce .csv only (blocks .exe/.html/.svg stored-XSS + polyglots)
    if Path(safe).suffix.lower() != ".csv":
        raise HTTPException(400, "only .csv uploads are accepted")
    # Basic filename hygiene: no empty / dotfiles / overlong names
    if safe in {".", ".."} or safe.startswith(".") or len(safe) > 128:
        raise HTTPException(400, "invalid filename")
    # Never trust the user-supplied filename for storage: randomize the
    # server-side name (prevents overwrite/predictable-path attacks) while
    # keeping the .csv suffix.
    stem = Path(safe).stem[:48]
    cleaned = "".join(c if (c.isalnum() or c in "-_") else "_" for c in stem) or "upload"
    dest = dest_dir / f"{cleaned}_{_secrets.token_hex(8)}.csv"
    # Stream with an enforced cap BEFORE unbounded memory consumption
    # (item 15): never await a full unbounded file.read().
    try:
        chunks: list[bytes] = []
        total = 0
        while True:
            piece = await file.read(64 * 1024)
            if not piece:
                break
            total += len(piece)
            if total > MAX_BUILDER_UPLOAD_BYTES:
                raise HTTPException(400, "csv too large (max 5 MB)")
            chunks.append(piece)
        raw = b"".join(chunks)
    finally:
        try:
            await file.close()
        except Exception:
            pass
    if not raw.strip():
        raise HTTPException(400, "csv is empty")
    # Refuse obvious non-CSV binaries (NUL bytes / MZ header)
    if b"\x00" in raw[:4096] or raw[:2] == b"MZ":
        raise HTTPException(400, "not a csv file")
    dest.write_bytes(raw)
    try:
        profile = profile_csv(dest)
    except Exception:
        # Malformed CSV must not leave orphaned upload files behind.
        try:
            dest.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    lint = lint_mapping(profile["proposed_mapping"], profile["columns"])
    return {**profile, "lint": lint, "csv_path": str(dest)}


@router.post("/pack-builder/insight")
async def pack_builder_insight(body: dict[str, Any]) -> dict[str, Any]:
    from src.domains.builder.pack_builder import first_insight_from_csv

    csv_path = body.get("csv_path")
    if not csv_path:
        raise HTTPException(400, "csv_path required")
    resolved = _resolve_builder_csv(str(csv_path))
    return first_insight_from_csv(
        resolved,
        mapping=body.get("mapping"),
        pack_id=str(body.get("pack_id") or "builder_preview"),
        display_name=str(body.get("display_name") or "Preview pack"),
        out_root=REPO_ROOT / "data" / "builder_packs",
    )


@router.post("/marketplace/install/{pack_id}")
async def marketplace_install(pack_id: str, role: str = Depends(get_role)) -> dict[str, Any]:
    from src.api.rbac import require_perm
    from src.domains.marketplace import install_vertical

    require_perm(role, "marketplace:install", open_mode_ok=True)
    try:
        return install_vertical(pack_id)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e


@router.get("/billing/usage")
async def usage(tenant_id: str | None = None) -> dict[str, Any]:
    from src.frontline.billing import usage_dashboard
    from src.ops.tenant import get_tenant

    actual_tenant = get_tenant() or tenant_id or "default"
    return usage_dashboard(actual_tenant)


@router.post("/billing/checkout")
async def billing_checkout(body: dict[str, Any]) -> dict[str, Any]:
    from src.frontline.billing import stripe_checkout
    from src.ops.tenant import get_tenant

    actual_tenant = get_tenant() or str(body.get("tenant_id") or "default")
    return stripe_checkout(
        actual_tenant,
        plan=str(body.get("plan") or "pilot"),
        amount_cents=int(body.get("amount_cents") or 9900),
    )


@router.post("/billing/webhook")
async def billing_webhook(request: Request, body: dict[str, Any]) -> dict[str, Any]:
    from src.frontline.billing import stripe_webhook

    raw: bytes | None = None
    try:
        raw = await request.body()
    except Exception:
        raw = None
    sig = request.headers.get("stripe-signature") or request.headers.get(
        "Stripe-Signature"
    )
    try:
        return stripe_webhook(body, raw_body=raw, signature=sig)
    except PermissionError as e:
        raise HTTPException(status_code=401, detail=str(e)) from e
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/seats")
async def seats_assign(body: dict[str, Any], role: str = Depends(get_role)) -> dict[str, Any]:
    from src.api.rbac import require_perm
    from src.frontline.billing import assign_seat
    from src.ops.tenant import get_tenant

    require_perm(role, "seat:admin")
    actual_tenant = get_tenant() or str(body.get("tenant_id") or "default")
    try:
        return assign_seat(
            actual_tenant,
            str(body.get("user_id") or "user"),
            str(body.get("role") or "agent"),
        )
    except PermissionError as e:
        raise HTTPException(403, str(e)) from e


@router.post("/keys")
async def keys_create(
    request: Request,
    body: dict[str, Any],
    role: str = Depends(get_role),
    actor: str = Depends(get_actor),
    x_frontline_session: str | None = Header(default=None),
) -> dict[str, Any]:
    from src.api.rbac import PERMS, claims_from_headers, require_perm
    from src.security.scoped_keys import (
        check_key_mint_rate_limit,
        create_scoped_key,
        log_key_mint_audit,
    )

    claims = claims_from_headers(x_frontline_session, request.cookies)
    client_ip = request.client.host if request.client else "unknown"
    principal = str(claims.get("sub") or actor or "unknown")

    requested_scopes = list(body.get("scopes") or ["kpi:read"])
    requested_tenant = str(body.get("tenant_id") or "").strip()
    if not requested_tenant:
        raise HTTPException(400, "tenant_id required")

    # 1. Rate limiting
    allowed, retry_after = check_key_mint_rate_limit(principal)
    if not allowed:
        log_key_mint_audit(
            principal=principal,
            requested_scopes=requested_scopes,
            granted_scopes=[],
            tenant_id=requested_tenant,
            ip=client_ip,
            success=False,
            error="rate_limit_exceeded",
        )
        raise HTTPException(
            status_code=429,
            detail="key mint rate limit exceeded",
            headers={"Retry-After": str(retry_after)},
        )

    # 2. Scope restriction
    caller_scopes = claims.get("scopes")
    if caller_scopes is not None:
        caller_set = set(caller_scopes)
        for s in requested_scopes:
            if s not in caller_set or s.startswith("admin:"):
                log_key_mint_audit(
                    principal=principal,
                    requested_scopes=requested_scopes,
                    granted_scopes=[],
                    tenant_id=requested_tenant,
                    ip=client_ip,
                    success=False,
                    error="scope_escalation_denied",
                )
                raise HTTPException(
                    status_code=403,
                    detail=f"scope_escalation_denied: scope {s!r} exceeds granted scopes",
                )

    # 3. Minter authorization
    try:
        require_perm(role, "admin:keys")
    except HTTPException:
        log_key_mint_audit(
            principal=principal,
            requested_scopes=requested_scopes,
            granted_scopes=[],
            tenant_id=requested_tenant,
            ip=client_ip,
            success=False,
            error="forbidden",
        )
        raise

    # 4. Single-tenant: keys may only be minted for this process tenant.
    from src.ops.tenant import get_tenant

    process_tenant = get_tenant()
    caller_tenant = claims.get("tenant_id") or process_tenant
    if requested_tenant != process_tenant or (
        caller_tenant and caller_tenant != requested_tenant
    ):
        log_key_mint_audit(
            principal=principal,
            requested_scopes=requested_scopes,
            granted_scopes=[],
            tenant_id=requested_tenant,
            ip=client_ip,
            success=False,
            error="tenant_mismatch",
        )
        raise HTTPException(
            status_code=403,
            detail=(
                f"single_tenant: cannot mint for {requested_tenant!r} "
                f"(process tenant is {process_tenant!r})"
            ),
        )

    # 5. Create scoped key
    try:
        res = create_scoped_key(requested_scopes, tenant_id=requested_tenant)
        log_key_mint_audit(
            principal=principal,
            requested_scopes=requested_scopes,
            granted_scopes=requested_scopes,
            tenant_id=requested_tenant,
            ip=client_ip,
            success=True,
        )
        return res
    except ValueError as e:
        log_key_mint_audit(
            principal=principal,
            requested_scopes=requested_scopes,
            granted_scopes=[],
            tenant_id=requested_tenant,
            ip=client_ip,
            success=False,
            error=str(e),
        )
        raise HTTPException(400, str(e)) from e


@router.get("/webhooks/catalog")
async def webhooks_catalog() -> dict[str, Any]:
    from src.frontline.webhook_catalog import list_catalog

    return {"events": list_catalog()}


@router.post("/webhooks/tick")
async def webhooks_tick(body: dict[str, Any] | None = None) -> dict[str, Any]:
    from src.frontline.webhook_catalog import fire_due_tick

    body = body or {}
    return fire_due_tick(force=True, channel=str(body.get("channel") or "slack"))


@router.post("/nl/query")
async def nl_query(body: dict[str, Any]) -> dict[str, Any]:
    from src.enterprise.grounded_nl import grounded_query

    return grounded_query(
        str(body.get("question") or ""),
        pack_id=str(body.get("pack_id") or "automotive_nhtsa"),
        plant=body.get("plant") if isinstance(body.get("plant"), dict) else None,
    )


@router.post("/views")
async def views_save(body: dict[str, Any]) -> dict[str, Any]:
    from src.frontline.saved_views import save_view

    return save_view(str(body.get("name") or "untitled"), body.get("spec") or {})


@router.post("/reports")
async def reports_build(body: dict[str, Any]) -> dict[str, Any]:
    from src.frontline.reports import build_report

    return build_report(str(body.get("title") or "Report"), kpi_ids=body.get("kpi_ids"), notes=str(body.get("notes") or ""))


@router.post("/reports/digest")
async def reports_digest(body: dict[str, Any] | None = None) -> dict[str, Any]:
    from src.frontline.reports import run_scheduled_digest

    body = body or {}
    return run_scheduled_digest(body.get("report_id"))


@router.get("/ops/slo")
async def ops_slo() -> dict[str, Any]:
    from src.observability.slo import evaluate_slos, job_queue_status, slo_dashboard

    return {"slo": slo_dashboard(), "jobs": job_queue_status(), "eval": evaluate_slos()}


@router.get("/ops/workers")
async def ops_workers() -> dict[str, Any]:
    from src.jobs.registry import list_workers

    return {"workers": list_workers()}


@router.post("/compliance/{key}/toggle")
async def compliance_toggle(key: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    from src.frontline.compliance_packs import toggle_pack

    body = body or {}
    try:
        return toggle_pack(key, bool(body.get("enabled", True)))
    except KeyError as e:
        raise HTTPException(404, str(e)) from e


@router.get("/compliance/{key}/template")
async def compliance_template(key: str) -> dict[str, Any]:
    from src.frontline.compliance_packs import evidence_template

    try:
        return evidence_template(key, record_id="NHTSA-100001", investigation_id="inv_0001", interaction_id="int_demo")
    except KeyError as e:
        raise HTTPException(404, str(e)) from e


@router.get("/copq/rank")
async def copq_rank(window_days: int = 7, include_simulated: bool = False) -> dict[str, Any]:
    from src.qubot.retrievers import live_risk_by_dollar

    rows = live_risk_by_dollar(window_days, include_simulated=include_simulated)
    return {"clusters": rows}


@router.get("/copq/roi")
async def copq_roi(issues_caught: int = 0, copq_per_issue: float = 0, lead_time_weeks: float = 0) -> dict[str, Any]:
    from src.frontline.copq import roi_attribution

    return roi_attribution(
        issues_caught=issues_caught,
        copq_per_issue=copq_per_issue,
        lead_time_weeks=lead_time_weeks,
    )


@router.get("/analytics/hotspots-map")
async def hotspots_map(pack_id: str | None = None, window_days: int = 90) -> dict[str, Any]:
    from src.frontline.analytics import geographic_hotspots

    return geographic_hotspots(pack_id=pack_id, window_days=window_days)


@router.get("/widget/skew-widget.js")
async def widget_script() -> FileResponse:
    path = REPO_ROOT / "dashboard" / "public" / "skew-widget.js"
    if not path.is_file():
        raise HTTPException(404, "widget missing")
    return FileResponse(path, media_type="application/javascript")


@router.get("/scoped")
async def scoped_probe(x_scoped_key: str | None = Header(default=None, alias="X-Scoped-Key")) -> dict[str, Any]:
    from src.security.scoped_keys import authenticate_scoped

    result = authenticate_scoped(x_scoped_key or "", "kpi:read")
    if not result.get("ok"):
        raise HTTPException(403, result.get("reason") or "denied")
    from src.frontline.provenance import list_kpis

    return {"ok": True, "kpis": list_kpis()}
