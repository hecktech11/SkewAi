"""Skew AI (skewai) FastAPI app.

Mounts the interactions, packs, and frontline routers. Adds CORS for the
dashboard dev server, slowapi rate-limiting on the start/simulate endpoints,
and a health check. Product brand is Skew AI; API path prefixes remain
``/api/frontline/*`` for compatibility.

Run with:
    uvicorn src.api.main:app --reload --port 8000
"""

from __future__ import annotations

import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from starlette.types import ASGIApp, Receive, Scope, Send

from src.api.frontline_gate import frontline_enabled, is_frontline_surface_path
from src.api.limiter import limiter
from src.api.problems import install_openapi_problem, problem, register_exception_handlers
from src.api.routes import frontline, interactions, packs
from src.config import REPO_ROOT, settings
from src.domains.active_pack import resolve_active_pack_id

# Stable pilot API version (URL stays /api/...; bump only on breaking changes).
API_VERSION = "1"


class FrontlineEnabledASGI:
    """Pure ASGI gate for HTTP **and** WebSocket scopes.

    ``BaseHTTPMiddleware`` never runs for ``websocket`` scopes, so the path
    check for ``/ws/*`` must live here (or in each WS handler).
    When FRONTLINE_ENABLED=0: ``/api/*`` → 503, ``/ws/*`` → close without
    accept. ``/health``, ``/health/ready``, and ``/ui`` stay up.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] in ("http", "websocket"):
            path = scope.get("path") or ""
            if not frontline_enabled() and is_frontline_surface_path(path):
                if scope["type"] == "http":
                    response = JSONResponse(
                        status_code=503,
                        content=problem(
                            503,
                            "Skew AI is disabled (FRONTLINE_ENABLED=0).",
                            extra={"frontline_enabled": False},
                        ),
                    )
                    await response(scope, receive, send)
                    return
                # WebSocket: refuse before accept (policy violation / try later).
                await send({"type": "websocket.close", "code": 1013})
                return
        await self.app(scope, receive, send)


from contextlib import asynccontextmanager

from src.security.harden import is_production_like


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # SOC/production fail-closed: refuse open mode + weak secrets when hardened.
    from src.security.harden import validate_startup_security

    validate_startup_security()
    # ML semantic fail-closed validation (audit F-007):
    from src.ml_runtime.embedding_runtime import validate_startup_config

    validate_startup_config()
    try:
        from src.observability.degradation import init_default_ladders

        init_default_ladders()
    except Exception:
        pass
    # Production-like: refuse to serve if pack/gazetteer/DB/secrets are dead.
    # Local tests and demos skip this; override with FRONTLINE_READINESS_SKIP_STARTUP=1.
    if is_production_like() and os.getenv(
        "FRONTLINE_READINESS_SKIP_STARTUP", ""
    ).strip().lower() not in {"1", "true", "yes", "on"}:
        _assert_startup_readiness()
    # Secret redaction in logs (item 50): API keys / session secrets must
    # never appear in Docker logs, access logs, or debug output.
    try:
        from src.security.secrets import install_secret_redaction

        install_secret_redaction()
    except Exception:
        pass
    # Ledger WAL replay (audit X.5): safety actions delivered from the local
    # WAL during a DB outage rejoin the chain on recovery. Best-effort.
    try:
        from src.ledger import replay_ledger_wal

        replay_ledger_wal()
    except Exception:
        pass
    # Feature #51: SIGTERM begins call drain (reject new contacts, finish active).
    try:
        from src.ops.drain import install_sigterm_handler

        install_sigterm_handler()
    except Exception:
        pass
    # Startup: close stale active interaction rows not in the live registry.
    reaper_task = None
    try:
        from src.api.routes.interactions import reap_orphans, reaper_loop

        await reap_orphans()
        # Periodic sweep: a customer WS that drops now keeps the contact
        # resumable, so something must finalize it if the browser never returns.
        import asyncio as _asyncio

        reaper_task = _asyncio.create_task(reaper_loop())
    except Exception:
        pass
    hb_task = None
    try:
        from src.jobs.registry import heartbeat, register_worker

        rec = register_worker(host="api")
        import asyncio as _asyncio

        async def _heartbeat_loop() -> None:
            while True:
                await _asyncio.sleep(30)
                heartbeat(rec["worker_id"])

        hb_task = _asyncio.create_task(_heartbeat_loop())
    except Exception:
        pass
    worker_task = None
    _inline_off = os.getenv("FRONTLINE_DISABLE_INLINE_WORKER", "").strip().lower() in {
        "1", "true", "yes", "on",
    }
    if not os.getenv("FRONTLINE_TEST_ISOLATION") and not _inline_off:
        try:
            from src.jobs.worker import worker_tick as _worker_tick

            import asyncio as _asyncio

            async def _job_worker() -> None:
                last_fairness = 0.0
                while True:
                    await _asyncio.sleep(5)
                    now = _asyncio.get_event_loop().time()
                    due = (now - last_fairness) >= 60.0
                    try:
                        await _asyncio.to_thread(_worker_tick, run_fairness=due)
                    except Exception:
                        pass
                    if due:
                        last_fairness = now

            worker_task = _asyncio.create_task(_job_worker())
        except Exception:
            pass
    try:
        yield
    finally:
        if worker_task is not None:
            worker_task.cancel()
            try:
                await worker_task
            except BaseException:
                pass
        if hb_task is not None:
            hb_task.cancel()
            try:
                await hb_task
            except BaseException:
                pass
        if reaper_task is not None:
            reaper_task.cancel()
            try:
                await reaper_task
            except BaseException:
                pass


# Production-like: disable interactive docs / OpenAPI (anonymous attack surface).
# Local demos keep /docs. Override with FRONTLINE_OPENAPI_PUBLIC=1 if needed.
_docs_public = (
    os.getenv("FRONTLINE_OPENAPI_PUBLIC", "").strip().lower() in {"1", "true", "yes", "on"}
    or not is_production_like()
)

app = FastAPI(
    title="Skew AI — Voice-of-Customer Intelligence Platform",
    description=(
        "Skew AI (skewai): domain-agnostic voice-of-customer intelligence. "
        "Plug in a Domain Pack and get an AI voice agent, a swarm of triage/RCA "
        "agents, and the Qubot v2 auditor.\n\n"
        "## Auth\n"
        "When `FRONTLINE_API_KEY` is set, send `X-API-Key` or `Authorization: Bearer`.\n\n"
        "## Versioning\n"
        f"Current API version **{API_VERSION}** (header `API-Version`). "
        "Paths under `/api/...` are v1 without a URL prefix. Breaking changes will "
        "introduce `/api/v2/...`.\n\n"
        "## Errors\n"
        "Errors use RFC 7807-shaped JSON (`type`, `title`, `status`, `detail`) "
        "while keeping FastAPI's `detail` string for existing clients.\n\n"
        "## Pagination\n"
        "Collection endpoints accept `limit`, `offset`, and optional `cursor`, "
        "and return a `pagination` object (`has_more`, `next_offset`, `next_cursor`)."
    ),
    version="2.0.0",
    lifespan=_lifespan,
    docs_url="/docs" if _docs_public else None,
    redoc_url="/redoc" if _docs_public else None,
    openapi_url="/openapi.json" if _docs_public else None,
    openapi_tags=[
        {"name": "interactions", "description": "Live contacts and takeover"},
        {"name": "packs", "description": "Domain pack discovery and activation"},
        {"name": "frontline", "description": "Skew AI ops: cases, investigations, audits"},
        {"name": "enterprise", "description": "Enterprise ops explorers"},
        {"name": "v3-platform", "description": "v3 OS: learning, experiments, governance"},
        {"name": "platform56", "description": "56-feature surfaces"},
        {"name": "meta", "description": "Health and discovery"},
    ],
)

# Wire slowapi
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
from src.security.identifiers import InvalidIdentifier

@app.exception_handler(InvalidIdentifier)
async def _invalid_identifier(_request, exc: InvalidIdentifier):
    return JSONResponse({"detail": str(exc)}, status_code=400)

register_exception_handlers(app)
install_openapi_problem(app)

# ── CORS (env-configurable; safe local defaults) ─────────────────────────────
_DEFAULT_CORS = [
    "http://127.0.0.1:8787",
    "http://localhost:8787",
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "http://127.0.0.1:8000",
    "http://localhost:8000",
]


def _cors_origins() -> list[str]:
    """Allowlist origins. Never accept ``*`` with credentials (M11)."""
    raw = os.getenv("CORS_ALLOW_ORIGINS", "").strip()
    if raw:
        origins = [o.strip() for o in raw.split(",") if o.strip()]
    else:
        origins = list(_DEFAULT_CORS)
    cleaned: list[str] = []
    for o in origins:
        if o == "*":
            # Credentialed CORS + wildcard is unsafe; drop and keep defaults.
            continue
        cleaned.append(o)
    return cleaned or list(_DEFAULT_CORS)


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key", "X-Frontline-Session", "X-Frontline-Role"],
)
# Pure ASGI (not BaseHTTPMiddleware) so /ws/* is gated too.
app.add_middleware(FrontlineEnabledASGI)  # type: ignore[arg-type]
from src.api.identifier_guard import IdentifierGuardASGI

app.add_middleware(IdentifierGuardASGI)  # type: ignore[arg-type]

# Security headers (CSP, frame deny, nosniff, HSTS) — SOC 2 CC6 baseline.
from src.security.headers import SecurityHeadersASGI

app.add_middleware(SecurityHeadersASGI)  # type: ignore[arg-type]


class ObservabilityASGI:
    """Request correlation + timing (item 47).

    Assigns every HTTP request an ``X-Request-ID`` (incoming value honored
    when well-formed), binds it to the log context, and records a timed
    span. Pure ASGI so it also stamps WebSocket scopes. Never raises into
    the app — observability failures must not crash business logic.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        try:
            from src.observability.otel import bind_request, new_request_id

            headers = dict(
                (k.decode("latin-1").lower(), v.decode("latin-1"))
                for k, v in (scope.get("headers") or [])
            )
            incoming = (headers.get("x-request-id") or "").strip()
            rid = incoming[:64] if incoming else new_request_id()
            bind_request(rid)
            scope["state"] = {**(scope.get("state") or {}), "request_id": rid}
        except Exception:
            rid = ""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        import time as _time

        _start = _time.perf_counter()

        async def send_with_obs(message: dict) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers") or [])
                if rid:
                    headers.append((b"x-request-id", rid.encode("ascii")))
                message = {**message, "headers": headers}
                try:
                    from src.observability.metrics import observe as _observe

                    _observe(
                        "http_request_ms",
                        (_time.perf_counter() - _start) * 1000.0,
                    )
                except Exception:
                    pass
            await send(message)

        try:
            await self.app(scope, receive, send_with_obs)
        except Exception:
            raise


app.add_middleware(ObservabilityASGI)  # type: ignore[arg-type]


class ApiVersionASGI:
    """Attach stable API-Version header to every HTTP response."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_version(message: dict) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers") or [])
                headers.append((b"api-version", API_VERSION.encode("ascii")))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_version)


app.add_middleware(ApiVersionASGI)  # type: ignore[arg-type]


class V1AliasASGI:
    """10/10 API versioning: /api/v1/* is an alias of canonical /api/*.

    Routers stay single-source; this rewrite runs before routing so clients
    that pin /v1/ never break. Non-v1 paths pass through untouched.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            path = scope.get("path") or ""
            if path == "/api/v1" or path.startswith("/api/v1/"):
                scope = {**scope, "path": "/api/" + path[len("/api/v1/"):].lstrip("/"),
                         "raw_path": scope.get("raw_path")}
        except Exception:
            pass
        await self.app(scope, receive, send)


app.add_middleware(V1AliasASGI)  # type: ignore[arg-type]


# ── Routers ──────────────────────────────────────────────────────────────────
from src.api.routes import enterprise as enterprise_routes
from src.api.routes import hardening as hardening_routes
from src.api.routes import pipeline as pipeline_routes
from src.api.routes import oidc_auth as oidc_auth_routes
from src.api.routes import platform56 as platform56_routes
from src.api.routes import v3_platform as v3_routes

app.include_router(interactions.router)
app.include_router(packs.router)
app.include_router(frontline.router)
app.include_router(hardening_routes.router)
app.include_router(oidc_auth_routes.router)
app.include_router(platform56_routes.router)
app.include_router(enterprise_routes.router)
app.include_router(v3_routes.router)
app.include_router(pipeline_routes.router)

# Blueprint §8.2: WebSocket endpoints live at the app root (not under /api/...).
app.add_api_websocket_route(
    "/ws/interaction/{interaction_id}",
    interactions.interaction_ws,
)
app.add_api_websocket_route(
    "/ws/twilio/{interaction_id}",
    interactions.twilio_ws,
)
app.add_api_websocket_route("/ws/console", interactions.console_ws)


# ── Health + root (always public — no API key) ───────────────────────────────


def _check_authenticated(request: Request) -> bool:
    try:
        from src.api.auth import _extract_key, check_api_key

        key = _extract_key(
            authorization=request.headers.get("authorization"),
            x_api_key=request.headers.get("x-api-key"),
        )
        if key:
            check_api_key(
                authorization=request.headers.get("authorization"),
                x_api_key=request.headers.get("x-api-key"),
                allow_open=False,
            )
            return True
    except Exception:
        pass
    try:
        from src.api.rbac import session_token_from_cookies, verify_session

        tok = session_token_from_cookies(getattr(request, "cookies", {}) or {}) or request.headers.get(
            "x-frontline-session"
        )
        if tok:
            verify_session(tok)
            return True
    except Exception:
        pass
    return False


@app.get("/", tags=["meta"])
async def root(request: Request) -> dict:
    # Match real policy (production-like / AUTH_REQUIRED), not key presence alone.
    from src.api.auth import auth_required as auth_is_required
    from src.security.harden import is_production_like

    auth = auth_is_required()
    is_auth = _check_authenticated(request)
    docs_on = (
        os.getenv("FRONTLINE_OPENAPI_PUBLIC", "").strip().lower()
        in {"1", "true", "yes", "on"}
        or (not is_production_like() and is_auth)
    )
    res: dict[str, Any] = {
        "name": "Skew AI",
        "brand": "skewai",
        "version": "2.0.0",
        "api_version": API_VERSION,
        "auth_required": auth,
        "auth": {
            "type": "api_key",
            "headers": ["X-API-Key", "Authorization: Bearer"],
            "required": auth,
        },
    }
    if not is_production_like() or is_auth:
        res["active_pack"] = resolve_active_pack_id()
        res["frontline_enabled"] = frontline_enabled()
        res["docs"] = "/docs" if docs_on else None
        res["openapi"] = "/openapi.json" if docs_on else None
        res["redoc"] = "/redoc" if docs_on else None
        res["links"] = {
            "health": "/health",
            "health_ready": "/health/ready",
            "interactions": "/api/interactions",
            "cases": "/api/frontline/cases",
            "packs": "/api/packs",
            "enterprise": "/api/frontline/enterprise",
            "v3_learning": "/api/v3/learning/proposals",
            "v3_experiments": "/api/v3/experiments",
            "v3_governance": "/api/v3/governance/deployments",
        }
    return res


@app.get("/health", tags=["meta"])
async def health(request: Request) -> dict:
    """Deep-ish health: process up + ops DB reachability + active pack loadable."""
    from src.data.warehouse import ops_con
    from src.domains.loader import load_pack

    pack_id = resolve_active_pack_id()
    db_ok = False
    pack_ok = False
    detail: dict = {}
    try:
        def _ping() -> None:
            with ops_con(read_only=True) as con:
                con.execute("SELECT 1").fetchone()

        from src.data.warehouse import ops_in_thread

        await ops_in_thread(_ping)
        db_ok = True
    except Exception as e:
        detail["db_error"] = type(e).__name__
    pack = None
    try:
        pack = load_pack(pack_id)
        pack_ok = True
    except Exception as e:
        detail["pack_error"] = type(e).__name__
    # Readiness (audit 0.5): a booted process with a missing model, empty
    # gazetteers, or no cost_model would silently run degraded forever.
    # Readiness fails CLOSED (503) unless explicitly waived per check via
    # FRONTLINE_READINESS_WAIVE=comma,list.
    readiness = _readiness_report(pack_id, pack, db_ok=db_ok)
    if not readiness["ready"]:
        detail["readiness"] = readiness

    from src.api.auth import auth_required as auth_is_required
    from src.jobs.registry import backend_name, worker_count
    from src.security.harden import is_production_like

    auth_req = auth_is_required()
    prod = is_production_like()
    is_auth = _check_authenticated(request)

    status = "ok" if (db_ok and pack_ok and readiness["ready"]) else "degraded"

    # In production or when auth is required, unauthenticated callers receive
    # a sanitized health response without internal architecture fingerprints.
    if (prod or auth_req) and not is_auth:
        from fastapi.responses import JSONResponse

        public_body = {
            "status": "ok" if readiness["ready"] else "not_ready",
            "api_version": API_VERSION,
            "active_pack": pack_id,
            "auth_required": auth_req,
        }
        if not readiness["ready"]:
            return JSONResponse(status_code=503, content=public_body)
        return public_body

    n_workers = worker_count()
    body = {
        "status": status,
        "api_version": API_VERSION,
        "active_pack": pack_id,
        "frontline_enabled": frontline_enabled(),
        "db_ok": db_ok,
        "pack_ok": pack_ok,
        "single_worker": n_workers <= 1,
        "worker_count": n_workers,
        "orchestrator_registry": "in_process_ws+distributed_close_claims",
        "worker_registry": backend_name(),
        "security": {
            "auth_required": auth_req,
            "production_like": prod,
            "audit_log": True,
            "note": (
                "Engineering controls only. Not a SOC 2 Type II attestation."
            ),
        },
        "llm_available": settings.llm_available,
        "auth_required": auth_req,
        "v3_runtime_routing": False,
        "ml_severity_shipped": False,
        "enterprise_readiness": (
            "single_tenant_hardened"
            if auth_req
            else "single_tenant_pilot_open"
        ),
    }
    try:
        from src.ml_runtime.embedding_runtime import health_payload as _emb_health

        body["embedding"] = _emb_health()
    except Exception as e:
        body["embedding"] = {"mode": "unknown", "error": type(e).__name__}
    if detail:
        body["detail"] = detail
    body["readiness"] = readiness

    if not readiness["ready"]:
        from fastapi.responses import JSONResponse

        body["status"] = "not_ready"
        return JSONResponse(status_code=503, content=body)
    return body


@app.get("/health/ready", tags=["meta"])
async def health_ready() -> dict:
    """Readiness probe: pack, gazetteer lookup, ops DB, domain warehouse, secrets.

    Returns 200 only when every required component is live. Otherwise 503 with
    ``failing_component`` set to the first failed check name.
    """
    from src.data.warehouse import ops_con, ops_in_thread
    from src.domains.loader import load_pack

    pack_id = resolve_active_pack_id()
    db_ok = False
    pack = None
    try:
        def _ping() -> None:
            with ops_con(read_only=True) as con:
                con.execute("SELECT 1").fetchone()

        await ops_in_thread(_ping)
        db_ok = True
    except Exception:
        db_ok = False
    try:
        pack = load_pack(pack_id)
    except Exception:
        pack = None
    report = _readiness_report(pack_id, pack, db_ok=db_ok)
    body = {
        "status": "ready" if report["ready"] else "not_ready",
        "ready": report["ready"],
        "active_pack": pack_id,
        "failing_component": report.get("failing_component"),
        "failing_components": report.get("failing_components") or [],
        "checks": report["checks"],
    }
    if not report["ready"]:
        return JSONResponse(status_code=503, content=body)
    return body


def _assert_startup_readiness() -> None:
    """Refuse to serve live traffic when a required component is missing."""
    from src.data.warehouse import ops_con
    from src.domains.loader import load_pack

    pack_id = resolve_active_pack_id()
    db_ok = False
    pack = None
    try:
        with ops_con(read_only=True) as con:
            con.execute("SELECT 1").fetchone()
        db_ok = True
    except Exception:
        db_ok = False
    try:
        pack = load_pack(pack_id)
    except Exception:
        pack = None
    report = _readiness_report(pack_id, pack, db_ok=db_ok)
    if report["ready"]:
        return
    failed = report.get("failing_components") or []
    raise RuntimeError(
        "Readiness failed; refusing to serve live traffic. "
        f"failing_component={report.get('failing_component')}; "
        f"failing_components={failed}"
    )


def _readiness_report(pack_id: str, pack: Any | None, *, db_ok: bool) -> dict[str, Any]:
    """Per-artifact readiness (audit 0.5). Fails closed unless waived.

    Waivers: FRONTLINE_READINESS_WAIVE=commaname list, e.g.
    "triage_model,cost_model" for a deliberately rules-only pilot.
    """
    from typing import Any as _Any

    waived = {
        w.strip().lower()
        for w in (os.getenv("FRONTLINE_READINESS_WAIVE") or "").split(",")
        if w.strip()
    }
    checks: dict[str, dict[str, _Any]] = {}

    def _check(name: str, ok: bool, note: str) -> None:
        checks[name] = {
            "ok": bool(ok) or name in waived,
            "waived": name in waived and not ok,
            "note": note,
        }

    _check("database", db_ok, "ops warehouse reachable")
    if pack is None:
        _check("pack", False, f"pack {pack_id} failed to load")
        _check("gazetteers", False, "no pack")
        _check("gazetteer_lookup", False, "no pack")
        _check("triage", False, "no pack")
        _check("cost_model", False, "no pack")
    else:
        _check("pack", True, f"pack {pack_id} loaded")
        try:
            gaz = getattr(pack, "gazetteers", {}) or {}
            terms = sum(len(getattr(g, "values", []) or []) for g in gaz.values())
        except Exception:
            gaz = {}
            terms = 0
        _check("gazetteers", terms > 0, f"{terms} gazetteer terms")
        lookup_ok = False
        lookup_note = "no gazetteer values to look up"
        try:
            for g in gaz.values():
                values = list(getattr(g, "values", []) or [])
                if not values:
                    continue
                probe = str(values[0])
                got = g.lookup(probe)
                lookup_ok = got is not None
                lookup_note = f"lookup({probe!r}) -> {got!r}"
                break
        except Exception as e:
            lookup_ok = False
            lookup_note = f"lookup failed: {type(e).__name__}"
        _check("gazetteer_lookup", lookup_ok if terms > 0 else False, lookup_note)
        try:
            sev = pack.manifest.severity
            artifact = getattr(sev, "model_artifact", None)
            if artifact:
                from pathlib import Path as _P

                from src.config import REPO_ROOT as _RR

                p = _P(str(artifact))
                exists = p.is_file() or (_RR / p).is_file() or (_RR / "domains" / p).is_file()
                _check("triage", exists, f"model artifact {artifact}")
            else:
                rules = list(getattr(sev, "rules", []) or [])
                _check("triage", bool(rules), "rules-only (no artifact declared)")
        except Exception as e:
            _check("triage", False, f"triage inspect failed: {type(e).__name__}")
        try:
            cm = pack.manifest.cost_model
            _check("cost_model", cm is not None, "pack cost_model present")
        except Exception:
            _check("cost_model", False, "no cost_model on manifest")
    try:
        from src.ml_runtime.embedding_runtime import (
            embedding_mode,
            try_semantic_embedder,
            validate_startup_config,
        )

        mode = embedding_mode()
        if mode == "semantic":
            try:
                cfg = validate_startup_config()
                _check(
                    "semantic_embedder",
                    cfg.get("ok", False),
                    "; ".join(cfg.get("errors", [])) or "semantic config validated",
                )
            except Exception as e:
                _check(
                    "semantic_embedder",
                    False,
                    f"semantic validation failed: {e}",
                )
        else:
            _check("semantic_embedder", True, f"not required in {mode} mode")
    except Exception as e:
        _check("semantic_embedder", False, f"inspect failed: {type(e).__name__}")
    try:
        from src.ledger.merkle import latest_head

        latest_head()
        _check("merkle_log", True, "tree-head table reachable")
    except Exception as e:
        _check("merkle_log", False, f"unreachable: {type(e).__name__}")
    try:
        from src.data.warehouse import domain_con as _domain_con

        with _domain_con(pack_id, read_only=True) as dcon:
            dcon.execute("SELECT 1").fetchone()
        _check("domain_warehouse", True, "domain warehouse queryable")
    except Exception as e:
        _check("domain_warehouse", False, type(e).__name__)
    try:
        from src.api.auth import _configured_key
        from src.security.harden import is_production_like, key_strength_problems

        if is_production_like():
            key = _configured_key()
            problems = key_strength_problems(key) if key else ["FRONTLINE_API_KEY is empty"]
            _check(
                "secrets",
                not problems,
                "; ".join(problems) if problems else "required secrets present",
            )
        else:
            _check("secrets", True, "not required outside production-like")
    except Exception as e:
        _check("secrets", False, f"inspect failed: {type(e).__name__}")
    failed = [name for name, c in checks.items() if not c["ok"]]
    ready = not failed
    return {
        "ready": ready,
        "checks": checks,
        "failing_components": failed,
        "failing_component": failed[0] if failed else None,
    }


# Serve built dashboard (Docker / pilot one-box). Prefer SPA at /ui.
_DASH_DIST = REPO_ROOT / "dashboard" / "dist"


class _StreamedStaticFiles(StaticFiles):
    """Disable sendfile/pathsend so ASGI middleware can wrap ``send``.

    Uvicorn advertises ``http.response.pathsend``. Starlette FileResponse then
    sends headers (with Content-Length) and a pathsend frame. Wrapped ``send``
    in our header/obs middleware drops that extension, so clients see
    IncompleteRead / a 200 with an empty body. Streaming the bytes instead
    keeps /ui/ working behind those wrappers.
    """

    async def __call__(self, scope, receive, send):  # type: ignore[override]
        if scope.get("type") == "http":
            extensions = dict(scope.get("extensions") or {})
            extensions.pop("http.response.pathsend", None)
            extensions.pop("http.response.zerocopysend", None)
            scope = {**scope, "extensions": extensions}
        await super().__call__(scope, receive, send)


if _DASH_DIST.is_dir():
    app.mount("/ui", _StreamedStaticFiles(directory=str(_DASH_DIST), html=True), name="dashboard")
