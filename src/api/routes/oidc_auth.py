"""Public Google / OIDC routes (no API-key gate — the browser starts here)."""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse, RedirectResponse

from src.api.auth import require_api_key
from src.api.limiter import limiter
from src.api.rbac import get_actor, require_perm_dep
from src.frontline.oidc import (
    begin_login,
    consume_handoff,
    finish_callback,
    provider_status,
    safe_post_login,
    save_local_provider,
)

router = APIRouter(prefix="/api/frontline", tags=["auth-oidc"])


def _session_cookie(resp: JSONResponse | RedirectResponse, token: str, *, max_age: int = 3600) -> None:
    """Set the session cookie with production-compatible attributes (item 19).

    - ``__Host-`` prefix + Secure + HttpOnly + SameSite=Lax + Path=/ when
      production-like (browsers enforce the prefix constraints);
    - unprefixed name over local http dev (Secure cookies would not store).
    Login always mints and sets a FRESH token, rotating any pre-existing
    session (fixation defense); logout clears both names.
    """
    if not token:
        return
    from src.api.rbac import session_cookie_name

    try:
        from src.security.harden import is_production_like

        _secure = is_production_like()
    except Exception:
        _secure = False
    resp.set_cookie(
        key=session_cookie_name(),
        value=token,
        httponly=True,
        secure=_secure,
        samesite="lax",
        path="/",
        max_age=max(60, int(max_age)),
    )


@router.get("/auth/oidc/status")
async def oidc_status(
    request: Request,
    actor: str = Depends(get_actor),
) -> dict[str, Any]:
    st = provider_status()
    from src.api.auth import is_open_mode

    is_auth = bool(actor and actor != "anonymous" or is_open_mode())
    client_id_val = (
        st["client_id"]
        if is_auth
        else ("***configured***" if st["configured"] else "")
    )
    return {
        "configured": st["configured"],
        "provider": st["provider"],
        "issuer": st["issuer"],
        "client_id": client_id_val,
        "start_path": st["start_path"],
        "scopes": st["scopes"],
        "authorization_endpoint": st["authorization_endpoint"],
        "redirect_uri": st["redirect_uri"],
        "source": st.get("source"),
    }


@router.put(
    "/auth/oidc/config",
    dependencies=[Depends(require_api_key)],
)
async def oidc_config_put(
    body: dict[str, Any],
    _role: str = Depends(require_perm_dep("pack:edit", open_mode_ok=True)),
) -> dict[str, Any]:
    """Save Google OAuth client for this machine.

    Authentication (API key) AND authorization (admin ``pack:edit``) are both
    required (item 12): an ordinary API key resolves to the ``service``
    principal, which must not rewrite the IdP configuration. Open local demos
    keep working via ``open_mode_ok``; production-like deploys refuse local
    provider writes entirely (see ``save_local_provider``).
    """
    try:
        st = save_local_provider(
            client_id=str(body.get("client_id") or ""),
            client_secret=str(body.get("client_secret") or ""),
            redirect_uri=str(body.get("redirect_uri") or ""),
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return {
        "configured": st["configured"],
        "provider": st["provider"],
        "client_id": st["client_id"],
        "redirect_uri": st["redirect_uri"],
        "source": st.get("source"),
        "start_path": st["start_path"],
    }


@router.get("/auth/oidc/start")
async def oidc_start(next: str | None = Query(default=None)) -> RedirectResponse:
    try:
        begun = begin_login(next_url=next)
    except RuntimeError:
        raise HTTPException(
            status_code=503,
            detail="Google sign-in is not configured. Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET.",
        ) from None
    return RedirectResponse(begun["authorize_url"], status_code=302)


@router.get("/auth/oidc/callback")
async def oidc_callback(
    request: Request,
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
) -> RedirectResponse:
    dest_err = safe_post_login(None, page="signin")
    dest_ok = safe_post_login(None, page="command")
    if error:
        return RedirectResponse(f"{dest_err}?error={quote(str(error))}", status_code=302)
    if not code or not state:
        return RedirectResponse(f"{dest_err}?error=missing_code", status_code=302)
    try:
        done = finish_callback(code=code, state=state)
    except ValueError as exc:
        return RedirectResponse(f"{dest_err}?error={quote(str(exc))}", status_code=302)
    nxt = done.get("next_url") or dest_ok
    if nxt.endswith("#signin"):
        nxt = nxt[: -len("signin")] + "command"
    # SECURITY: Use URL fragment instead of query string so proxies, access logs,
    # and Referer headers never leak the handoff ID.
    if "#" in nxt:
        base_url, frag = nxt.split("#", 1)
        loc = f"{base_url}#{frag}&handoff={done['handoff']}" if frag else f"{base_url}#handoff={done['handoff']}"
    else:
        loc = f"{nxt}#handoff={done['handoff']}"
    resp = RedirectResponse(loc, status_code=302)
    _session_cookie(resp, str(done.get("token") or ""))
    return resp


@router.post("/auth/logout")
async def auth_logout(request: Request = None) -> JSONResponse:  # type: ignore[assignment]
    from src.api.rbac import SESSION_COOKIE_HOST, SESSION_COOKIE_LEGACY, revoke_session, session_token_from_cookies

    # Revoke active session token so it cannot be replayed even before exp
    if request:
        token = session_token_from_cookies(getattr(request, "cookies", {}) or {})
        if not token:
            auth = (request.headers.get("authorization") or request.headers.get("x-frontline-session") or "").strip()
            if auth.lower().startswith("bearer "):
                token = auth[7:].strip()
            elif auth:
                token = auth
        if token:
            revoke_session(token)

    resp = JSONResponse({"signed_in": False, "revoked": True})
    # Invalidate both cookie names (rotation pair); the client keeps no token.
    try:
        from src.security.harden import is_production_like

        _secure = is_production_like()
    except Exception:
        _secure = False
    for name in (SESSION_COOKIE_HOST, SESSION_COOKIE_LEGACY):
        resp.delete_cookie(name, path="/", secure=_secure, samesite="lax")
    return resp


@router.post("/auth/oidc/complete")
async def oidc_complete(body: dict[str, Any]) -> JSONResponse:
    try:
        out = consume_handoff(str(body.get("handoff") or ""))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    token = out.pop("token", "")
    resp = JSONResponse(out)
    _session_cookie(resp, token)
    return resp


def _map_role(role_name: str) -> str:
    # SECURITY: self-supplied display titles must never elevate. Signup/login
    # always mint "agent"; elevation requires an admin session via the
    # (currently closed) session-mint flow or OIDC IdP mapping. Kept for
    # backwards-compat display only — always returns agent.
    return "agent"


def _hash_password(password: str, salt: str) -> str:
    import hashlib

    try:
        raw = hashlib.scrypt(
            password.encode("utf-8"), salt=salt.encode("utf-8"), n=16384, r=8, p=1
        ).hex()
        return f"scrypt:{raw}"
    except Exception:
        return hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt.encode("utf-8"), 600_000
        ).hex()


def _verify_password(password: str, salt: str, expected_hash: str) -> bool:
    import hashlib
    import hmac

    if expected_hash.startswith("scrypt:"):
        try:
            raw = hashlib.scrypt(
                password.encode("utf-8"), salt=salt.encode("utf-8"), n=16384, r=8, p=1
            ).hex()
            return hmac.compare_digest(expected_hash, f"scrypt:{raw}")
        except Exception:
            return False
    calc_100k = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000
    ).hex()
    calc_600k = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), 600_000
    ).hex()
    return hmac.compare_digest(expected_hash, calc_100k) or hmac.compare_digest(
        expected_hash, calc_600k
    )


@router.post("/auth/signup")
@limiter.limit("10 per minute")
async def auth_signup(request: Request, body: dict[str, Any]) -> JSONResponse:
    import secrets
    from datetime import datetime, timezone
    import hmac
    from src.api.rbac import issue_session
    from src.data.warehouse import ops_con, ops_in_thread
    from src.security.harden import is_production_like

    allow_public = os.getenv(
        "FRONTLINE_ALLOW_PUBLIC_SIGNUP", "1" if not is_production_like() else "0"
    ).strip().lower() in {"1", "true", "yes"}
    if not allow_public:
        raise HTTPException(
            status_code=403,
            detail="Public registration is disabled in this environment. Contact an administrator.",
        )

    name = str(body.get("name") or "").strip()[:120]
    email = str(body.get("email") or "").strip().lower()[:254]
    company = str(body.get("company") or "").strip()[:120]
    # Display title only — never a privilege (see _map_role).
    title = str(body.get("role") or "Operations").strip()[:120]
    password = str(body.get("password") or "")

    if not name or not email or not password:
        raise HTTPException(status_code=400, detail="Name, email, and password are required.")
    if "@" not in email or "." not in email:
        raise HTTPException(status_code=400, detail="A valid email address is required.")
    if len(password) < 12:
        raise HTTPException(status_code=400, detail="Password must be at least 12 characters long.")

    salt = secrets.token_hex(16)
    pwd_hash = _hash_password(password, salt)
    user_id = f"usr_{secrets.token_hex(8)}"
    now = datetime.now(timezone.utc)

    def _create_user():
        with ops_con() as con:
            # Ensure table exists
            con.execute("""
            CREATE TABLE IF NOT EXISTS frontline_users (
                user_id            VARCHAR PRIMARY KEY,
                name               VARCHAR NOT NULL,
                email              VARCHAR UNIQUE NOT NULL,
                company            VARCHAR,
                role               VARCHAR NOT NULL,
                password_hash      VARCHAR NOT NULL,
                salt               VARCHAR NOT NULL,
                created_at         TIMESTAMP NOT NULL,
                last_login_at      TIMESTAMP
            );
            """)
            found = con.execute("SELECT email FROM frontline_users WHERE email = ?", [email]).fetchone()
            if found:
                raise HTTPException(status_code=409, detail="An account with this email address already exists.")
            con.execute(
                """
                INSERT INTO frontline_users (
                    user_id, name, email, company, role, password_hash, salt, created_at, last_login_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [user_id, name, email, company, title, pwd_hash, salt, now, now]
            )

    try:
        await ops_in_thread(_create_user)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Database error during registration.")

    # Always agent — elevation requires admin action, never self-selection.
    session_data = issue_session(
        subject=email,
        role="agent",
        ttl_s=86400,
        extra={"name": name, "company": company, "title": title},
    )
    token = session_data["token"]
    resp = JSONResponse({
        "success": True,
        "subject": email,
        "user": {
            "name": name,
            "email": email,
            "company": company,
            "role": title,
        },
    })
    # Cookie lifetime must match the minted token (open mode clamps TTL).
    import time as _time

    _session_cookie(resp, token, max_age=max(60, int(session_data["exp"] - _time.time())))
    return resp


@router.post("/auth/login")
@limiter.limit("10 per minute")
async def auth_login(request: Request, body: dict[str, Any]) -> JSONResponse:
    import secrets
    from datetime import datetime, timezone
    import hmac
    from src.api.rbac import issue_session
    from src.data.warehouse import ops_con, ops_in_thread

    email = str(body.get("email") or "").strip().lower()[:254]
    password = str(body.get("password") or "")
    remember = bool(body.get("remember", True))

    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password are required.")

    # SECURITY: removed hardcoded demo backdoor (pilot.sre@skew.ai /
    # alex.mercer@skew.ai + static passwords → admin). Demo logins must use
    # real DB accounts or OIDC; no static credentials in source.

    def _lookup_user():
        with ops_con() as con:
            con.execute("""
            CREATE TABLE IF NOT EXISTS frontline_users (
                user_id            VARCHAR PRIMARY KEY,
                name               VARCHAR NOT NULL,
                email              VARCHAR UNIQUE NOT NULL,
                company            VARCHAR,
                role               VARCHAR NOT NULL,
                password_hash      VARCHAR NOT NULL,
                salt               VARCHAR NOT NULL,
                created_at         TIMESTAMP NOT NULL,
                last_login_at      TIMESTAMP
            );
            """)
            row = con.execute(
                """
                SELECT user_id, name, email, company, role, password_hash, salt
                FROM frontline_users WHERE email = ?
                """,
                [email]
            ).fetchone()
            if row:
                con.execute("UPDATE frontline_users SET last_login_at = ? WHERE email = ?", [datetime.now(timezone.utc), email])
            return row

    try:
        user_row = await ops_in_thread(_lookup_user)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Database error.")

    if not user_row:
        # Generic message — do not distinguish unknown email vs bad password.
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    user_id, name, email_val, company, title, expected_hash, salt = user_row
    if not _verify_password(password, salt, expected_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    # Always agent — stored display title never elevates.
    ttl = 86400 if remember else 3600
    session_data = issue_session(
        subject=email_val,
        role="agent",
        ttl_s=ttl,
        extra={"name": name, "company": company, "title": title},
    )
    token = session_data["token"]
    resp = JSONResponse({
        "success": True,
        "subject": email_val,
        "user": {
            "name": name,
            "email": email_val,
            "company": company,
            "role": title,
        },
    })
    import time as _time

    _session_cookie(resp, token, max_age=max(60, int(session_data["exp"] - _time.time())))
    return resp
