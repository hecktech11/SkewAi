"""RBAC then SSO surface (feature #50).

Roles: agent / supervisor / auditor / service / admin. Pilot uses signed session tokens.
Bare ``X-Frontline-Role`` cannot elevate above agent (FIND-003). Elevated roles
require a signed session.

Session minting (FIND-002):
- Open mode: always issues ``agent`` only (short TTL).
- Hardened: never signs with the literal ``dev-only`` fallback.
- Elevated roles require an admin issuer session (or explicit bootstrap env).

Service principal (H1 reduce):
- Validated shared API key without a session is ``service`` (not full admin).
- Dangerous ops (pack disk edit, drain, jobs run-next, DSR delete) need admin.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import time
from typing import Any

from fastapi import Depends, Header, HTTPException, Request

ROLES = frozenset({"agent", "supervisor", "auditor", "service", "admin", "dsr_officer"})
ELEVATED = frozenset({"supervisor", "auditor", "admin", "dsr_officer"})

#: Session cookie names. ``__Host-`` prefix (item 19) is required where
#: compatible: production-like deploys set Secure, so the prefixed name is
#: enforced by browsers (no Domain, Path=/, Secure). Local http dev keeps
#: the unprefixed name (Secure cookies would never be stored over http).
#: Readers accept both so rotation never locks operators out.
SESSION_COOKIE_HOST = "__Host-frontline_session"
SESSION_COOKIE_LEGACY = "frontline_session"


def session_cookie_name() -> str:
    """Cookie name to SET for new sessions (env-dependent)."""
    try:
        from src.security.harden import is_production_like

        if is_production_like():
            return SESSION_COOKIE_HOST
    except Exception:
        pass
    return SESSION_COOKIE_LEGACY


def session_token_from_cookies(cookies: Any) -> str:
    """Extract the session token, accepting both cookie names.

    The ``__Host-`` variant wins when both are present (it is the only one
    a production-like server sets).
    """
    try:
        get = cookies.get if hasattr(cookies, "get") else (lambda k: None)
        return (
            (get(SESSION_COOKIE_HOST) or "").strip()
            or (get(SESSION_COOKIE_LEGACY) or "").strip()
        )
    except Exception:
        return ""

# Permission matrix (pilot)
PERMS: dict[str, frozenset[str]] = {
    "agent": frozenset({"contact:write", "case:read"}),
    "service": frozenset(
        {
            "contact:write",
            "case:read",
            "takeover",
            "audit:read",
            "ledger:read",
            "channel:ingest",
            "biometrics:match",
            "ops:read",
        }
    ),
    "dsr_officer": frozenset({"dsr:export", "case:read"}),
    "supervisor": frozenset(
        {
            "contact:write",
            "case:read",
            "case:write",
            "takeover",
            "approval:decide",
            "dsr:export",
            "seat:admin",
            "channel:ingest",
            "biometrics:match",
            "routing:control",
            "ops:read",
            "ledger:read",
            "audit:read",
        }
    ),
    "auditor": frozenset({"case:read", "audit:read", "ledger:read", "dsr:export", "ops:read"}),
    "admin": frozenset(
        {
            "*",
            "dsr:export",
            "dsr:delete",
            "marketplace:install",
            "session:mint",
            "admin:keys",
            "deploy:activate",
            "deploy:create",
            "ops:read",
            "ops:write",
            "seat:admin",
            "pack:edit",
            "pack:activate",
            "ops:drain",
            "jobs:run",
            "channel:ingest",
            "biometrics:match",
        }
    ),
}


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _is_open_mode() -> bool:
    from src.api.auth import is_open_mode

    return is_open_mode()


def _auth_required() -> bool:
    from src.api.auth import auth_required

    return auth_required()


def _secret() -> bytes:
    """HMAC material for session tokens.

    Requires explicit SESSION_SECRET whenever authentication is required.
    Never alias to FRONTLINE_API_KEY in any mode (separate cryptographic principals).
    Open mode without auth falls back to dev-only.
    """
    session_sec = (os.getenv("SESSION_SECRET") or "").strip()
    api_key = (os.getenv("FRONTLINE_API_KEY") or "").strip()

    if session_sec and api_key and session_sec == api_key:
        raise HTTPException(
            status_code=503,
            detail="SESSION_SECRET must not match FRONTLINE_API_KEY (cannot alias service API key to session signing key).",
        )

    try:
        from src.security.harden import is_production_like

        if is_production_like() or _env_bool("PILOT_HARDENED", False) or _env_bool("SOC2_MODE", False):
            if not session_sec:
                raise HTTPException(
                    status_code=503,
                    detail="SESSION_SECRET must be configured for session tokens in production-like mode.",
                )
            return session_sec.encode()
    except HTTPException:
        raise
    except Exception:
        pass

    if session_sec:
        return session_sec.encode()

    if _auth_required() or _env_bool("PILOT_HARDENED", False):
        raise HTTPException(
            status_code=503,
            detail=(
                "SESSION_SECRET must be configured for "
                "session tokens when auth is required (refusing dev-only fallback)."
            ),
        )
    return b"dev-only"


def issue_session(
    subject: str,
    role: str,
    *,
    issuer_role: str | None = None,
    ttl_s: int = 3600,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Mint a signed session. Elevated roles need an admin issuer (or bootstrap)."""
    requested = (role or "agent").strip().lower()
    if requested not in ROLES or requested == "service":
        # service is a machine principal, not a mintable end-user session role
        if requested == "service":
            requested = "agent"
        elif requested not in ROLES:
            requested = "agent"

    if _is_open_mode():
        # FIND-002: open mode never elevates
        requested = "agent"
        ttl_s = min(int(ttl_s), 900)
    else:
        issuer = (issuer_role or "agent").strip().lower()
        if issuer not in ROLES:
            issuer = "agent"
        bootstrap = _env_bool("FRONTLINE_BOOTSTRAP_ADMIN", False)
        try:
            from src.security.harden import is_production_like

            if is_production_like():
                bootstrap = False
        except Exception:
            pass
        if requested in ELEVATED:
            allowed = issuer == "admin" or (
                requested == "admin" and bootstrap
            )
            if not allowed:
                raise HTTPException(
                    status_code=403,
                    detail=(
                        f"role {requested!r} requires an admin session issuer "
                        f"(or FRONTLINE_BOOTSTRAP_ADMIN=1 for one-time admin mint)"
                    ),
                )

    exp = int(time.time()) + max(60, int(ttl_s))
    body = {"sub": subject, "role": requested, "exp": exp}
    if extra:
        # SECURITY: never allow caller-supplied extra to overwrite reserved
        # claims (sub/role/exp/idp). Previously extra={"role": <display>}
        # clobbered the validated role, letting open-mode signup mint
        # arbitrary roles including "admin".
        for k, v in extra.items():
            if k in {"sub", "subject", "role", "exp", "iat", "idp", "iss"}:
                continue
            body[k] = v
    raw = json.dumps(body, separators=(",", ":"), sort_keys=True)
    sig = hmac.new(_secret(), raw.encode(), hashlib.sha256).hexdigest()
    token = f"{raw}|{sig}"
    res = {"token": token, "role": requested, "subject": subject, "exp": exp}
    if extra:
        for k, v in extra.items():
            if k in {"token", "role", "subject", "exp", "sub"}:
                continue
            res[k] = v
    return res


def issue_idp_session(subject: str, *, ttl_s: int = 3600) -> dict[str, Any]:
    """Mint an agent session after IdP verification. Never elevates. No issuer role."""
    name = (subject or "").strip() or "idp-user"
    exp = int(time.time()) + max(60, int(ttl_s))
    body = {"sub": name, "role": "agent", "exp": exp, "idp": "oidc"}
    raw = json.dumps(body, separators=(",", ":"), sort_keys=True)
    sig = hmac.new(_secret(), raw.encode(), hashlib.sha256).hexdigest()
    token = f"{raw}|{sig}"
    return {"token": token, "role": "agent", "subject": name, "exp": exp}


_REVOKED_SESSION_SIGS: set[str] = set()


def revoke_session(token: str) -> None:
    """Revoke a session token on logout, adding it to the revocation denylist."""
    if not token or not isinstance(token, str):
        return
    tok = token.strip()
    if not tok:
        return
    sig = tok.rsplit("|", 1)[1] if "|" in tok else hashlib.sha256(tok.encode()).hexdigest()
    _REVOKED_SESSION_SIGS.add(sig)

    try:
        from src.data.warehouse import ops_con

        exp_val = None
        if "|" in tok:
            try:
                body = json.loads(tok.rsplit("|", 1)[0])
                exp_val = int(body.get("exp") or 0)
            except Exception:
                pass
        with ops_con() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS revoked_sessions (
                    token_sig VARCHAR PRIMARY KEY,
                    revoked_at TIMESTAMP WITH TIME ZONE,
                    exp BIGINT
                )
                """
            )
            con.execute(
                "INSERT OR REPLACE INTO revoked_sessions (token_sig, revoked_at, exp) VALUES (?, CURRENT_TIMESTAMP, ?)",
                [sig, exp_val],
            )
    except Exception:
        pass


def is_session_revoked(token: str) -> bool:
    """Check whether a session token has been revoked."""
    if not token or not isinstance(token, str):
        return False
    tok = token.strip()
    if not tok:
        return False
    sig = tok.rsplit("|", 1)[1] if "|" in tok else hashlib.sha256(tok.encode()).hexdigest()
    if sig in _REVOKED_SESSION_SIGS:
        return True
    try:
        from src.data.warehouse import ops_con

        with ops_con(read_only=True) as con:
            row = con.execute(
                "SELECT 1 FROM revoked_sessions WHERE token_sig = ?",
                [sig],
            ).fetchone()
            if row:
                _REVOKED_SESSION_SIGS.add(sig)
                return True
    except Exception:
        pass
    return False


def verify_session(token: str) -> dict[str, Any]:
    if "|" not in token:
        raise HTTPException(status_code=401, detail="invalid session token")
    if is_session_revoked(token):
        raise HTTPException(status_code=401, detail="session has been revoked")
    raw, sig = token.rsplit("|", 1)
    expect = hmac.new(_secret(), raw.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expect, sig):
        raise HTTPException(status_code=401, detail="bad session signature")
    body = json.loads(raw)
    if int(body.get("exp") or 0) < time.time():
        raise HTTPException(status_code=401, detail="session expired")
    # Fail closed: unknown/spoofed role strings (e.g. legacy tokens where
    # display-title overwrote the claim) never elevate — treat as agent.
    _r = str(body.get("role") or "agent")
    if _r not in ROLES:
        body["role"] = "agent"
    return body


def role_from_headers(
    x_frontline_role: str | None = None,
    x_frontline_session: str | None = None,
    api_key: str | None = None,
) -> str:
    """Resolve role.

    - Signed session: trust session role.
    - DSR API key (FRONTLINE_DSR_API_KEY): dsr_officer role.
    - Auth required (shared API key validated by Depends): service principal
      (not full admin — H1). Opt-in full admin via FRONTLINE_SERVICE_IS_ADMIN=1.
    - Open mode / unauthenticated: agent only; bare ``X-Frontline-Role`` cannot elevate.
    """
    if x_frontline_session:
        return str(verify_session(x_frontline_session).get("role") or "agent")
    # FIND-003: bare header never elevates
    if x_frontline_role:
        role = x_frontline_role.strip().lower()
        if role not in ROLES:
            raise HTTPException(status_code=400, detail=f"unknown role: {role}")
        if role in ELEVATED or role == "service":
            role = "agent"
    else:
        role = "agent"

    # Distinguish dedicated DSR key vs shared service key
    dsr_key = os.getenv("FRONTLINE_DSR_API_KEY", "").strip()
    if api_key and dsr_key and secrets.compare_digest(api_key.strip(), dsr_key):
        return "dsr_officer"

    # Shared API key → limited service principal (or legacy full admin opt-in)
    if not _is_open_mode() and _auth_required():
        if _env_bool("FRONTLINE_SERVICE_IS_ADMIN", False):
            return "admin"
        return "service"
    return role


def subject_from_headers(
    x_frontline_session: str | None = None,
    api_key: str | None = None,
    *,
    default: str = "service",
) -> str:
    """Identity for audit fields — never trust client body.author under harden."""
    if isinstance(x_frontline_session, str) and x_frontline_session:
        body = verify_session(x_frontline_session)
        return str(body.get("sub") or body.get("role") or default)[:80]
    dsr_key = os.getenv("FRONTLINE_DSR_API_KEY", "").strip()
    if api_key and dsr_key and secrets.compare_digest(api_key.strip(), dsr_key):
        return "dsr_officer"
    if not _is_open_mode() and _auth_required():
        return default[:80]
    return "operator"


def claims_from_headers(
    x_frontline_session: str | None = None,
    cookies: Any = None,
) -> dict[str, Any]:
    """Extract claims dictionary from session token (header or cookie)."""
    token = (x_frontline_session or "").strip()
    if not token and cookies is not None:
        token = session_token_from_cookies(cookies)
    if token:
        try:
            return verify_session(token)
        except Exception:
            return {}
    return {}


def require_perm(role: str, perm: str, *, open_mode_ok: bool = False) -> None:
    """Raise 403 when *role* is not granted *perm*.

    ``open_mode_ok=True`` skips the check in local open mode (unsigned demo
    operator is always ``agent``). Takeover is never skipped — seizing a live
    customer call is gated even on a laptop.
    """
    _never_open = frozenset({
        "dsr:export", "dsr:delete", "admin:keys",
        "deploy:activate",
    })
    if open_mode_ok and perm in _never_open:
        open_mode_ok = False
    if open_mode_ok and _is_open_mode():
        return
    grants = PERMS.get(role, frozenset())
    if "*" in grants or perm in grants:
        return
    raise HTTPException(status_code=403, detail=f"role {role} lacks {perm}")


def has_perm(role: str, perm: str, *, open_mode_ok: bool = False) -> bool:
    """Return True if *role* is granted *perm*, False otherwise."""
    try:
        require_perm(role, perm, open_mode_ok=open_mode_ok)
        return True
    except HTTPException:
        return False


def require_perm_dep(perm: str, *, open_mode_ok: bool = False):
    """FastAPI dependency that resolves the role and enforces *perm*."""

    async def _dep(role: str = Depends(get_role)) -> str:
        require_perm(role, perm, open_mode_ok=open_mode_ok)
        return role

    _dep.__name__ = f"require_{perm.replace(':', '_')}"
    return _dep


def _websocket_api_key(websocket: Any) -> str:
    """The API key this connection actually authenticated with.

    Prefers the principal stored by ``authenticate_websocket`` (the only place
    a first-frame key is ever seen) and falls back to the handshake headers for
    connections authenticated by the ``require_ws_api_key`` dependency.
    """
    try:
        from src.api.auth import ws_principal

        principal = ws_principal(websocket)
        if principal is not None and principal.api_key:
            return principal.api_key
    except Exception:
        pass
    try:
        headers = websocket.headers
    except Exception:
        return ""
    try:
        key = (headers.get("x-api-key") or "").strip()
        if key:
            return key
        auth = (headers.get("authorization") or "").strip()
    except Exception:
        return ""
    if not auth:
        return ""
    parts = auth.split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return auth


def role_from_websocket(websocket: Any) -> str:
    """Resolve RBAC role from a WebSocket (session cookie / header / credential).

    The authenticated credential is passed through to ``role_from_headers``:
    without it, a validated DSR key fell through to the shared service
    principal, which carries the ``takeover`` permission (R04).
    """
    try:
        from src.api.auth import ws_principal

        p = ws_principal(websocket)
        if p is not None and p.credential == "session" and p.role:
            return p.role
    except Exception:
        pass

    session = ""
    try:
        session = (websocket.headers.get("x-frontline-session") or "").strip()
    except Exception:
        session = ""
    if not session:
        try:
            session = session_token_from_cookies(websocket.cookies)
        except Exception:
            session = ""
    role_hdr = None
    try:
        role_hdr = websocket.headers.get("x-frontline-role")
    except Exception:
        role_hdr = None
    return role_from_headers(
        role_hdr,
        session or None,
        api_key=_websocket_api_key(websocket) or None,
    )


def oidc_discovery() -> dict[str, Any]:
    from src.frontline.oidc import provider_status

    st = provider_status()
    if not st["configured"] and not os.getenv("OIDC_ISSUER", "").strip():
        return {
            "configured": False,
            "provider": "google",
            "note": "Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET for Google sign-in",
            "start_path": "/api/frontline/auth/oidc/start",
            "roles": sorted(ROLES),
        }
    return {
        "configured": st["configured"],
        "provider": st["provider"],
        "issuer": st["issuer"],
        "authorization_endpoint": st["authorization_endpoint"],
        "token_endpoint": st["token_endpoint"],
        "jwks_uri": st["jwks_uri"],
        "start_path": st["start_path"],
        "roles": sorted(ROLES),
    }


def _extract_header_key(request: Request, x_api_key: str | None, authorization: str | None) -> str:
    key = (x_api_key or "").strip()
    if not key and authorization:
        parts = authorization.strip().split(None, 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            key = parts[1].strip()
        else:
            key = authorization.strip()
    return key


async def get_role(
    request: Request,
    x_frontline_role: str | None = Header(default=None),
    x_frontline_session: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None),
) -> str:
    key = _extract_header_key(request, x_api_key, authorization)
    return role_from_headers(
        x_frontline_role,
        x_frontline_session or session_token_from_cookies(request.cookies),
        api_key=key,
    )


async def get_actor(
    request: Request,
    x_frontline_session: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None),
) -> str:
    key = _extract_header_key(request, x_api_key, authorization)
    return subject_from_headers(
        x_frontline_session or session_token_from_cookies(request.cookies) or None,
        api_key=key,
    )
