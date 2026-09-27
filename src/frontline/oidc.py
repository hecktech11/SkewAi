"""Google / OIDC authorization-code login.

Browser never sees the client secret. Tests inject HTTP + JWKS so they do not
call Google.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlencode, urlparse

from src.config import REPO_ROOT
from src.data.timeutil import utc_now
from src.data.warehouse import ops_con
from src.ids import new_ulid

GOOGLE_ISSUER = "https://accounts.google.com"
GOOGLE_AUTH = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN = "https://oauth2.googleapis.com/token"
GOOGLE_JWKS = "https://www.googleapis.com/oauth2/v3/certs"
HANDOFF_TTL_S = 120
DEFAULT_LOCAL_OIDC = REPO_ROOT / "data" / "oidc_local.json"

# Test hooks — never used in production.
_http_post: Callable[..., dict[str, Any]] | None = None
_http_get_json: Callable[[str], dict[str, Any]] | None = None


def set_http_hooks(
    post: Callable[..., dict[str, Any]] | None = None,
    get_json: Callable[[str], dict[str, Any]] | None = None,
) -> None:
    global _http_post, _http_get_json
    _http_post = post
    _http_get_json = get_json


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def local_oidc_path() -> Path:
    raw = _env("OIDC_LOCAL_PATH")
    return Path(raw) if raw else DEFAULT_LOCAL_OIDC


def load_local_provider() -> dict[str, str]:
    path = local_oidc_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        "client_id": str(data.get("client_id") or "").strip(),
        "client_secret": str(data.get("client_secret") or "").strip(),
        "redirect_uri": str(data.get("redirect_uri") or "").strip(),
    }


def save_local_provider(
    *,
    client_id: str,
    client_secret: str,
    redirect_uri: str = "",
) -> dict[str, Any]:
    """Persist Google OAuth client to a gitignored local file (dev/pilot only)."""
    from src.security.harden import is_production_like

    if is_production_like():
        raise RuntimeError("oidc_local_config_forbidden")
    cid = (client_id or "").strip()
    secret = (client_secret or "").strip()
    if len(cid) < 8 or len(secret) < 8:
        raise ValueError("google_credentials_incomplete")
    path = local_oidc_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "client_id": cid,
        "client_secret": secret,
        "redirect_uri": (redirect_uri or "").strip(),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return provider_status()


def provider_status() -> dict[str, Any]:
    local = load_local_provider()
    client_id = _env("GOOGLE_CLIENT_ID") or _env("OIDC_CLIENT_ID") or local.get("client_id", "")
    client_secret = (
        _env("GOOGLE_CLIENT_SECRET") or _env("OIDC_CLIENT_SECRET") or local.get("client_secret", "")
    )
    issuer = _env("OIDC_ISSUER") or (GOOGLE_ISSUER if client_id else "")
    google = (not _env("OIDC_ISSUER")) or "accounts.google.com" in issuer
    redirect = (
        _env("OIDC_REDIRECT_URI")
        or local.get("redirect_uri")
        or "http://127.0.0.1:8000/api/frontline/auth/oidc/callback"
    )
    return {
        "configured": bool(client_id and client_secret),
        "provider": "google" if google else "oidc",
        "issuer": issuer or GOOGLE_ISSUER,
        "client_id": client_id,
        "authorization_endpoint": GOOGLE_AUTH if google else f"{issuer.rstrip('/')}/authorize",
        "token_endpoint": GOOGLE_TOKEN if google else f"{issuer.rstrip('/')}/token",
        "jwks_uri": GOOGLE_JWKS if google else f"{issuer.rstrip('/')}/jwks",
        "redirect_uri": redirect,
        "start_path": "/api/frontline/auth/oidc/start",
        "scopes": "openid email profile",
        "source": (
            "env"
            if (_env("GOOGLE_CLIENT_ID") or _env("OIDC_CLIENT_ID"))
            else ("local" if local.get("client_id") else "none")
        ),
    }


def _ensure(con) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS oidc_auth_state (
            state VARCHAR PRIMARY KEY,
            nonce VARCHAR,
            next_url VARCHAR,
            created_at TIMESTAMP,
            used BOOLEAN,
            code_verifier VARCHAR,
            browser_binding VARCHAR
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS oidc_handoffs (
            handoff_id VARCHAR PRIMARY KEY,
            token VARCHAR,
            subject VARCHAR,
            email VARCHAR,
            role VARCHAR,
            exp INTEGER,
            created_at TIMESTAMP,
            used BOOLEAN,
            browser_binding VARCHAR
        )
        """
    )
    try:
        cols = {str(r[0]) for r in con.execute("DESCRIBE oidc_auth_state").fetchall()}
    except Exception:
        cols = set()
    if cols and "code_verifier" not in cols:
        con.execute("ALTER TABLE oidc_auth_state ADD COLUMN code_verifier VARCHAR")
    if cols and "browser_binding" not in cols:
        con.execute("ALTER TABLE oidc_auth_state ADD COLUMN browser_binding VARCHAR")
    try:
        cols_h = {str(r[0]) for r in con.execute("DESCRIBE oidc_handoffs").fetchall()}
    except Exception:
        cols_h = set()
    if cols_h and "browser_binding" not in cols_h:
        con.execute("ALTER TABLE oidc_handoffs ADD COLUMN browser_binding VARCHAR")


def allowed_next_origins() -> set[str]:
    allowed = {
        "http://127.0.0.1:8787",
        "http://localhost:8787",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    }
    extra = _env("CORS_ALLOW_ORIGINS")
    if extra:
        for part in extra.split(","):
            p = part.strip().rstrip("/")
            if p:
                allowed.add(p)
    ui = _env("FRONTLINE_UI_ORIGIN")
    if ui:
        allowed.add(ui.rstrip("/"))
    return allowed


def safe_post_login(next_url: str | None, *, page: str = "command") -> str:
    """Return an in-app URL. Rejects open redirects."""
    dest = page if page in {"command", "signin"} else "command"
    fallback = f"/ui/#{dest}"
    raw = (next_url or "").strip()
    if not raw:
        return fallback
    parsed = urlparse(raw)
    if not parsed.scheme or not parsed.netloc:
        return fallback
    origin = f"{parsed.scheme}://{parsed.netloc}"
    if origin not in allowed_next_origins():
        return fallback
    return f"{origin}/ui/#{dest}"


def _pkce_pair() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(48)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def begin_login(*, next_url: str | None = None, browser_binding: str | None = None) -> dict[str, str]:
    cfg = provider_status()
    if not cfg["configured"]:
        raise RuntimeError("google_oauth_not_configured")
    state = secrets.token_urlsafe(24)
    nonce = secrets.token_urlsafe(24)
    verifier, challenge = _pkce_pair()
    nxt = safe_post_login(next_url, page="command")
    with ops_con() as con:
        _ensure(con)
        con.execute(
            """
            INSERT INTO oidc_auth_state
            (state, nonce, next_url, created_at, used, code_verifier, browser_binding)
            VALUES (?, ?, ?, ?, FALSE, ?, ?)
            """,
            [state, nonce, nxt, utc_now(), verifier, browser_binding],
        )
    params = {
        "client_id": cfg["client_id"],
        "redirect_uri": cfg["redirect_uri"],
        "response_type": "code",
        "scope": cfg["scopes"],
        "state": state,
        "nonce": nonce,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "access_type": "online",
        "include_granted_scopes": "true",
        "prompt": "select_account",
    }
    return {
        "authorize_url": f"{cfg['authorization_endpoint']}?{urlencode(params)}",
        "state": state,
        "nonce": nonce,
        "next_url": nxt,
        "browser_binding": browser_binding or "",
    }


def pop_state(state: str) -> dict[str, Any] | None:
    if not state:
        return None
    with ops_con() as con:
        _ensure(con)
        row = con.execute(
            """
            SELECT state, nonce, next_url, used, code_verifier, created_at, browser_binding
            FROM oidc_auth_state WHERE state = ?
            """,
            [state],
        ).fetchone()
        if not row or row[3]:
            return None
        created = row[5]
        try:
            from datetime import datetime, timezone, timedelta

            if created is not None:
                if isinstance(created, str):
                    created_dt = datetime.fromisoformat(created)
                else:
                    created_dt = created
                if created_dt.tzinfo is None:
                    created_dt = created_dt.replace(tzinfo=timezone.utc)
                if datetime.now(timezone.utc) - created_dt > timedelta(minutes=10):
                    con.execute("UPDATE oidc_auth_state SET used = TRUE WHERE state = ?", [state])
                    return None
        except Exception:
            pass
        con.execute("UPDATE oidc_auth_state SET used = TRUE WHERE state = ?", [state])
        return {
            "state": row[0],
            "nonce": row[1],
            "next_url": row[2],
            "code_verifier": row[4] or "",
            "browser_binding": row[6] or "",
        }


def _b64url_decode(data: str) -> bytes:
    pad = "=" * ((4 - len(data) % 4) % 4)
    return base64.urlsafe_b64decode(data + pad)


def _b64url_json(data: str) -> dict[str, Any]:
    return json.loads(_b64url_decode(data).decode("utf-8"))


def verify_id_token(
    token: str,
    *,
    audience: str,
    issuer: str,
    nonce: str | None = None,
    jwks: dict[str, Any] | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Verify a Google-style RS256 ID token. Pure when *jwks* is passed."""
    parts = (token or "").split(".")
    if len(parts) != 3:
        raise ValueError("id_token_malformed")
    header = _b64url_json(parts[0])
    payload = _b64url_json(parts[1])
    if header.get("alg") != "RS256":
        raise ValueError("id_token_alg")
    allowed_iss = {issuer, issuer.rstrip("/"), f"{issuer.rstrip('/')}/", GOOGLE_ISSUER}
    if payload.get("iss") not in allowed_iss:
        raise ValueError("id_token_iss")
    if payload.get("aud") != audience:
        raise ValueError("id_token_aud")
    now_i = int(now if now is not None else time.time())
    exp = int(payload.get("exp") or 0)
    if exp <= now_i:
        raise ValueError("id_token_exp")
    nbf = payload.get("nbf")
    if nbf is not None and int(nbf) > now_i:
        raise ValueError("id_token_nbf")
    iat = payload.get("iat")
    if iat is not None and int(iat) > now_i + 60:
        raise ValueError("id_token_iat")
    if nonce and payload.get("nonce") != nonce:
        raise ValueError("id_token_nonce")

    keys = (jwks or _load_jwks()).get("keys") or []
    kid = header.get("kid")
    if not kid:
        raise ValueError("id_token_kid")
    chosen = next((k for k in keys if k.get("kid") == kid), None)
    if not chosen:
        raise ValueError("id_token_kid")
    _verify_rs256(parts[0], parts[1], parts[2], chosen)
    return payload


def _load_jwks() -> dict[str, Any]:
    raw = _env("OIDC_JWKS_JSON")
    if raw:
        return json.loads(raw)
    cfg = provider_status()
    url = cfg["jwks_uri"]
    if _http_get_json:
        return _http_get_json(url)
    import httpx

    r = httpx.get(url, timeout=15.0)
    r.raise_for_status()
    return r.json()


def _verify_rs256(h_b64: str, p_b64: str, s_b64: str, jwk: dict[str, Any]) -> None:
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding
    from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicNumbers

    n = int.from_bytes(_b64url_decode(jwk["n"]), "big")
    e = int.from_bytes(_b64url_decode(jwk["e"]), "big")
    key = RSAPublicNumbers(e, n).public_key(default_backend())
    try:
        key.verify(
            _b64url_decode(s_b64),
            f"{h_b64}.{p_b64}".encode("ascii"),
            padding.PKCS1v15(),
            hashes.SHA256(),
        )
    except Exception as exc:
        raise ValueError("id_token_sig") from exc


def exchange_code(code: str, *, code_verifier: str = "") -> dict[str, Any]:
    cfg = provider_status()
    local = load_local_provider()
    data = {
        "code": code,
        "client_id": cfg["client_id"],
        "client_secret": (
            _env("GOOGLE_CLIENT_SECRET")
            or _env("OIDC_CLIENT_SECRET")
            or local.get("client_secret", "")
        ),
        "redirect_uri": cfg["redirect_uri"],
        "grant_type": "authorization_code",
    }
    if code_verifier:
        data["code_verifier"] = code_verifier
    if _http_post:
        return _http_post(cfg["token_endpoint"], data)
    import httpx

    r = httpx.post(cfg["token_endpoint"], data=data, timeout=15.0)
    r.raise_for_status()
    return r.json()


def finish_callback(*, code: str, state: str, browser_binding: str | None = None) -> dict[str, Any]:
    st = pop_state(state)
    if not st:
        raise ValueError("oidc_state_invalid")
    expected_binding = str(st.get("browser_binding") or "")
    if expected_binding:
        if not browser_binding or not hmac.compare_digest(expected_binding, str(browser_binding)):
            raise ValueError("oidc_browser_binding_mismatch")
    tokens = exchange_code(code, code_verifier=str(st.get("code_verifier") or ""))
    id_token = tokens.get("id_token")
    if not id_token:
        raise ValueError("oidc_missing_id_token")
    cfg = provider_status()
    claims = verify_id_token(
        str(id_token),
        audience=cfg["client_id"],
        issuer=cfg["issuer"],
        nonce=st["nonce"],
    )
    email = str(claims.get("email") or "").strip()
    if email and claims.get("email_verified") is False:
        raise ValueError("google_email_unverified")
    subject = email or str(claims.get("name") or claims.get("sub") or "google-user")
    from src.api.rbac import issue_idp_session

    sess = issue_idp_session(subject)
    hid = "hf_" + new_ulid()
    final_binding = browser_binding or expected_binding or ""
    with ops_con() as con:
        _ensure(con)
        con.execute(
            """
            INSERT INTO oidc_handoffs
            (handoff_id, token, subject, email, role, exp, created_at, used, browser_binding)
            VALUES (?, ?, ?, ?, ?, ?, ?, FALSE, ?)
            """,
            [
                hid,
                sess["token"],
                sess["subject"],
                email,
                sess["role"],
                sess["exp"],
                utc_now(),
                final_binding,
            ],
        )
    return {
        "handoff": hid,
        "token": sess["token"],
        "next_url": st["next_url"],
        "subject": sess["subject"],
        "email": email,
        "role": sess["role"],
        "browser_binding": final_binding,
    }


def handoff_expired(created_at, *, now=None, ttl_s: int = HANDOFF_TTL_S) -> bool:
    """True when a handoff is older than ttl_s. Pure."""
    if created_at is None:
        return True
    current = now or utc_now()
    try:
        created = created_at.replace(tzinfo=None)
        cur = current.replace(tzinfo=None)
    except Exception:
        return True
    return (cur - created).total_seconds() > max(1, int(ttl_s))


def consume_handoff(handoff_id: str, *, browser_binding: str | None = None) -> dict[str, Any]:
    hid = (handoff_id or "").strip()
    if not hid:
        raise ValueError("handoff_missing")
    with ops_con() as con:
        _ensure(con)
        row = con.execute(
            """
            SELECT token, subject, email, role, exp, used, created_at, browser_binding
            FROM oidc_handoffs WHERE handoff_id = ?
            """,
            [hid],
        ).fetchone()
        if not row:
            raise ValueError("handoff_unknown")
        if row[5]:
            raise ValueError("handoff_used")
        if handoff_expired(row[6]):
            con.execute("UPDATE oidc_handoffs SET used = TRUE WHERE handoff_id = ?", [hid])
            raise ValueError("handoff_expired")
        expected_binding = str(row[7] or "")
        if expected_binding:
            if not browser_binding or not hmac.compare_digest(expected_binding, str(browser_binding)):
                raise ValueError("handoff_binding_mismatch")
        con.execute("UPDATE oidc_handoffs SET used = TRUE WHERE handoff_id = ?", [hid])
    return {
        "token": row[0],
        "subject": row[1],
        "email": row[2],
        "role": row[3],
        "exp": row[4],
        "provider": "google",
    }
