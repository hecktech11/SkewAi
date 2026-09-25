"""Production / SOC-oriented runtime hardening helpers.

SOC 2 Type II is an auditor attestation over time — this module only enforces
*engineering* fail-closed controls that support Security criteria (CC6, CC7).
"""

from __future__ import annotations

import os
from typing import Any


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def is_production_like() -> bool:
    """True when deploy should refuse open mode and weak secrets."""
    env = (os.getenv("ENV") or os.getenv("APP_ENV") or "").strip().lower()
    if env in {"production", "prod", "staging", "stage"}:
        return True
    return _env_bool("PILOT_HARDENED", False) or _env_bool("SOC2_MODE", False)


# ── Fail-closed network/key policy (remediation item 1 + 50) ──────────────

#: Minimum API-key strength when authentication actually protects a deployment
#: (production-like mode or a wildcard bind). 32 bytes ≈ 256 bits of entropy
#: when randomly generated; the check is on UTF-8 byte length.
MIN_API_KEY_BYTES = 32

#: Minimum session-secret strength in production-like mode.
MIN_SESSION_SECRET_BYTES = 32

#: Exact placeholder values that must never authenticate a deployment.
#: Exact match only (case-insensitive) so real secrets containing these
#: substrings (e.g. "pilot-secret-test") are not false-flagged.
WEAK_KEY_VALUES = frozenset({
    "dev-only", "changeme", "change-me", "password", "secret", "test",
    "placeholder", "apikey", "api-key", "api_key", "12345678", "default",
    "example", "letmein", "qwerty", "admin", "frontline",
})

#: Binds that expose the process beyond loopback.
WILDCARD_HOSTS = frozenset({"0.0.0.0", "::", "0:0:0:0:0:0:0:0"})


def api_host() -> str:
    """Configured bind host (live env read so tests can monkeypatch)."""
    return (os.getenv("API_HOST") or "127.0.0.1").strip() or "127.0.0.1"


def is_wildcard_bind(host: str | None = None) -> bool:
    """True when the API binds beyond loopback (LAN/container-wide)."""
    return (host or api_host()).strip() in WILDCARD_HOSTS


def auth_explicitly_required() -> bool:
    """True only when FRONTLINE_AUTH_REQUIRED is explicitly enabled."""
    return _env_bool("FRONTLINE_AUTH_REQUIRED", False)


def key_strength_problems(key: str) -> list[str]:
    """Return human-readable problems with *key* (empty = strong enough)."""
    problems: list[str] = []
    k = (key or "").strip()
    if not k:
        return ["FRONTLINE_API_KEY is empty"]
    if len(k.encode("utf-8")) < MIN_API_KEY_BYTES:
        problems.append(
            f"FRONTLINE_API_KEY must be at least {MIN_API_KEY_BYTES} bytes "
            f"(got {len(k.encode('utf-8'))}). Generate with: "
            "python -c 'import secrets; print(secrets.token_urlsafe(32))'"
        )
    if k.lower() in WEAK_KEY_VALUES:
        problems.append(
            "FRONTLINE_API_KEY uses a placeholder value; set a generated secret."
        )
    if len(set(k)) < 8:
        problems.append("FRONTLINE_API_KEY has too little entropy (repeated characters).")
    return problems


def locker_key_permission_problems() -> list[str]:
    """Private locker keys must be owner-only (0600). Returns problems."""
    problems: list[str] = []
    try:
        from pathlib import Path

        from src.config import REPO_ROOT

        key_dir = REPO_ROOT / "data" / "locker_keys"
        if not key_dir.is_dir():
            return []
        for pem in sorted(key_dir.glob("*.pem")):
            if "private" not in pem.name and "key" not in pem.name:
                continue
            try:
                mode = pem.stat().st_mode & 0o777
            except OSError:
                continue
            if mode & 0o077:
                problems.append(
                    f"locker private key {pem.name} has permissions "
                    f"{oct(mode)}; must be 0600 (run: chmod 600 {pem})."
                )
    except Exception:
        pass
    return problems


def validate_startup_security() -> dict[str, Any]:
    """Validate security env at process start.

    Fail-closed cases (raise ``RuntimeError``):
      - production-like deploy with open mode, missing/weak API key, weak
        session secret, or world-readable locker private keys;
      - wildcard bind (``API_HOST=0.0.0.0``/``::``) without explicitly enabled
        authentication (``FRONTLINE_AUTH_REQUIRED=1`` + strong key), except an
        explicit non-production local-pilot acknowledgment
        (``FRONTLINE_OPEN_BIND_ACK=1``).

    Loopback binds (127.0.0.1/localhost) without auth remain allowed outside
    production-like mode as the intentional local-dev escape hatch.

    Returns a status dict. Raises ``RuntimeError`` when misconfigured.
    """
    from src.api.auth import _configured_key, auth_required, is_open_mode

    open_mode = is_open_mode()
    key = _configured_key()
    session = (os.getenv("SESSION_SECRET") or "").strip()
    auth_req = auth_required()
    prod = is_production_like()
    host = api_host()
    wildcard = is_wildcard_bind(host)
    auth_explicit = auth_explicitly_required()

    status: dict[str, Any] = {
        "production_like": prod,
        "auth_required": auth_req,
        "open_mode": open_mode,
        "api_host": host,
        "wildcard_bind": wildcard,
        "api_key_configured": bool(key),
        "session_secret_configured": bool(session) or bool(key),
        "ok": True,
        "warnings": [],
    }

    # ── Wildcard bind gate (item 1): LAN/container-wide listeners must be
    # explicitly authenticated. This runs before first request (lifespan).
    if wildcard and not prod:
        if not (auth_explicit and key and not key_strength_problems(key)):
            ack = _env_bool("FRONTLINE_OPEN_BIND_ACK", False)
            if not ack:
                raise RuntimeError(
                    "Security startup check failed:\n- "
                    f"API_HOST={host} binds beyond loopback but authenticated "
                    "mode is not explicitly enabled. Refusing to start open on "
                    "a shared network. Fix one of:\n"
                    "  * FRONTLINE_AUTH_REQUIRED=1 + FRONTLINE_API_KEY "
                    f"(<{MIN_API_KEY_BYTES} bytes, generated), or\n"
                    "  * bind loopback only (API_HOST=127.0.0.1) for local dev, or\n"
                    "  * set FRONTLINE_OPEN_BIND_ACK=1 to explicitly acknowledge "
                    "an unauthenticated local pilot (never on shared networks)."
                )
            status["warnings"].append(
                "Wildcard bind without authentication explicitly acknowledged "
                "via FRONTLINE_OPEN_BIND_ACK=1 (local pilot only)."
            )

    # ── Pinned Trust Root Gate (F-001): refuse startup if required and missing
    trust_root_req = os.getenv("FRONTLINE_TRUST_ROOT_REQUIRED", "1").strip().lower() in {"1", "true", "yes", "on"}
    if trust_root_req:
        try:
            from src.security.trust_roots import TrustRootNotFound, TrustRootStore

            store = TrustRootStore()
            store.get_active_public_key()
            status["trust_root_configured"] = True
        except (TrustRootNotFound, Exception) as e:
            status["ok"] = False
            err = (
                f"Security startup check failed: no trust root configured (FRONTLINE_TRUST_ROOT_REQUIRED=1): {e}. "
                "Register a trust root in the DB, set FRONTLINE_TRUST_ROOT_PATH, or set FRONTLINE_TRUST_ROOT_REQUIRED=0."
            )
            problems = status.setdefault("errors", [])
            problems.append(err)
            raise RuntimeError(err)

    if prod:
        problems: list[str] = []
        if open_mode or not auth_req:
            problems.append(
                "Open mode is not allowed when ENV=production/staging or "
                "PILOT_HARDENED=1 / SOC2_MODE=1. Set FRONTLINE_AUTH_REQUIRED=1 "
                "and FRONTLINE_API_KEY, and do not set FRONTLINE_OPEN_MODE=1."
            )
        problems.extend(key_strength_problems(key))
        if session:
            if len(session.encode("utf-8")) < MIN_SESSION_SECRET_BYTES:
                problems.append(
                    f"SESSION_SECRET must be at least {MIN_SESSION_SECRET_BYTES} "
                    "bytes in hardened/production mode."
                )
            if session.lower() in WEAK_KEY_VALUES:
                problems.append("Refusing weak SESSION_SECRET value in hardened mode.")
            if key and session == key:
                problems.append(
                    "SESSION_SECRET must not match FRONTLINE_API_KEY (cannot alias service API key to session signing key)."
                )
        else:
            problems.append("SESSION_SECRET must be explicitly configured in hardened/production mode.")
        if _env_bool("FRONTLINE_BOOTSTRAP_ADMIN", False):
            problems.append(
                "FRONTLINE_BOOTSTRAP_ADMIN must not be enabled in hardened/production mode."
            )
        if _env_bool("FRONTLINE_SERVICE_IS_ADMIN", False):
            problems.append(
                "FRONTLINE_SERVICE_IS_ADMIN must not be enabled in hardened/production mode (service key must not escalate to full admin)."
            )
        problems.extend(locker_key_permission_problems())
        if problems:
            status["ok"] = False
            status["errors"] = problems
            raise RuntimeError("Security startup check failed:\n- " + "\n- ".join(problems))

    # Non-production: weak-key hygiene is advisory, except wildcard binds
    # which are already gated above. Still surface actionable warnings.
    if not prod and open_mode:
        status["warnings"].append(
            "Running in open mode (no API key). Fine for local demos only."
        )
    if auth_req and not session:
        status["warnings"].append(
            "SESSION_SECRET is unset; set SESSION_SECRET for session authentication."
        )
    if not prod and key and (auth_explicit or wildcard):
        for p in key_strength_problems(key):
            status["warnings"].append(p)
    return status
