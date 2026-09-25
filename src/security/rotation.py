"""Key + secret rotation with chain continuity (review §6).

  - Ed25519 locker keys rotate; old signatures still verify (pubkey ring)
  - SESSION_SECRET rotation with overlap window (dual-HMAC accept)
  - admin-action audit hook (pack switch, settings change, mapping upload)
  - WS origin check helper + /v1/ version alias map

No live rotation daemon — deterministic helpers + file layout that ops
and tests can drive.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import REPO_ROOT


def key_ring_dir() -> Path:
    raw = (os.getenv("LOCKER_KEY_RING_DIR") or "").strip()
    p = Path(raw).expanduser() if raw else (REPO_ROOT / "data" / "locker_keys" / "ring")
    return p.resolve() if p.is_absolute() else (REPO_ROOT / p).resolve()


def rotation_manifest_path() -> Path:
    return key_ring_dir() / "rotation.json"


def record_rotation(*, old_key_id: str, new_key_id: str, actor: str = "ops") -> dict[str, Any]:
    """Append a rotation event; old pubs stay in ring so old sigs verify."""
    d = key_ring_dir()
    d.mkdir(parents=True, exist_ok=True)
    mp = rotation_manifest_path()
    try:
        manifest = json.loads(mp.read_text(encoding="utf-8")) if mp.is_file() else {"rotations": []}
    except Exception:
        manifest = {"rotations": []}
    evt = {"old_key_id": old_key_id, "new_key_id": new_key_id, "actor": actor,
           "at": datetime.now(timezone.utc).isoformat()}
    manifest["rotations"].append(evt)
    manifest["active_key_id"] = new_key_id
    # retired keys remain listed for verification
    retired = set(manifest.get("retired_key_ids", []))
    if old_key_id:
        retired.add(old_key_id)
    manifest["retired_key_ids"] = sorted(retired)
    mp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return evt


def verify_with_ring(payload: bytes, signature: bytes, public_keys: list[Any]) -> bool:
    """True if ANY ring pubkey verifies (old signatures survive rotation)."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    for raw in public_keys:
        try:
            if isinstance(raw, Ed25519PublicKey):
                pub = raw
            elif isinstance(raw, (bytes, bytearray)):
                if len(raw) == 32:
                    pub = Ed25519PublicKey.from_public_bytes(bytes(raw))
                elif b"BEGIN PUBLIC KEY" in raw:
                    pub = serialization.load_pem_public_key(bytes(raw))
                else:
                    pub = serialization.load_der_public_key(bytes(raw))
            elif isinstance(raw, str):
                pub = serialization.load_pem_public_key(raw.encode("ascii"))
            else:
                continue
            pub.verify(signature, payload)
            return True
        except Exception:
            continue
    return False


# ── SESSION_SECRET overlap ────────────────────────────────────────────────

def _hmac_hex(secret: str, msg: str) -> str:
    return hmac.new(secret.encode(), msg.encode(), hashlib.sha256).hexdigest()


def sign_session(msg: str, *, primary: str | None = None) -> str:
    secret = (primary or (os.getenv("SESSION_SECRET") or "")).strip()
    api_key = (os.getenv("FRONTLINE_API_KEY") or "").strip()
    if secret and api_key and secret == api_key:
        raise RuntimeError("SESSION_SECRET must not match FRONTLINE_API_KEY")
    if not secret:
        from src.api.auth import auth_required
        from src.security.harden import is_production_like

        if is_production_like() or auth_required():
            raise RuntimeError("SESSION_SECRET required when authentication is enabled")
        secret = "dev-only"
    return _hmac_hex(secret, msg)


def verify_session(msg: str, sig: str, *, primary: str | None = None,
                   secondary: str | None = None) -> bool:
    """Accept primary OR previous secret during rotation overlap."""
    prev = (secondary if secondary is not None else os.getenv("SESSION_SECRET_PREVIOUS", "")).strip()
    primary_sec = (primary or (os.getenv("SESSION_SECRET") or "")).strip()
    api_key = (os.getenv("FRONTLINE_API_KEY") or "").strip()
    if primary_sec and api_key and primary_sec == api_key:
        return False
    if not primary_sec:
        from src.api.auth import auth_required
        from src.security.harden import is_production_like

        if is_production_like() or auth_required():
            return False
        primary_sec = "dev-only"
    cands = [primary_sec]
    if prev:
        cands.append(prev)
    return any(hmac.compare_digest(_hmac_hex(s, msg), sig) for s in cands if s)


# ── Admin-action audit + WS origin + versioning ───────────────────────────

_ADMIN_ACTIONS = {"pack_switch", "settings_change", "mapping_upload",
                  "key_rotation", "retention_change", "connector_config"}


def is_admin_action(action: str) -> bool:
    return str(action or "") in _ADMIN_ACTIONS


def admin_audit_row(*, actor: str, action: str, detail: str = "") -> dict[str, Any]:
    if not is_admin_action(action):
        raise ValueError(f"not an admin action: {action}")
    return {"actor": actor, "action": action, "detail": detail[:1000],
            "at": datetime.now(timezone.utc).isoformat()}


def ws_origin_allowed(origin: str | None, *, allowed: list[str] | None = None) -> bool:
    if allowed is None:
        raw = (os.getenv("WS_ALLOWED_ORIGINS") or "http://127.0.0.1:8787,http://localhost:8787").strip()
        allowed = [o.strip() for o in raw.split(",") if o.strip()]
    if not origin:
        return False  # fail closed on socket upgrade without Origin
    o = origin.strip().rstrip("/")
    return o in [a.rstrip("/") for a in allowed]


# /api/* canonical; /api/v1/* alias. Pure map so routers stay single-source.
def v1_alias(path: str) -> str:
    p = "/" + (path or "").lstrip("/")
    if p.startswith("/api/v1/"):
        return "/api/" + p[len("/api/v1/"):]
    return p


__all__ = [
    "key_ring_dir", "rotation_manifest_path", "record_rotation",
    "verify_with_ring", "sign_session", "verify_session",
    "is_admin_action", "admin_audit_row", "ws_origin_allowed", "v1_alias",
]
