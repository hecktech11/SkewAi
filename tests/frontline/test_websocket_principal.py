"""WebSocket principal resolution (R04).

Authentication accepts the dedicated DSR key, so authorization must see a
dsr_officer — not fall back to the shared service principal, which carries the
takeover permission used to send supervisor turns into a live contact. The
validated identity has to survive the trip from authentication to RBAC.
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from src.api.auth import authenticate_websocket, check_api_key
from src.api.rbac import issue_session, require_perm, role_from_websocket

SERVICE_KEY = "test-service-secret-key-32-chars-ws!!"
DSR_KEY = "test-dsr-secret-key-32-chars-ws-only!"
SESSION_SECRET = "test-session-secret-key-32-chars-ws!!"


@pytest.fixture
def auth_env(monkeypatch):
    monkeypatch.setenv("FRONTLINE_AUTH_REQUIRED", "1")
    monkeypatch.setenv("FRONTLINE_API_KEY", SERVICE_KEY)
    monkeypatch.setenv("FRONTLINE_DSR_API_KEY", DSR_KEY)
    monkeypatch.setenv("SESSION_SECRET", SESSION_SECRET)
    monkeypatch.delenv("FRONTLINE_OPEN_MODE", raising=False)
    monkeypatch.delenv("FRONTLINE_SERVICE_IS_ADMIN", raising=False)


class _State:
    pass


class FakeWebSocket:
    """Minimal stand-in for the parts of starlette's WebSocket we touch."""

    def __init__(self, headers=None, cookies=None, first_frame=None, query=None):
        self.headers = {k.lower(): v for k, v in (headers or {}).items()}
        self.cookies = cookies or {}
        self.query_params = query or {}
        self.state = _State()
        self._first_frame = first_frame
        self.accepted = False
        self.closed_with = None

        class _ClientState:
            name = "CONNECTING"

        self.client_state = _ClientState()

    async def accept(self):
        self.accepted = True
        self.client_state.name = "CONNECTED"

    async def receive_json(self):
        if self._first_frame is None:
            raise RuntimeError("no frame queued")
        return self._first_frame

    async def close(self, code=1000):
        self.closed_with = code


def test_dsr_key_in_handshake_headers_resolves_dsr_officer(auth_env):
    ws = FakeWebSocket(headers={"X-API-Key": DSR_KEY})

    # Authentication accepts this credential...
    check_api_key(x_api_key=DSR_KEY)

    # ...so authorization must see the DSR officer it belongs to.
    assert role_from_websocket(ws) == "dsr_officer"


def test_dsr_principal_over_websocket_cannot_take_over_a_contact(auth_env):
    ws = FakeWebSocket(headers={"X-API-Key": DSR_KEY})

    role = role_from_websocket(ws)

    with pytest.raises(HTTPException) as exc:
        require_perm(role, "takeover")
    assert exc.value.status_code == 403


def test_service_key_in_handshake_headers_still_resolves_service(auth_env):
    ws = FakeWebSocket(headers={"X-API-Key": SERVICE_KEY})

    assert role_from_websocket(ws) == "service"
    require_perm("service", "takeover")  # unchanged: service keeps takeover


def test_bearer_dsr_key_resolves_dsr_officer(auth_env):
    ws = FakeWebSocket(headers={"Authorization": f"Bearer {DSR_KEY}"})

    assert role_from_websocket(ws) == "dsr_officer"


async def test_dsr_key_in_first_auth_frame_resolves_dsr_officer(auth_env):
    """Browser clients authenticate with a frame, not a header — the principal
    must be carried forward from there too."""
    ws = FakeWebSocket(first_frame={"type": "auth", "api_key": DSR_KEY})

    await authenticate_websocket(ws)

    assert role_from_websocket(ws) == "dsr_officer"
    with pytest.raises(HTTPException):
        require_perm(role_from_websocket(ws), "takeover")


async def test_service_key_in_first_auth_frame_resolves_service(auth_env):
    ws = FakeWebSocket(first_frame={"type": "auth", "api_key": SERVICE_KEY})

    await authenticate_websocket(ws)

    assert role_from_websocket(ws) == "service"


async def test_authenticate_websocket_reports_the_verified_credential(auth_env):
    ws = FakeWebSocket(first_frame={"type": "auth", "api_key": DSR_KEY})

    principal = await authenticate_websocket(ws)

    assert principal.credential == "dsr"
    assert principal.role == "dsr_officer"


def test_check_api_key_reports_which_credential_matched(auth_env):
    assert check_api_key(x_api_key=DSR_KEY).credential == "dsr"
    assert check_api_key(x_api_key=SERVICE_KEY).credential == "service"


def test_signed_session_cookie_still_wins_over_the_key(auth_env):
    """A signed session is the stronger identity; the key must not downgrade it."""
    token = issue_session("sup_usr", "supervisor", issuer_role="admin")["token"]
    ws = FakeWebSocket(
        headers={"X-API-Key": DSR_KEY},
        cookies={"frontline_session": token},
    )

    assert role_from_websocket(ws) == "supervisor"


async def test_wrong_key_in_first_frame_is_rejected(auth_env):
    ws = FakeWebSocket(first_frame={"type": "auth", "api_key": "not-the-key"})

    with pytest.raises(HTTPException) as exc:
        await authenticate_websocket(ws)
    assert exc.value.status_code == 401
