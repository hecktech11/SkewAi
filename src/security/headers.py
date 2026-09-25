"""HTTP security headers middleware (SOC 2 CC6 network/app hardening baseline)."""

from __future__ import annotations

from starlette.types import ASGIApp, Receive, Scope, Send


# Baseline CSP: allow self + Google Fonts already used by dashboard index.html.
# Tighten further when self-hosting fonts.
_CSP = (
    "default-src 'self'; "
    "script-src 'self'; "
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
    "font-src 'self' https://fonts.gstatic.com data:; "
    "img-src 'self' data: blob:; "
    "connect-src 'self' ws://127.0.0.1:* ws://localhost:*; "
    "frame-ancestors 'none'; "
    "base-uri 'self'; "
    "form-action 'self'"
)


def _hsts_enabled() -> bool:
    """HSTS only in production-like deploys (FIND-R08 — avoid breaking local HTTP)."""
    try:
        from src.security.harden import is_production_like

        return is_production_like()
    except Exception:
        return False


class SecurityHeadersASGI:
    """Attach security headers to every HTTP response."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: dict) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers") or [])
                extra = [
                    (b"x-content-type-options", b"nosniff"),
                    (b"x-frame-options", b"DENY"),
                    (b"referrer-policy", b"no-referrer"),
                    (
                        b"permissions-policy",
                        b"camera=(), microphone=(self), geolocation=()",
                    ),
                    (b"content-security-policy", _CSP.encode("ascii")),
                ]
                if _hsts_enabled():
                    extra.append(
                        (
                            b"strict-transport-security",
                            b"max-age=31536000; includeSubDomains",
                        )
                    )
                existing = {h[0].lower() for h in headers}
                for k, v in extra:
                    if k not in existing:
                        headers.append((k, v))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_headers)
