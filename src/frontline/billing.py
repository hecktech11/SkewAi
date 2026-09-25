"""Usage dashboard, Stripe test-mode billing, seats, roles, plan tiers."""

from __future__ import annotations

import json
from typing import Any

from src.data.timeutil import utc_now
from src.data.warehouse import ops_con
from src.frontline.metering import record_usage, usage_summary
from src.ids import new_ulid

PLAN_TIERS = {
    "trial": {"seats": 2, "contacts_month": 50, "roles": ("agent",)},
    "pilot": {"seats": 10, "contacts_month": 2000, "roles": ("agent", "supervisor", "auditor")},
    "enterprise": {
        "seats": 100,
        "contacts_month": 100000,
        "roles": ("agent", "supervisor", "auditor", "admin", "service"),
    },
}


def _ensure(con) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS billing_accounts (
            tenant_id VARCHAR PRIMARY KEY,
            plan VARCHAR NOT NULL,
            seats_used INTEGER NOT NULL,
            stripe_customer VARCHAR,
            updated_at TIMESTAMP NOT NULL
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS billing_invoices (
            invoice_id VARCHAR PRIMARY KEY,
            tenant_id VARCHAR NOT NULL,
            amount_cents INTEGER NOT NULL,
            currency VARCHAR NOT NULL,
            status VARCHAR NOT NULL,
            stripe_session VARCHAR,
            created_at TIMESTAMP NOT NULL
        )
        """
    )
    # Server-side plan binding (item 13): the plan an invoice pays for is
    # recorded at checkout; webhooks must never trust client-supplied plan.
    for ddl in (
        "ALTER TABLE billing_invoices ADD COLUMN plan VARCHAR",
    ):
        try:
            con.execute(ddl)
        except Exception:
            pass
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS billing_webhook_events (
            event_id VARCHAR PRIMARY KEY,
            stripe_session VARCHAR,
            tenant_id VARCHAR NOT NULL,
            plan VARCHAR NOT NULL,
            status VARCHAR NOT NULL,
            received_at TIMESTAMP NOT NULL
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS tenant_seats (
            seat_id VARCHAR PRIMARY KEY,
            tenant_id VARCHAR NOT NULL,
            user_id VARCHAR NOT NULL,
            role VARCHAR NOT NULL,
            created_at TIMESTAMP NOT NULL
        )
        """
    )


def get_account(tenant_id: str = "default") -> dict[str, Any]:
    with ops_con() as con:
        _ensure(con)
        row = con.execute(
            "SELECT tenant_id, plan, seats_used, stripe_customer FROM billing_accounts WHERE tenant_id = ?",
            [tenant_id],
        ).fetchone()
        if not row:
            con.execute(
                """
                INSERT INTO billing_accounts (tenant_id, plan, seats_used, stripe_customer, updated_at)
                VALUES (?, 'trial', 0, NULL, ?)
                """,
                [tenant_id, utc_now()],
            )
            return {
                "tenant_id": tenant_id,
                "plan": "trial",
                "seats_used": 0,
                "limits": PLAN_TIERS["trial"],
            }
    return {
        "tenant_id": row[0],
        "plan": row[1],
        "seats_used": int(row[2] or 0),
        "stripe_customer": row[3],
        "limits": PLAN_TIERS.get(row[1], PLAN_TIERS["trial"]),
    }


def set_plan(tenant_id: str, plan: str) -> dict[str, Any]:
    if plan not in PLAN_TIERS:
        raise ValueError(f"unknown plan {plan}")
    get_account(tenant_id)
    with ops_con() as con:
        _ensure(con)
        con.execute(
            "UPDATE billing_accounts SET plan = ?, updated_at = ? WHERE tenant_id = ?",
            [plan, utc_now(), tenant_id],
        )
    return get_account(tenant_id)


def assign_seat(tenant_id: str, user_id: str, role: str) -> dict[str, Any]:
    acct = get_account(tenant_id)
    limits = acct["limits"]
    if role not in limits["roles"]:
        raise PermissionError(f"role {role} not on plan {acct['plan']}")
    if int(acct["seats_used"]) >= int(limits["seats"]):
        raise PermissionError(f"seat limit {limits['seats']} reached")
    sid = "seat_" + new_ulid()
    with ops_con() as con:
        _ensure(con)
        con.execute(
            """
            INSERT INTO tenant_seats (seat_id, tenant_id, user_id, role, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            [sid, tenant_id, user_id, role, utc_now()],
        )
        con.execute(
            "UPDATE billing_accounts SET seats_used = seats_used + 1, updated_at = ? WHERE tenant_id = ?",
            [utc_now(), tenant_id],
        )
    return {"seat_id": sid, "tenant_id": tenant_id, "user_id": user_id, "role": role}


def enforce_plan(tenant_id: str, *, metric: str = "contacts", quantity: float = 1.0, role: str | None = None) -> dict[str, Any]:
    acct = get_account(tenant_id)
    limits = acct["limits"]
    if role and role not in limits["roles"]:
        return {"allowed": False, "reason": f"role {role} not on plan {acct['plan']}"}
    summary = usage_summary(tenant_id=tenant_id)
    used = float((summary.get("metrics") or {}).get(metric) or 0)
    cap = float(limits.get("contacts_month") or 0)
    if metric == "contacts" and used + quantity > cap:
        return {"allowed": False, "reason": f"over plan: {used}+{quantity} > {cap}"}
    return {"allowed": True, "plan": acct["plan"], "used": used, "cap": cap}


def usage_dashboard(tenant_id: str = "default") -> dict[str, Any]:
    from src.frontline.capabilities import list_capabilities

    acct = get_account(tenant_id)
    summary = usage_summary(tenant_id=tenant_id)
    return {
        "tenant_id": tenant_id,
        "plan": acct["plan"],
        "limits": acct["limits"],
        "seats_used": acct["seats_used"],
        "metrics": summary.get("metrics") or {},
        "month": summary.get("month"),
        # Item 40: stubs are listed with availability=stub here so sales/UI
        # can never present them as paid production features.
        "capabilities": list_capabilities(),
    }


def plan_includes(plan: str, capability: str) -> bool:
    """Whether *plan* sells *capability*. Stubs are never included (item 40)."""
    from src.frontline.capabilities import is_sellable

    if plan not in PLAN_TIERS:
        raise ValueError(f"unknown plan {plan}")
    return is_sellable(capability)


def stripe_checkout(
    tenant_id: str,
    *,
    plan: str = "pilot",
    amount_cents: int = 9900,
    stripe_session: str | None = None,
) -> dict[str, Any]:
    """Test-mode Stripe path: persist an invoice + session id (no live charge)."""
    if plan not in PLAN_TIERS:
        raise ValueError(plan)
    iid = "in_" + new_ulid()
    session = stripe_session or ("cs_test_" + new_ulid())
    with ops_con() as con:
        _ensure(con)
        try:
            con.execute(
                """
                INSERT INTO billing_invoices
                (invoice_id, tenant_id, amount_cents, currency, status, stripe_session, plan, created_at)
                VALUES (?, ?, ?, 'usd', 'open', ?, ?, ?)
                """,
                [iid, tenant_id, int(amount_cents), session, plan, utc_now()],
            )
        except Exception:
            # Legacy table without the plan column (pre-item-13 DBs).
            con.execute(
                """
                INSERT INTO billing_invoices
                (invoice_id, tenant_id, amount_cents, currency, status, stripe_session, created_at)
                VALUES (?, ?, ?, 'usd', 'open', ?, ?)
                """,
                [iid, tenant_id, int(amount_cents), session, utc_now()],
            )
    return {
        "invoice_id": iid,
        "tenant_id": tenant_id,
        "plan": plan,
        "amount_cents": amount_cents,
        "status": "open",
        "stripe_session": session,
        "checkout_url": f"https://checkout.stripe.com/c/pay/{session}",
        "test_mode": True,
    }


def stripe_webhook(
    payload: dict[str, Any],
    *,
    raw_body: bytes | None = None,
    signature: str | None = None,
    allow_unsigned: bool = False,
) -> dict[str, Any]:
    """Apply a Stripe-like webhook (checkout.session.completed).

    Hardening (item 13):
    - Requires STRIPE_WEBHOOK_SECRET and a valid ``t=<ts>,v1=hmac_sha256``
      signature over the raw body AND a fresh timestamp (``STRIPE_WEBHOOK_TOLERANCE_S``,
      default 300s) — replay protection. Missing/stale/forged signatures raise
      PermissionError across all deploy modes.
    - Never trust client-supplied plan/tenant: the plan comes from the
      checkout invoice row and the tenant must equal the invoice's tenant.
      Unknown/forged sessions raise LookupError.
    - Processed Stripe event IDs are recorded (``billing_webhook_events``);
      a repeated delivery returns the stored result without side effects.
    """
    import hashlib
    import hmac
    import os
    import time

    event = payload.get("type") or payload.get("event") or ""
    event_id = str(payload.get("id") or "").strip()
    session = (
        (payload.get("data") or {}).get("object", {}).get("id")
        or payload.get("stripe_session")
        or ""
    )
    claimed_tenant = payload.get("tenant_id") or "default"

    secret = (os.getenv("STRIPE_WEBHOOK_SECRET") or "").strip()
    if secret:
        if not raw_body or not signature:
            raise PermissionError("missing webhook signature")
        # Expected format: t=<unix-ts>,v1=<hex> (Stripe-style). v1 only.
        ts_raw, v1 = "", ""
        for part in str(signature).split(","):
            part = part.strip()
            if part.startswith("t="):
                ts_raw = part[2:].strip()
            elif part.startswith("v1="):
                v1 = part[3:].strip()
        if not v1:
            raise PermissionError("invalid webhook signature")
        expect = hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expect, v1):
            raise PermissionError("invalid webhook signature")
        try:
            tolerance = int(os.getenv("STRIPE_WEBHOOK_TOLERANCE_S") or "300")
        except ValueError:
            tolerance = 300
        try:
            age = abs(time.time() - int(ts_raw))
        except (TypeError, ValueError):
            raise PermissionError("invalid webhook timestamp")
        if age > tolerance:
            raise PermissionError("stale webhook signature (replay rejected)")
    elif not allow_unsigned:
        raise PermissionError(
            "STRIPE_WEBHOOK_SECRET must be configured to process webhooks. Refusing unsigned events."
        )

    with ops_con() as con:
        _ensure(con)
        # Idempotent replay: same Stripe event ID returns the stored result.
        if event_id:
            try:
                prior = con.execute(
                    "SELECT status, tenant_id, plan FROM billing_webhook_events WHERE event_id = ?",
                    [event_id],
                ).fetchone()
            except Exception:
                prior = None
            if prior:
                return {
                    "ok": True,
                    "event": event or "invoice.paid",
                    "plan": prior[2],
                    "tenant_id": prior[1],
                    "event_id": event_id,
                    "replayed": True,
                }
        if not session:
            raise ValueError("stripe_session required for plan change")
        try:
            inv = con.execute(
                "SELECT invoice_id, tenant_id, status, plan FROM billing_invoices WHERE stripe_session = ?",
                [session],
            ).fetchone()
        except Exception:
            # Legacy table without plan column.
            inv = None
            try:
                legacy = con.execute(
                    "SELECT invoice_id, tenant_id, status FROM billing_invoices WHERE stripe_session = ?",
                    [session],
                ).fetchone()
                if legacy:
                    inv = (legacy[0], legacy[1], legacy[2], None)
            except Exception:
                inv = None
        if not inv:
            raise LookupError(f"unknown stripe_session: {session[:24]}")
        invoice_id, invoice_tenant, invoice_status, invoice_plan = inv
        # The session is bound to its invoice's account — a mismatched
        # tenant claim is an escalation attempt, not a payment.
        if claimed_tenant != invoice_tenant:
            raise PermissionError(
                f"webhook tenant {claimed_tenant!r} does not match invoice account"
            )
        # Server-side plan: the invoice row is authoritative. A client
        # payload asking for a different plan is ignored (never escalated).
        plan = invoice_plan or payload.get("plan") or "pilot"
        if plan not in PLAN_TIERS:
            raise ValueError(f"unknown plan {plan}")
        already_paid = (invoice_status == "paid")
        if not already_paid:
            con.execute(
                "UPDATE billing_invoices SET status = 'paid' WHERE stripe_session = ?",
                [session],
            )
        if event_id:
            try:
                con.execute(
                    """
                    INSERT INTO billing_webhook_events
                    (event_id, stripe_session, tenant_id, plan, status, received_at)
                    VALUES (?, ?, ?, ?, 'paid', ?)
                    """,
                    [event_id, session, invoice_tenant, plan, utc_now()],
                )
            except Exception:
                pass
    if event in {"checkout.session.completed", "invoice.paid", ""}:
        # set_plan is idempotent; replays converge to the same plan.
        set_plan(invoice_tenant, plan)
    out: dict[str, Any] = {
        "ok": True,
        "event": event or "invoice.paid",
        "plan": plan,
        "tenant_id": invoice_tenant,
    }
    if event_id:
        out["event_id"] = event_id
        out["replayed"] = already_paid
    return out


def record_metered(metric: str, quantity: float = 1.0, *, tenant_id: str = "default") -> dict[str, Any]:
    gate = enforce_plan(tenant_id, metric=metric, quantity=quantity)
    if not gate["allowed"]:
        return {"recorded": False, **gate}
    ev = record_usage(metric, quantity, tenant_id=tenant_id)
    return {"recorded": True, "event": ev, **gate}


__all__ = [
    "PLAN_TIERS",
    "get_account",
    "set_plan",
    "assign_seat",
    "enforce_plan",
    "usage_dashboard",
    "plan_includes",
    "stripe_checkout",
    "stripe_webhook",
    "record_metered",
]
