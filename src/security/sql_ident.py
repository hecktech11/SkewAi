"""Allowlisted SQL identifiers for dynamic table/column fragments (M13 / B608).

Values (WHERE ? =) stay parameterized. Only *names* go through this helper.
"""

from __future__ import annotations

import re

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Ops / domain tables referenced via f-string fragments in shipped code.
SAFE_TABLES = frozenset(
    {
        "cases",
        "case_notes",
        "contact_issues",
        "agent_actions",
        "interaction_turns",
        "interaction_version_stamps",
        "risk_snapshots",
        "interactions",
        "investigations",
        "investigation_hypotheses",
        "recorded_fixes",
        "job_queue",
        "approvals",
        "report_subscriptions",
        "usage_events",
        "alert_dedup",
        "alert_dead_letter",
        "erasure_drill_reports",
        "connector_deliveries",
        "contact_memory",
        "llm_spend",
        "canonical_identity",
        "entity_observations",
    }
)

SAFE_COLUMNS = frozenset(
    {
        "interaction_id",
        "case_id",
        "entity_1",
        "entity_2",
        "entity_3",
        "category",
        "region",
        "status",
        "pack_id",
        "created_at",
        "started_at",
        "followup_draft",
        "tenant_id",
        "fired_at",
        "ts",
        "prev_hash",
        "row_hash",
        "assignee",
        "sla_due_at",
        # Chain-preserving erasure only (dsr.tombstone — fixed internal DDL,
        # never request input).
        "text",
        "input_summary",
        "output_summary",
        "body_json",
        "erased",
        "description",
        "body",
        "hash_version",
        "content_hash",
        "claims",
    }
)

# Schema-migration only (warehouse APPLY); never from request input.
SAFE_ALTER_COLUMNS = frozenset(
    {
        "prev_hash",
        "row_hash",
        "tenant_id",
        "assignee",
        "sla_due_at",
        "case_kind",
        "customer_ref",
        "degraded_ledger",
        "csat",
        "customer_resolved",
        "enrichment_partial",
        "erased",
        "entity_key",
        "provenance",
        "hash_version",
        "content_hash",
        "claims",
    }
)

# SET-clause field fragments built only from fixed allowlisted keys in ops.py
SAFE_CASE_UPDATE_FIELDS = frozenset({"status", "followup_draft"})


def safe_ident(name: str, allow: frozenset[str], *, kind: str = "identifier") -> str:
    """Return name if it is in ``allow`` and looks like a plain SQL identifier."""
    raw = (name or "").strip()
    if not raw or not _IDENT_RE.match(raw):
        raise ValueError(f"invalid SQL {kind}: {name!r}")
    if raw not in allow:
        raise ValueError(f"SQL {kind} not allowlisted: {name!r}")
    return raw


def safe_table(name: str) -> str:
    return safe_ident(name, SAFE_TABLES, kind="table")


def safe_column(name: str) -> str:
    return safe_ident(name, SAFE_COLUMNS, kind="column")


__all__ = [
    "safe_ident",
    "safe_table",
    "safe_column",
    "SAFE_TABLES",
    "SAFE_COLUMNS",
    "SAFE_CASE_UPDATE_FIELDS",
    "SAFE_ALTER_COLUMNS",
]
