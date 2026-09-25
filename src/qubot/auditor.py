"""Qubot v2 Auditor — pack-aware, audit-first agent verification.

The auditor runs after every contact (post_contact_audit playbook) and produces
a markdown report at `reports/qubot/contacts/{interaction_id}.md`. The headline
feature is the **groundedness audit** (§11.3): for every `agent_actions` row of
the audited contact, re-query the domain warehouse to confirm each cited
evidence ID exists AND matches the claimed scope; parse the output_summary and
follow-up draft for uncited IDs.

Verdict per action: `grounded | unverifiable | mismatch`. Any mismatch → report
flagged red, case marked `needs_review`, and a `groundedness_mismatch` ops alert
fires. Deterministic code, not LLM judgment — fully domain-agnostic because it
operates on the canonical schema.

Daily digest (`daily_frontline_digest` playbook) aggregates case_funnel +
live_risk + agent_performance + open investigation_status into a daily ops brief.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import settings
from src.data.warehouse import domain_con, ops_con
from src.frontline.alerts import alert_groundedness_mismatch
from src.qubot.retrievers import (
    agent_performance,
    case_funnel,
    contact_audit,
    investigation_status,
    live_risk,
)

REPORTS_DIR = settings.frontline_db_path.parent.parent / "reports" / "qubot" / "contacts"
DIGESTS_DIR = settings.frontline_db_path.parent.parent / "reports" / "qubot" / "digests"


# ── Groundedness audit ────────────────────────────────────────────────────────


@dataclass
class ActionVerdict:
    """Per-action groundedness verdict."""

    action_id: str
    agent: str
    action_type: str
    verdict: str = "unverifiable"  # grounded | unverifiable | mismatch
    evidence_checked: int = 0
    evidence_confirmed: int = 0
    evidence_confirmed_snapshot: int = 0
    evidence_confirmed_live: int = 0
    mismatched_ids: list[str] = field(default_factory=list)
    uncited_ids_in_output: list[str] = field(default_factory=list)
    source_drifted_ids: list[str] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    detail: str = ""


@dataclass
class AuditResult:
    """Result of the post-contact audit."""

    interaction_id: str
    pack_id: str
    pack_version: str
    context_version: str = "platform"
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # Snapshot semantics (item 3): evidence is verified against the historical
    # state pinned at action time (``as_of``), not the live warehouse. Re-audit
    # after source edits reproduces the historical verdict; live-vs-pin
    # differences surface separately as source drift.
    as_of: str = ""
    snapshot_verified: int = 0
    live_verified: int = 0

    # Headline verdict
    overall_verdict: str = "grounded"  # grounded | mismatch | source-drifted
    total_actions: int = 0
    grounded_actions: int = 0
    unverifiable_actions: int = 0
    mismatch_actions: int = 0

    action_verdicts: list[ActionVerdict] = field(default_factory=list)

    # Severity sanity check
    severity_triage: str = ""
    severity_safety_flags: dict[str, bool] = field(default_factory=dict)
    severity_sane: bool = True
    severity_note: str = ""

    # Frustration curve
    peak_frustration: float = 0.0
    frustration_curve: list[dict[str, Any]] = field(default_factory=list)

    # Supervised segment log
    supervised_segments: list[dict[str, Any]] = field(default_factory=list)

    # Flags / recommendations
    flags: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    report_path: Path | None = None


# ── Evidence verification ───────────────────────────────────────────────────
#
# Snapshot-first (item 3): every verifier accepts an optional ``snapshot_row``
# (the pinned historical row for this action). When present, existence AND
# scope are checked against the snapshot — never the live warehouse — so later
# database edits cannot rewrite a historical audit. Live state is used only
# when no pin exists, and the provenance ("snapshot" vs "live") is returned
# to the caller for labeling.


def _snapshot_rows_for_action(action_id: str) -> dict[str, dict[str, Any]]:
    """action pinned evidence_id -> historical row dict (parsed body_json)."""
    try:
        from src.qubot.evidence_pin import snapshots_for_action

        out: dict[str, dict[str, Any]] = {}
        for snap in snapshots_for_action(action_id):
            eid = str(snap.get("evidence_id") or "")
            if not eid:
                continue
            try:
                row = json.loads(snap.get("body_json") or "{}")
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(row, dict):
                out[eid] = row
        return out
    except Exception:
        return {}


def _scope_mismatch_advisory(
    adv: dict[str, Any], customer_entities: dict[str, str | None]
) -> str | None:
    """Return mismatch reason or None when the advisory scope fits."""
    for slot, scope_col in [
        ("entity_1", "scope_entity_1"),
        ("entity_2", "scope_entity_2"),
        ("entity_3", "scope_entity_3"),
        ("category", "scope_category"),
    ]:
        scope_val = adv.get(scope_col)
        if scope_val is None:
            continue  # advisory doesn't scope on this dimension
        customer_val = customer_entities.get(slot)
        if customer_val is None:
            return (
                f"advisory scopes {scope_col}='{scope_val}' "
                f"but customer {slot} is empty"
            )
        if str(scope_val).upper() != str(customer_val).upper():
            return (
                f"advisory {scope_col}='{scope_val}' != "
                f"customer {slot}='{customer_val}'"
            )
    return None


def _verify_advisory_id(
    pack_id: str,
    advisory_id: str,
    customer_entities: dict[str, str | None],
    *,
    snapshot_row: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """Re-query the domain warehouse: does the advisory exist AND match the customer's entities?"""
    if snapshot_row is not None:
        reason = _scope_mismatch_advisory(snapshot_row, customer_entities)
        if reason is None:
            return True, "snapshot:ok"
        return False, f"snapshot:{reason}"
    try:
        with domain_con(pack_id) as con:
            cur = con.execute(
                """
                SELECT advisory_id, scope_entity_1, scope_entity_2, scope_entity_3, scope_category
                FROM advisories WHERE advisory_id = ?
                """,
                [advisory_id],
            )
            cols = [d[0] for d in con.description] if con.description else []
            row = cur.fetchone()
    except FileNotFoundError:
        return False, "live:domain warehouse not found"
    if not row:
        return False, "live:advisory_id not in advisories table"
    adv = dict(zip(cols, row))

    # Scope check: if advisory scopes to a specific value, the customer must have that value.
    reason = _scope_mismatch_advisory(adv, customer_entities)
    if reason is not None:
        return False, f"live:{reason}"
    return True, "live:ok"


def _verify_record_id(
    pack_id: str,
    record_id: str,
    customer_entities: dict[str, str | None] | None = None,
    *,
    snapshot_row: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """Does this record exist AND match the customer's scope (when given)?"""
    if snapshot_row is not None:
        if customer_entities:
            for slot in ("category", "entity_2", "entity_3"):
                cust = (customer_entities.get(slot) or "").strip()
                val = str(snapshot_row.get(slot) or "").strip()
                if cust and val and cust.upper() != val.upper():
                    return False, (
                        f"snapshot:record {slot}='{val}' != "
                        f"customer {slot}='{cust}'"
                    )
        return True, "snapshot:ok"
    try:
        with domain_con(pack_id) as con:
            row = con.execute(
                "SELECT record_id, category, entity_2, entity_3 FROM records WHERE record_id = ?",
                [record_id],
            ).fetchone()
            if row is None:
                return False, "live:record_id not in records table"
            if customer_entities:
                cols = [d[0] for d in con.description]
                rec = dict(zip(cols, row))
                for slot in ("category", "entity_2", "entity_3"):
                    cust = (customer_entities.get(slot) or "").strip()
                    val = str(rec.get(slot) or "").strip()
                    if cust and val and cust.upper() != val.upper():
                        return False, f"live:record {slot}='{val}' != customer {slot}='{cust}'"
        return True, "live:ok"
    except FileNotFoundError:
        return False, "live:domain warehouse not found"


def _verify_cluster_id(
    pack_id: str,
    cluster_id: int | str,
    *,
    snapshot_row: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    if snapshot_row is not None:
        return True, "snapshot:ok"
    try:
        with domain_con(pack_id) as con:
            row = con.execute(
                "SELECT cluster_id FROM clusters WHERE cluster_id = ? AND pack_id = ?",
                [int(cluster_id), pack_id],
            ).fetchone()
        return (row is not None), ("live:ok" if row else "live:cluster_id not found")
    except (FileNotFoundError, ValueError, TypeError):
        return False, "live:invalid cluster_id"


def _verify_investigation_id(
    _pack_id: str,
    investigation_id: str,
    *,
    snapshot_row: dict[str, Any] | None = None,
) -> bool:
    """Investigations live in the ops warehouse."""
    if snapshot_row is not None:
        return True
    with ops_con(read_only=True) as con:
        row = con.execute(
            "SELECT investigation_id FROM investigations WHERE investigation_id = ?",
            [investigation_id],
        ).fetchone()
    return row is not None


def _verify_evidence_id(
    pack_id: str,
    evidence_id: str,
    customer_entities: dict[str, str | None],
    *,
    snapshots: dict[str, dict[str, Any]] | None = None,
) -> tuple[bool, str]:
    """Dispatch to the right verifier based on the ID's shape.

    ID shapes (canonical schema):
      - inv_XXXX                -> investigation (ops warehouse)
      - int_XXXX                -> interaction (ops warehouse)
      - case_XXXX               -> case (ops warehouse)
      - digits only             -> cluster id (domain warehouse)
      - NHTSA-XXXX / CFPB-XXXX  -> record id (domain warehouse)
      - \\d{2}V-\\d+              -> advisory id (domain warehouse, NHTSA recall format)
      - other advisory-shaped   -> advisory id (domain warehouse)
    """
    eid = str(evidence_id).strip()
    if not eid:
        return False, "empty evidence_id"
    snap = (snapshots or {}).get(eid)

    # Investigation ids: inv_XXXX
    if eid.startswith("inv_"):
        ok = _verify_investigation_id(pack_id, eid, snapshot_row=snap)
        return ok, ("snapshot:investigation" if snap is not None
                    else "live:investigation") if ok else "live:investigation-missing"

    # Interaction ids: int_XXXX (rare in evidence_ids, but supported)
    if eid.startswith("int_"):
        if snap is not None:
            return True, "snapshot:interaction"
        with ops_con(read_only=True) as con:
            row = con.execute(
                "SELECT interaction_id FROM interactions WHERE interaction_id = ?", [eid]
            ).fetchone()
        return row is not None, "live:interaction"

    # Diagnostic question ids: dq_XXXX (spoken read-back cites the question)
    if eid.startswith("dq_"):
        if snap is not None:
            return True, "snapshot:diagnostic_question"
        try:
            with ops_con(read_only=True) as con:
                row = con.execute(
                    "SELECT question_id FROM diagnostic_questions WHERE question_id = ?",
                    [eid],
                ).fetchone()
                if row is None:
                    row = con.execute(
                        "SELECT question_id FROM diagnostic_answers WHERE question_id = ?",
                        [eid],
                    ).fetchone()
        except Exception:
            row = None
        return row is not None, "live:diagnostic_question"

    # Case ids: case_XXXX
    if eid.startswith("case_"):
        if snap is not None:
            return True, "snapshot:case"
        with ops_con(read_only=True) as con:
            row = con.execute(
                "SELECT case_id FROM cases WHERE case_id = ?", [eid]
            ).fetchone()
        return row is not None, "live:case"

    # Numeric cluster ids (the Investigator emits str(cluster_id) in evidence_ids)
    if eid.isdigit():
        ok, msg = _verify_cluster_id(pack_id, eid, snapshot_row=snap)
        return ok, f"cluster:{msg}" if not ok else ("snapshot:cluster" if snap is not None else "live:cluster")

    # Advisory ids — common patterns: '19V-12345', 'NHTSA-12345' (recall format).
    # Try advisories first ONLY if the ID looks like an advisory (contains 'V-'
    # or matches the canonical recall pattern).
    import re
    if re.match(r"^\d{2}[A-Z]-\d+$", eid) or "V-" in eid:
        return _verify_advisory_id(pack_id, eid, customer_entities, snapshot_row=snap)

    # Otherwise try records (e.g. NHTSA-100001, CFPB-12345) WITH scope check.
    ok, msg = _verify_record_id(pack_id, eid, customer_entities, snapshot_row=snap)
    if ok:
        return True, "snapshot:record" if snap is not None else "live:record"

    # Last resort: try advisories (covers arbitrary advisory_id formats).
    ok, msg = _verify_advisory_id(pack_id, eid, customer_entities, snapshot_row=snap)
    if ok:
        return ok, "snapshot:advisory" if snap is not None else "live:advisory"
    return False, f"unknown evidence_id shape: {eid} ({msg})"


# ── Uncited ID extraction ────────────────────────────────────────────────────


def _extract_ids_from_text(text: str) -> list[str]:
    """Find ID-like tokens in output text. Catches things like '19V-12345',
    'NHTSA-100001', 'inv_0007', or bare integers in 'cluster #14' phrases.

    Conservative on purpose: we'd rather miss a citation than false-flag a
    plain word like 'CR-V' or a date like '2026-W22' as an uncited ID.
    """
    import re

    if not text:
        return []
    ids: list[str] = []
    # Pattern 1: NHTSA/CFPB-style record ids (case-insensitive, normalized
    # to upper for matching; catches 'nhtsa-100' lowercase hallucinations).
    for m in re.finditer(r"\b([A-Za-z]{3,}-\d{3,}(?:-\d{2,})*)\b", text):
        ids.append(m.group(1).upper())
    # Pattern 2: recall-style advisory ids like '19V-12345' (2 digits + letter + dash + digits)
    for m in re.finditer(r"\b(\d{2}[A-Za-z]-\d{3,})\b", text):
        ids.append(m.group(1).upper())
    # Pattern 3: inv_XXXX (and similar prefixes like int_XXXX, case_XXXX)
    for m in re.finditer(r"\b(inv_\d+)\b", text, flags=re.IGNORECASE):
        ids.append(m.group(1).lower())
    # Pattern 3b: int_XXXX (interaction hallucinations must not pass as grounded)
    for m in re.finditer(r"\b(int_[A-Za-z0-9]+)\b", text, flags=re.IGNORECASE):
        ids.append(m.group(1))
    # Pattern 4: case_XXXX
    for m in re.finditer(r"\b(case_[A-Za-z0-9]{4,})\b", text, flags=re.IGNORECASE):
        ids.append(m.group(1))
    # Pattern 5: 'cluster #14' → '14' (also bare 'cluster 14')
    for m in re.finditer(r"\bcluster\s*#?\s*(\d+)\b", text, flags=re.IGNORECASE):
        ids.append(m.group(1))
    return ids


# ── Severity sanity ───────────────────────────────────────────────────────────


def _check_severity_sanity(case_row: dict[str, Any], safety_flags: dict[str, bool]) -> tuple[bool, str]:
    """Severity sanity: if any safety flag is set, severity MUST be Critical."""
    if not safety_flags:
        return True, "no safety flags; severity unchallenged"
    sev = (case_row or {}).get("severity", "")
    if any(safety_flags.values()) and sev != "Critical":
        return False, f"safety_flags set but severity='{sev}' (expected Critical)"
    return True, "severity consistent with safety flags"


# ── Main audit ────────────────────────────────────────────────────────────────


async def audit_interaction(
    interaction_id: str,
    *,
    write_report: bool = True,
    as_of: str | None = None,
) -> AuditResult:
    """Run the post_contact_audit playbook for a single interaction.

    Returns an AuditResult; if `write_report` is True, also writes the markdown
    report to `reports/qubot/contacts/{interaction_id}.md`.

    Snapshot semantics (item 3): evidence existence/scope is verified against
    the historical rows pinned at action-write time (the ``as_of`` snapshot),
    not the live warehouse. ``as_of`` defaults to the audit time and is
    recorded on the result. Re-running an audit after source edits reproduces
    the historical verdict; live-vs-pin differences are reported separately as
    source drift. Live state is consulted only for evidence IDs that were
    never pinned, and those verdicts are labeled ``live:``.
    """
    from src.data.timeutil import utc_now

    as_of = as_of or utc_now().replace(tzinfo=None).isoformat()
    data = contact_audit(interaction_id)
    interaction = data["interaction"]
    turns = data["turns"]
    actions = data["actions"]
    case = data["case"]

    pack_id = interaction["pack_id"]
    pack_version = interaction.get("pack_version", "")
    customer_entities = {
        "entity_1": interaction.get("entity_1"),
        "entity_2": interaction.get("entity_2"),
        "entity_3": interaction.get("entity_3"),
        "category": interaction.get("category"),
    }

    result = AuditResult(
        interaction_id=interaction_id,
        pack_id=pack_id,
        pack_version=pack_version,
        total_actions=len(actions),
        peak_frustration=interaction.get("peak_frustration") or 0.0,
        as_of=as_of,
    )

    # ── Groundedness audit per action ────────────────────────────────────────
    safety_flags: dict[str, bool] = {}
    if case and case.get("safety_flags"):
        sf = case["safety_flags"]
        if isinstance(sf, str):
            try:
                sf = json.loads(sf)
            except json.JSONDecodeError:
                sf = {}
        safety_flags = sf or {}
    result.severity_safety_flags = safety_flags

    for action in actions:
        verdict = _audit_single_action(
            pack_id, action, customer_entities, as_of=as_of, turns=turns
        )
        result.snapshot_verified += verdict.evidence_confirmed_snapshot
        result.live_verified += verdict.evidence_confirmed_live
        result.action_verdicts.append(verdict)
        if verdict.verdict == "grounded":
            result.grounded_actions += 1
        elif verdict.verdict == "unverifiable":
            result.unverifiable_actions += 1
            try:
                from src.frontline.validation_queue import enqueue_insight

                enqueue_insight(
                    kind="unverifiable",
                    interaction_id=interaction_id,
                    summary=verdict.detail or verdict.action_type,
                    action_id=verdict.action_id,
                    case_id=(case or {}).get("case_id"),
                )
            except Exception:
                pass
        elif verdict.verdict == "source-drifted":
            result.mismatch_actions += 1
            result.overall_verdict = "source-drifted"
            result.flags.append(
                f"source-drifted: {', '.join(verdict.source_drifted_ids[:5])}"
            )
        else:  # mismatch
            result.mismatch_actions += 1
            result.overall_verdict = "mismatch"
            result.flags.append(
                f"{verdict.agent}/{verdict.action_type} (action {verdict.action_id}): {verdict.detail}"
            )
            # Fire ops alert (deduped per interaction per day)
            await alert_groundedness_mismatch(
                interaction_id=interaction_id,
                action_id=verdict.action_id,
                detail=verdict.detail,
            )
            # Mark case needs_review
            if case and case.get("case_id"):
                _mark_case_needs_review(case["case_id"])

    # ── Severity sanity ──────────────────────────────────────────────────────
    sane, note = _check_severity_sanity(case, safety_flags)
    result.severity_sane = sane
    result.severity_note = note
    result.severity_triage = (case or {}).get("severity", "")
    if not sane:
        result.flags.append(f"severity sanity: {note}")

    # ── Frustration curve ─────────────────────────────────────────────────────
    result.frustration_curve = [
        {"seq": t.get("seq"), "speaker": t.get("speaker"),
         "frustration_score": t.get("frustration_score")}
        for t in turns
        if t.get("speaker") == "customer"
    ]

    # ── Supervised segment log ───────────────────────────────────────────────
    result.supervised_segments = [
        {
            "seq": t.get("seq"),
            "text": t.get("text"),
            "ts": str(t.get("ts")) if t.get("ts") else "",
        }
        for t in turns
        if t.get("speaker") == "supervisor"
    ]

    # ── Recommendations ──────────────────────────────────────────────────────
    if result.mismatch_actions:
        result.recommendations.append(
            f"Review {result.mismatch_actions} action(s) with mismatched evidence — "
            f"the case is marked needs_review."
        )
    if not result.severity_sane:
        result.recommendations.append("Reconcile triage severity with safety flags.")
    if interaction.get("supervised"):
        result.recommendations.append(
            f"Review {len(result.supervised_segments)} supervisor turn(s) for tone/quality."
        )
    if interaction.get("enrichment_partial"):
        result.flags.append("enrichment_partial")
        result.recommendations.append(
            "Enrichment was partial (timeout or agent failure); treat the brief as incomplete."
        )

    # ── Write the report ─────────────────────────────────────────────────────
    if write_report:
        result.report_path = _write_contact_report(result, data)
        try:
            from src.qubot.report_rotation import rotate_reports

            rotate_reports()
        except Exception:
            pass  # rotation must never break audits

    return result


def _audit_single_action(
    pack_id: str,
    action: dict[str, Any],
    customer_entities: dict[str, str | None],
    *,
    as_of: str | None = None,
    turns: list[dict[str, Any]] | None = None,
) -> ActionVerdict:
    """Audit one agent_actions row.

    Evidence is verified against the ``as_of`` snapshot (pinned rows) first;
    the live warehouse is only a fallback for never-pinned IDs. ``as_of`` is
    recorded for provenance; pins are immutable so any re-audit reproduces the
    historical verdict.
    """
    action_id = action.get("action_id", "")
    agent = action.get("agent", "")
    action_type = action.get("action_type", "")
    evidence_ids = action.get("evidence_ids") or []
    if isinstance(evidence_ids, str):
        try:
            evidence_ids = json.loads(evidence_ids)
        except json.JSONDecodeError:
            evidence_ids = []

    verdict = ActionVerdict(
        action_id=action_id,
        agent=agent,
        action_type=action_type,
        evidence_checked=len(evidence_ids),
    )

    # ── Check that every cited evidence ID exists + matches the scope ──────
    # Snapshot-first (item 3): historical pins, not live state.
    snapshots = _snapshot_rows_for_action(action_id)
    mismatched: list[str] = []
    confirmed = 0
    confirmed_snapshot = 0
    confirmed_live = 0
    for eid in evidence_ids:
        ok, msg = _verify_evidence_id(
            pack_id, str(eid), customer_entities, snapshots=snapshots
        )
        if ok:
            confirmed += 1
            if msg.startswith("snapshot:"):
                confirmed_snapshot += 1
            else:
                confirmed_live += 1
        else:
            mismatched.append(f"{eid} ({msg})")
    verdict.evidence_confirmed = confirmed
    verdict.evidence_confirmed_snapshot = confirmed_snapshot
    verdict.evidence_confirmed_live = confirmed_live
    verdict.mismatched_ids = mismatched

    # ── Check that every ID mentioned in the output is also in evidence_ids ──
    output_text = (action.get("output_summary") or "") + " " + (action.get("input_summary") or "")
    cited_in_text = _extract_ids_from_text(output_text)
    evidence_set = {str(e) for e in evidence_ids}
    # The action's own case_id and interaction_id are recorded on the row, not in evidence_ids;
    # exclude them from the uncited list to avoid false positives.
    own_case_id = action.get("case_id")
    if own_case_id:
        evidence_set.add(str(own_case_id))
    own_interaction_id = action.get("interaction_id")
    if own_interaction_id:
        evidence_set.add(str(own_interaction_id))
    uncited = [c for c in cited_in_text if c not in evidence_set]
    verdict.uncited_ids_in_output = uncited

    from src.qubot.evidence_pin import detect_source_drift, snapshots_for_action
    from src.qubot.claims import audit_bound_claims, load_bound_claims

    drifted = detect_source_drift(action_id)
    verdict.source_drifted_ids = [str(d.get("evidence_id")) for d in drifted]

    planted = load_bound_claims(action_id)
    if planted:
        claim_audit = audit_bound_claims(
            planted, snapshots_for_action(action_id), turns=turns
        )
        if not claim_audit.ok:
            verdict.unsupported_claims = list(claim_audit.rejected)
            verdict.verdict = "mismatch"
            verdict.detail = "unsupported-claim: " + "; ".join(claim_audit.rejected[:3])
            return verdict

    # ── Final verdict ────────────────────────────────────────────────────────
    if verdict.source_drifted_ids:
        verdict.verdict = "source-drifted"
        verdict.detail = "source-drifted: " + ", ".join(verdict.source_drifted_ids[:3])
    elif mismatched:
        unknown = all(
            "unknown evidence_id shape" in m or "empty evidence_id" in m for m in mismatched
        )
        if unknown:
            verdict.verdict = "unverifiable"
            verdict.detail = f"unverifiable evidence: {', '.join(mismatched[:3])}"
        else:
            verdict.verdict = "mismatch"
            verdict.detail = f"mismatched evidence: {', '.join(mismatched[:3])}"
    elif uncited:
        verdict.verdict = "mismatch"
        verdict.detail = f"uncited IDs in output: {', '.join(uncited[:3])}"
    elif evidence_ids and confirmed == len(evidence_ids):
        verdict.verdict = "grounded"
        verdict.detail = f"all {confirmed} evidence id(s) verified"
    elif not evidence_ids:
        # No evidence cited. Only flag if there are uncited IDs AFTER excluding
        # the action's own case_id (already handled above). If uncited is empty,
        # the action is grounded — it just doesn't claim any external evidence.
        if uncited:
            verdict.verdict = "mismatch"
            verdict.detail = f"no evidence_ids but output mentions: {', '.join(uncited[:3])}"
        else:
            verdict.verdict = "grounded"
            verdict.detail = "no evidence claimed"
    else:
        verdict.verdict = "unverifiable"
        verdict.detail = "some evidence could not be verified"
    embedding_flags = _audit_embedding_versions(action)
    if embedding_flags:
        extra = "; ".join(embedding_flags[:3])
        verdict.detail = (verdict.detail + " | " if verdict.detail else "") + extra
        if any(
            "mismatch" in f or "missing_embedding" in f or "unregistered" in f or "missing embedding-version" in f
            for f in embedding_flags
        ):
            verdict.verdict = "mismatch"
    return verdict


def _audit_embedding_versions(action: dict[str, Any]) -> list[str]:
    """Defense-in-depth: flag mixed embedding versions on semantic decisions.

    Hash-era rows without ``emb_q=`` metadata are left alone. Qubot never
    rewrites the action.
    """
    import re

    from src.ml_runtime.embedding_space import HASH_EMBEDDING_VERSION

    text = f"{action.get('input_summary') or ''} {action.get('output_summary') or ''}"
    if "emb_q=" not in text and "emb_c=" not in text:
        return []
    q_m = re.search(r"emb_q=(\S+)", text)
    c_m = re.search(r"emb_c=(\S+)", text)
    match_m = re.search(r"match=(\S+)", text)
    build_m = re.search(r"cluster_build=(\S+)", text)
    flags: list[str] = []
    qv = q_m.group(1) if q_m else ""
    cv = c_m.group(1) if c_m else ""
    status = match_m.group(1) if match_m else ""
    atype = action.get("action_type") or ""
    if atype in {"similar_search", "cluster_matched", "brief_written", "embedding_recorded"}:
        if status in {"semantic", "performed"} and (not qv or not cv or cv == "-"):
            flags.append("missing embedding-version metadata on semantic decision")
        if qv and cv and cv not in {"-", ""} and qv != cv:
            flags.append(f"embedding version mismatch query vs candidate")
        if "novel_reason=missing_embedding" in text:
            flags.append("novelty caused only by missing_embedding")
        if qv.startswith("blake2b-512-v1") and build_m and "minilm" in (build_m.group(1) or "").lower():
            flags.append("hash query matched against semantic cluster build")
        if qv.startswith("all-MiniLM") and "blake2b-512-v1" in cv:
            flags.append("semantic query matched against hash-era vectors")
        registered = qv == HASH_EMBEDDING_VERSION or qv.startswith("all-MiniLM-L6-v2@") or qv.startswith("toy-semantic")
        if qv and not registered:
            flags.append("unregistered model revision")
        if cv and cv not in {"-", ""}:
            cre = cv == HASH_EMBEDDING_VERSION or cv.startswith("all-MiniLM-L6-v2@") or cv.startswith("toy-semantic")
            if not cre:
                flags.append("unregistered candidate model revision")
    return flags


def _mark_case_needs_review(case_id: str) -> None:
    """Mark a case needs_review (only if it's currently 'open' or 'pending_followup').

    Also tags case_kind='audit_review' (audit 7.2) so fleet statistics —
    anomaly counts, cluster rebuilds, economics, the customer funnel — never
    mistake an audit finding for a customer complaint — and opens a
    review-queue row with an SLA + null owner so the finding has someone and
    somewhen (board #10).
    """
    with ops_con() as con:
        # Use status='pending_followup' to signal needs_review; the original
        # status is preserved via the followup_draft (deterministic in v2).
        try:
            con.execute(
                "UPDATE cases SET status = 'pending_followup', case_kind = 'audit_review'"
                " WHERE case_id = ? AND status = 'open'",
                [case_id],
            )
        except Exception:
            # Pre-case_kind schema.
            con.execute(
                "UPDATE cases SET status = 'pending_followup' WHERE case_id = ? AND status = 'open'",
                [case_id],
            )
        try:
            row = con.execute(
                "SELECT interaction_id FROM cases WHERE case_id = ?", [case_id]
            ).fetchone()
            iid = str(row[0]) if row and row[0] else None
        except Exception:
            iid = None
    try:
        from src.frontline.validation_queue import create_review

        create_review(interaction_id=iid, case_id=case_id, reason="audit_mismatch")
    except Exception:
        pass


# ── Report writing ──────────────────────────────────────────────────────────


def _write_contact_report(result: AuditResult, data: dict[str, Any]) -> Path:
    """Write the markdown audit report."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"{result.interaction_id}.md"

    interaction = data["interaction"]
    turns = data["turns"]
    actions = data["actions"]
    case = data["case"] or {}

    overall_emoji = "✅" if result.overall_verdict == "grounded" else "🚩"

    lines: list[str] = []
    lines.append(f"# Contact Audit Report — {result.interaction_id}")
    lines.append("")
    lines.append(f"**Pack:** `{result.pack_id}` (v{result.pack_version})  ")
    lines.append(f"**Generated:** {result.generated_at.isoformat()}  ")
    lines.append(f"**Overall verdict:** {overall_emoji} **{result.overall_verdict.upper()}**")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Channel:** {interaction.get('channel')}")
    lines.append(f"- **Started:** {interaction.get('started_at')}")
    lines.append(f"- **Ended:** {interaction.get('ended_at')}")
    lines.append(f"- **Outcome:** {interaction.get('outcome')}")
    lines.append(f"- **Supervised:** {interaction.get('supervised')}")
    lines.append(f"- **Peak frustration:** {result.peak_frustration:.2f}")
    lines.append(f"- **LLM calls:** {interaction.get('llm_calls', 0)}")
    if interaction.get("enrichment_partial"):
        lines.append(
            "- **Enrichment:** PARTIAL (timeout or agent failure; brief may be incomplete)"
        )
    if case:
        lines.append(f"- **Case:** `{case.get('case_id')}` (severity={case.get('severity')}, priority=P{case.get('priority')})")
        if case.get("investigation_id"):
            lines.append(f"- **Investigation:** `{case.get('investigation_id')}`")
    lines.append("")

    # ── Timeline table ──────────────────────────────────────────────────────
    lines.append("## Timeline")
    lines.append("")
    lines.append("| Seq | Speaker | Text | Latency | LLM | Frustration |")
    lines.append("|---|---|---|---|---|---|")
    for t in turns:
        text = (t.get("text") or "").replace("|", "\\|").replace("\n", " ")[:120]
        fr = t.get("frustration_score")
        fr_str = f"{fr:.2f}" if fr is not None else "—"
        lines.append(
            f"| {t.get('seq')} | {t.get('speaker')} | {text} | "
            f"{t.get('latency_ms') or '—'}ms | {'yes' if t.get('llm_used') else 'no'} | {fr_str} |"
        )
    lines.append("")

    # ── Agent actions + groundedness ────────────────────────────────────────
    lines.append("## Agent Actions & Groundedness")
    lines.append("")
    lines.append(f"Total: {result.total_actions} | Grounded: {result.grounded_actions} | "
                 f"Unverifiable: {result.unverifiable_actions} | **Mismatch: {result.mismatch_actions}**")
    lines.append("")
    lines.append("| Action | Agent | Type | Verdict | Evidence | Detail |")
    lines.append("|---|---|---|---|---|---|")
    for v in result.action_verdicts:
        verdict_emoji = {"grounded": "✅", "unverifiable": "⚠️", "mismatch": "🚩"}.get(v.verdict, "?")
        ev = f"{v.evidence_confirmed}/{v.evidence_checked}"
        detail = v.detail.replace("|", "\\|")[:140]
        lines.append(
            f"| `{v.action_id[:12]}` | {v.agent} | {v.action_type} | "
            f"{verdict_emoji} {v.verdict} | {ev} | {detail} |"
        )
    lines.append("")

    # ── Frustration curve ──────────────────────────────────────────────────
    if result.frustration_curve:
        lines.append("## Frustration Curve")
        lines.append("")
        lines.append("```")
        max_bar = 40
        for pt in result.frustration_curve:
            score = pt.get("frustration_score") or 0.0
            bar_len = int(score * max_bar)
            lines.append(f"  turn {pt.get('seq'):>3}: {'█' * bar_len} {score:.2f}")
        lines.append("```")
        lines.append("")

    # ── Supervised segment ─────────────────────────────────────────────────
    if result.supervised_segments:
        lines.append("## Supervised Segment")
        lines.append("")
        for seg in result.supervised_segments:
            lines.append(f"**[supervisor, turn {seg.get('seq')}]** {seg.get('text')}")
            lines.append("")
        lines.append("")

    # ── Flags + recommendations ────────────────────────────────────────────
    if result.flags:
        lines.append("## 🚩 Flags")
        lines.append("")
        for f in result.flags:
            lines.append(f"- {f}")
        lines.append("")
    if result.recommendations:
        lines.append("## Recommendations")
        lines.append("")
        for r in result.recommendations:
            lines.append(f"- {r}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ── Daily digest ─────────────────────────────────────────────────────────────


def write_daily_digest(window_days: int = 1) -> Path:
    """Build the daily_frontline_digest playbook output."""
    DIGESTS_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = DIGESTS_DIR / f"daily_{today}.md"

    funnel = case_funnel(window_days=window_days)
    risk = live_risk(window_days=7)
    perf = agent_performance(window_days=window_days)
    invs = investigation_status(window_days=30)

    lines: list[str] = []
    lines.append(f"# Skew AI Daily Digest — {today}")
    lines.append("")
    lines.append(f"_Window: last {window_days} day(s). Generated {datetime.now(timezone.utc).isoformat()}_")
    lines.append("")

    lines.append("## Contact Funnel")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    for k, v in funnel.items():
        lines.append(f"| {k.replace('_', ' ').title()} | {v} |")
    lines.append("")

    lines.append("## Live Risk (top clusters, last 7 days)")
    lines.append("")
    if not risk:
        lines.append("_No live cluster risk in the window._")
        lines.append("")
    else:
        lines.append("| Cluster | Live Cases | Critical | Lead-Time (wks) | Last Case |")
        lines.append("|---|---|---|---|---|")
        for cl in risk[:10]:
            lines.append(
                f"| #{cl.get('cluster_id')} | {cl.get('live_case_count')} | "
                f"{cl.get('critical_count', 0)} | {cl.get('lead_time_weeks', '—')} | "
                f"{cl.get('last_case_at')} |"
            )
        lines.append("")

    lines.append("## Agent Performance")
    lines.append("")
    if not perf:
        lines.append("_No agent activity in the window._")
        lines.append("")
    else:
        lines.append("| Agent | Actions | Errors | Error% | p50 (ms) | p95 (ms) |")
        lines.append("|---|---|---|---|---|---|")
        for a in perf:
            lines.append(
                f"| {a.get('agent')} | {a.get('action_count')} | {a.get('error_count', 0)} | "
                f"{a.get('error_rate_pct', 0)}% | {a.get('p50_duration_ms', '—')} | "
                f"{a.get('p95_duration_ms', '—')} |"
            )
        lines.append("")

    lines.append("## Open Investigations")
    lines.append("")
    open_invs = [i for i in invs if i.get("status") == "open"]
    if not open_invs:
        lines.append("_No open investigations._")
        lines.append("")
    else:
        lines.append("| Investigation | Cluster | Cases | Days Open | Title |")
        lines.append("|---|---|---|---|---|")
        for i in open_invs:
            lines.append(
                f"| `{i.get('investigation_id')}` | #{i.get('cluster_id')} | "
                f"{i.get('case_count')} | {i.get('days_open', '—')} | {i.get('title', '')[:60]} |"
            )
        lines.append("")

    lines.append("## Handoffs (audit 2.4)")
    lines.append("")
    try:
        from src.data.warehouse import ops_con as _ops_con

        with _ops_con(read_only=True) as _con:
            _h = _con.execute(
                """
                SELECT action_type, COUNT(*) FROM agent_actions
                WHERE ts >= now() - INTERVAL (? || ' days')
                  AND action_type IN (
                    'handoff_offer_emitted', 'handoff_accepted',
                    'handoff_unfulfilled')
                GROUP BY action_type
                """,
                [str(window_days)],
            ).fetchall()
        _hm = {str(a): int(n) for a, n in _h}
        lines.append(
            f"offered={_hm.get('handoff_offer_emitted', 0)} "
            f"accepted={_hm.get('handoff_accepted', 0)} "
            f"unfulfilled={_hm.get('handoff_unfulfilled', 0)}"
        )
    except Exception:
        lines.append("_Handoff stats unavailable._")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


__all__ = [
    "audit_interaction",
    "write_daily_digest",
    "AuditResult",
    "ActionVerdict",
    "REPORTS_DIR",
    "DIGESTS_DIR",
]
