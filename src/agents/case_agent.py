"""Case Agent — record + follow-up + investigation link.

Creates the `cases` row from the slot frame + Triage + Sentinel + Investigator
outputs; drafts the follow-up email (LLM-grounded: may only reference ids
present in `evidence_ids`, verified before saving); hands the case number back
to Intake for readback.

Investigation auto-linking (deterministic rule):
  - If the case has a cluster_match_id:
      * link to the open investigation for that cluster if one exists; otherwise
      * if this is the Nth case (FRONTLINE_INVESTIGATION_MIN_CASES, default 3)
        on that cluster within the rolling window, auto-open a new
        investigation (title from cluster top terms), fire an
        investigation_opened alert, and have Intake mention it in the
        closing script.
"""

from __future__ import annotations

import asyncio
import json
import threading
from datetime import datetime, timedelta, timezone
from typing import Any

from src.agents.base import Agent
from src.config import settings
from src.data.warehouse import ops_con, ops_in_thread
from src.ids import new_ulid
from src.ledger import record_action
from src.ledger.writer import record_action_on_con

# Serialize open/link for a cluster within this process (M2).
_investigation_open_lock = threading.Lock()
# Same-loop serialization for the async close path (item 31): the threading
# lock covers sync sections; this covers awaits around them. Cross-process
# safety comes from the distributed close claim (src/jobs/registry).
_investigation_open_alock = asyncio.Lock()


def _case_id() -> str:
    return "case_" + new_ulid()


def _investigation_id(seq: int) -> str:
    return f"inv_{seq:04d}"


def _now() -> datetime:
    from src.data.timeutil import utc_now

    return utc_now()


class CaseAgent(Agent):
    """Create the case; draft follow-up; link to investigation."""

    name = "case"

    async def run(self, **kwargs: Any) -> dict[str, Any]:
        ctx = self.ctx

        # Idempotent re-close (item 7): a preset case_id that already exists
        # must never allocate a second case — finalize/update the existing one.
        # A preset id WITHOUT a row (stale/planted) falls through to fresh
        # creation so hangup can never strand a contact on a phantom case.
        if ctx.case_id and self._case_row(ctx.case_id) is not None:
            return await self.reclose_existing(ctx.case_id)

        # ── Compute evidence list ─────────────────────────────────────────
        evidence_ids: list[str] = []
        if ctx.advisory_match and ctx.advisory_match.get("advisory_id"):
            evidence_ids.append(ctx.advisory_match["advisory_id"])
        if ctx.investigation_brief:
            if ctx.investigation_brief.get("cluster_id") is not None:
                evidence_ids.append(str(ctx.investigation_brief["cluster_id"]))
            evidence_ids.extend(
                r.get("record_id", "")
                for r in ctx.investigation_brief.get("similar_records", [])
                if r.get("record_id")
            )

        # ── Allocate case_id FIRST so follow-up + ledger never see <pending>
        case_id = _case_id()
        ctx.case_id = case_id

        # ── Investigation auto-linking BEFORE draft so the letter can
        # honestly claim "under active investigation" only when linked ──
        investigation_id = None
        investigation_opened = False
        cluster_id = ctx.investigation_brief.get("cluster_id") if ctx.investigation_brief else None
        if cluster_id is not None:
            async with _investigation_open_alock:
                investigation_id = await ops_in_thread(
                    self._link_or_open_investigation, cluster_id
                )
            investigation_opened = investigation_id is not None and not self._was_existing(
                cluster_id, investigation_id
            )
        ctx.investigation_id = investigation_id

        # Hypotheses from elicitation answers (item 35): when the contact is
        # linked to an investigation AND the customer answered diagnostic
        # questions, record each answer as a proposed hypothesis with
        # provenance (question_id + answer_id) — the brief alone is not
        # where hypotheses live.
        if investigation_id:
            try:
                from src.enterprise.investigation_workspace import add_hypothesis
                from src.frontline.elicitation import answers_for_interaction

                for ans in answers_for_interaction(ctx.interaction_id):
                    body = (
                        f"Customer answer [{ans.get('question_id')}/"
                        f"{ans.get('answer_id')}]: "
                        f"{(ans.get('prompt') or '').strip()[:200]} → "
                        f"{(ans.get('answer') or '').strip()[:300]}"
                    )
                    try:
                        add_hypothesis(investigation_id, body, status="open")
                    except Exception:
                        continue
            except Exception:
                pass

        # ── Draft follow-up (deterministic; LLM polish is optional) ───────
        followup = self._draft_followup(evidence_ids)

        # ── Safety floor (C-OPEN-1): any safety flag → Critical + P1 ──────
        if any(ctx.safety_flags.values()):
            ctx.severity = "Critical"
            ctx.severity_source = ctx.severity_source or "rules"
            ctx.priority = 1

        # ── Insert case + case_created ledger in one transaction (M3) ─────
        case_action = self._action(
            action_type="case_created",
            input_summary=f"slots={ctx.slots}, severity={ctx.severity}, priority=P{ctx.priority}",
            output_summary=f"case_id={case_id}",
            evidence_ids=evidence_ids,
            case_id=case_id,
        )
        desc_plain = (ctx.slots.get("description") or "")[:500]
        # No plaintext fallback: when encryption is unavailable the case is
        # still written, but with a marker instead of the customer's words (R29).
        from src.security.pii import store_subject_text

        desc_stored = store_subject_text(
            ctx.interaction_id, desc_plain, field="cases.description_summary"
        )
        insert_args = [
            case_id,
            ctx.interaction_id,
            ctx.pack.id,
            _now(),
            ctx.slots.get("category"),
            desc_stored,
            _now(),
            ctx.severity,
            ctx.severity_source,
            ctx.priority,
            json.dumps(ctx.safety_flags),
            ctx.advisory_match.get("advisory_id") if ctx.advisory_match else None,
            cluster_id,
            (
                ctx.investigation_brief.get("similar_record_count", 0)
                if ctx.investigation_brief
                else 0
            ),
            investigation_id,
            "open",
            followup,
            getattr(ctx, "case_kind", None) or "customer",
            getattr(ctx, "customer_ref", None),
        ]

        def _insert_case() -> None:
            with ops_con() as con:
                con.execute("BEGIN TRANSACTION")
                try:
                    try:
                        con.execute(
                            """
                            INSERT INTO cases (
                                case_id, interaction_id, pack_id, created_at,
                                category, description_summary, onset, severity,
                                severity_source, priority, safety_flags,
                                advisory_match_id, cluster_match_id, similar_record_count,
                                investigation_id, status, followup_draft,
                                case_kind, customer_ref
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            insert_args,
                        )
                    except Exception:
                        # Pre-case_kind schema: fall back to the legacy column set.
                        con.execute(
                            """
                            INSERT INTO cases (
                                case_id, interaction_id, pack_id, created_at,
                                category, description_summary, onset, severity,
                                severity_source, priority, safety_flags,
                                advisory_match_id, cluster_match_id, similar_record_count,
                                investigation_id, status, followup_draft
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            insert_args[:-2],
                        )
                    record_action_on_con(con, case_action)
                    con.execute("COMMIT")
                except Exception:
                    try:
                        con.execute("ROLLBACK")
                    except Exception:
                        pass
                    raise

        await ops_in_thread(_insert_case)

        ctx.case_id = case_id
        ctx.investigation_id = investigation_id

        # Grounded remedy/next-step when advisory matched (close the loop)
        try:
            from src.frontline.remedy import offer_and_ledger

            offer = offer_and_ledger(
                ctx.interaction_id,
                advisory_match=ctx.advisory_match,
                case_id=case_id,
                pack_display_name=getattr(ctx.pack, "display_name", "support")
                or "support",
            )
            if offer:
                ctx.remedy_offer = offer  # type: ignore[attr-defined]
        except Exception:
            pass

        # Multi-issue split from description when multiple problems expressed.
        # Do not silently no-op: ledger failures so Qubot/ops can see them.
        try:
            from src.frontline.multi_issue import attach_multi_issues_from_description

            desc = ctx.slots.get("description") or ""
            multi = attach_multi_issues_from_description(
                ctx.interaction_id,
                pack_id=ctx.pack.id,
                description=desc,
                primary_case_id=case_id,
                category=ctx.slots.get("category"),
            )
            if multi:
                ctx.linked_issues = multi  # type: ignore[attr-defined]
        except Exception as e:
            try:
                record_action(self._action(
                    action_type="case_created",
                    input_summary="multi_issue_attach_failed",
                    output_summary=f"{type(e).__name__}: {e}"[:500],
                    evidence_ids=[],
                    case_id=case_id,
                    ok=False,
                    error=f"{type(e).__name__}: {e}"[:300],
                ))
            except Exception:
                pass

        record_action(self._action(
            action_type="followup_drafted",
            input_summary="",
            output_summary=followup[:500],
            evidence_ids=evidence_ids,
            case_id=case_id,
        ))

        # Outbound connector: case_created (never raises into contact path).
        try:
            from src.frontline.connectors import dispatch_case_created

            await dispatch_case_created(
                case_id=case_id,
                interaction_id=ctx.interaction_id,
                pack_id=ctx.pack.id if ctx.pack else None,
                investigation_id=investigation_id,
                cluster_id=cluster_id,
                severity=ctx.severity,
                priority=ctx.priority,
                category=ctx.slots.get("category"),
            )
        except Exception:
            pass

        if investigation_id:
            if investigation_opened:
                record_action(self._action(
                    action_type="investigation_opened",
                    input_summary=(
                        f"cluster_id={cluster_id} reached min cases "
                        f"({settings.investigation_min_cases})"
                    ),
                    output_summary=f"investigation_id={investigation_id}",
                    evidence_ids=[str(cluster_id)] if cluster_id is not None else [],
                    case_id=case_id,
                ))
                title = f"Cluster {cluster_id}"
                if ctx.investigation_brief and ctx.investigation_brief.get(
                    "cluster_top_terms"
                ):
                    terms = ctx.investigation_brief["cluster_top_terms"]
                    if isinstance(terms, list) and terms:
                        title = (
                            f"Cluster {cluster_id}: "
                            f"{' '.join(str(t) for t in terms[:3])}"
                        )
                try:
                    from src.frontline.alerts import alert_investigation_opened

                    await alert_investigation_opened(
                        investigation_id,
                        cluster_id,
                        title,
                        pack_id=ctx.pack.id if ctx.pack else None,
                    )
                except Exception:
                    pass
                try:
                    from src.frontline.connectors import dispatch_investigation_opened

                    await dispatch_investigation_opened(
                        investigation_id=investigation_id,
                        cluster_id=cluster_id,
                        title=title,
                        case_id=case_id,
                        interaction_id=ctx.interaction_id,
                        pack_id=ctx.pack.id if ctx.pack else None,
                    )
                except Exception:
                    pass
            else:
                record_action(self._action(
                    action_type="investigation_linked",
                    input_summary=f"cluster_id={cluster_id} has open investigation",
                    output_summary=f"investigation_id={investigation_id}",
                    evidence_ids=[str(cluster_id)] if cluster_id is not None else [],
                    case_id=case_id,
                ))

        return {
            "case_id": case_id,
            "investigation_id": investigation_id,
            "investigation_opened": investigation_opened,
            "followup_draft": followup,
            "evidence_ids": evidence_ids,
            "remedy_offer": getattr(ctx, "remedy_offer", None),
            "linked_issues": getattr(ctx, "linked_issues", None) or [],
        }

    @staticmethod
    def _case_row(case_id: str) -> dict[str, Any] | None:
        with ops_con(read_only=True) as con:
            try:
                cur = con.execute(
                    "SELECT * FROM cases WHERE case_id = ?", [case_id]
                )
                row = cur.fetchone()
            except Exception:
                return None
            if not row:
                return None
            return dict(zip([d[0] for d in cur.description], row))

    async def reclose_existing(self, case_id: str) -> dict[str, Any]:
        """Idempotent re-close of an already-created case (item 7).

        Refreshes severity/priority from the current context, ledgers a
        ``case_status_updated`` action, and returns the SAME case_id —
        repeated hangup/close events never create a second case.
        """
        ctx = self.ctx
        row = self._case_row(case_id) or {}
        if any(ctx.safety_flags.values()):
            ctx.severity = "Critical"
            ctx.severity_source = ctx.severity_source or "rules"
            ctx.priority = 1

        def _touch() -> None:
            with ops_con() as con:
                con.execute(
                    """
                    UPDATE cases
                    SET severity = ?, severity_source = ?, priority = ?
                    WHERE case_id = ?
                    """,
                    [
                        ctx.severity or row.get("severity") or "Medium",
                        ctx.severity_source or row.get("severity_source") or "rules",
                        ctx.priority or row.get("priority") or 3,
                        case_id,
                    ],
                )

        await ops_in_thread(_touch)
        ctx.case_id = case_id
        ctx.investigation_id = row.get("investigation_id")
        record_action(self._action(
            action_type="case_status_updated",
            input_summary="idempotent re-close: existing case reused",
            output_summary=(
                f"case_id={case_id} kept; no duplicate created "
                f"(severity={ctx.severity} P{ctx.priority})"
            ),
            evidence_ids=[],
            case_id=case_id,
        ))
        return {
            "case_id": case_id,
            "investigation_id": row.get("investigation_id"),
            "investigation_opened": False,
            "followup_draft": row.get("followup_draft") or "",
            "evidence_ids": [],
            "remedy_offer": getattr(ctx, "remedy_offer", None),
            "linked_issues": getattr(ctx, "linked_issues", None) or [],
            "reused": True,
        }
    def _draft_followup(self, evidence_ids: list[str]) -> str:
        """Grounded follow-up email. References only ids in evidence_ids."""
        ctx = self.ctx
        e1 = ctx.slots.get("entity_1", "")
        e2 = ctx.slots.get("entity_2", "")
        e3 = ctx.slots.get("entity_3", "")
        cat = ctx.slots.get("category", "")
        subject = f"Your support case — {e2} {e3} ({cat})"

        # Optional LLM opener (capped); body stays template-grounded.
        opener = f"Thanks for contacting support about your {e1} {e2} {e3}."
        try:
            from src.ai.narration import phrase_followup_draft

            res = phrase_followup_draft(
                case_id=ctx.case_id or "<pending>",
                category=cat or "issue",
                severity=str(ctx.severity or "Medium"),
                description=str(ctx.slots.get("description") or ""),
            )
            if res.text:
                opener = res.text
        except Exception:
            pass

        lines = [
            f"Subject: {subject}",
            "",
            "Hello,",
            "",
            opener,
            f"We've logged case {ctx.case_id or '<pending>'} for the {cat or 'issue'} you reported.",
            "",
        ]
        if ctx.advisory_match:
            adv = ctx.advisory_match
            lines.append(
                f"We found a matching known issue ({adv.get('advisory_id')}): "
                f"{adv.get('summary') or adv.get('scope_summary')}."
            )
            if adv.get("remedy"):
                lines.append(f"Recommended remedy: {adv['remedy']}")
            if adv.get("url"):
                lines.append(f"More info: {adv['url']}")
            lines.append("")
        if ctx.investigation_brief and ctx.investigation_brief.get("cluster_id") is not None:
            ib = ctx.investigation_brief
            lines.append(
                f"Your case matches a cluster of {ib.get('cluster_count', 0)} similar reports "
                f"(cluster #{ib.get('cluster_id')})."
            )
            if ib.get("lead_time_weeks") is not None:
                lines.append(
                    f"Historically, similar clusters preceded an advisory by "
                    f"{ib['lead_time_weeks']} weeks."
                )
            # Only claim active investigation when one is actually linked
            if ctx.investigation_id:
                lines.append("This issue is under active investigation.")
            lines.append("")
        lines.append("A specialist will follow up within one business day.")
        lines.append("")
        lines.append("Reference IDs (for your records): " + ", ".join(evidence_ids))
        return "\n".join(lines)

    # ── Investigation auto-linking ────────────────────────────────────────
    def _link_or_open_investigation(self, cluster_id: int) -> str | None:
        """Return an investigation_id for this cluster; open a new one if
        the Nth case threshold is crossed.

        Serialized with a process lock + re-check so two concurrent threshold
        crossings cannot open two open investigations for the same cluster
        (M2). Cross-process races are covered by the slice claim (audit 4.5):
        a claim loser never inserts — it links to the winner's row.

        Locking discipline: ``ops_con`` holds a NON-REENTRANT process lock
        for the connection lifetime, so the slice claim (which opens its own
        connection) is acquired BETWEEN two short connections, never nested
        inside one — nesting self-deadlocks.
        """
        ctx = self.ctx
        try:
            from src.jobs.registry import acquire_slice_claim

            def _claim() -> bool:
                return acquire_slice_claim(
                    f"inv:{ctx.pack.id}:{cluster_id}", f"case-{ctx.interaction_id}"
                )
        except Exception:
            def _claim() -> bool:
                return True

        def _bump(con, inv_id: str) -> str:
            con.execute(
                """
                UPDATE investigations
                SET case_count = case_count + 1, last_case_at = ?
                WHERE investigation_id = ?
                """,
                [_now(), inv_id],
            )
            return inv_id

        with _investigation_open_lock:
            # Phase A (short connection): link to an existing open row, else
            # check the rolling threshold. No claim held here.
            with ops_con() as con:
                inv_id = self._find_open_investigation(con, cluster_id, ctx.pack.id)
                if inv_id:
                    return _bump(con, inv_id)
                window_start = _now() - timedelta(days=7)
                count_row = con.execute(
                    """
                    SELECT COUNT(*) FROM cases
                    WHERE pack_id = ? AND cluster_match_id = ?
                      AND created_at >= ?
                    """,
                    [ctx.pack.id, cluster_id, window_start],
                ).fetchone()
                recent_count = count_row[0] if count_row else 0
                if recent_count + 1 < settings.investigation_min_cases:
                    return None
            # Phase B (NO connection held): claim the fleet-wide open right.
            # Claiming only now that we intend to insert keeps the TTL window
            # from blocking openers that ultimately don't insert.
            claimed = _claim()
            # Phase C (fresh connection): re-check, then insert-or-link.
            with ops_con() as con:
                inv_id = self._find_open_investigation(con, cluster_id, ctx.pack.id)
                if inv_id:
                    return _bump(con, inv_id)
                if not claimed:
                    return None  # winner owns the open; link on next contact
                seq_row = con.execute("SELECT nextval('investigation_seq')").fetchone()
                seq = seq_row[0] if seq_row else 1
                inv_id = _investigation_id(seq)

                title = f"Cluster {cluster_id}"
                if ctx.investigation_brief and ctx.investigation_brief.get(
                    "cluster_top_terms"
                ):
                    terms = ctx.investigation_brief["cluster_top_terms"]
                    if isinstance(terms, list) and terms:
                        title = (
                            f"Cluster {cluster_id}: "
                            f"{' '.join(str(t) for t in terms[:3])}"
                        )
                if ctx.slots.get("category"):
                    title += f" ({ctx.slots['category']})"

                con.execute(
                    """
                    INSERT INTO investigations
                    (investigation_id, pack_id, cluster_id, title, status,
                     opened_at, last_case_at, case_count)
                    VALUES (?, ?, ?, ?, 'open', ?, ?, 1)
                    """,
                    [inv_id, ctx.pack.id, cluster_id, title, _now(), _now()],
                )
                return inv_id

    @staticmethod
    def _find_open_investigation(con, cluster_id: int, pack_id: str) -> str | None:
        row = con.execute(
            """
            SELECT investigation_id FROM investigations
            WHERE cluster_id = ? AND pack_id = ? AND status = 'open'
            ORDER BY opened_at DESC LIMIT 1
            """,
            [cluster_id, pack_id],
        ).fetchone()
        return row[0] if row else None

    def _was_existing(self, cluster_id: int, investigation_id: str) -> bool:
        """Was this investigation pre-existing (i.e. linked, not opened)?
        Heuristic: case_count > 1 means it existed before this case."""
        with ops_con(read_only=True) as con:
            row = con.execute(
                "SELECT case_count FROM investigations WHERE investigation_id = ?",
                [investigation_id],
            ).fetchone()
            if not row:
                return False
            return row[0] > 1
