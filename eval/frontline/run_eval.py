"""Frontline v2 eval runner — replay personas through the orchestrator.

Replays each persona through the orchestrator in text mode against committed
fixture DBs (offline, LLM stubbed like v1's harness). CI gates (per blueprint
§15.1):
  - slot-fill ≥ 90% (cooperative/vague)
  - safety escalation = 100%
  - advisory notification on planted matches = 100%
  - frustration flag on angry persona = 100%
  - investigation auto-open on planted Nth case = 100%
  - uncited claims = 0 (from audit)
  - ledger completeness = 100%
  - turns-to-completion ≤ 12

All gates must pass on both packs — that's the genericity proof in CI.

Usage:
    python -m eval.frontline.run_eval
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from src.agents.orchestrator import OrchestratorHooks, create_interaction
from src.data.warehouse import ops_con, reset_ops_db, init_ops_db
from src.domains.loader import list_packs, load_pack
from eval.frontline.personas import Persona, personas_for_pack
from eval.frontline.report import GateResult, format_eval_report


# ── Pack seeding dispatcher ──────────────────────────────────────────────────
# Each pack has its own seed script (the Pack Builder will eventually replace
# these). We dispatch by pack_id so the finance pack gets finance data, not
# automotive data.

_PACK_SEEDERS = {
    "automotive_nhtsa": None,  # filled below via lazy import
    "finance_cfpb": None,
}


def _seed_pack_domain(pack_id: str) -> None:
    """Seed the domain warehouse for a pack using its dedicated seed script."""
    if pack_id == "automotive_nhtsa":
        from scripts.seed_domains import build as build_auto
        build_auto(pack_id)
    elif pack_id == "finance_cfpb":
        from scripts.seed_finance_cfpb import build as build_fin
        build_fin(pack_id)
    else:
        # Try the automotive seeder as a last resort (covers _template etc.)
        try:
            from scripts.seed_domains import build as build_auto
            build_auto(pack_id)
        except Exception as e:
            raise FileNotFoundError(f"no seeder for pack '{pack_id}': {e}")


# ── Hooks that capture every emit for assertions ────────────────────────────


class _EvalHooks(OrchestratorHooks):
    """Captures agent turns + activities + handoff offers for assertions."""

    def __init__(self) -> None:
        self.agent_turns: list[str] = []
        self.activities: list[dict[str, Any]] = []
        self.slots_updates: list[dict[str, str]] = []
        self.handoff_offers: int = 0
        self.ended: dict[str, Any] | None = None

    async def emit_customer_turn(self, text: str, meta: dict[str, Any]) -> None:
        if meta.get("speaker") in ("agent", "supervisor"):
            self.agent_turns.append(text)

    async def emit_activity(self, payload: dict[str, Any]) -> None:
        self.activities.append(dict(payload))

    async def emit_slots_update(self, slots: dict[str, str]) -> None:
        self.slots_updates.append({k: v for k, v in slots.items() if not k.startswith("__")})

    async def emit_handoff_offer(self) -> None:
        self.handoff_offers += 1

    async def emit_interaction_ended(self, payload: dict[str, Any]) -> None:
        self.ended = dict(payload)


# ── Run one persona ───────────────────────────────────────────────────────────


async def run_persona(persona: Persona, pack_id: str) -> dict[str, Any]:
    """Run one persona through the orchestrator. Returns captured state.

    All observations are black-box (orchestrator state + emitted turns), and
    expectations come from ``persona.expected`` — gates must not re-derive
    outcomes from the same structures the agents populate. ``prefilled_slots``
    captures any memory-prefilled slots before turn 1 so slot-fill measures
    NEW information from this contact only.
    """
    hooks = _EvalHooks()
    orch, _ = await create_interaction(channel="web_text", pack_id=pack_id, hooks=hooks)
    required_names = [s.name for s in orch.ctx.pack.required_slots()]
    prefilled = {k for k in required_names if orch.ctx.slots.get(k)}
    turns_to_escalation: int | None = None
    for idx, turn in enumerate(persona.turns, start=1):
        if orch.ctx.state in ("DONE", "ABANDONED"):
            break
        await orch.handle_customer_turn(turn)
        if turns_to_escalation is None and any(orch.ctx.safety_flags.values()):
            turns_to_escalation = idx
    if (
        persona.expected.should_complete
        and orch.ctx.slots.get("__confirm_pending__")
        and orch.ctx.state not in ("DONE", "ABANDONED")
    ):
        await orch.handle_customer_turn("Yes, that's right.")
    if orch.ctx.state != "DONE":
        await orch.hangup()

    brief = orch.ctx.investigation_brief or {}
    # Slot-fill on NEW information: prefilled memory does not count.
    new_required = [s for s in required_names if s not in prefilled]
    new_filled = [s for s in new_required if orch.ctx.slots.get(s)]
    slot_fill_new = (len(new_filled) / len(new_required)) if new_required else 1.0

    return {
        "interaction_id": orch.ctx.interaction_id,
        "state": orch.ctx.state,
        "slots": dict(orch.ctx.slots),
        "slot_fill_pct": _slot_fill_pct(orch),
        "slot_fill_new_pct": slot_fill_new,
        "prefilled_slots": sorted(prefilled),
        "advisory_match": orch.ctx.advisory_match is not None,
        "advisory_id": (orch.ctx.advisory_match or {}).get("advisory_id"),
        "cluster_id": brief.get("cluster_id"),
        "investigation_opened": orch.ctx.investigation_id is not None,
        "investigation_id": orch.ctx.investigation_id,
        "frustration_flagged": orch.ctx.frustration_flagged,
        "peak_frustration": orch.ctx.peak_frustration,
        "safety_flags": dict(orch.ctx.safety_flags),
        "turns_to_escalation": turns_to_escalation,
        "case_id": orch.ctx.case_id,
        "n_customer_turns": sum(1 for t in orch.ctx.turns if t["speaker"] == "customer"),
        "n_total_turns": len(orch.ctx.turns),
        "agent_turns": hooks.agent_turns,
        "activities": hooks.activities,
        "handoff_offers": hooks.handoff_offers,
    }


def _slot_fill_pct(orch) -> float:
    required = orch.ctx.pack.required_slots()
    if not required:
        return 1.0
    filled = sum(1 for s in required if orch.ctx.slots.get(s.name))
    return filled / len(required)


async def run_investigation_auto_open_gate(
    *,
    pack_id: str,
    trigger_persona: Persona,
    min_cases: int,
) -> GateResult:
    """Investigation gate with isolated ops state and honest attempt counts.

    Resets the ops warehouse so earlier eval personas cannot contribute cases.
    Runs the trigger persona up to ``min_cases`` times (success path opens on
    the Nth similar case). Reports actual attempts / successful runs, not a
    fixed planned count when breaking early.
    """
    # Isolate from prior personas in this eval_pack run.
    reset_ops_db()
    init_ops_db()

    planned = max(1, int(min_cases))
    attempts = 0
    successes = 0
    opened = False
    last_error = ""

    for _ in range(planned):
        attempts += 1
        try:
            pr = await run_persona(trigger_persona, pack_id)
            if pr.get("error"):
                last_error = str(pr.get("error"))
                continue
            successes += 1
            if pr.get("investigation_opened"):
                opened = True
                break
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            # Continue so counters stay honest; do not crash the whole pack eval.

    detail = (
        f"attempts={attempts} successes={successes} planned={planned} "
        f"investigation_opened={opened}"
    )
    if last_error and not opened:
        detail += f" last_error={last_error[:120]}"
    return GateResult(
        name="investigation_auto_open",
        pack_id=pack_id,
        passed=opened,
        expected=True,
        actual=opened,
        detail=detail,
    )


# ── Per-pack eval ─────────────────────────────────────────────────────────────


async def eval_pack(pack_id: str) -> list[GateResult]:
    """Run all personas against one pack; return gate results."""
    # Reset + seed
    reset_ops_db()
    init_ops_db()
    try:
        _seed_pack_domain(pack_id)
    except FileNotFoundError as e:
        # Pack doesn't have a seed script
        return [GateResult(
            name="pack_available", pack_id=pack_id, passed=False,
            detail=f"no domain warehouse builder for pack '{pack_id}': {e}",
        )]

    personas = personas_for_pack(pack_id)
    results: list[GateResult] = []

    # ── Run all personas ────────────────────────────────────────────────────
    persona_results: dict[str, dict[str, Any]] = {}
    for persona in personas:
        try:
            persona_results[persona.name] = await run_persona(persona, pack_id)
        except Exception as e:
            persona_results[persona.name] = {"error": f"{type(e).__name__}: {e}"}

    # ── Gate: slot-fill from persona.expected (cooperative + vague) ───────
    # Expectations are defined INDEPENDENTLY on the persona (not re-derived
    # from the same slot dict Intake populates); slot-fill measures NEW
    # information (memory prefill subtracted).
    by_name = {p.name: p for p in personas}
    for pname in ("cooperative", "vague"):
        pr = persona_results.get(pname, {})
        if "error" in pr:
            results.append(GateResult(
                name=f"slot_fill_{pname}", pack_id=pack_id, passed=False,
                detail=pr["error"], expected="≥90%", actual="error",
            ))
            continue
        want = (by_name.get(pname).expected.min_slot_fill_pct if by_name.get(pname) else 0.9) or 0.9
        pct = pr.get("slot_fill_new_pct", pr.get("slot_fill_pct", 0.0))
        passed = pct >= want
        results.append(GateResult(
            name=f"slot_fill_{pname}", pack_id=pack_id, passed=passed,
            expected=f"≥{want*100:.0f}% new-info", actual=f"{pct*100:.0f}%",
            detail=f"{pname} filled {pct*100:.0f}% new slots (prefill={pr.get('prefilled_slots')})",
        ))

    # ── Gate: safety escalation within 1 turn ─────────────────────────────
    pr = persona_results.get("safety_critical", {})
    if "error" not in pr:
        escalated = any(pr.get("safety_flags", {}).values())
        tte = pr.get("turns_to_escalation")
        passed = bool(escalated and tte is not None and tte <= 1)
        results.append(GateResult(
            name="safety_escalation", pack_id=pack_id, passed=passed,
            expected="escalate on turn ≤1", actual=f"turns_to_escalation={tte}",
            detail=f"safety_flags={pr.get('safety_flags', {})}, turns_to_escalation={tte}",
        ))

    # ── Gate: advisory notification (independently re-verified) ───────────
    # The gate does NOT trust the ctx flag alone: it re-queries the warehouse
    # for the advisory and checks scope overlap itself (no planted evidence
    # via the same SQL the matcher used).
    pr = persona_results.get("advisory_match", {})
    if "error" not in pr:
        matched = pr.get("advisory_match", False)
        verified = _independently_verify_advisory(
            pack_id, pr.get("advisory_id"), pr.get("slots") or {}
        )
        passed = bool(matched and verified)
        results.append(GateResult(
            name="advisory_notification", pack_id=pack_id, passed=passed,
            expected=True, actual=matched,
            detail=f"advisory_match={matched}, independent_scope_verify={verified}",
        ))

    # ── Gate: frustration on angry persona (behavioral, not lexicon) ──────
    # Lexicon-vs-lexicon would be tautological (angry turns contain lexicon
    # words by construction). The gate requires OBSERVED behavior: at least
    # one handoff offer emitted to the customer.
    pr = persona_results.get("angry", {})
    if "error" not in pr:
        flagged = pr.get("frustration_flagged", False)
        offers = int(pr.get("handoff_offers", 0))
        passed = bool(flagged and offers >= 1)
        results.append(GateResult(
            name="frustration_flag", pack_id=pack_id, passed=passed,
            expected="flagged + handoff_offers≥1",
            actual=f"flagged={flagged}, offers={offers}",
            detail=f"peak_frustration={pr.get('peak_frustration', 0):.2f}, "
                   f"handoff_offers={offers}",
        ))

    # ── Gates that require cooperative persona rows still in the ops DB ────
    # Must run BEFORE investigation_auto_open, which resets ops for isolation.
    pr = persona_results.get("cooperative", {})
    if "error" not in pr:
        n_turns = pr.get("n_customer_turns", 99)
        passed = n_turns <= 12
        results.append(GateResult(
            name="turns_to_completion", pack_id=pack_id, passed=passed,
            expected="≤12", actual=n_turns,
            detail=f"cooperative took {n_turns} customer turns",
        ))

    pr = persona_results.get("cooperative", {})
    if "error" not in pr and pr.get("interaction_id"):
        iid = pr["interaction_id"]
        complete = _check_ledger_completeness(iid)
        results.append(GateResult(
            name="ledger_completeness", pack_id=pack_id, passed=complete,
            expected=True, actual=complete,
            detail=(
                "every agent turn has a prior ledger row"
                if complete
                else f"ledger incomplete or missing agent turns for {iid}"
            ),
        ))

    pr = persona_results.get("cooperative", {})
    if "error" not in pr and pr.get("interaction_id"):
        iid = pr["interaction_id"]
        uncited = await _check_uncited_claims(iid)
        results.append(GateResult(
            name="uncited_claims", pack_id=pack_id, passed=(uncited == 0),
            expected=0, actual=uncited,
            detail=(
                f"audit found {uncited} action(s) with uncited IDs"
                if uncited >= 0
                else f"audit unavailable for {iid}"
            ),
        ))

    # ── Negative controls (item 41): fixtures that can genuinely fail ────
    # off_topic must NOT escalate safety or notify advisories (it must be
    # redirected and completed); abandoner must leave NO case behind.
    pr = persona_results.get("off_topic", {})
    if "error" not in pr:
        neg_ok = (
            not any((pr.get("safety_flags") or {}).values())
            and not pr.get("advisory_match", False)
            and pr.get("state") in ("DONE", "ABANDONED")
        )
        results.append(GateResult(
            name="negative_off_topic", pack_id=pack_id, passed=bool(neg_ok),
            expected="no escalate/notify + terminal",
            actual=f"flags={pr.get('safety_flags')}, notified={pr.get('advisory_match')}, state={pr.get('state')}",
            detail="off-topic must be redirected, never escalated",
        ))
    pr = persona_results.get("abandoner", {})
    if "error" not in pr:
        neg_ok = pr.get("case_id") is None and pr.get("state") == "ABANDONED"
        results.append(GateResult(
            name="negative_abandoner", pack_id=pack_id, passed=bool(neg_ok),
            expected="case_id None + ABANDONED",
            actual=f"case_id={pr.get('case_id')}, state={pr.get('state')}",
            detail="hangup must not orphan a case",
        ))

    # ── Gate: investigation auto-open (resets ops — last DB-mutating gate) ─
    from src.config import settings

    min_cases = settings.investigation_min_cases
    trigger_persona = next((p for p in personas if p.name == "investigation_trigger"), None)
    if trigger_persona is None:
        results.append(GateResult(
            name="investigation_auto_open", pack_id=pack_id, passed=False,
            expected=True, actual=False,
            detail="no 'investigation_trigger' persona defined for this pack",
        ))
    else:
        gate = await run_investigation_auto_open_gate(
            pack_id=pack_id,
            trigger_persona=trigger_persona,
            min_cases=min_cases,
        )
        # Pin the matched cluster to the fixture's known cluster (item 41):
        # automotive brake fixture is cluster 14, finance double-charge is 41.
        # A gate that opens on ANY cluster cannot detect cluster drift.
        expected_cluster = {"automotive_nhtsa": 14, "finance_cfpb": 41}.get(pack_id)
        if gate.passed and expected_cluster is not None:
            got = _last_trigger_cluster(pack_id)
            gate.detail += f" cluster_id={got} (expected {expected_cluster})"
            if got is not None and int(got) != expected_cluster:
                gate.passed = False
                gate.actual = got
        results.append(gate)

    if pack_id == "automotive_nhtsa":
        from eval.frontline.recall_lead_gate import recall_lead_gate

        results.append(recall_lead_gate())

    return results


def _last_trigger_cluster(pack_id: str) -> int | None:
    """Most recent investigation-linked cluster for *pack_id* (eval pin)."""
    try:
        with ops_con(read_only=True) as con:
            row = con.execute(
                """
                SELECT cluster_id FROM investigations
                WHERE pack_id = ? ORDER BY opened_at DESC LIMIT 1
                """,
                [pack_id],
            ).fetchone()
    except Exception:
        return None
    if not row or row[0] is None:
        return None
    try:
        return int(row[0])
    except (TypeError, ValueError):
        return None


def _independently_verify_advisory(
    pack_id: str, advisory_id: str | None, slots: dict[str, Any]
) -> bool:
    """Gate-side re-verification: the advisory must exist AND scope-match.

    Uses a direct warehouse read with the gate's own scope logic — never the
    matcher's in-memory flag — so planted evidence cannot self-certify.
    """
    if not advisory_id:
        return False
    try:
        from src.data.warehouse import domain_con

        with domain_con(pack_id) as con:
            cur = con.execute(
                """
                SELECT scope_entity_1, scope_entity_2, scope_entity_3, scope_category
                FROM advisories WHERE advisory_id = ?
                """,
                [advisory_id],
            )
            row = cur.fetchone()
    except Exception:
        return False
    if not row:
        return False
    scope = dict(zip(
        ("scope_entity_1", "scope_entity_2", "scope_entity_3", "scope_category"), row
    ))
    for slot, col in (
        ("entity_1", "scope_entity_1"), ("entity_2", "scope_entity_2"),
        ("entity_3", "scope_entity_3"), ("category", "scope_category"),
    ):
        want = scope.get(col)
        if want is None:
            continue
        got = (slots.get(slot) or "").strip()
        if not got or str(want).upper() != str(got).upper():
            return False
    return True


def _check_ledger_completeness(interaction_id: str) -> bool:
    """Verify every agent turn has a matching agent_actions row.

    Fails closed when the interaction has no agent turns (missing id / wiped DB).
    """
    with ops_con(read_only=True) as con:
        turns = con.execute(
            "SELECT text FROM interaction_turns WHERE interaction_id = ? AND speaker = 'agent'",
            [interaction_id],
        ).fetchall()
        actions = con.execute(
            "SELECT output_summary FROM agent_actions WHERE interaction_id = ?",
            [interaction_id],
        ).fetchall()
    if not turns:
        # Vacuous pass is a false green when the interaction was wiped or never stored.
        return False
    action_summaries = [(a[0] or "") for a in actions]
    for (turn_text,) in turns:
        # Each agent turn's text should appear in SOME action's output_summary,
        # OR be the greeting/goodbye (which are in greeting_emitted/goodbye_emitted).
        if not any(turn_text[:60] in s for s in action_summaries):
            return False
    return True


async def _check_uncited_claims(interaction_id: str) -> int:
    """Return mismatch count, or -1 if audit cannot run (fail closed for gate)."""
    try:
        from src.qubot.auditor import audit_interaction
        result = await audit_interaction(interaction_id, write_report=False)
        return int(result.mismatch_actions)
    except Exception:
        return -1


# ── Main ──────────────────────────────────────────────────────────────────────


# Packs gated in CI / make eval-frontline. Scaffold packs (e.g. `_template`) are
# intentionally excluded so overall exit code reflects real verticals only.
EVAL_PACKS: tuple[str, ...] = ("automotive_nhtsa", "finance_cfpb")


def packs_for_eval() -> list[str]:
    """Return real vertical packs that exist on disk, in EVAL_PACKS order."""
    available = set(list_packs())
    return [p for p in EVAL_PACKS if p in available]


async def main() -> int:
    all_results: list[GateResult] = []
    packs = packs_for_eval()
    if not packs:
        print("No eval packs found (expected automotive_nhtsa and/or finance_cfpb).")
        return 1
    for pack_id in packs:
        print(f"\n→ Evaluating pack: {pack_id}")
        try:
            results = await eval_pack(pack_id)
        except Exception as e:
            results = [GateResult(
                name="eval_completed", pack_id=pack_id, passed=False,
                detail=f"eval crashed: {type(e).__name__}: {e}",
            )]
        all_results.extend(results)
        for r in results:
            mark = "✅" if r.passed else "❌"
            print(f"  {mark} {r.name}: {r.detail}")

    report = format_eval_report(all_results)
    print()
    print(report)

    # Also write to reports/
    from datetime import datetime, timezone
    from pathlib import Path
    reports_dir = Path("reports/qubot/eval")
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"eval_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.md"
    path.write_text(report, encoding="utf-8")
    print(f"\nEval report written to: {path}")

    return 0 if all_results and all(r.passed for r in all_results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
