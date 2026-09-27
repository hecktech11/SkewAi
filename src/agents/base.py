"""Agent base contract.

Every v2 agent (intake, sentiment, triage, sentinel, investigator, case)
implements `Agent`. The orchestrator constructs agents per-interaction with
a pack + interaction context; agents run, write to the ledger, and return a
structured result. They never emit directly to the customer — that's the
orchestrator's job.

Design (carried from v1's `src/agents/base.py`):
  - Agents are stateless between turns; per-interaction state lives in the
    `InteractionContext` passed in.
  - The ledger is the audit spine; every side-effecting decision goes through
    `record_action` (via `action_timer`) BEFORE the output is emitted.
  - No LangChain / agent frameworks — each agent is a small, testable function.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.domains.loader import LoadedPack
from src.ledger import AgentAction


class TurnPersistenceError(RuntimeError):
    """A turn was not durably stored. Callers must not treat it as accepted."""


def _ctx_utc_now() -> datetime:
    from src.data.timeutil import utc_now

    return utc_now()


@dataclass
class InteractionContext:
    """Per-interaction state shared across agents.

    The orchestrator owns this; agents read + extend it. Everything here is
    persisted to the ops warehouse tables (interactions, interaction_turns,
    cases, agent_actions).
    """

    interaction_id: str
    pack: LoadedPack
    channel: str = "web_text"                 # web_voice | web_text | simulated
    started_at: datetime = field(default_factory=lambda: _ctx_utc_now())

    # ── Slot state ────────────────────────────────────────────────────────
    slots: dict[str, str] = field(default_factory=dict)          # entity_1/2/3, category, description
    slot_attempts: dict[str, int] = field(default_factory=dict)   # re-ask counters

    # ── Transcript ───────────────────────────────────────────────────────
    turns: list[dict[str, Any]] = field(default_factory=list)    # {seq, speaker, text, ts, frustration?}

    # ── Enrichment outputs (filled by Triage/Sentinel/Investigator) ──────
    severity: str = "Low"                                         # Low | Medium | Critical
    severity_source: str = "rules"                               # model | rules
    priority: int = 3                                             # 1-4
    safety_flags: dict[str, bool] = field(default_factory=dict)
    advisory_match: dict[str, Any] | None = None                  # {advisory_id, scope_summary, remedy, url, source}
    investigation_brief: dict[str, Any] | None = None             # {cluster_id, similar_count, lead_time_weeks, ...}
    frustration_score: float = 0.0                                # rolling avg last 3 customer turns
    peak_frustration: float = 0.0
    frustration_flagged: bool = False
    handoff_offered: bool = False

    # ── Case outputs ─────────────────────────────────────────────────────
    case_id: str | None = None
    followup_draft: str | None = None
    investigation_id: str | None = None

    # ── Bookkeeping ──────────────────────────────────────────────────────
    llm_calls: int = 0
    supervised: bool = False
    state: str = "GREETING"
    # sha256 identity for returning-customer case attach (audit 1.1).
    customer_ref: str | None = None
    # Supervisor identity holding a takeover (board: simultaneous takeovers).
    takeover_claimed_by: str | None = None
    # True when the close was forced while supervised (board: human_resolved).
    closed_while_supervised: bool = False
    # Pre-close self-critique result (set by Orchestrator._move_to_closing).
    self_critique: dict[str, Any] | None = None
    # Enrichment ran (even partially) — hangup after this point must close
    # with a case, never abandon evidence (audit 6.1).
    enrichment_done: bool = False
    enrichment_partial: bool = False
    enrichment_degraded: bool = False
    # Kill-switch script captured while SUPERVISED (audit 3.3): emitted if
    # the supervisor releases with a safety escalation pending.
    pending_safety_script: str | None = None
    # customer | simulated | audit_review — live-risk and COPQ exclude non-customer.
    case_kind: str = "customer"
    # ── Voice Tier-A wiring state ─────────────────────────────────────
    region: str | None = None
    locale: str = "en-US"
    voice_consent_required: bool = False
    voice_consent_obtained: bool = False
    voice_consent_script: str = ""
    voice_consent_version: str = "policy.v1"
    drive_mode: bool = False
    drive_mode_offered: bool = False
    confirmation_phase: str = "collection"  # collection | pending | done
    confirmation_attempts: int = 0
    vin: str | None = None
    last_agent_action_id: str | None = None
    last_agent_text: str = ""
    last_agent_word_markers: list[dict[str, Any]] = field(default_factory=list)
    last_turn_latency_ms: float = 0.0
    last_asr_confidence: float | None = None
    # Versioned human-control snapshot. Bumped on takeover and release.
    control_generation: int = 0
    # Utterances already handed to the channel, keyed by utterance id.
    # Playback identity stays on the one currently speaking so a later
    # generated turn cannot be reconciled as if the caller heard it.
    emitted_utterances: dict[str, dict[str, Any]] = field(default_factory=dict)
    playing_utterance_id: str | None = None
    human_message_ids: set[str] = field(default_factory=set)

    # ── Slot helpers ────────────────────────────────────────────────────
    def required_slots_remaining(self) -> list[str]:
        return [
            s.name
            for s in self.pack.required_slots()
            if not self.slots.get(s.name)
        ]

    def count_turn(self) -> int:
        """Customer turns that consume FRONTLINE_MAX_TURNS.

        Supervisor / agent turns are excluded so a takeover cannot burn the
        budget, and confirmation / elicitation replies still count because
        they are recorded as speaker='customer'.
        """
        return count_turn(self)

    def has_required_slots(self) -> bool:
        return not self.required_slots_remaining()

    def record_turn(self, speaker: str, text: str, **extra: Any) -> dict[str, Any]:
        if getattr(self, "_erased", False):
            raise TurnPersistenceError(f"interaction {self.interaction_id} has been erased")
        try:
            from src.frontline.dsr import is_interaction_erased
            if is_interaction_erased(self.interaction_id):
                self._erased = True
                raise TurnPersistenceError(f"interaction {self.interaction_id} has been erased")
        except TurnPersistenceError:
            raise
        except Exception:
            pass
        seq = len(self.turns) + 1
        turn = {
            "turn_id": f"{self.interaction_id}_t{seq}",
            "seq": seq,
            "speaker": speaker,
            "text": text,
            "ts": _ctx_utc_now(),
            **extra,
        }
        # Durable write before the turn is visible in memory. A failed insert
        # must not look like an accepted turn to resume, the console, or retry.
        from src.data.turns import persist_turn

        try:
            persist_turn(self.interaction_id, turn)
        except Exception as exc:
            raise TurnPersistenceError(
                f"turn {turn.get('turn_id')} was not stored"
            ) from exc
        self.turns.append(turn)
        return turn


def count_turn(ctx: InteractionContext) -> int:
    """Single turn-budget counter used by every collection path.

    Counts speaker='customer' only. Supervisor and agent turns do not consume
    FRONTLINE_MAX_TURNS. Confirmation, readback replies, and elicitation
    answers are customer turns and therefore count.
    """
    return sum(1 for t in ctx.turns if (t.get("speaker") or "") == "customer")


# ── Agent contract ──────────────────────────────────────────────────────────


class Agent(ABC):
    """Base class for all v2 agents.

    Agents are constructed once per interaction with the interaction context.
    `run()` is called by the orchestrator when the agent's trigger condition
    is met. Agents return a structured result dict; they MUST NOT emit directly
    to the customer.
    """

    name: str = "agent"              # intake | sentiment | triage | sentinel | investigator | case

    def __init__(self, ctx: InteractionContext) -> None:
        self.ctx = ctx

    @property
    def pack(self) -> LoadedPack:
        return self.ctx.pack

    @abstractmethod
    async def run(self, **kwargs: Any) -> dict[str, Any]:
        """Execute the agent's task. Returns a structured result dict.

        Implementations MUST record every action to the ledger before any
        observable side effect (using `action_timer` or `record_action`).
        """
        ...

    def _action(
        self,
        action_type: str,
        input_summary: str = "",
        output_summary: str = "",
        evidence_ids: list[str] | None = None,
        claims: list[Any] | None = None,
        case_id: str | None = None,
        ok: bool = True,
        error: str | None = None,
        duration_ms: int | None = None,
    ) -> AgentAction:
        """Build an AgentAction bound to this agent + interaction."""
        return AgentAction(
            interaction_id=self.ctx.interaction_id,
            agent=self.name,
            action_type=action_type,
            input_summary=input_summary,
            output_summary=output_summary,
            evidence_ids=evidence_ids or [],
            claims=claims or [],
            case_id=case_id,
            ok=ok,
            error=error,
            duration_ms=duration_ms,
        )
