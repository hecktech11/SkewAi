"""Pydantic schema for a Domain Pack manifest (`pack.yaml`).

A pack is a directory `domains/<pack_id>/` containing:
  - pack.yaml            — this manifest
  - taxonomy.yaml        — category tree (referenced from pack.yaml)
  - gazetteers/*.csv     — frequency-ranked entity value lists
  - context/*            — Qubot context pack additions (schemas/semantics)
  - playbooks/*.yaml     — optional domain-specific Qubot playbooks
  - data/mapping.yaml    — source column → canonical column mapping
  - demo/*               — demo script + eval personas
  - models/*             — optional trained artifacts (severity classifier etc.)

The schema is intentionally narrow: three entity slots + one category slot + a
free-text description. Anything beyond that is out of scope for v2 (see
blueprint §21, "Over-generalizing kills the timeline"). This is what makes the
genericity proof tractable.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


# ── Slot frame ──────────────────────────────────────────────────────────────


class SlotSpec(BaseModel):
    """One slot in the intake agent's slot frame.

    Slots are filled in `order`. The intake agent asks exactly one question
    per turn for the highest-priority missing required slot.
    """

    name: str = Field(..., description="Slot key, e.g. 'entity_1', 'category', 'description'")
    label: str = Field(..., description="Human label, e.g. 'Vehicle year'")
    prompt: str = Field(..., description="Question template asked when this slot is missing")
    required: bool = True
    max_re_asks: int = Field(default=2, ge=0)
    # validation strategy
    validation: Literal["gazetteer", "regex", "year-range", "free-text"] = "free-text"
    # for gazetteer validation: name of the CSV in gazetteers/ (without .csv)
    gazetteer: str | None = None
    # for regex validation: a Python regex string
    regex: str | None = None
    # for year-range validation: [min, max]
    year_range: tuple[int, int] | None = None

    @field_validator("gazetteer")
    @classmethod
    def _gazetteer_requires_gazetteer_validation(cls, v, info):
        if info.data.get("validation") == "gazetteer" and not v:
            raise ValueError("validation='gazetteer' requires the `gazetteer` field to be set")
        return v


# ── Entity labels ───────────────────────────────────────────────────────────


class EntityLabels(BaseModel):
    """Labels for the three canonical entity slots.

    Automotive: year/make/model. Finance: product/sub_product/company.
    SaaS: plan/feature/platform. The labels are pack-defined; the slot keys
    (entity_1/2/3) are fixed.
    """

    entity_1: str = Field(..., description="Label for entity_1, e.g. 'Year'")
    entity_2: str = Field(..., description="Label for entity_2, e.g. 'Make'")
    entity_3: str = Field(..., description="Label for entity_3, e.g. 'Model'")


# ── Safety ──────────────────────────────────────────────────────────────────


class SafetySpec(BaseModel):
    """Pack-defined safety escalation.

    Automotive: fire/crash/injury. Finance: fraud/identity-theft/discrimination.
    """

    escalation_lexicon: list[str] = Field(
        ..., description="Substrings that trigger immediate safety escalation"
    )
    safety_questions: list[str] = Field(
        default_factory=list,
        description="Questions the intake agent must always ask, regardless of slots",
    )
    escalation_script: str = Field(
        ..., description="Static script read verbatim on safety escalation (no LLM)"
    )


# ── Advisory match ──────────────────────────────────────────────────────────


class AdvisoryMatchSpec(BaseModel):
    """Parameterized SQL template over the advisories table.

    The SQL MUST be read-only and parameterized. The loader lints it to reject
    write statements (INSERT/UPDATE/DELETE/DROP/ALTER/CREATE).
    """

    sql_template: str = Field(
        ...,
        description=(
            "Parameterized SQL. Bind params :entity_1, :entity_2, :entity_3, :category. "
            "Returns columns: advisory_id, issued_at, scope_summary, remedy, url, source."
        ),
    )
    readback_fields: list[str] = Field(
        default_factory=lambda: ["advisory_id", "scope_summary", "remedy", "url"],
        description="Fields read back to the customer verbatim (from SQL, no LLM)",
    )

    @field_validator("sql_template")
    @classmethod
    def _no_write_statements(cls, v: str) -> str:
        forbidden = ("INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE", "MERGE")
        upper = v.upper()
        for kw in forbidden:
            # word-boundary-ish check
            if kw in upper.split():
                raise ValueError(f"advisory_match.sql_template contains forbidden keyword: {kw}")
        return v


# ── Severity ────────────────────────────────────────────────────────────────


class SeverityRule(BaseModel):
    """One rule in the ordered severity rule list."""

    when: str = Field(..., description="Boolean expression over slot + safety flags, e.g. 'safety.fire'")
    severity: Literal["Low", "Medium", "Critical"]
    reason: str = ""


class SeveritySpec(BaseModel):
    """Severity model path (ML runtime artifact) OR ordered rule list.

    Automotive keeps the v1 XGBoost model. Other packs default to rules.
    """

    model_artifact: str | None = Field(
        default=None, description="Ref into ML runtime, e.g. 'automotive_nhtsa/severity_xgb'"
    )
    rules: list[SeverityRule] = Field(default_factory=list)
    # priority matrix: any safety flag → P1 always (hardcoded in triage.py)
    priority_matrix: dict[str, int] = Field(
        default_factory=lambda: {"safety": 1, "Critical": 1, "Medium": 2, "Low": 3}
    )


# ── Cost model + region centroids (item 33/34) ───────────────────────────────


class CostModel(BaseModel):
    """Pack-configured cost figures for financial_impact (item 33).

    No env-var guessing: every number is declared here, in the pack, where
    operators can review it. ``exposure_multiplier`` replaces the old
    arbitrary ``*10`` recall multiplier with a named, pack-owned assumption.
    """

    cost_per_case: float = 250.0
    recall_cost_per_unit: float = 900.0
    exposure_multiplier: float = 1.0
    currency: str = "USD"


class BookingPolicy(BaseModel):
    """Pack-owned authorization limits for booking (board #10).

    The booking backend is a labeled stub, but the POLICY is real and
    enforced: at most ``max_per_case`` bookings per case, and any booking
    when ``require_supervisor`` is true (or the cap is hit) returns
    ``requires_approval`` instead of booking — a supervisor approves via
    the normal takeover flow.
    """

    max_per_case: int = 1
    require_supervisor: bool = False


# ── Top-level manifest ──────────────────────────────────────────────────────


class PackManifest(BaseModel):
    """The full Domain Pack manifest.

    Validated against this schema by `src.domains.loader.load_pack()`. After
    loading, the loader computes a `pack_version` hash stamped on every
    interaction, case, and audit report.
    """

    id: str = Field(..., description="Pack id, must match the directory name under domains/")
    display_name: str
    locale: str = Field(
        default="en-US",
        description="BCP-47 locale for STT/TTS and prompt locale (pack-driven, not hardcoded)",
    )
    greeting: str = Field(..., description="Canned greeting (no LLM)")
    goodbye: str = Field(..., description="Canned goodbye (no LLM)")
    refusal_topics: list[str] = Field(
        default_factory=list,
        description="Topics the agent politely refuses to discuss (off-topic guard)",
    )
    entities: EntityLabels
    slot_frame: list[SlotSpec] = Field(..., min_length=1)
    safety: SafetySpec
    advisory_match: AdvisoryMatchSpec
    severity: SeveritySpec
    taxonomy_ref: str = Field(default="taxonomy.yaml")
    # optional: relative path to the per-pack domain DuckDB (overrides default)
    domain_db: str | None = None
    # optional: pack-owned cost figures (financial_impact reads these; item 33)
    cost_model: CostModel = Field(default_factory=CostModel)
    # optional: booking authorization limits (board #10)
    booking_policy: BookingPolicy = Field(default_factory=BookingPolicy)
    # optional: evidence-backed region centroids for hotspot maps
    # (region -> [lon, lat]); absent regions fall back to labeled demo coords
    region_centroids: dict[str, list[float]] = Field(default_factory=dict)

    @field_validator("slot_frame")
    @classmethod
    def _slot_frame_must_have_description(cls, v: list[SlotSpec]) -> list[SlotSpec]:
        names = {s.name for s in v}
        if "description" not in names:
            raise ValueError("slot_frame must include a 'description' slot (free-text)")
        return v

    @field_validator("id")
    @classmethod
    def _id_no_path_separators(cls, v: str) -> str:
        if "/" in v or "\\" in v:
            raise ValueError("pack id must not contain path separators")
        return v
