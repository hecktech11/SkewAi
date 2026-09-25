"""Canonical data model — two databases, one contract.

This module defines:
  1. The **ops warehouse** schema (written at runtime): interactions,
     interaction_turns, cases, investigations, agent_actions.
  2. The **domain warehouse** canonical schema (read-only at runtime,
     built per-pack by ingestion): records, advisories, weekly_anomalies,
     clusters, cluster_assignments, backtest_results.

The domain warehouse tables are also exposed as **canonical views** over
existing v1 NHTSA tables (complaints → records, recalls → advisories) so v1
Qubot keeps working unchanged.

Ledger invariant (enforced in src/ledger/): every output shown to the customer
or console — including supervisor turns — MUST have its `agent_actions` row
written **before** it is emitted. No ledger row, no output.
"""

from __future__ import annotations

# ── Ops warehouse schema ────────────────────────────────────────────────────
# Written at runtime by the orchestrator and agents. Lives in
# `data/frontline.duckdb`.

OPS_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS interactions (
    interaction_id     VARCHAR PRIMARY KEY,        -- 'int_' + ULID
    pack_id            VARCHAR NOT NULL,
    pack_version       VARCHAR NOT NULL,           -- stamped at start
    started_at         TIMESTAMP NOT NULL,
    ended_at           TIMESTAMP,
    channel            VARCHAR NOT NULL,           -- web_voice | web_text | simulated
    -- canonical entity slots (pack-labeled)
    entity_1           VARCHAR,
    entity_2           VARCHAR,
    entity_3           VARCHAR,
    category           VARCHAR,
    description        TEXT,
    status             VARCHAR NOT NULL,          -- active | completed | abandoned | escalated
    outcome            VARCHAR,                   -- case_created | advisory_notified | escalated_safety | incomplete | abandoned_with_slots | handed_off | human_resolved
    enrichment_partial BOOLEAN NOT NULL DEFAULT FALSE,
    supervised         BOOLEAN NOT NULL DEFAULT FALSE,  -- a human took over at some point
    peak_frustration   DOUBLE,
    -- denormalized for the console: latest frustration
    last_frustration   DOUBLE,
    peak_frustration_turn INT,
    -- audit
    llm_calls          INTEGER NOT NULL DEFAULT 0,
    customer_ref       VARCHAR,                    -- sha256 of phone/email/account (returning-customer link; never raw PII)
    degraded_ledger    BOOLEAN NOT NULL DEFAULT FALSE,  -- safety output delivered from WAL fallback
    csat               INTEGER,                        -- post-call outcome signal 1-5 (board #9)
    customer_resolved  BOOLEAN,                        -- customer says the issue is resolved (board #9)
    schema_version     INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_interactions_status ON interactions(status, started_at);
CREATE INDEX IF NOT EXISTS idx_interactions_pack ON interactions(pack_id, started_at);
-- Failure post-mortems + escalated filters (enterprise root-cause / copilot)
CREATE INDEX IF NOT EXISTS idx_interactions_status_outcome ON interactions(status, outcome);
-- Supervisor copilot "most frustrated" ORDER BY peak_frustration DESC
CREATE INDEX IF NOT EXISTS idx_interactions_peak_fr ON interactions(peak_frustration);
-- Entity fingerprint lookups / memory-adjacent filters
CREATE INDEX IF NOT EXISTS idx_interactions_entities ON interactions(entity_1, entity_2, entity_3);

CREATE TABLE IF NOT EXISTS interaction_turns (
    turn_id            VARCHAR PRIMARY KEY,
    interaction_id     VARCHAR NOT NULL,             -- FK to interactions (enforced in app code)
    seq                INTEGER NOT NULL,
    speaker            VARCHAR NOT NULL,           -- customer | agent | supervisor
    text               TEXT NOT NULL,
    ts                 TIMESTAMP NOT NULL,
    latency_ms         INTEGER,
    llm_used           BOOLEAN NOT NULL DEFAULT FALSE,
    frustration_score  DOUBLE,                    -- NULL for agent/supervisor turns
    erased             BOOLEAN NOT NULL DEFAULT FALSE,
    UNIQUE(interaction_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_turns_interaction ON interaction_turns(interaction_id, seq);
-- Risk scorer: COUNT(*) WHERE interaction_id=? AND speaker='customer'
CREATE INDEX IF NOT EXISTS idx_turns_ix_speaker ON interaction_turns(interaction_id, speaker);

CREATE TABLE IF NOT EXISTS cases (
    case_id            VARCHAR PRIMARY KEY,        -- 'case_' + ULID
    interaction_id     VARCHAR NOT NULL,             -- FK to interactions (enforced in app code)
    pack_id            VARCHAR NOT NULL,
    created_at         TIMESTAMP NOT NULL,
    category           VARCHAR,
    description_summary TEXT,
    onset              TIMESTAMP,
    severity           VARCHAR NOT NULL,           -- Low | Medium | Critical
    severity_source    VARCHAR NOT NULL,           -- model | rules
    priority           INTEGER NOT NULL,           -- 1-4 (1 = highest)
    safety_flags       JSON,                       -- pack-defined keys, e.g. {fire: true}
    advisory_match_id  VARCHAR,
    cluster_match_id   INTEGER,
    similar_record_count INTEGER DEFAULT 0,
    investigation_id   VARCHAR,                    -- FK to investigations (nullable)
    status             VARCHAR NOT NULL DEFAULT 'open',  -- open | pending_followup | closed
    followup_draft     TEXT,
    case_kind          VARCHAR NOT NULL DEFAULT 'customer',
    -- 'customer' = real contact case (fleet corpus); 'audit_review' = Qubot
    -- mismatch review (excluded from anomaly/cluster/economics/funnel stats)
    customer_ref       VARCHAR,                    -- sha256 identity copied from interactions (returning-customer link)
    schema_version     INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_cases_status ON cases(status, severity);
CREATE INDEX IF NOT EXISTS idx_cases_severity ON cases(severity, created_at);
CREATE INDEX IF NOT EXISTS idx_cases_investigation ON cases(investigation_id);
CREATE INDEX IF NOT EXISTS idx_cases_pack ON cases(pack_id, created_at);
-- Critical FK path: case by interaction (audit, risk, timeline, connector export)
CREATE INDEX IF NOT EXISTS idx_cases_interaction ON cases(interaction_id);
-- live_risk + investigation auto-link COUNT by pack/cluster/window
CREATE INDEX IF NOT EXISTS idx_cases_cluster_pack_created
    ON cases(pack_id, cluster_match_id, created_at);
-- Open/pending queue ordered by recency
CREATE INDEX IF NOT EXISTS idx_cases_status_created ON cases(status, created_at);
-- Category filter / similar-case copilot
CREATE INDEX IF NOT EXISTS idx_cases_category ON cases(category);

-- Operator notes on cases (pilot human-in-the-loop trail).
CREATE TABLE IF NOT EXISTS case_notes (
    note_id            VARCHAR PRIMARY KEY,
    case_id            VARCHAR NOT NULL,
    author             VARCHAR NOT NULL DEFAULT 'operator',
    body               TEXT NOT NULL,
    created_at         TIMESTAMP NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_case_notes_case ON case_notes(case_id, created_at);

CREATE TABLE IF NOT EXISTS investigations (
    investigation_id   VARCHAR PRIMARY KEY,        -- 'inv_' + sequence
    pack_id            VARCHAR NOT NULL,
    cluster_id         INTEGER NOT NULL,
    title              VARCHAR NOT NULL,           -- deterministic: cluster top terms + category
    status             VARCHAR NOT NULL DEFAULT 'open',  -- open | monitoring | closed
    opened_at          TIMESTAMP NOT NULL,
    last_case_at       TIMESTAMP,
    case_count         INTEGER NOT NULL DEFAULT 0,
    assignee           VARCHAR,
    sla_due_at         TIMESTAMP,
    schema_version     INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS investigation_hypotheses (
    hypothesis_id      VARCHAR PRIMARY KEY,
    investigation_id   VARCHAR NOT NULL,
    body               TEXT NOT NULL,
    status             VARCHAR NOT NULL DEFAULT 'open',
    created_at         TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS recorded_fixes (
    fix_id             VARCHAR PRIMARY KEY,
    pack_id            VARCHAR NOT NULL,
    investigation_id   VARCHAR,
    entity_2           VARCHAR,
    entity_3           VARCHAR,
    category           VARCHAR,
    fixed_at           TIMESTAMP NOT NULL,
    note               TEXT
);

CREATE INDEX IF NOT EXISTS idx_investigations_status ON investigations(status, opened_at);
CREATE INDEX IF NOT EXISTS idx_investigations_cluster ON investigations(cluster_id, pack_id);

-- Investigation sequence (used to mint inv_0001, inv_0002, ...)
CREATE SEQUENCE IF NOT EXISTS investigation_seq START 1;

CREATE TABLE IF NOT EXISTS agent_actions (
    action_id          VARCHAR PRIMARY KEY,        -- ULID
    interaction_id     VARCHAR NOT NULL,             -- FK to interactions (enforced in app code)
    case_id            VARCHAR,
    agent              VARCHAR NOT NULL,           -- intake | sentiment | triage | sentinel | investigator | case | orchestrator | supervisor
    action_type        VARCHAR NOT NULL,
    input_summary      VARCHAR(500),
    output_summary     VARCHAR(500),
    evidence_ids       JSON,                       -- list of record/advisory/cluster/investigation ids
    ok                 BOOLEAN NOT NULL DEFAULT TRUE,
    error              VARCHAR,
    duration_ms        INTEGER,
    ts                 TIMESTAMP NOT NULL,
    schema_version     INTEGER NOT NULL DEFAULT 1,
    prev_hash          VARCHAR,                    -- hash chain: previous row_hash (tamper-evident)
    row_hash           VARCHAR,                    -- sha256 of this row + prev_hash
    hash_version       INTEGER NOT NULL DEFAULT 1, -- hash version (1: legacy, 2: with claims & content_hash)
    content_hash       VARCHAR,                    -- sha256 of canonical row content (preserved across erasure)
    claims             VARCHAR,                    -- serialized claims payload
    erased             BOOLEAN NOT NULL DEFAULT FALSE
    -- chain-preserving erasure (7.3): PII content tombstoned, hashes kept
);

CREATE INDEX IF NOT EXISTS idx_actions_interaction ON agent_actions(interaction_id, ts);
CREATE INDEX IF NOT EXISTS idx_actions_agent ON agent_actions(agent, ts);
CREATE INDEX IF NOT EXISTS idx_actions_type ON agent_actions(action_type, ts);
-- Windowed agent_performance / digests filter by ts alone
CREATE INDEX IF NOT EXISTS idx_actions_ts ON agent_actions(ts);
-- Failed-action scans in root-cause (ok=false is selective when rare)
CREATE INDEX IF NOT EXISTS idx_actions_ok_ts ON agent_actions(ok, ts);

-- Alert dedup table: at most one alert per (event, cluster/interaction) per day.
CREATE TABLE IF NOT EXISTS alert_dedup (
    dedup_key          VARCHAR PRIMARY KEY,         -- f"{event}:{cluster_or_interaction}:{date}"
    event              VARCHAR NOT NULL,
    ref_id             VARCHAR NOT NULL,           -- cluster_id or interaction_id
    fired_at           TIMESTAMP NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_alert_dedup_ref ON alert_dedup(event, ref_id);

-- Failed webhook deliveries after exhausted retries (pilot reliability).
CREATE TABLE IF NOT EXISTS alert_dead_letter (
    dead_letter_id     VARCHAR PRIMARY KEY,
    event              VARCHAR NOT NULL,
    ref_id             VARCHAR NOT NULL,
    interaction_id     VARCHAR,
    payload_json       TEXT NOT NULL,
    error              TEXT,
    attempts           INTEGER NOT NULL,
    created_at         TIMESTAMP NOT NULL,
    last_attempt_at    TIMESTAMP NOT NULL,
    status             VARCHAR NOT NULL DEFAULT 'pending'  -- pending | replayed | discarded
);

CREATE INDEX IF NOT EXISTS idx_alert_dl_status ON alert_dead_letter(status, created_at);

-- Weekly crypto-shred drill reports (GDPR erasure honesty).
CREATE TABLE IF NOT EXISTS erasure_drill_reports (
    report_id          VARCHAR PRIMARY KEY,
    ran_at             TIMESTAMP NOT NULL,
    interaction_id     VARCHAR,
    passed             BOOLEAN NOT NULL,
    chain_ok           BOOLEAN,
    payload_unreadable BOOLEAN,
    dek_destroyed      BOOLEAN,
    error              VARCHAR,
    detail_json        VARCHAR
);

-- Outbound connector delivery log (generic HTTP / outbox case-export).
CREATE TABLE IF NOT EXISTS connector_deliveries (
    delivery_id        VARCHAR PRIMARY KEY,
    event              VARCHAR NOT NULL,           -- case_created | investigation_opened | manual_export
    ref_id             VARCHAR NOT NULL,           -- primary id (case_id or investigation_id)
    interaction_id     VARCHAR,
    case_id            VARCHAR,
    investigation_id   VARCHAR,
    payload_json       TEXT NOT NULL,
    sink               VARCHAR NOT NULL,           -- outbox | http | outbox+http
    status             VARCHAR NOT NULL,           -- success | pending | failed | replayed
    attempts           INTEGER NOT NULL DEFAULT 0,
    error              TEXT,
    outbox_path        VARCHAR,
    http_status        INTEGER,
    created_at         TIMESTAMP NOT NULL,
    last_attempt_at    TIMESTAMP NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_conn_dlv_status ON connector_deliveries(status, created_at);
CREATE INDEX IF NOT EXISTS idx_conn_dlv_event ON connector_deliveries(event, ref_id);

-- Cross-contact entity memory (long-term recall by entity fingerprint).
CREATE TABLE IF NOT EXISTS contact_memory (
    memory_id          VARCHAR PRIMARY KEY,
    pack_id            VARCHAR NOT NULL,
    entity_key         VARCHAR NOT NULL,
    entity_1           VARCHAR,
    entity_2           VARCHAR,
    entity_3           VARCHAR,
    last_interaction_id VARCHAR,
    last_case_id       VARCHAR,
    last_severity      VARCHAR,
    last_category      VARCHAR,
    last_outcome       VARCHAR,
    peak_frustration   DOUBLE,
    open_case_count    INTEGER NOT NULL DEFAULT 0,
    interaction_count  INTEGER NOT NULL DEFAULT 1,
    note_summary       TEXT,
    first_seen_at      TIMESTAMP NOT NULL,
    last_seen_at       TIMESTAMP NOT NULL,
    UNIQUE (pack_id, entity_key)
);

CREATE INDEX IF NOT EXISTS idx_memory_pack_key ON contact_memory(pack_id, entity_key);
CREATE INDEX IF NOT EXISTS idx_memory_last_seen ON contact_memory(last_seen_at);

-- Reusable multi-step scenario playbooks (enterprise scenario builder).
CREATE TABLE IF NOT EXISTS scenarios (
    scenario_id        VARCHAR PRIMARY KEY,
    pack_id            VARCHAR NOT NULL,
    name               VARCHAR NOT NULL,
    description        TEXT,
    steps_json         TEXT NOT NULL,
    created_at         TIMESTAMP NOT NULL,
    updated_at         TIMESTAMP NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_scenarios_pack ON scenarios(pack_id, updated_at);

-- Optional risk snapshot history for predictive escalation engine.
CREATE TABLE IF NOT EXISTS risk_snapshots (
    snapshot_id        VARCHAR PRIMARY KEY,
    interaction_id     VARCHAR NOT NULL,
    ts                 TIMESTAMP NOT NULL,
    escalation_prob    DOUBLE NOT NULL,
    churn_prob         DOUBLE NOT NULL,
    sentiment_trend    DOUBLE NOT NULL,
    est_resolution_turns DOUBLE,
    factors_json       TEXT
);

CREATE INDEX IF NOT EXISTS idx_risk_ix ON risk_snapshots(interaction_id, ts);

-- ── Frontline v3 OS foundation: learning / experiments / governance ────────

-- Continuous learning: improvement proposals from failure/audit signals.
CREATE TABLE IF NOT EXISTS learning_proposals (
    proposal_id        VARCHAR PRIMARY KEY,
    created_at         TIMESTAMP NOT NULL,
    updated_at         TIMESTAMP NOT NULL,
    status             VARCHAR NOT NULL DEFAULT 'proposed',
    -- proposed | approved | rejected | deployed
    weakness_class     VARCHAR NOT NULL,
    title              VARCHAR NOT NULL,
    detail             TEXT,
    evidence_ids       TEXT,                       -- JSON list
    suggested_change   TEXT NOT NULL,
    impact_score       DOUBLE NOT NULL DEFAULT 0.5,
    source             VARCHAR NOT NULL DEFAULT 'root_cause',
    interaction_id     VARCHAR,
    reviewed_by        VARCHAR,
    reviewed_at        TIMESTAMP,
    review_note        TEXT
);

CREATE INDEX IF NOT EXISTS idx_learn_status ON learning_proposals(status, created_at);
CREATE INDEX IF NOT EXISTS idx_learn_class ON learning_proposals(weakness_class, created_at);

-- Versioned AI artifacts (prompt/workflow/routing/safety/retrieval policies as data).
CREATE TABLE IF NOT EXISTS ai_artifacts (
    artifact_id        VARCHAR PRIMARY KEY,
    kind               VARCHAR NOT NULL,           -- prompt | workflow | routing | safety_policy | retrieval
    name               VARCHAR NOT NULL,
    version            VARCHAR NOT NULL,
    body_json          TEXT NOT NULL,
    created_at         TIMESTAMP NOT NULL,
    notes              TEXT,
    UNIQUE (kind, name, version)
);

CREATE INDEX IF NOT EXISTS idx_artifacts_kind ON ai_artifacts(kind, name, version);

-- Experiments: control vs candidate with offline comparison metrics.
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id      VARCHAR PRIMARY KEY,
    name               VARCHAR NOT NULL,
    description        TEXT,
    mode               VARCHAR NOT NULL DEFAULT 'ab',  -- ab | shadow | canary
    status             VARCHAR NOT NULL DEFAULT 'draft', -- draft | running | completed | rolled_back
    control_artifact_id VARCHAR,
    candidate_artifact_id VARCHAR,
    created_at         TIMESTAMP NOT NULL,
    completed_at       TIMESTAMP,
    metrics_json       TEXT
);

CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status, created_at);

CREATE TABLE IF NOT EXISTS experiment_trials (
    trial_id           VARCHAR PRIMARY KEY,
    experiment_id      VARCHAR NOT NULL,
    arm                VARCHAR NOT NULL,           -- control | candidate
    interaction_id     VARCHAR,
    resolution_ok      BOOLEAN,
    escalated          BOOLEAN,
    latency_ms         INTEGER,
    groundedness_ok    BOOLEAN,
    hallucination_flag BOOLEAN,
    cost_units         DOUBLE,
    created_at         TIMESTAMP NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_trials_exp ON experiment_trials(experiment_id, arm);

-- Governance: soft-activate configuration deployments (append-only history).
CREATE TABLE IF NOT EXISTS config_deployments (
    deployment_id      VARCHAR PRIMARY KEY,
    label              VARCHAR NOT NULL,
    created_at         TIMESTAMP NOT NULL,
    activated_at       TIMESTAMP,
    deactivated_at     TIMESTAMP,
    status             VARCHAR NOT NULL DEFAULT 'inactive', -- active | inactive | rolled_back
    pack_version       VARCHAR,
    artifact_versions_json TEXT,                   -- map kind/name -> version
    activated_by       VARCHAR,
    note               TEXT
);

CREATE INDEX IF NOT EXISTS idx_deploy_status ON config_deployments(status, created_at);

-- Per-interaction version stamp (which pack/deployment/experiment/policy ran).
CREATE TABLE IF NOT EXISTS interaction_version_stamps (
    interaction_id     VARCHAR PRIMARY KEY,
    stamped_at         TIMESTAMP NOT NULL,
    pack_id            VARCHAR,
    pack_version       VARCHAR,
    deployment_id      VARCHAR,
    experiment_id      VARCHAR,
    artifact_versions_json TEXT,
    model_policy       VARCHAR NOT NULL DEFAULT 'deterministic'
);

CREATE INDEX IF NOT EXISTS idx_stamps_pack ON interaction_version_stamps(pack_id, stamped_at);

CREATE TABLE IF NOT EXISTS handoff_requests (
    handoff_id         VARCHAR PRIMARY KEY,        -- 'hfr_' + ULID
    interaction_id     VARCHAR NOT NULL,
    status             VARCHAR NOT NULL DEFAULT 'pending',
    -- pending -> claimed (supervisor took it) | unfulfilled (SLA timeout) |
    -- done (served) | cancelled (customer hung up)
    prior_state        VARCHAR,
    sla_due_at         TIMESTAMP NOT NULL,
    claimed_by         VARCHAR,
    created_at         TIMESTAMP NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_handoff_interaction ON handoff_requests(interaction_id, status);
CREATE INDEX IF NOT EXISTS idx_handoff_due ON handoff_requests(status, sla_due_at);

CREATE TABLE IF NOT EXISTS callback_requests (
    callback_id        VARCHAR PRIMARY KEY,        -- 'cbk_' + ULID
    interaction_id     VARCHAR NOT NULL,
    partial_slots      JSON,
    reason             VARCHAR NOT NULL,           -- e.g. 'incomplete_slots_max_turns'
    status             VARCHAR NOT NULL DEFAULT 'open',
    created_at         TIMESTAMP NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_callback_status ON callback_requests(status, created_at);

CREATE TABLE IF NOT EXISTS slice_claims (
    slice_key          VARCHAR PRIMARY KEY,
    owner              VARCHAR NOT NULL,
    claimed_at         TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS novel_candidates (
    novel_id           VARCHAR PRIMARY KEY,        -- 'nvl_' + ULID
    interaction_id     VARCHAR NOT NULL,
    pack_id            VARCHAR NOT NULL,
    category           VARCHAR,
    entity_2           VARCHAR,
    entity_3           VARCHAR,
    top_score          DOUBLE,
    status             VARCHAR NOT NULL DEFAULT 'open',
    -- open -> clustered (absorbed by a rebuild) | dismissed (engineer triage)
    created_at         TIMESTAMP NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_novel_status ON novel_candidates(pack_id, status, created_at);

CREATE TABLE IF NOT EXISTS embedding_shadow_comparisons (
    comparison_id VARCHAR PRIMARY KEY,
    interaction_id VARCHAR,
    pack_id VARCHAR,
    query_embedding_version VARCHAR,
    candidate_embedding_version VARCHAR,
    cluster_build_id VARCHAR,
    hash_top_ids VARCHAR,
    semantic_top_ids VARCHAR,
    rank_overlap DOUBLE,
    hash_top_score DOUBLE,
    semantic_top_score DOUBLE,
    hash_cluster_id INTEGER,
    semantic_cluster_id INTEGER,
    hash_novel BOOLEAN,
    semantic_novel BOOLEAN,
    semantic_status VARCHAR,
    latency_ms INTEGER,
    created_at TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS turn_dedup (
    interaction_id     VARCHAR NOT NULL,
    client_turn_id     VARCHAR NOT NULL,
    seen_at            TIMESTAMP NOT NULL,
    PRIMARY KEY (interaction_id, client_turn_id)
);

CREATE TABLE IF NOT EXISTS cluster_feedback (
    feedback_id        VARCHAR PRIMARY KEY,        -- 'cfb_' + ULID
    cluster_id         INTEGER,
    cluster_uid        VARCHAR,
    pack_id            VARCHAR NOT NULL,
    verdict            VARCHAR NOT NULL,           -- 'wrong' | 'novel' | 'correct'
    note               TEXT,
    author             VARCHAR,
    created_at         TIMESTAMP NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_cluster_feedback ON cluster_feedback(pack_id, cluster_id, created_at);

CREATE TABLE IF NOT EXISTS intercept_cooldown (
    slice_key          VARCHAR PRIMARY KEY,        -- pack|category|entity_2
    last_fired_at      TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS review_queue (
    review_id          VARCHAR PRIMARY KEY,        -- 'rvw_' + ULID
    interaction_id     VARCHAR,
    case_id            VARCHAR,
    reason             VARCHAR NOT NULL,           -- e.g. 'audit_mismatch'
    status             VARCHAR NOT NULL DEFAULT 'open',
    -- open -> assigned -> resolved | false_alarm
    owner              VARCHAR,
    sla_due_at         TIMESTAMP,
    verdict            VARCHAR,
    created_at         TIMESTAMP NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_review_status ON review_queue(status, sla_due_at);

CREATE TABLE IF NOT EXISTS canonical_identity (
    canonical_id       VARCHAR PRIMARY KEY,        -- 'cid_' + ULID
    identity_key       VARCHAR NOT NULL UNIQUE,    -- e.g. 'vin:1HGCM82633A004352' or 'serial:...'
    vin                VARCHAR,
    make               VARCHAR,
    model              VARCHAR,
    year               INTEGER,
    identity_status    VARCHAR NOT NULL DEFAULT 'unverified', -- 'unverified', 'candidate_checksum_valid', 'customer_confirmed', 'externally_verified'
    first_observed_at  TIMESTAMP NOT NULL,
    last_verified_at   TIMESTAMP,
    source             VARCHAR NOT NULL,           -- 'nhtsa', 'warranty', 'service', 'voice_intake', etc.
    metadata_json      VARCHAR                     -- arbitrary attributes, warranty date, trim, engine
);

CREATE INDEX IF NOT EXISTS idx_canonical_vin ON canonical_identity(vin);
CREATE INDEX IF NOT EXISTS idx_canonical_make_model ON canonical_identity(make, model, year);

CREATE TABLE IF NOT EXISTS entity_observations (
    observation_id     VARCHAR PRIMARY KEY,        -- 'obs_' + ULID
    interaction_id     VARCHAR NOT NULL,
    canonical_id       VARCHAR,                    -- foreign link when resolved
    raw_spoken_text    VARCHAR,
    extracted_vin      VARCHAR,
    confidence         DOUBLE DEFAULT 1.0,
    vin_status         VARCHAR NOT NULL,           -- 'observed', 'candidate_checksum_valid', 'invalid_checksum', 'confirmed', 'corrected'
    source_channel     VARCHAR NOT NULL,           -- 'voice', 'text', 'batch'
    observed_at        TIMESTAMP NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_obs_interaction ON entity_observations(interaction_id);
CREATE INDEX IF NOT EXISTS idx_obs_canonical ON entity_observations(canonical_id);

CREATE TABLE IF NOT EXISTS trust_roots (
    key_id         TEXT PRIMARY KEY,
    public_key_pem TEXT NOT NULL,
    created_at     TIMESTAMP NOT NULL,
    revoked_at     TIMESTAMP NULL,
    created_by     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS key_mint_audit (
    audit_id           VARCHAR PRIMARY KEY,
    principal          VARCHAR NOT NULL,
    requested_scopes   VARCHAR NOT NULL,
    granted_scopes     VARCHAR NOT NULL,
    tenant_id          VARCHAR NOT NULL,
    ip                 VARCHAR,
    success            BOOLEAN NOT NULL,
    error              VARCHAR,
    created_at         TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS dsr_export_audit (
    audit_id           VARCHAR PRIMARY KEY,
    principal          VARCHAR NOT NULL,
    scope              VARCHAR NOT NULL,
    record_count       INTEGER NOT NULL,
    ip                 VARCHAR,
    created_at         TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS frontline_users (
    user_id            VARCHAR PRIMARY KEY,
    name               VARCHAR NOT NULL,
    email              VARCHAR UNIQUE NOT NULL,
    company            VARCHAR,
    role               VARCHAR NOT NULL,
    password_hash      VARCHAR NOT NULL,
    salt               VARCHAR NOT NULL,
    created_at         TIMESTAMP NOT NULL,
    last_login_at      TIMESTAMP
);
"""


# ── Domain warehouse canonical views ──────────────────────────────────────
# Read-only at runtime. Built per-pack. For automotive_nhtsa these are
# views over the existing v1 NHTSA tables (complaints → records,
# recalls → advisories) so v1 Qubot keeps working unchanged.

DOMAIN_VIEWS_SQL = """
-- Canonical records view (was complaints in v1).
-- Built per-pack; this is the contract the agents query.
CREATE TABLE IF NOT EXISTS records (
    record_id          VARCHAR PRIMARY KEY,
    occurred_at         TIMESTAMP,
    received_at        TIMESTAMP NOT NULL,
    entity_1           VARCHAR,                    -- pack-labeled (year / product / ...)
    entity_2           VARCHAR,
    entity_3           VARCHAR,
    category           VARCHAR,
    subcategory        VARCHAR,
    text               TEXT NOT NULL,
    severity_label     VARCHAR,
    region             VARCHAR,
    source             VARCHAR,                    -- NHTSA | CFPB | internal
    embedding          FLOAT[],                    -- hash-era bag-of-hash vector (blake2b-512-v1); not semantic
    entity_key         VARCHAR,                    -- canonical cross-source join key (declared in mapping.yaml)
    provenance         VARCHAR NOT NULL DEFAULT 'observed'
    -- observed | fixture | computed | inferred | legacy
);

CREATE INDEX IF NOT EXISTS idx_records_received ON records(received_at);
CREATE INDEX IF NOT EXISTS idx_records_category ON records(category);
CREATE INDEX IF NOT EXISTS idx_records_entities ON records(entity_1, entity_2, entity_3);

CREATE TABLE IF NOT EXISTS advisories (
    advisory_id        VARCHAR PRIMARY KEY,
    issued_at          TIMESTAMP NOT NULL,
    scope_entity_1     VARCHAR,
    scope_entity_2     VARCHAR,
    scope_entity_3     VARCHAR,
    scope_category     VARCHAR,
    summary            TEXT NOT NULL,
    remedy             TEXT,
    url                VARCHAR,
    source             VARCHAR                     -- NHTSA | CFPB | internal
);

CREATE INDEX IF NOT EXISTS idx_advisories_scope ON advisories(scope_entity_2, scope_category);
CREATE INDEX IF NOT EXISTS idx_advisories_issued ON advisories(issued_at);
CREATE INDEX IF NOT EXISTS idx_advisories_scope_e1 ON advisories(scope_entity_1, scope_entity_2);

CREATE TABLE IF NOT EXISTS clusters (
    cluster_id         INTEGER PRIMARY KEY,
    pack_id             VARCHAR NOT NULL,
    cluster_uid        VARCHAR,                    -- globally unique identity (item 4: 'clu_'+ULID); legacy integer ids kept for compat
    signature          VARCHAR,                    -- stable content signature (board #5: category|entity|top-terms sha)
    top_terms          JSON,
    category           VARCHAR,
    record_count       INTEGER NOT NULL DEFAULT 0,
    first_seen         TIMESTAMP,
    last_seen          TIMESTAMP
);

CREATE TABLE IF NOT EXISTS cluster_assignments (
    record_id          VARCHAR NOT NULL,
    cluster_id         INTEGER NOT NULL,           -- FK to clusters (enforced in app code)
    distance           DOUBLE,
    PRIMARY KEY (record_id, cluster_id)
);

CREATE INDEX IF NOT EXISTS idx_assignments_cluster ON cluster_assignments(cluster_id);
CREATE INDEX IF NOT EXISTS idx_clusters_pack ON clusters(pack_id, category);

CREATE TABLE IF NOT EXISTS weekly_anomalies (
    pack_id             VARCHAR NOT NULL,
    iso_week            VARCHAR NOT NULL,           -- e.g. '2026-W22'
    category            VARCHAR,
    entity_2            VARCHAR,
    record_count       INTEGER NOT NULL,
    baseline_mean       DOUBLE,
    baseline_std       DOUBLE,
    z_score             DOUBLE,
    is_anomaly          BOOLEAN NOT NULL DEFAULT FALSE,
    p_value             DOUBLE,                     -- one-sided quasi-Poisson tail (item 10)
    p_bh                DOUBLE,                     -- Benjamini-Hochberg adjusted p (item 10)
    baseline_weeks      INTEGER,                    -- leave-one-out history depth
    method              VARCHAR,                    -- quasi-poisson | zero-baseline | low-history-heuristic | insufficient-history
    PRIMARY KEY (pack_id, iso_week, category, entity_2)
);

CREATE INDEX IF NOT EXISTS idx_weekly_anom_pack_week ON weekly_anomalies(pack_id, iso_week);

CREATE TABLE IF NOT EXISTS backtest_results (
    cluster_id         INTEGER NOT NULL,           -- FK to clusters (enforced in app code)
    advisory_id        VARCHAR NOT NULL,           -- FK to advisories (enforced in app code)
    lead_time_weeks    INTEGER,                    -- how many weeks before the advisory the cluster spiked
    matched            BOOLEAN NOT NULL DEFAULT FALSE,
    pack_id            VARCHAR,                    -- owning pack (item 43: scopes cleanup per pack)
    provenance         VARCHAR NOT NULL DEFAULT 'computed',
    -- 'fixture' = seeded demo row, never observed evidence;
    -- 'computed' = written by run_backtest from warehouse state
    match_basis        VARCHAR,                    -- e.g. 'category+entity_2+temporal'
    PRIMARY KEY (cluster_id, advisory_id)
);

CREATE INDEX IF NOT EXISTS idx_backtest_pack_matched ON backtest_results(pack_id, matched);

CREATE TABLE IF NOT EXISTS cluster_lineage (
    old_cluster_uid    VARCHAR NOT NULL,
    new_cluster_uid    VARCHAR NOT NULL,
    pack_id            VARCHAR NOT NULL,
    overlap            DOUBLE NOT NULL,              -- member Jaccard overlap (1.0 = signature-identical)
    correspondence_kind VARCHAR,                     -- same_embedding | cross_embedding
    created_at         TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (old_cluster_uid, new_cluster_uid)
);

CREATE INDEX IF NOT EXISTS idx_lineage_new ON cluster_lineage(new_cluster_uid);

-- Versioned embedding sidecar. records.embedding stays hash-era.
CREATE TABLE IF NOT EXISTS record_embeddings (
    record_id          VARCHAR NOT NULL,
    embedding_version  VARCHAR NOT NULL,
    native_dimension   INTEGER NOT NULL,
    output_dimension   INTEGER NOT NULL,
    artifact_sha256    VARCHAR,
    embedding          FLOAT[],
    status             VARCHAR NOT NULL,
    error_code         VARCHAR,
    error_message      VARCHAR,
    created_at         TIMESTAMP DEFAULT current_timestamp,
    updated_at         TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (record_id, embedding_version)
);
CREATE INDEX IF NOT EXISTS idx_record_embeddings_version
    ON record_embeddings(embedding_version, status);

CREATE TABLE IF NOT EXISTS cluster_builds (
    build_id                  VARCHAR PRIMARY KEY,
    pack_id                   VARCHAR NOT NULL,
    embedding_version         VARCHAR NOT NULL,
    cluster_algorithm_version VARCHAR NOT NULL,
    parameters_json           VARCHAR NOT NULL,
    source_cutoff             TIMESTAMP,
    status                    VARCHAR NOT NULL,
    coverage_json             VARCHAR,
    created_at                TIMESTAMP DEFAULT current_timestamp,
    activated_at              TIMESTAMP
);
CREATE TABLE IF NOT EXISTS cluster_build_clusters (
    build_id           VARCHAR NOT NULL,
    cluster_id         INTEGER NOT NULL,
    cluster_uid        VARCHAR NOT NULL,
    embedding_version  VARCHAR NOT NULL,
    centroid           FLOAT[],
    top_terms          VARCHAR,
    category           VARCHAR,
    record_count       INTEGER NOT NULL,
    first_seen         TIMESTAMP,
    last_seen          TIMESTAMP,
    signature          VARCHAR,
    PRIMARY KEY (build_id, cluster_id)
);
CREATE TABLE IF NOT EXISTS cluster_build_assignments (
    build_id           VARCHAR NOT NULL,
    record_id          VARCHAR NOT NULL,
    cluster_id         INTEGER NOT NULL,
    embedding_version  VARCHAR NOT NULL,
    distance           DOUBLE,
    PRIMARY KEY (build_id, record_id)
);

CREATE TABLE IF NOT EXISTS exposure (
    pack_id            VARCHAR NOT NULL,
    iso_week           VARCHAR NOT NULL,           -- e.g. '2026-W22'
    category           VARCHAR,
    entity_2           VARCHAR,
    exposure_units     DOUBLE NOT NULL,            -- vehicles-in-operation / active accounts / units shipped
    unit               VARCHAR NOT NULL DEFAULT 'units',
    unit_type          VARCHAR NOT NULL DEFAULT 'vehicles_in_operation', -- 'vehicles_in_operation' | 'units_sold' | 'policy_count'
    source             VARCHAR NOT NULL DEFAULT 'ihs_polk',              -- 'ihs_polk' | 'internal_sales' | 'telematics_active'
    updated_at         TIMESTAMP DEFAULT current_timestamp,
    -- Rates beat raw counts: anomaly z-scores divide by exposure when a row
    -- exists, and slices without exposure are labeled 'unnormalised'.
    PRIMARY KEY (pack_id, iso_week, category, entity_2)
);

CREATE INDEX IF NOT EXISTS idx_exposure_pack_week ON exposure(pack_id, iso_week);
"""
