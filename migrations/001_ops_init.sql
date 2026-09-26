-- Alembic-ready initial ops schema (feature #45).
-- target: ops
-- Applied manually or via future Alembic env; DuckDB already has equivalent DDL.

CREATE TABLE IF NOT EXISTS interactions (
    interaction_id TEXT PRIMARY KEY,
    pack_id TEXT,
    pack_version TEXT,
    channel TEXT,
    status TEXT,
    outcome TEXT,
    entity_1 TEXT,
    entity_2 TEXT,
    entity_3 TEXT,
    category TEXT,
    description TEXT,
    enrichment_partial BOOLEAN DEFAULT FALSE,
    supervised BOOLEAN DEFAULT FALSE,
    peak_frustration DOUBLE PRECISION,
    last_frustration DOUBLE PRECISION,
    peak_frustration_turn INTEGER,
    llm_calls INTEGER DEFAULT 0,
    customer_ref TEXT,
    degraded_ledger BOOLEAN DEFAULT FALSE,
    csat INTEGER,
    customer_resolved BOOLEAN,
    erased BOOLEAN DEFAULT FALSE,
    started_at TIMESTAMPTZ,
    ended_at TIMESTAMPTZ,
    schema_version INTEGER DEFAULT 1,
    tenant_id TEXT DEFAULT 'default'
);

CREATE TABLE IF NOT EXISTS cases (
    case_id TEXT PRIMARY KEY,
    interaction_id TEXT,
    pack_id TEXT,
    severity TEXT,
    status TEXT,
    category TEXT,
    cluster_id INTEGER,
    description_summary TEXT,
    followup_draft TEXT,
    case_kind TEXT DEFAULT 'customer',
    customer_ref TEXT,
    similar_record_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ,
    schema_version INTEGER DEFAULT 1,
    tenant_id TEXT DEFAULT 'default'
);

CREATE TABLE IF NOT EXISTS agent_actions (
    action_id TEXT PRIMARY KEY,
    interaction_id TEXT,
    agent TEXT,
    action_type TEXT,
    input_summary TEXT,
    output_summary TEXT,
    created_at TIMESTAMPTZ,
    prev_hash TEXT,
    row_hash TEXT,
    erased BOOLEAN DEFAULT FALSE,
    hash_version INTEGER DEFAULT 1,
    content_hash TEXT,
    claims TEXT,
    tenant_id TEXT DEFAULT 'default'
);

CREATE INDEX IF NOT EXISTS idx_cases_tenant ON cases (tenant_id);
CREATE INDEX IF NOT EXISTS idx_interactions_tenant ON interactions (tenant_id);
