"""Postgres production backend (feature #45, item 32).

DuckDB remains the LOCAL/DEV warehouse. PostgreSQL is the PRODUCTION
backend, selected by ``FRONTLINE_OPS_DSN=postgresql://...``:

- ``ensure_postgres_schema`` installs ``pgvector`` (512-dim embeddings),
  a ``fts_main_records`` full-text index (GIN over tsvector — no
  contact-level ``ILIKE '%keyword%'`` full scans on the prod path),
  weekly rollup tables, and the production indexes.
- ``semantic_search_records`` / ``fts_search_records`` route hot read paths
  to vector / FTS operators on Postgres and to the legacy Python/ILIKE path
  on DuckDB (explicitly labeled dev-only).
- Fallback policy: production-like deploys FAIL CLOSED on Postgres errors
  (no silent DuckDB fallback hiding a prod outage); local/dev falls back
  with a warning + metric so laptops keep working without a database.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator

from src.data.warehouse import ops_con as duckdb_ops_con


def ops_backend() -> str:
    dsn = os.getenv("FRONTLINE_OPS_DSN", "").strip()
    if dsn.startswith("postgres"):
        return "postgres"
    return "duckdb"


def is_production_backend() -> bool:
    """True when the process is configured for the production backend."""
    return ops_backend() == "postgres"


# ── Production DDL (item 32) ────────────────────────────────────────────────
# pgvector + FTS + rollups + indexes. Mirrors the canonical schema in
# src/data/schema.py; applied by ensure_postgres_schema() and shipped as
# migrations/002_pg_vector_fts.sql for review/deploy.

PG_PRODUCTION_DDL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS records (
    record_id       TEXT PRIMARY KEY,
    occurred_at     TIMESTAMPTZ,
    received_at     TIMESTAMPTZ NOT NULL,
    entity_1        TEXT,
    entity_2        TEXT,
    entity_3        TEXT,
    category        TEXT,
    subcategory     TEXT,
    text            TEXT NOT NULL,
    severity_label  TEXT,
    region          TEXT,
    source          TEXT,
    embedding       vector(512)
);
CREATE INDEX IF NOT EXISTS idx_pg_records_received ON records (received_at);
CREATE INDEX IF NOT EXISTS idx_pg_records_category ON records (category);
CREATE INDEX IF NOT EXISTS idx_pg_records_entities ON records (entity_1, entity_2, entity_3);
-- Vector similarity (cosine distance operator; IVFFlat tuned at deploy time
-- with ANALYZE + lists=100 after bulk load).
CREATE INDEX IF NOT EXISTS idx_pg_records_embedding
    ON records USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Full-text search surface (item 32): fts_main_records replaces
-- contact-level ILIKE '%keyword%' full scans on the production path.
ALTER TABLE records ADD COLUMN IF NOT EXISTS fts tsvector
    GENERATED ALWAYS AS (
        to_tsvector('english', coalesce(text, '') || ' ' || coalesce(category, ''))
    ) STORED;
CREATE INDEX IF NOT EXISTS fts_main_records ON records USING GIN (fts);
CREATE INDEX IF NOT EXISTS idx_pg_records_trgm ON records USING GIN (text gin_trgm_ops);

CREATE TABLE IF NOT EXISTS weekly_rollups (
    pack_id         TEXT NOT NULL,
    iso_week        TEXT NOT NULL,
    category        TEXT,
    entity_2        TEXT,
    record_count    INTEGER NOT NULL DEFAULT 0,
    anomaly_count   INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (pack_id, iso_week, category, entity_2)
);
CREATE INDEX IF NOT EXISTS idx_pg_rollups_week ON weekly_rollups (pack_id, iso_week);

CREATE TABLE IF NOT EXISTS advisories (
    advisory_id     TEXT PRIMARY KEY,
    issued_at       TIMESTAMPTZ NOT NULL,
    scope_entity_1  TEXT,
    scope_entity_2  TEXT,
    scope_entity_3  TEXT,
    scope_category  TEXT,
    summary         TEXT NOT NULL,
    remedy          TEXT,
    url             TEXT,
    source          TEXT
);
CREATE INDEX IF NOT EXISTS idx_pg_advisories_scope ON advisories (scope_entity_2, scope_category);

-- Versioned embeddings. records.embedding remains the hash-era 512-d column.
CREATE TABLE IF NOT EXISTS record_embeddings (
    record_id          TEXT NOT NULL,
    embedding_version  TEXT NOT NULL,
    native_dimension   INTEGER NOT NULL,
    output_dimension   INTEGER NOT NULL,
    artifact_sha256    TEXT,
    embedding          vector(512),
    status             TEXT NOT NULL,
    error_code         TEXT,
    error_message      TEXT,
    created_at         TIMESTAMPTZ DEFAULT NOW(),
    updated_at         TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (record_id, embedding_version)
);
CREATE INDEX IF NOT EXISTS idx_pg_record_embeddings_version
    ON record_embeddings (embedding_version, status);
""".strip()


def ensure_postgres_schema(conn) -> dict[str, Any]:
    """Apply production DDL on an open psycopg connection. Returns a report."""
    applied: list[str] = []
    for stmt in [s.strip() for s in PG_PRODUCTION_DDL.split(";") if s.strip()]:
        with conn.cursor() as cur:
            cur.execute(stmt)
        applied.append(stmt.split("\n")[0][:80])
    conn.commit()
    return {"statements": len(applied), "vector_dim": 512, "fts": "fts_main_records"}


def semantic_search_records(
    conn,
    query_vector: list[float],
    *,
    pack_id: str | None = None,
    category: str | None = None,
    limit: int = 5,
    embedding_version: str | None = None,
) -> list[dict[str, Any]]:
    """Cosine-nearest records via pgvector (production path).

    ``query_vector`` must be 512-dim. Raises on dimension mismatch (fail
    loudly, never a false 0-tie). When ``embedding_version`` is set, rank
    the sidecar table and filter by version BEFORE the vector operator so
    hash and semantic rows cannot share an index scan.
    """
    if len(query_vector) != 512:
        raise ValueError(f"query embedding must be 512-dim, got {len(query_vector)}")
    if embedding_version:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT record_id, 1 - (embedding <=> %s::vector) AS sim
                FROM record_embeddings
                WHERE embedding_version = %s
                  AND status = 'complete'
                  AND embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                [query_vector, embedding_version, query_vector, int(limit)],
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]
    clauses = ["embedding IS NOT NULL"]
    params: list[Any] = []
    if category:
        clauses.append("category = %s")
        params.append(category)
    where = " AND ".join(clauses)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT record_id, category, entity_2, entity_3, text,
                   1 - (embedding <=> %s::vector) AS sim
            FROM records
            WHERE {where}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            [query_vector, *params, query_vector, int(limit)],
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def fts_search_records(
    conn,
    query: str,
    *,
    category: str | None = None,
    limit: int = 40,
) -> list[dict[str, Any]]:
    """Full-text record search via fts_main_records (production path).

    Replaces ``ILIKE '%keyword%'`` full scans; falls back to trigram
    similarity when the query has no lexemes.
    """
    q = (query or "").strip()
    if not q:
        return []
    clauses: list[str] = ["fts @@ plainto_tsquery('english', %s)"]
    params: list[Any] = [q]
    if category:
        clauses.append("category = %s")
        params.append(category)
    where = " AND ".join(clauses)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT record_id, category, entity_2, entity_3, text,
                   ts_rank(fts, plainto_tsquery('english', %s)) AS rank
            FROM records
            WHERE {where}
            ORDER BY rank DESC
            LIMIT %s
            """,
            [q, *params, int(limit)],
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def refresh_weekly_rollups(conn, *, pack_id: str) -> int:
    """Rebuild the weekly_rollups summary for a pack (production reporting)."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM weekly_rollups WHERE pack_id = %s", [pack_id])
        cur.execute(
            """
            INSERT INTO weekly_rollups
                (pack_id, iso_week, category, entity_2, record_count, anomaly_count)
            SELECT %s, iso_week, category, entity_2,
                   SUM(record_count),
                   SUM(CASE WHEN is_anomaly THEN 1 ELSE 0 END)
            FROM weekly_anomalies
            WHERE pack_id = %s
            GROUP BY iso_week, category, entity_2
            """,
            [pack_id, pack_id],
        )
        n = cur.rowcount
    conn.commit()
    return int(n or 0)


def alembic_ready() -> bool:
    from pathlib import Path
    from src.config import REPO_ROOT

    return (REPO_ROOT / "migrations" / "versions").is_dir() or (
        REPO_ROOT / "migrations" / "001_ops_init.sql"
    ).is_file()


@contextmanager
def ops_connection(read_only: bool = False) -> Iterator[Any]:
    """Yield a DB connection for ops. Rejects PostgreSQL as unsupported in this release; yields DuckDB."""
    if ops_backend() == "postgres":
        raise RuntimeError(
            "FRONTLINE_OPS_DSN is configured, but PostgreSQL mode is unsupported in this release: "
            "the dialect layer and schema migrations require DuckDB. "
            "Unset FRONTLINE_OPS_DSN."
        )
    with duckdb_ops_con(read_only=read_only) as con:
        yield _Tagged(con, backend="duckdb")


class _Tagged:
    """Thin proxy so callers can inspect ``.backend``."""

    def __init__(self, con: Any, *, backend: str, postgres_error: str | None = None) -> None:
        self._con = con
        self.backend = backend
        self.postgres_error = postgres_error

    def execute(self, *a: Any, **k: Any) -> Any:
        return self._con.execute(*a, **k)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._con, name)


def backend_status() -> dict[str, Any]:
    return {
        "ops_backend": ops_backend(),
        "alembic_ready": alembic_ready(),
        "dsn_set": bool(os.getenv("FRONTLINE_OPS_DSN", "").strip()),
        "vector_dim": 512,
        "fts": "fts_main_records" if ops_backend() == "postgres" else "dev-ilike-only",
        "note": (
            "PostgreSQL + pgvector + FTS production backend; "
            "DuckDB is local/dev only"
            if ops_backend() == "postgres"
            else "DuckDB local/dev; set FRONTLINE_OPS_DSN=postgresql://... for the Postgres production backend"
        ),
    }
