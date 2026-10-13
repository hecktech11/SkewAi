"""Versioned embedding sidecar. Never overwrites records.embedding in place.

DuckDB: FLOAT[] with application-level dimension checks.
PostgreSQL: vector(512) plus embedding_version so indexes cannot mix models.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Iterable, Sequence

from src.data.timeutil import utc_now
from src.data.warehouse import _rollback_quietly, apply_domain_schema, domain_con
from src.ml_runtime.embedding_space import (
    OUTPUT_DIMENSION,
    EmbeddedVector,
    EmbeddingDimensionError,
    as_embedded,
)

RECORD_EMBEDDINGS_DDL = """
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
    model_key          VARCHAR,
    padding_strategy   VARCHAR,
    tokenizer_sha256   VARCHAR,
    truncation_length  INTEGER,
    sequence_length    INTEGER,
    inference_ms       DOUBLE,
    library_version    VARCHAR,
    sanitized          BOOLEAN,
    created_at         TIMESTAMP DEFAULT current_timestamp,
    updated_at         TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (record_id, embedding_version)
)
"""

SHADOW_COMPARISONS_DDL = """
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
)
"""

PG_RECORD_EMBEDDINGS_DDL = """
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
CREATE INDEX IF NOT EXISTS idx_pg_record_embeddings_vec
    ON record_embeddings USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
"""

_VALID_STATUS = frozenset({"complete", "pending", "failed", "quarantined"})


def ensure_record_embeddings(con) -> None:
    con.execute(RECORD_EMBEDDINGS_DDL)
    try:
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_record_embeddings_version "
            "ON record_embeddings(embedding_version, status)"
        )
    except Exception:
        _rollback_quietly(con)
    for ddl in (
        "ALTER TABLE record_embeddings ADD COLUMN model_key VARCHAR",
        "ALTER TABLE record_embeddings ADD COLUMN padding_strategy VARCHAR",
        "ALTER TABLE record_embeddings ADD COLUMN tokenizer_sha256 VARCHAR",
        "ALTER TABLE record_embeddings ADD COLUMN truncation_length INTEGER",
        "ALTER TABLE record_embeddings ADD COLUMN sequence_length INTEGER",
        "ALTER TABLE record_embeddings ADD COLUMN inference_ms DOUBLE",
        "ALTER TABLE record_embeddings ADD COLUMN library_version VARCHAR",
        "ALTER TABLE record_embeddings ADD COLUMN sanitized BOOLEAN",
    ):
        try:
            con.execute(ddl)
        except Exception:
            _rollback_quietly(con)


def ensure_shadow_table(con) -> None:
    con.execute(SHADOW_COMPARISONS_DDL)


def _validate_vector(vec: EmbeddedVector) -> None:
    if vec.output_dimension != OUTPUT_DIMENSION or len(vec.values) != OUTPUT_DIMENSION:
        raise EmbeddingDimensionError(
            f"sidecar requires {OUTPUT_DIMENSION}-d vectors, got {len(vec.values)}"
        )


def upsert_embedding(
    pack_id: str,
    record_id: str,
    vec: EmbeddedVector,
    *,
    artifact_sha256: str = "",
    status: str = "complete",
    error_code: str | None = None,
    error_message: str | None = None,
    model_key: str = "",
    tokenizer_sha256: str = "",
    truncation_length: int | None = None,
    sequence_length: int | None = None,
    inference_ms: float | None = None,
    library_version: str = "",
    sanitized: bool = False,
) -> None:
    if status not in _VALID_STATUS:
        raise ValueError(f"invalid embedding status {status!r}")
    if status == "complete":
        _validate_vector(vec)
        values: list[float] | None = list(vec.values)
        native = vec.native_dimension
        output = vec.output_dimension
        version = vec.embedding_version
    else:
        values = list(vec.values) if vec.values else None
        native = vec.native_dimension
        output = vec.output_dimension
        version = vec.embedding_version
    now = utc_now().replace(tzinfo=None)
    with domain_con(pack_id, read_only=False) as con:
        apply_domain_schema(con)
        ensure_record_embeddings(con)
        con.execute(
            """
            INSERT INTO record_embeddings (
                record_id, embedding_version, native_dimension, output_dimension,
                artifact_sha256, embedding, status, error_code, error_message,
                model_key, padding_strategy, tokenizer_sha256, truncation_length,
                sequence_length, inference_ms, library_version, sanitized,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (record_id, embedding_version) DO UPDATE SET
                native_dimension = excluded.native_dimension,
                output_dimension = excluded.output_dimension,
                artifact_sha256 = excluded.artifact_sha256,
                embedding = excluded.embedding,
                status = excluded.status,
                error_code = excluded.error_code,
                error_message = excluded.error_message,
                model_key = excluded.model_key,
                padding_strategy = excluded.padding_strategy,
                tokenizer_sha256 = excluded.tokenizer_sha256,
                truncation_length = excluded.truncation_length,
                sequence_length = excluded.sequence_length,
                inference_ms = excluded.inference_ms,
                library_version = excluded.library_version,
                sanitized = excluded.sanitized,
                updated_at = excluded.updated_at
            """,
            [
                record_id,
                version,
                native,
                output,
                artifact_sha256 or None,
                values,
                status,
                error_code,
                (error_message or "")[:300] or None,
                model_key or vec.model_key or None,
                vec.padding_strategy or None,
                tokenizer_sha256 or None,
                truncation_length,
                sequence_length,
                inference_ms,
                library_version or None,
                bool(sanitized),
                now,
                now,
            ],
        )


def mark_embedding_status(
    pack_id: str,
    record_id: str,
    embedding_version: str,
    *,
    status: str,
    error_code: str | None = None,
    error_message: str | None = None,
    native_dimension: int = OUTPUT_DIMENSION,
    output_dimension: int = OUTPUT_DIMENSION,
) -> None:
    if status not in _VALID_STATUS:
        raise ValueError(f"invalid embedding status {status!r}")
    now = utc_now().replace(tzinfo=None)
    with domain_con(pack_id, read_only=False) as con:
        apply_domain_schema(con)
        ensure_record_embeddings(con)
        con.execute(
            """
            INSERT INTO record_embeddings (
                record_id, embedding_version, native_dimension, output_dimension,
                embedding, status, error_code, error_message, created_at, updated_at
            ) VALUES (?, ?, ?, ?, NULL, ?, ?, ?, ?, ?)
            ON CONFLICT (record_id, embedding_version) DO UPDATE SET
                status = excluded.status,
                error_code = excluded.error_code,
                error_message = excluded.error_message,
                updated_at = excluded.updated_at
            """,
            [
                record_id,
                embedding_version,
                native_dimension,
                output_dimension,
                status,
                error_code,
                (error_message or "")[:300] or None,
                now,
                now,
            ],
        )


def get_embedding(
    pack_id: str,
    record_id: str,
    embedding_version: str,
) -> EmbeddedVector | None:
    with domain_con(pack_id, read_only=True) as con:
        try:
            row = con.execute(
                """
                SELECT embedding, native_dimension, output_dimension, status,
                       model_key, padding_strategy
                FROM record_embeddings
                WHERE record_id = ? AND embedding_version = ?
                """,
                [record_id, embedding_version],
            ).fetchone()
        except Exception:
            return None
    if not row or row[3] != "complete" or row[0] is None:
        return None
    values = list(row[0])
    return as_embedded(
        values,
        embedding_version,
        native_dimension=int(row[1]),
        output_dimension=int(row[2]),
        model_key=str(row[4] or ""),
        padding_strategy=str(row[5] or ""),
    )


def get_embeddings_for_records(
    pack_id: str,
    record_ids: Sequence[str],
    embedding_version: str,
) -> dict[str, EmbeddedVector]:
    if not record_ids:
        return {}
    found: dict[str, EmbeddedVector] = {}
    with domain_con(pack_id, read_only=True) as con:
        for i in range(0, len(record_ids), 400):
            chunk = list(record_ids[i : i + 400])
            ph = ",".join("?" for _ in chunk)
            try:
                rows = con.execute(
                    f"""
                    SELECT record_id, embedding, native_dimension, output_dimension,
                           model_key, padding_strategy
                    FROM record_embeddings
                    WHERE embedding_version = ? AND status = 'complete'
                      AND record_id IN ({ph})
                    """,
                    [embedding_version, *chunk],
                ).fetchall()
            except Exception:
                return {}
            for rid, emb, native, output, model_key, padding in rows:
                if emb is None:
                    continue
                found[str(rid)] = as_embedded(
                    list(emb),
                    embedding_version,
                    native_dimension=int(native),
                    output_dimension=int(output),
                    model_key=str(model_key or ""),
                    padding_strategy=str(padding or ""),
                )
    return found


def ensure_pack_embedding_schema(pack_id: str) -> None:
    with domain_con(pack_id, read_only=False) as con:
        apply_domain_schema(con)
        ensure_record_embeddings(con)


def list_eligible_record_ids(pack_id: str, *, limit: int | None = None) -> list[str]:
    sql = "SELECT record_id FROM records WHERE text IS NOT NULL AND length(trim(text)) > 0 ORDER BY record_id"
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    with domain_con(pack_id, read_only=True) as con:
        try:
            rows = con.execute(sql).fetchall()
        except Exception:
            return []
    return [str(r[0]) for r in rows]


def completed_ids(pack_id: str, embedding_version: str) -> set[str]:
    with domain_con(pack_id, read_only=True) as con:
        try:
            rows = con.execute(
                """
                SELECT record_id FROM record_embeddings
                WHERE embedding_version = ? AND status = 'complete'
                """,
                [embedding_version],
            ).fetchall()
        except Exception:
            return set()
    return {str(r[0]) for r in rows}


def coverage_for_version(pack_id: str, embedding_version: str) -> dict[str, int]:
    eligible = list_eligible_record_ids(pack_id)
    done = completed_ids(pack_id, embedding_version)
    with domain_con(pack_id, read_only=True) as con:
        try:
            counts = dict(
                con.execute(
                    """
                    SELECT status, COUNT(*) FROM record_embeddings
                    WHERE embedding_version = ? GROUP BY status
                    """,
                    [embedding_version],
                ).fetchall()
            )
        except Exception:
            counts = {}
    return {
        "eligible": len(eligible),
        "complete": int(counts.get("complete") or len(done)),
        "pending": int(counts.get("pending") or 0),
        "failed": int(counts.get("failed") or 0),
        "quarantined": int(counts.get("quarantined") or 0),
    }


def iter_records_for_backfill(
    pack_id: str,
    embedding_version: str,
    *,
    batch_size: int,
    include_retry: bool = True,
) -> Iterable[list[tuple[str, str]]]:
    """Yield (record_id, text) batches that still need this version.

    Does not load the full corpus into memory.
    """
    done = completed_ids(pack_id, embedding_version)
    skip_status = {"complete"}
    if not include_retry:
        skip_status |= {"quarantined"}
    already: set[str] = set(done)
    if not include_retry:
        with domain_con(pack_id, read_only=True) as con:
            try:
                for (rid,) in con.execute(
                    """
                    SELECT record_id FROM record_embeddings
                    WHERE embedding_version = ? AND status = 'quarantined'
                    """,
                    [embedding_version],
                ).fetchall():
                    already.add(str(rid))
            except Exception:
                pass
    offset = 0
    page = max(50, int(batch_size) * 4)
    while True:
        with domain_con(pack_id, read_only=True) as con:
            try:
                rows = con.execute(
                """
                SELECT record_id, text FROM records
                WHERE text IS NOT NULL
                ORDER BY record_id
                LIMIT ? OFFSET ?
                """,
                    [page, offset],
                ).fetchall()
            except Exception:
                break
        if not rows:
            break
        offset += len(rows)
        batch: list[tuple[str, str]] = []
        for rid, text in rows:
            if str(rid) in already:
                continue
            t = str(text or "").strip()
            batch.append((str(rid), t))
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
        if len(rows) < page:
            break


def pg_semantic_search_sql() -> str:
    """Postgres nearest-neighbour MUST filter version before ranking."""
    return """
        SELECT record_id, 1 - (embedding <=> %s::vector) AS sim
        FROM record_embeddings
        WHERE embedding_version = %s
          AND status = 'complete'
          AND embedding IS NOT NULL
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """


__all__ = [
    "RECORD_EMBEDDINGS_DDL",
    "SHADOW_COMPARISONS_DDL",
    "PG_RECORD_EMBEDDINGS_DDL",
    "ensure_record_embeddings",
    "ensure_pack_embedding_schema",
    "ensure_shadow_table",
    "upsert_embedding",
    "mark_embedding_status",
    "get_embedding",
    "get_embeddings_for_records",
    "list_eligible_record_ids",
    "completed_ids",
    "coverage_for_version",
    "iter_records_for_backfill",
    "pg_semantic_search_sql",
]
