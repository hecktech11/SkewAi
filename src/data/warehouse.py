"""Warehouse — manages the two DuckDB databases.

  1. **Ops warehouse** (`data/frontline.duckdb`) — written at runtime by the
     orchestrator and agents. Schema from `OPS_SCHEMA_SQL`.
  2. **Domain warehouse** (per-pack, read-only at runtime) — built by
     ingestion. Schema from `DOMAIN_VIEWS_SQL`.

The ops warehouse is created lazily on first connection. The domain warehouse
is opened read-only after the pack loader has resolved its path.
"""

from __future__ import annotations

import asyncio
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, TypeVar

T = TypeVar("T")

import duckdb

from src.config import settings
from src.data.schema import DOMAIN_VIEWS_SQL, OPS_SCHEMA_SQL

# ── Connection caching ──────────────────────────────────────────────────────

# RLock (not Lock): cross-thread exclusion is identical, but same-thread
# nesting degrades to serialized contention instead of wedging the warehouse
# forever (audit 4.5 incident). Phased connections are still preferred.
_ops_lock = threading.RLock()
_ops_initialized = False


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _close_thread_ops_ro() -> None:
    """No-op retained for callers that still invoke the old RO-cache helper."""
    return


@contextmanager
def ops_con(read_only: bool = False) -> Iterator[duckdb.DuckDBPyConnection]:
    """Yield a connection to the ops warehouse. Applies schema on first open.

    Single-file DuckDB cannot mix concurrent read_only and read-write handles
    (and ``asyncio.to_thread`` worker threads used to leave TLS RO handles open,
    which broke writers). All access is serialized under ``_ops_lock`` with a
    short-lived write-capable connection. ``read_only`` is accepted for API
    compatibility but does not open a separate RO configuration.

    When FRONTLINE_OPS_DSN is configured for PostgreSQL (audit 6.1), delegates
    directly to the multi-writer Postgres pool to eliminate lock contention.
    """
    try:
        from src.data.postgres_backend import is_production_backend, ops_connection
        if is_production_backend():
            with ops_connection(read_only=read_only) as pcon:
                yield pcon
            return
    except Exception:
        pass

    global _ops_initialized
    path = settings.frontline_db_path
    _ensure_parent(path)

    with _ops_lock:
        if not _ops_initialized:
            con0 = duckdb.connect(str(path))
            try:
                apply_ops_schema(con0)
            finally:
                # Never leak the handle: a failed apply with an open
                # connection wedges every later connect on the file lock.
                con0.close()
            _ops_initialized = True

        con = None
        for attempt in range(5):
            try:
                con = duckdb.connect(str(path), read_only=False)
                break
            except Exception as exc:
                if "Could not set lock on file" in str(exc) and attempt < 4:
                    time.sleep(0.04 * (2 ** attempt))
                    continue
                raise
        try:
            yield con
        finally:
            if con is not None:
                con.close()


async def ops_in_thread(fn: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
    """Run a sync function that uses ``ops_con`` off the asyncio event loop.

    DuckDB I/O still serializes on ``_ops_lock``; this just keeps the lock
    (and the query) from stalling live WebSocket frames.
    """
    if kwargs:
        return await asyncio.to_thread(lambda: fn(*args, **kwargs))
    return await asyncio.to_thread(fn, *args)


@contextmanager
def domain_con(pack_id: str, read_only: bool = True) -> Iterator[duckdb.DuckDBPyConnection]:
    """Yield a read-only connection to a pack's domain warehouse.

    If the file doesn't exist and read_only=True, raises FileNotFoundError —
    callers should build it first via ``scripts.seed_domains`` /
    ``make frontline-db``, or load a source CSV with
    ``src.domains.mapping_ingest.ingest_mapped_csv`` /
    ``python -m scripts.ingest_nhtsa``.
    """
    from src.security.identifiers import safe_pack_id

    pack_id = str(safe_pack_id(pack_id))
    path = settings.domain_db_path(pack_id)
    if not path.exists():
        # Try the pack's own domain_db override
        from src.domains.loader import load_pack
        try:
            pack = load_pack(pack_id)
            path = pack.domain_db_path()
        except FileNotFoundError:
            pass
    if not path.exists():
        if read_only:
            raise FileNotFoundError(f"Domain warehouse not found: {path} (build it first)")
        _ensure_parent(path)
    con = duckdb.connect(str(path), read_only=read_only)
    try:
        if not read_only:
            # Apply the canonical schema (CREATE IF NOT EXISTS).
            apply_domain_schema(con)
        yield con
    finally:
        con.close()
        if not read_only:
            try:
                from scripts.migrate import stamp_schema_current

                stamp_schema_current(path, target="domain")
            except Exception:
                pass


# ── Init helpers (for tests + Makefile) ─────────────────────────────────────


def init_ops_db() -> None:
    """Force-create the ops warehouse with schema applied."""
    global _ops_initialized
    path = settings.frontline_db_path
    _ensure_parent(path)
    with _ops_lock:
        con = duckdb.connect(str(path))
        try:
            apply_ops_schema(con)
        finally:
            con.close()
        _ops_initialized = True
    try:
        from scripts.migrate import stamp_schema_current

        stamp_schema_current(path, target="ops")
    except Exception:
        pass


def reset_ops_db() -> None:
    """Drop and recreate the ops warehouse (used by eval + tests).

    Refuses to wipe the default pilot path ``data/frontline.duckdb`` unless
    ``FRONTLINE_ALLOW_DEFAULT_DB_RESET=1`` is set (intentional ``make frontline-db``).
    Tests/CI must set ``FRONTLINE_DB_PATH`` to an isolated temp file.
    """
    global _ops_initialized
    from src.config import allow_default_db_reset, is_default_pilot_ops_path

    path = settings.frontline_db_path
    if is_default_pilot_ops_path(path) and not allow_default_db_reset():
        raise RuntimeError(
            f"refusing to reset pilot ops DB at {path}; "
            "set FRONTLINE_DB_PATH to an isolated path for tests, or set "
            "FRONTLINE_ALLOW_DEFAULT_DB_RESET=1 for intentional local rebuilds "
            "(make frontline-db)."
        )
    _close_thread_ops_ro()
    with _ops_lock:
        if path.exists():
            path.unlink()
        # Also remove WAL
        for suffix in (".wal", ".tmp"):
            extra = path.with_suffix(path.suffix + suffix)
            if extra.exists():
                extra.unlink()
        _ops_initialized = False
    init_ops_db()


def ensure_domain_db_reset_allowed(path: Path, *, force: bool = False) -> None:
    """Raise if *path* is a pilot domain DB and reset is not explicitly allowed.

    Safe no-op when the file does not exist, when *force* is True, when
    ``FRONTLINE_ALLOW_DEFAULT_DB_RESET`` / ``FRONTLINE_ALLOW_DOMAIN_DB_RESET``
    is set, or when *path* is outside the default pilot ``data/domains/`` tree
    (isolated test/CI warehouses).
    """
    from src.config import allow_domain_db_reset, is_default_pilot_domain_path

    if not path.exists():
        return
    if force or allow_domain_db_reset():
        return
    if not is_default_pilot_domain_path(path):
        return
    raise RuntimeError(
        f"refusing to delete pilot domain DB at {path}; "
        "pass force=True / --force, or set FRONTLINE_ALLOW_DEFAULT_DB_RESET=1 "
        "(or FRONTLINE_ALLOW_DOMAIN_DB_RESET=1) for intentional rebuilds "
        "(make frontline-db / make seed-domains)."
    )


def unlink_domain_db(path: Path, *, force: bool = False) -> None:
    """Delete a domain warehouse file (and WAL sidecars) only when allowed."""
    ensure_domain_db_reset_allowed(path, force=force)
    if path.exists():
        path.unlink()
    for suffix in (".wal", ".tmp"):
        extra = path.with_suffix(path.suffix + suffix)
        if extra.exists():
            extra.unlink()


def init_domain_db(pack_id: str) -> None:
    """Force-create a pack's domain warehouse with the canonical schema."""
    with domain_con(pack_id, read_only=False) as con:
        pass  # schema applied in the context manager


def _strip_sql_comments(sql: str) -> str:
    """Strip SQL line comments (-- ...) so they don't break statement splitting."""
    out_lines = []
    for line in sql.splitlines():
        # Remove everything after `--` (but only if it's not inside a string literal —
        # our schema has no such cases).
        idx = line.find("--")
        if idx >= 0:
            line = line[:idx]
        out_lines.append(line)
    return "\n".join(out_lines)


def apply_domain_schema(con) -> None:
    for stmt in _strip_sql_comments(DOMAIN_VIEWS_SQL).strip().split(";"):
        s = stmt.strip()
        if s:
            con.execute(s)
    # Forward-compatible columns for domain DBs built before items 4/43
    # (backtest pack scoping + provenance; cluster history table below).
    # Fixed DDL constants (not user input) — no identifier allowlist needed.
    for ddl in (
        "ALTER TABLE backtest_results ADD COLUMN pack_id VARCHAR",
        "ALTER TABLE backtest_results ADD COLUMN provenance VARCHAR",
        "ALTER TABLE backtest_results ADD COLUMN match_basis VARCHAR",
        "ALTER TABLE clusters ADD COLUMN cluster_uid VARCHAR",
        "ALTER TABLE clusters ADD COLUMN signature VARCHAR",
        "ALTER TABLE weekly_anomalies ADD COLUMN p_value DOUBLE",
        "ALTER TABLE weekly_anomalies ADD COLUMN p_bh DOUBLE",
        "ALTER TABLE weekly_anomalies ADD COLUMN baseline_weeks INTEGER",
        "ALTER TABLE weekly_anomalies ADD COLUMN method VARCHAR",
        "ALTER TABLE records ADD COLUMN entity_key VARCHAR",
        "ALTER TABLE records ADD COLUMN provenance VARCHAR",
        "ALTER TABLE records ADD COLUMN embedding FLOAT[]",
        "ALTER TABLE cluster_lineage ADD COLUMN correspondence_kind VARCHAR",
    ):
        try:
            con.execute(ddl)
        except Exception:
            pass
    # Legacy rows predate provenance tracking: label them honestly so they
    # can never masquerade as observed/computed evidence (item 2).
    try:
        con.execute(
            "UPDATE backtest_results SET provenance = 'legacy' "
            "WHERE provenance IS NULL"
        )
    except Exception:
        pass
    try:
        con.execute(
            "UPDATE records SET provenance = 'legacy' WHERE provenance IS NULL"
        )
    except Exception:
        pass
    for _idx_ddl in (
        "CREATE INDEX IF NOT EXISTS idx_records_entity_key ON records(entity_key)",
        "CREATE INDEX IF NOT EXISTS idx_records_source ON records(source)",
    ):
        try:
            con.execute(_idx_ddl)
        except Exception:
            pass
    try:
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_backtest_pack_matched "
            "ON backtest_results(pack_id, matched)"
        )
    except Exception:
        pass
    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS cluster_versions (
                cluster_uid      VARCHAR PRIMARY KEY,
                cluster_id       INTEGER,
                pack_id          VARCHAR NOT NULL,
                version          INTEGER NOT NULL DEFAULT 1,
                supersedes_uid   VARCHAR,
                member_count     INTEGER NOT NULL DEFAULT 0,
                top_terms        VARCHAR,
                category         VARCHAR,
                created_at       TIMESTAMP DEFAULT current_timestamp
            )
            """
        )
    except Exception:
        pass
    try:
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_cluster_versions_pack "
            "ON cluster_versions(pack_id, cluster_id)"
        )
    except Exception:
        pass
    try:
        from src.ml_runtime.embedding_store import ensure_record_embeddings
        from src.ml_runtime.cluster_builds import ensure_cluster_build_tables

        ensure_record_embeddings(con)
        ensure_cluster_build_tables(con)
    except Exception:
        pass


def apply_ops_schema(con) -> None:
    for stmt in _strip_sql_comments(OPS_SCHEMA_SQL).strip().split(";"):
        s = stmt.strip()
        if s:
            con.execute(s)
    # Forward-compatible columns (older DuckDB files created before hash-chain).
    from src.security.sql_ident import SAFE_ALTER_COLUMNS, safe_ident, safe_table

    for col, typ in (
        ("prev_hash", "VARCHAR"),
        ("row_hash", "VARCHAR"),
    ):
        try:
            c = safe_ident(col, SAFE_ALTER_COLUMNS, kind="column")
            t = safe_table("agent_actions")
            # typ is a fixed constant from this loop, not user input
            con.execute(f"ALTER TABLE {t} ADD COLUMN {c} {typ}")
        except Exception:
            pass
    for col, typ in (
        ("assignee", "VARCHAR"),
        ("sla_due_at", "TIMESTAMP"),
    ):
        try:
            c = safe_ident(col, SAFE_ALTER_COLUMNS, kind="column")
            t = safe_table("investigations")
            con.execute(f"ALTER TABLE {t} ADD COLUMN {c} {typ}")
        except Exception:
            pass
    for table, cols in (
        ("cases", (("case_kind", "VARCHAR"), ("customer_ref", "VARCHAR"))),
        (
            "interactions",
            (("customer_ref", "VARCHAR"), ("degraded_ledger", "BOOLEAN"),
             ("csat", "INTEGER"), ("customer_resolved", "BOOLEAN"),
             ("enrichment_partial", "BOOLEAN")),
        ),
        (
            "agent_actions",
            (
                ("erased", "BOOLEAN"),
                ("hash_version", "INTEGER DEFAULT 1"),
                ("content_hash", "VARCHAR"),
                ("claims", "VARCHAR"),
            ),
        ),
        ("interaction_turns", (("erased", "BOOLEAN"),)),
    ):
        for col, typ in cols:
            try:
                c = safe_ident(col, SAFE_ALTER_COLUMNS, kind="column")
                t = safe_table(table)
                con.execute(f"ALTER TABLE {t} ADD COLUMN {c} {typ}")
            except Exception:
                pass
    # Version-dependent indexes AFTER the ALTERs above: creating them in the
    # static DDL would hard-fail apply on pre-column databases (and a failed
    # apply must never wedge the warehouse).
    for _idx_ddl in (
        "CREATE INDEX IF NOT EXISTS idx_cases_customer"
        " ON cases(customer_ref, category, created_at)",
        "CREATE INDEX IF NOT EXISTS idx_cases_kind ON cases(case_kind, status)",
    ):
        try:
            con.execute(_idx_ddl)
        except Exception:
            pass
    # LLM daily spend ledger (Phase 1 narration cap).
    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_spend (
                day_key            VARCHAR PRIMARY KEY,  -- YYYY-MM-DD UTC
                turn_count         INTEGER NOT NULL DEFAULT 0,
                cost_units         DOUBLE NOT NULL DEFAULT 0.0,
                updated_at         TIMESTAMP NOT NULL
            )
            """
        )
    except Exception:
        pass
    # Refresh planner stats so new indexes are chosen after bulk loads / upgrades.
    from src.security.sql_ident import safe_table

    for table in (
        "interactions",
        "interaction_turns",
        "cases",
        "agent_actions",
        "investigations",
        "contact_memory",
        "connector_deliveries",
        "alert_dead_letter",
        "llm_spend",
    ):
        try:
            con.execute(f"ANALYZE {safe_table(table)}")
        except Exception:
            pass
