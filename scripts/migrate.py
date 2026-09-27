"""Ordered, repeatable SQL migrations (item 48).

Runtime code applies ``CREATE TABLE IF NOT EXISTS`` idempotently; THIS runner
is the deployment-ordering record for production (especially Postgres, where
the app never runs DDL):

    python -m scripts.migrate --db ops            # ops warehouse
    python -m scripts.migrate --db domain --pack automotive_nhtsa
    python -m scripts.migrate --db postgres       # FRONTLINE_OPS_DSN

Each ``migrations/NNN_*.sql`` file runs ONCE per database, tracked in
``schema_migrations``. Re-runs are no-ops. Files must be append-only and
idempotent (``IF NOT EXISTS`` / guarded ALTERs) so a half-applied deploy can
safely retry.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from src.config import REPO_ROOT

MIGRATIONS_DIR = REPO_ROOT / "migrations"
_VERSION_RE = re.compile(r"^(\d{3})_.+\.sql$")


def available_migrations(*, target: str) -> list[tuple[str, Path]]:
    """Migrations scoped to *target* (ops|domain|postgres).

    Scope comes from the file's ``-- target:`` header (comma-separated);
    files without a header apply everywhere.
    """
    out: list[tuple[str, Path]] = []
    for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        m = _VERSION_RE.match(path.name)
        if not m:
            continue
        header = path.read_text(encoding="utf-8").splitlines()[:5]
        tags: set[str] | None = None
        for line in header:
            mm = re.match(r"\s*--\s*target\s*:\s*(.+)", line)
            if mm:
                tags = {t.strip() for t in mm.group(1).split(",")}
                break
        if tags is not None and target not in tags and "all" not in tags:
            continue
        out.append((m.group(1), path))
    return sorted(out)


def _ensure_log_duckdb(con) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version VARCHAR PRIMARY KEY,
            name VARCHAR NOT NULL,
            applied_at TIMESTAMP DEFAULT current_timestamp
        )
        """
    )


def migration_head(*, target: str = "ops") -> str:
    """Latest available migration version for *target* ('' when none)."""
    migs = available_migrations(target=target)
    return migs[-1][0] if migs else ""


def stamp_schema_current(db_path: str | Path, *, target: str = "ops") -> list[str]:
    """Record all available versions as applied (for builders that write
    current-schema warehouses directly, e.g. fixture seeds)."""
    import duckdb

    path = Path(db_path)
    if not path.is_file():
        return []
    stamped: list[str] = []
    con = duckdb.connect(str(path))
    try:
        _ensure_log_duckdb(con)
        for version, name in available_migrations(target=target):
            try:
                con.execute(
                    "INSERT INTO schema_migrations (version, name) VALUES (?, ?)",
                    [version, name.name],
                )
                stamped.append(version)
            except Exception:
                pass
    finally:
        con.close()
    return stamped


def require_schema_current(db_path: str | Path, *, target: str = "ops") -> None:
    """Refuse ingest work when the warehouse is behind migration HEAD.

    Audit 0.5: ingest scripts must not write into an auto-created schema
    that migrations later alter. Raises RuntimeError naming the missing
    versions; operators run ``python -m scripts.migrate`` first.
    """
    import duckdb

    path = Path(db_path)
    if not path.is_file():
        return  # fresh file: schema DDL applies on open, then migrate
    want = {v for v, _ in available_migrations(target=target)}
    if not want:
        return
    con = duckdb.connect(str(path), read_only=True)
    try:
        try:
            have = {
                str(r[0])
                for r in con.execute(
                    "SELECT version FROM schema_migrations"
                ).fetchall()
            }
        except Exception:
            have = set()
    finally:
        con.close()
    missing = sorted(want - have)
    if missing:
        raise RuntimeError(
            f"warehouse {path} is behind migration HEAD "
            f"(missing {missing}); run python -m scripts.migrate first"
        )


def _applied_duckdb(con) -> set[str]:
    _ensure_log_duckdb(con)
    try:
        return {str(r[0]) for r in con.execute("SELECT version FROM schema_migrations").fetchall()}
    except Exception:
        return set()


def _apply_duckdb(con, version: str, path: Path) -> None:
    sql = path.read_text(encoding="utf-8")
    # Strip line comments so statement splitting is safe.
    cleaned = "\n".join(
        line.split("--", 1)[0] for line in sql.splitlines()
    )
    for stmt in cleaned.split(";"):
        s = stmt.strip()
        if s:
            con.execute(s)
    con.execute(
        "INSERT INTO schema_migrations (version, name) VALUES (?, ?)",
        [version, path.name],
    )


def migrate_duckdb(db_path: Path, *, target: str = "ops") -> list[str]:
    """Apply pending migrations to a DuckDB file. Returns applied versions."""
    import duckdb

    applied: list[str] = []
    con = duckdb.connect(str(db_path))
    try:
        done = _applied_duckdb(con)
        for version, path in available_migrations(target=target):
            if version in done:
                continue
            _apply_duckdb(con, version, path)
            applied.append(version)
    finally:
        con.close()
    return applied


def validate_postgres_dsn_security(dsn: str) -> None:
    """Ensure remote PostgreSQL connections use verified TLS (verify-full or verify-ca).

    Loopback addresses (localhost, 127.0.0.1, ::1) and local Unix domain sockets
    are permitted without TLS for local testing.
    """
    if not dsn:
        raise ValueError("PostgreSQL DSN is empty.")

    host: str | None = None
    sslmode: str | None = None

    if dsn.startswith("postgres://") or dsn.startswith("postgresql://"):
        from urllib.parse import parse_qs, urlparse

        parsed = urlparse(dsn)
        host = parsed.hostname
        query = parse_qs(parsed.query)
        if "sslmode" in query:
            sslmode = query["sslmode"][0].lower()
    else:
        # Key-value format e.g. "host=... sslmode=..."
        for k, v1, v2 in re.findall(r'(\w+)=(?:"([^"]*)"|(\S+))', dsn):
            val = v1 or v2
            if k == "host":
                host = val
            elif k == "sslmode":
                sslmode = val.lower()

    # Determine if host is local loopback or local socket
    is_local = False
    if not host or host.startswith("/"):
        # Local Unix domain socket
        is_local = True
    elif host in ("localhost", "127.0.0.1", "::1", "127.0.0.1:5432") or host.startswith("127."):
        is_local = True

    if not is_local:
        if sslmode not in ("verify-full", "verify-ca"):
            raise ValueError(
                f"Remote PostgreSQL connection requires verified TLS (sslmode=verify-full or sslmode=verify-ca). "
                f"Got sslmode={sslmode!r} for host {host!r}."
            )


def migrate_postgres(dsn: str) -> list[str]:
    """Apply pending migrations to Postgres via psycopg. Returns applied."""
    validate_postgres_dsn_security(dsn)
    import psycopg

    applied: list[str] = []
    conn = psycopg.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    applied_at TIMESTAMPTZ DEFAULT NOW()
                )
                """
            )
            cur.execute("SELECT version FROM schema_migrations")
            done = {str(r[0]) for r in cur.fetchall()}
        for version, path in available_migrations(target="postgres"):
            if version in done:
                continue
            sql = path.read_text(encoding="utf-8")
            with conn.cursor() as cur:
                cur.execute(sql)
                cur.execute(
                    "INSERT INTO schema_migrations (version, name) VALUES (%s, %s)",
                    [version, path.name],
                )
            conn.commit()
            applied.append(version)
    finally:
        conn.close()
    return applied


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply ordered SQL migrations")
    parser.add_argument("--db", choices=["ops", "domain", "postgres"], default="ops")
    parser.add_argument("--pack", default="automotive_nhtsa")
    args = parser.parse_args(argv)
    if args.db == "postgres":
        import os

        dsn = (os.getenv("FRONTLINE_OPS_DSN") or "").strip()
        if not dsn.startswith("postgres"):
            print("FRONTLINE_OPS_DSN is not a postgres DSN; nothing to do.")
            return 2
        applied = migrate_postgres(dsn)
    elif args.db == "domain":
        from src.config import settings

        applied = migrate_duckdb(settings.domain_db_path(args.pack), target="domain")
    else:
        from src.config import settings

        applied = migrate_duckdb(settings.frontline_db_path, target="ops")
    if applied:
        print(f"applied migrations: {', '.join(applied)}")
    else:
        print("already up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
