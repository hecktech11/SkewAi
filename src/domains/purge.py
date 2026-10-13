"""Delete a source's records and everything keyed to them.

Re-ingesting a corpus under a new ``record_id`` scheme is not idempotent:
``upsert_records`` is ``INSERT OR REPLACE`` on ``record_id``, so rows written
under the old key are never touched and the corpus silently doubles. Every
count, rate and baseline in the product is then wrong, and nothing reports an
error. No delete path for ``records`` existed anywhere in the repo before
this module, which made the Phase 0 record-id change unsafe to ship.

Two warehouses are involved. ``records`` and its per-record side tables live
in the pack's domain DB; trust and supply attribution live in the ops DB.
Both are cleaned, and orphans are resolved against what actually remains in
``records`` rather than against the set just deleted — that also sweeps up
rows stranded by an earlier partial load.
"""

from __future__ import annotations

from typing import Any

from src.data.warehouse import domain_con, ops_con

#: Domain-DB tables with a ``record_id`` that must not outlive its record.
_DOMAIN_CHILD_TABLES = (
    "cluster_assignments",
    "record_embeddings",
    "cluster_build_assignments",
)

#: Ops-DB tables with a ``record_id`` that must not outlive its record.
#: Both are created lazily by their own modules, so absence is normal.
_OPS_CHILD_TABLES = (
    "record_trust",
    "record_supply",
)


def _table_exists(con, name: str) -> bool:
    row = con.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_name = ?",
        [name],
    ).fetchone()
    return row is not None


def purge_source(
    pack_id: str,
    source: str,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Delete every ``records`` row for ``source`` and all rows keyed to them.

    ``source`` is matched case-insensitively, because ``source_ingest`` upper-cases
    the tag it writes while ``mapping.yaml`` literals do not.

    With ``dry_run=True`` nothing is written and the counts describe what a
    real run would remove. Returns per-table counts plus ``embedding_runs``,
    the number of backfill checkpoints invalidated (their ``last_record_id``
    no longer exists, so resuming one would skip an arbitrary prefix).
    """
    tag = str(source or "").strip()
    if not tag:
        raise ValueError("purge_source requires a non-empty source")

    report: dict[str, Any] = {
        "pack_id": pack_id,
        "source": tag,
        "dry_run": dry_run,
        "records": 0,
    }

    # read_only on a dry run, so a coding slip cannot write.
    with domain_con(pack_id, read_only=dry_run) as con:
        report["records"] = con.execute(
            "SELECT COUNT(*) FROM records WHERE upper(source) = upper(?)", [tag]
        ).fetchone()[0]

        for table in _DOMAIN_CHILD_TABLES:
            if not _table_exists(con, table):
                report[table] = 0
                continue
            # Count against the records that are going away, not against the
            # whole table: a shared cluster also holds rows for other sources.
            report[table] = con.execute(
                f"""
                SELECT COUNT(*) FROM {table} t
                WHERE EXISTS (
                    SELECT 1 FROM records r
                    WHERE r.record_id = t.record_id AND upper(r.source) = upper(?)
                )
                """,
                [tag],
            ).fetchone()[0]

        if not dry_run:
            # Children first: once the parent row is gone there is nothing
            # left to identify the child by.
            for table in _DOMAIN_CHILD_TABLES:
                if not _table_exists(con, table):
                    continue
                con.execute(
                    f"""
                    DELETE FROM {table} t
                    WHERE EXISTS (
                        SELECT 1 FROM records r
                        WHERE r.record_id = t.record_id AND upper(r.source) = upper(?)
                    )
                    """,
                    [tag],
                )
            con.execute("DELETE FROM records WHERE upper(source) = upper(?)", [tag])

            # Sweep anything already orphaned, from this purge or an earlier
            # partial load. Cheap, and it keeps the invariant simple:
            # no side-table row without a record.
            for table in _DOMAIN_CHILD_TABLES:
                if not _table_exists(con, table):
                    continue
                con.execute(
                    f"""
                    DELETE FROM {table} t
                    WHERE NOT EXISTS (
                        SELECT 1 FROM records r WHERE r.record_id = t.record_id
                    )
                    """
                )

    # The ops DB cannot join to the domain DB, so it is filtered on pack_id.
    # record_trust and record_supply are per-pack, so this is exact for a
    # single-source pack and over-broad only where one pack mixes sources —
    # accepted, because a stale trust badge for a deleted record is worse
    # than a missing one (it is what the UI shows as provenance).
    with ops_con() as con:
        for table in _OPS_CHILD_TABLES:
            if not _table_exists(con, table):
                report[table] = 0
                continue
            report[table] = con.execute(
                f"SELECT COUNT(*) FROM {table} WHERE pack_id = ?", [pack_id]
            ).fetchone()[0]
            if not dry_run:
                con.execute(f"DELETE FROM {table} WHERE pack_id = ?", [pack_id])

        if _table_exists(con, "embedding_backfill_runs"):
            report["embedding_runs"] = con.execute(
                "SELECT COUNT(*) FROM embedding_backfill_runs WHERE pack_id = ?",
                [pack_id],
            ).fetchone()[0]
            if not dry_run:
                con.execute(
                    "DELETE FROM embedding_backfill_runs WHERE pack_id = ?", [pack_id]
                )
        else:
            report["embedding_runs"] = 0

    return report


def count_source(pack_id: str, source: str) -> int:
    """Rows in ``records`` for ``source`` (case-insensitive)."""
    with domain_con(pack_id) as con:
        return con.execute(
            "SELECT COUNT(*) FROM records WHERE upper(source) = upper(?)",
            [str(source or "").strip()],
        ).fetchone()[0]


__all__ = ["purge_source", "count_source"]
