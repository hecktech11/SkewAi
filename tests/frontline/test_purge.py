"""Purging a source must leave no row behind and no orphan keyed to it.

Re-ingesting under a new record_id scheme is not idempotent — upsert_records
is INSERT OR REPLACE on record_id, so old-key rows survive a reload and the
corpus silently doubles. These tests pin the delete path that makes the
Phase 0 record-id change safe.
"""

from __future__ import annotations

import pytest

from src.data.warehouse import domain_con, ops_con
from src.domains.purge import count_source, purge_source

PACK = "automotive_nhtsa"


@pytest.fixture
def seeded(tmp_path, monkeypatch):
    """Two sources of records plus a side-table row for each."""
    monkeypatch.setenv("DOMAIN_DB_PATH", str(tmp_path))
    with domain_con(PACK, read_only=False) as con:
        for rid, source in (
            ("NHTSA-1", "NHTSA"),
            ("NHTSA-2", "NHTSA"),
            ("FL-1", "FRONTLINE"),
        ):
            con.execute(
                """
                INSERT OR REPLACE INTO records
                    (record_id, received_at, category, text, source, entity_key)
                VALUES (?, TIMESTAMP '2026-01-01 00:00:00', 'SERVICE BRAKES', 'x', ?, 'k')
                """,
                [rid, source],
            )
            con.execute(
                "INSERT OR REPLACE INTO cluster_assignments (record_id, cluster_id) VALUES (?, 1)",
                [rid],
            )
    return tmp_path


def _ids(table: str) -> set[str]:
    with domain_con(PACK) as con:
        return {r[0] for r in con.execute(f"SELECT record_id FROM {table}").fetchall()}


def test_purge_removes_only_the_named_source(seeded):
    report = purge_source(PACK, "NHTSA")
    assert report["records"] == 2
    assert _ids("records") == {"FL-1"}


def test_purge_removes_the_side_table_rows_too(seeded):
    """An assignment outliving its record double-counts every cluster."""
    purge_source(PACK, "NHTSA")
    assert _ids("cluster_assignments") == {"FL-1"}


def test_purge_is_case_insensitive_on_source(seeded):
    """source_ingest upper-cases its tag; mapping.yaml literals do not."""
    assert purge_source(PACK, "nhtsa")["records"] == 2
    assert count_source(PACK, "NHTSA") == 0


def test_dry_run_reports_without_deleting(seeded):
    report = purge_source(PACK, "NHTSA", dry_run=True)
    assert report["records"] == 2
    assert report["dry_run"] is True
    assert count_source(PACK, "NHTSA") == 2
    assert _ids("cluster_assignments") == {"NHTSA-1", "NHTSA-2", "FL-1"}


def test_purge_sweeps_pre_existing_orphans(seeded):
    """A partial earlier load can leave an assignment with no record."""
    with domain_con(PACK, read_only=False) as con:
        con.execute(
            "INSERT OR REPLACE INTO cluster_assignments (record_id, cluster_id) VALUES ('GHOST', 1)"
        )
    purge_source(PACK, "NHTSA")
    assert "GHOST" not in _ids("cluster_assignments")


def test_purge_of_an_absent_source_is_a_no_op(seeded):
    report = purge_source(PACK, "DOES-NOT-EXIST")
    assert report["records"] == 0
    assert count_source(PACK, "NHTSA") == 2


def test_purge_requires_a_source(seeded):
    """Guards against a blank argument silently emptying the warehouse."""
    with pytest.raises(ValueError):
        purge_source(PACK, "")
    with pytest.raises(ValueError):
        purge_source(PACK, "   ")


def test_purge_is_idempotent(seeded):
    assert purge_source(PACK, "NHTSA")["records"] == 2
    assert purge_source(PACK, "NHTSA")["records"] == 0


def test_purge_clears_trust_and_backfill_checkpoints(seeded, reset_ops_db):
    """A checkpoint's last_record_id would skip an arbitrary prefix after a re-key."""
    with ops_con() as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS embedding_backfill_runs (
                run_id VARCHAR PRIMARY KEY, pack_id VARCHAR,
                embedding_version VARCHAR, last_record_id VARCHAR,
                total_records INTEGER, completed INTEGER, in_progress INTEGER,
                failed_permanent INTEGER, failed_retriable INTEGER,
                quarantined INTEGER, checkpoint_ts TIMESTAMP, status VARCHAR
            )
            """
        )
        con.execute(
            "INSERT INTO embedding_backfill_runs (run_id, pack_id, last_record_id) "
            "VALUES ('r1', ?, 'NHTSA-1')",
            [PACK],
        )
    report = purge_source(PACK, "NHTSA")
    assert report["embedding_runs"] == 1
    with ops_con() as con:
        remaining = con.execute(
            "SELECT COUNT(*) FROM embedding_backfill_runs WHERE pack_id = ?", [PACK]
        ).fetchone()[0]
    assert remaining == 0


def test_report_names_every_table_it_touched(seeded):
    report = purge_source(PACK, "NHTSA", dry_run=True)
    for key in ("records", "cluster_assignments", "record_embeddings", "record_trust"):
        assert key in report, f"{key} missing from the purge report"
