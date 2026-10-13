"""Phase 1 early warning: recall clock, censored weeks, scan cadence.

Synthetic rows prove the gates. The committed NHTSA sample is the eval gate
in eval.frontline.recall_lead_gate, not this file.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

_SAMPLE_MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "eval" / "frontline" / "fixtures" / "recall_lead" / "manifest.json"
)


def _insert_week(con, pack, prefix, day, n, *, category, entity_2, entity_3="", source="NHTSA"):
    for i in range(n):
        con.execute(
            """
            INSERT INTO records (
                record_id, occurred_at, received_at, entity_2, entity_3,
                category, text, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                f"{prefix}-{day:%Y%m%d}-{i}",
                day,
                day,
                entity_2,
                entity_3,
                category,
                f"{prefix} note {i}",
                source,
            ],
        )


def test_flat_rcl_row_uses_report_received_date():
    from src.domains.nhtsa_components import normalize_component
    from scripts.ingest_recalls import flat_line_to_advisory_row

    fields = [""] * 29
    fields[1] = "15V320"
    fields[2] = "Honda"
    fields[3] = "Accord"
    fields[4] = "2015"
    fields[6] = "AIR BAGS:FRONTAL"
    fields[10] = "V"
    fields[15] = "20150527"
    fields[19] = "Inflator rupture."
    fields[21] = "Replace the inflator."
    row = flat_line_to_advisory_row("\t".join(fields))
    assert row is not None
    assert row["RCDATE"] == "20150527"
    assert row["MAKETXT"] == "HONDA"
    assert row["COMPONENT"] == normalize_component("AIR BAGS:FRONTAL")
    assert row["ADVISORY_KEY"] == (
        f"15V320|2015|HONDA|ACCORD|{row['COMPONENT']}"
    )
    assert row["RCL_URL"].endswith("15V320")
    assert flat_line_to_advisory_row("\t".join([""] * 29)) is None


def test_recall_insert_follows_mapping_contract(tmp_path, monkeypatch):
    from src.data.warehouse import domain_con
    from scripts.ingest_recalls import (
        advisory_contract,
        flat_line_to_advisory_row,
        insert_advisories,
    )

    monkeypatch.setenv("DOMAIN_DB_PATH", str(tmp_path))
    fields = [""] * 29
    fields[1] = "15V320"
    fields[2] = "HONDA"
    fields[3] = "ACCORD"
    fields[4] = "2015"
    fields[6] = "SERVICE BRAKES, HYDRAULIC:FOUNDATION"
    fields[15] = "20150527"
    fields[19] = "Brake hose may leak."
    fields[21] = "Replace the hose."
    row = flat_line_to_advisory_row("\t".join(fields))
    assert row is not None
    contract = advisory_contract("automotive_nhtsa")
    assert contract["issued_at"] == "RCDATE"
    assert contract["scope_category"] == "COMPONENT"
    inserted = insert_advisories("phase1_rcl", [row], contract=contract)
    assert inserted["written"] == 1
    with domain_con("phase1_rcl", read_only=True) as con:
        stored = con.execute(
            """
            SELECT advisory_id, issued_at, scope_entity_2, scope_category, summary, url
            FROM advisories
            """
        ).fetchone()
    assert stored[0] == row["ADVISORY_KEY"]
    assert str(stored[1]).startswith("2015-05-27")
    assert stored[2] == "HONDA"
    assert stored[3] == row["COMPONENT"]
    assert stored[4] == "Brake hose may leak."
    assert "15V320" in stored[5]


def test_weekly_grain_migration_keeps_existing_rows(tmp_path):
    import duckdb

    from src.data.warehouse import apply_domain_schema

    path = tmp_path / "old.duckdb"
    con = duckdb.connect(str(path))
    con.execute(
        """
        CREATE TABLE weekly_anomalies (
            pack_id VARCHAR NOT NULL,
            iso_week VARCHAR NOT NULL,
            category VARCHAR,
            entity_2 VARCHAR,
            record_count INTEGER NOT NULL,
            baseline_mean DOUBLE,
            baseline_std DOUBLE,
            z_score DOUBLE,
            is_anomaly BOOLEAN NOT NULL DEFAULT FALSE,
            PRIMARY KEY (pack_id, iso_week, category, entity_2)
        )
        """
    )
    con.execute(
        """
        INSERT INTO weekly_anomalies (
            pack_id, iso_week, category, entity_2, record_count,
            baseline_mean, baseline_std, z_score, is_anomaly
        ) VALUES ('p', '2018-W01', 'SERVICE BRAKES', 'HONDA', 4, 1, 0.5, 1.5, FALSE)
        """
    )
    apply_domain_schema(con)
    info = {
        name: pk
        for name, pk in con.execute(
            "SELECT name, pk FROM pragma_table_info('weekly_anomalies')"
        ).fetchall()
    }
    assert info["entity_3"] > 0
    assert "censored" in info
    row = con.execute(
        "SELECT z_score, entity_3, COALESCE(censored, FALSE) FROM weekly_anomalies"
    ).fetchone()
    assert row[0] == 1.5
    assert row[1] == ""
    assert row[2] is False or row[2] == 0
    con.close()


def test_source_filter_and_censored_bit(tmp_path, monkeypatch):
    from src.data.warehouse import apply_domain_schema, domain_con
    from src.ml_runtime.anomalies import recompute_weekly_anomalies

    monkeypatch.setenv("DOMAIN_DB_PATH", str(tmp_path))
    pack = "phase1_src"
    start = datetime(2018, 1, 8)
    with domain_con(pack, read_only=False) as con:
        apply_domain_schema(con)
        for w in range(6):
            day = start + timedelta(weeks=w)
            _insert_week(
                con, pack, "n", day, 2,
                category="SERVICE BRAKES", entity_2="HONDA", source="NHTSA",
            )
            _insert_week(
                con, pack, "w", day, 2 if w < 5 else 20,
                category="SERVICE BRAKES", entity_2="HONDA", source="warranty",
            )
    warranty = recompute_weekly_anomalies(pack, source="warranty")
    assert any(r["is_anomaly"] for r in warranty)
    complaints = recompute_weekly_anomalies(pack, source="NHTSA")
    assert not any(r["is_anomaly"] for r in complaints)
    censored = recompute_weekly_anomalies(
        pack, source="warranty", censor_recent_days=4000
    )
    assert any(r.get("censored") for r in censored)
    assert not any(r["is_anomaly"] and r.get("censored") for r in censored)
    with domain_con(pack, read_only=True) as con:
        stored = con.execute(
            "SELECT COUNT(*) FROM weekly_anomalies WHERE censored = TRUE"
        ).fetchone()
    assert stored[0] >= 1


def test_model_grain_keeps_a_flat_model_quiet(tmp_path, monkeypatch):
    from src.data.warehouse import apply_domain_schema, domain_con
    from src.ml_runtime.anomalies import recompute_weekly_anomalies

    monkeypatch.setenv("DOMAIN_DB_PATH", str(tmp_path))
    pack = "phase1_model"
    start = datetime(2018, 1, 8)
    with domain_con(pack, read_only=False) as con:
        apply_domain_schema(con)
        for w in range(6):
            day = start + timedelta(weeks=w)
            _insert_week(
                con, pack, "a", day, 2 if w < 5 else 20,
                category="SERVICE BRAKES", entity_2="HONDA", entity_3="ACCORD",
            )
            _insert_week(
                con, pack, "c", day, 2,
                category="SERVICE BRAKES", entity_2="HONDA", entity_3="CIVIC",
            )
    scored = recompute_weekly_anomalies(pack, model_grain=True)
    accord = [r for r in scored if r.get("entity_3") == "ACCORD" and r["is_anomaly"]]
    civic = [r for r in scored if r.get("entity_3") == "CIVIC" and r["is_anomaly"]]
    assert accord
    assert not civic


def test_fixture_z_gone_and_flat_series_opens_nothing(reset_ops_db, tmp_path, monkeypatch):
    from src.data.warehouse import apply_domain_schema, domain_con, ops_con
    from src.frontline.fleet_scan import run_fleet_scan

    monkeypatch.setenv("DOMAIN_DB_PATH", str(tmp_path))
    monkeypatch.setenv("FRONTLINE_REPORTING_LAG_DAYS", "0")
    pack = "phase1_fleet"
    start = datetime(2018, 1, 8)
    with domain_con(pack, read_only=False) as con:
        apply_domain_schema(con)
        con.execute(
            """
            INSERT INTO weekly_anomalies (
                pack_id, iso_week, category, entity_2, record_count,
                baseline_mean, baseline_std, z_score, is_anomaly
            ) VALUES (?, '2018-W02', 'SERVICE BRAKES', 'HONDA', 4, 1.0, 0.5, 6.0, TRUE)
            """,
            [pack],
        )
        for w in range(6):
            day = start + timedelta(weeks=w)
            _insert_week(
                con, pack, "spike", day, 2 if w < 5 else 20,
                category="SERVICE BRAKES", entity_2="HONDA",
            )
            _insert_week(
                con, pack, "flat", day, 2,
                category="ENGINE", entity_2="HONDA",
            )
        now = start + timedelta(weeks=5)
        con.execute(
            """
            INSERT INTO clusters (
                cluster_id, pack_id, top_terms, category, record_count, first_seen, last_seen
            ) VALUES
            (1000, ?, '["grind"]', 'SERVICE BRAKES', 20, ?, ?),
            (1001, ?, '["tick"]', 'ENGINE', 12, ?, ?)
            """,
            [pack, start, now, pack, start, now],
        )
        for i in range(20):
            con.execute(
                "INSERT INTO cluster_assignments (record_id, cluster_id, distance) VALUES (?, 1000, 0.2)",
                [f"spike-{now:%Y%m%d}-{i}"],
            )
        for w in range(6):
            day = start + timedelta(weeks=w)
            for i in range(2):
                con.execute(
                    "INSERT INTO cluster_assignments (record_id, cluster_id, distance) VALUES (?, 1001, 0.2)",
                    [f"flat-{day:%Y%m%d}-{i}"],
                )
    report = run_fleet_scan(pack, rebuild_clusters=False, alert=False)
    opened = {row["cluster_id"] for row in report["investigations_opened"]}
    assert opened == {1000}
    assert report["anomaly_slices"] >= 1
    with domain_con(pack, read_only=True) as con:
        zs = [float(r[0]) for r in con.execute(
            "SELECT z_score FROM weekly_anomalies WHERE z_score IS NOT NULL"
        ).fetchall()]
    assert all(abs(z - 6.0) > 1e-6 for z in zs)
    with ops_con(read_only=True) as con:
        n = con.execute(
            "SELECT COUNT(*) FROM investigations WHERE pack_id = ?",
            [pack],
        ).fetchone()[0]
    assert n == 1


def test_recall_report_keeps_miss_and_unconfirmed(tmp_path, monkeypatch):
    from src.backtest.recall_lead import format_recall_lead_report, recall_lead_report
    from src.data.warehouse import apply_domain_schema, domain_con
    from scripts.ingest_recalls import advisory_contract, insert_advisories

    monkeypatch.setenv("DOMAIN_DB_PATH", str(tmp_path))
    pack = "phase1_lead"
    start = datetime(2018, 1, 8)
    spike_day = start + timedelta(weeks=5)
    issued = datetime(2018, 3, 12)
    with domain_con(pack, read_only=False) as con:
        apply_domain_schema(con)
        for w in range(6):
            day = start + timedelta(weeks=w)
            _insert_week(
                con, pack, "hit", day, 2 if w < 5 else 20,
                category="SERVICE BRAKES", entity_2="HONDA", entity_3="ACCORD",
            )
            _insert_week(
                con, pack, "lone", day, 2 if w < 5 else 20,
                category="STEERING", entity_2="HONDA", entity_3="ACCORD",
            )
        # Arrived after the miss recall. Must not count as a pre-recall rise.
        _insert_week(
            con, pack, "late", datetime(2018, 6, 4), 20,
            category="ENGINE", entity_2="HONDA", entity_3="ACCORD",
        )
    contract = advisory_contract("automotive_nhtsa")
    hit_id = "18V100|2018|HONDA|ACCORD|SERVICE BRAKES"
    miss_id = "18V200|2018|HONDA|ACCORD|ENGINE"
    insert_advisories(
        pack,
        [
            {
                "ADVISORY_KEY": hit_id,
                "RCDATE": "20180312",
                "YEARTXT": "2018",
                "MAKETXT": "HONDA",
                "MODELTXT": "ACCORD",
                "COMPONENT": "SERVICE BRAKES",
                "DESC_DEFECT": "Hose leak.",
                "CORRECTIVE_ACTION": "Replace.",
                "RCL_URL": "https://www.nhtsa.gov/recalls?nhtsaId=18V100",
            },
            {
                "ADVISORY_KEY": miss_id,
                "RCDATE": "20180115",
                "YEARTXT": "2018",
                "MAKETXT": "HONDA",
                "MODELTXT": "ACCORD",
                "COMPONENT": "ENGINE",
                "DESC_DEFECT": "Stall.",
                "CORRECTIVE_ACTION": "Reflash.",
                "RCL_URL": "https://www.nhtsa.gov/recalls?nhtsaId=18V200",
            },
        ],
        contract=contract,
    )
    report = recall_lead_report(pack, run_backtests=True)
    by_id = {r["advisory_id"]: r for r in report["recalls"]}
    assert by_id[hit_id]["outcome"] == "hit"
    assert by_id[hit_id]["lead_weeks"] > 0
    assert by_id[hit_id]["first_flagged_week"]
    assert by_id[miss_id]["outcome"] == "miss"
    assert report["miss_count"] >= 1
    assert report["unconfirmed_count"] >= 1
    assert any(r["status"] == "unconfirmed" and r["category"] == "STEERING" for r in report["unconfirmed"])
    text = format_recall_lead_report(report)
    assert "miss" in text
    assert "unconfirmed" in text
    assert hit_id in text
    # The late engine pile is after the miss date, so it cannot be the hit.
    assert spike_day < issued


def test_received_clock_ignores_rows_that_arrived_later():
    from src.ml_runtime.anomalies import count_weekly_slices

    recall = datetime(2018, 3, 12)
    rows = [
        {"received_at": datetime(2018, 2, 12), "occurred_at": datetime(2017, 1, 2),
         "category": "ENGINE", "entity_2": "HONDA", "source": "NHTSA"},
        {"received_at": datetime(2018, 6, 1), "occurred_at": datetime(2018, 1, 2),
         "category": "ENGINE", "entity_2": "HONDA", "source": "NHTSA"},
    ]
    counted = count_weekly_slices(rows, clock="received", as_of=recall)
    assert sum(counted.values()) == 1
    event = count_weekly_slices(rows, clock="event", as_of=recall)
    assert sum(event.values()) == 2


def test_worker_enqueues_scheduled_scan_on_cadence(reset_ops_db, monkeypatch):
    from src.data.warehouse import ops_con
    import src.jobs.worker as worker

    monkeypatch.setenv("FRONTLINE_SCAN_INTERVAL_S", "86400")
    worker._last_scan_mono = 0.0
    first = worker.worker_tick()
    assert first["scan"]
    assert {row["job_type"] for row in first["scan"]} == {"scheduled_scan"}
    second = worker.worker_tick()
    assert second["scan"] == []
    with ops_con() as con:
        con.execute("DELETE FROM job_queue WHERE job_type = 'scheduled_scan'")


def test_early_warning_payload_includes_spikes(reset_ops_db, seed_automotive_pack, monkeypatch):
    from fastapi.testclient import TestClient

    from src.api.main import app

    monkeypatch.delenv("FRONTLINE_API_KEY", raising=False)
    monkeypatch.setenv("FRONTLINE_ENABLED", "1")
    with TestClient(app) as client:
        body = client.get("/api/frontline/early-warning").json()
    assert "spikes" in body
    assert "censored_weeks" in body
    assert "spike_week" in body
    for spike in body["spikes"]:
        assert "record_count" in spike
        assert "baseline_mean" in spike
        assert "z_score" in spike
        assert spike.get("category") or spike.get("entity_2")


@pytest.mark.skipif(
    not _SAMPLE_MANIFEST.is_file(),
    reason="real NHTSA sample is written after the flat file is read",
)
def test_committed_sample_gate_passes():
    from eval.frontline.recall_lead_gate import recall_lead_gate

    gate = recall_lead_gate()
    assert gate.passed, gate.detail
