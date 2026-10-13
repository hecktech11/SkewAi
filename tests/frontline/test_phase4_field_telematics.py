"""Phase 4: field returns and telematics, scored apart from complaints.

Demo files stay fixture. The partner proof re-ingests the same mappings
as observed into a temp pack, then runs the fix check on that cohort.
"""

from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path

from src.config import REPO_ROOT


_MAP = REPO_ROOT / "domains" / "automotive_nhtsa" / "data"


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def test_fixture_demo_does_not_move_anomaly_counts(tmp_path, monkeypatch):
    from src.data.warehouse import apply_domain_schema, domain_con
    from src.domains.source_ingest import load_demo_private_sources
    from src.ml_runtime.anomalies import recompute_weekly_anomalies

    monkeypatch.setenv("DOMAIN_DB_PATH", str(tmp_path))
    pack = "phase4_fixture"
    start = datetime(2018, 1, 8)
    with domain_con(pack, read_only=False) as con:
        apply_domain_schema(con)
        for w in range(6):
            day = start + timedelta(weeks=w)
            for i in range(2):
                con.execute(
                    """
                    INSERT INTO records (
                        record_id, occurred_at, received_at, entity_2, entity_3,
                        category, text, source, provenance
                    ) VALUES (?, ?, ?, 'HONDA', 'ACCORD', 'SERVICE BRAKES', ?, 'NHTSA', 'observed')
                    """,
                    [f"obs-{day:%Y%m%d}-{i}", day, day, f"observed note {i}"],
                )
    loaded = load_demo_private_sources(pack)
    assert loaded["provenance"] == "fixture"
    assert loaded["field"]["upserted"] == 1
    assert loaded["telematics"]["upserted"] == 2
    with domain_con(pack, read_only=True) as con:
        counts = con.execute(
            """
            SELECT event_count FROM records
            WHERE source = 'TELEMATICS'
            ORDER BY event_count DESC
            """
        ).fetchall()
        prov = con.execute(
            """
            SELECT DISTINCT provenance FROM records
            WHERE source IN ('FIELD', 'TELEMATICS')
            """
        ).fetchall()
    assert [int(r[0]) for r in counts] == [3, 1]
    assert {r[0] for r in prov} == {"fixture"}
    scored = recompute_weekly_anomalies(pack)
    honda = [
        r for r in scored
        if r["category"] == "SERVICE BRAKES" and r["entity_2"] == "HONDA"
    ]
    assert sum(int(r["record_count"]) for r in honda) == 12
    assert not any(r["iso_week"] == "2018-W23" and int(r["record_count"]) > 0 for r in honda)


def test_partner_cohort_lists_sources_and_concludes(reset_ops_db, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from src.api.main import app
    from src.data.warehouse import apply_domain_schema, domain_con, ops_con
    from src.domains.source_ingest import ingest_source
    from src.frontline.live_intercept import open_or_link_investigation

    monkeypatch.setenv("DOMAIN_DB_PATH", str(tmp_path))
    monkeypatch.delenv("FRONTLINE_API_KEY", raising=False)
    monkeypatch.setenv("FRONTLINE_ENABLED", "1")
    pack = "phase4_partner"
    pivot = datetime(2018, 3, 5)
    field_rows = []
    telematics_rows = []
    service_rows = []
    for week in range(1, 9):
        before = pivot - timedelta(weeks=week)
        after = pivot + timedelta(weeks=week)
        for i in range(4):
            field_rows.append([
                f"FR-B-{week}-{i}", before.date().isoformat(), before.date().isoformat(),
                "VIN-PARTNER" if week == 1 and i == 0 else f"VIN-B-{week}-{i}",
                "HONDA", "ACCORD", "SERVICE BRAKES", "HOSE", "Hose seep on return", "seep",
            ])
        for i in range(2):
            field_rows.append([
                f"FR-C-{week}-{i}", before.date().isoformat(), before.date().isoformat(),
                f"VIN-C-{week}-{i}", "FORD", "F-150", "SERVICE BRAKES", "HOSE",
                "Control return", "seep",
            ])
        field_rows.append([
            f"FR-A-{week}", after.date().isoformat(), after.date().isoformat(),
            f"VIN-A-{week}", "HONDA", "ACCORD", "SERVICE BRAKES", "HOSE",
            "Quieter return", "seep",
        ])
        for i in range(2):
            field_rows.append([
                f"FR-D-{week}-{i}", after.date().isoformat(), after.date().isoformat(),
                f"VIN-D-{week}-{i}", "FORD", "F-150", "SERVICE BRAKES", "HOSE",
                "Control still flat", "seep",
            ])
    spike_day = (pivot - timedelta(weeks=1)).date().isoformat()
    next_day = (pivot - timedelta(weeks=1) + timedelta(days=1)).date().isoformat()
    for i in range(3):
        telematics_rows.append([
            f"TE-{i}", spike_day, "COHORT-A", "HONDA", "ACCORD", "SERVICE BRAKES", "P0501",
        ])
    telematics_rows.append([
        "TE-3", next_day, "COHORT-A", "HONDA", "ACCORD", "SERVICE BRAKES", "P0501",
    ])
    service_rows.append([
        "RO-1", spike_day, "2018", "HONDA", "ACCORD", "SERVICE BRAKES", "B123",
        "Tech confirmed hose seep", "CA",
    ])
    _write_csv(
        tmp_path / "field.csv",
        ["RETURN_ID", "RETURN_DATE", "INSPECTION_DATE", "IDENTITY", "MAKETXT",
         "MODELTXT", "COMPONENT", "PART", "TECH_NOTE", "SYMPTOM"],
        field_rows,
    )
    _write_csv(
        tmp_path / "telematics.csv",
        ["EVENT_ID", "EVENT_DATE", "COHORT", "MAKETXT", "MODELTXT", "COMPONENT", "FAULT_CODE"],
        telematics_rows,
    )
    _write_csv(
        tmp_path / "service.csv",
        ["RO_NUMBER", "VISIT_DATE", "MODEL_YR", "MAKE", "MODEL", "SYSTEM",
         "OP_CODE", "TECH_NOTES", "STATE"],
        service_rows,
    )
    with domain_con(pack, read_only=False) as con:
        apply_domain_schema(con)
        con.execute(
            """
            INSERT INTO records (
                record_id, occurred_at, received_at, entity_2, entity_3,
                category, text, source, entity_key, provenance
            ) VALUES (?, ?, ?, 'HONDA', 'ACCORD', 'SERVICE BRAKES', ?, 'NHTSA', ?, 'observed')
            """,
            [
                "nhtsa-1",
                pivot - timedelta(weeks=2),
                pivot - timedelta(weeks=2),
                "Complaint hose seep",
                "VIN-PARTNER|HONDA|ACCORD",
            ],
        )
    field = ingest_source(
        pack, "field", tmp_path / "field.csv",
        mapping_path=_MAP / "mapping.field.yaml", provenance="observed",
    )
    telematics = ingest_source(
        pack, "telematics", tmp_path / "telematics.csv",
        mapping_path=_MAP / "mapping.telematics.yaml", provenance="observed",
    )
    service = ingest_source(
        pack, "service", tmp_path / "service.csv",
        mapping_path=_MAP / "mapping.service.yaml", provenance="observed",
    )
    assert field["provenance"] == "observed"
    assert field["upserted"] > 0
    assert telematics["upserted"] == 2
    assert service["upserted"] == 1
    with domain_con(pack, read_only=True) as con:
        counts = [
            int(r[0]) for r in con.execute(
                "SELECT event_count FROM records WHERE source = 'TELEMATICS' ORDER BY event_count DESC"
            ).fetchall()
        ]
    assert counts == [3, 1]
    inv_id = open_or_link_investigation(
        pack_id=pack,
        cluster_id=0,
        title="Partner hose",
        category="SERVICE BRAKES",
        entity_2="HONDA",
        entity_3="ACCORD",
    )
    with TestClient(app) as client:
        brief = client.get(f"/api/frontline/investigations/{inv_id}/brief")
        assert brief.status_code == 200, brief.text
        by_source = brief.json()["records_by_source"]
        for source in ("NHTSA", "SERVICE", "FIELD", "TELEMATICS"):
            assert by_source[source], source
        closed = client.patch(
            f"/api/frontline/investigations/{inv_id}",
            json={
                "status": "closed",
                "fixed_at": "2018-03-05",
                "note": "Partner hose campaign",
                "control_entity_2": "FORD",
            },
        )
    assert closed.status_code == 200, closed.text
    measured = closed.json()["fix_effectiveness"]["measured"]
    assert measured["status"] == "improved"
    assert measured["event_weeks_after"] >= 8
    assert measured["window_days"] == 56
    assert measured["before_count"] > measured["after_count"]
    assert measured["diff_in_diff"] is not None
    assert measured["diff_in_diff"] < 0
    with ops_con(read_only=True) as con:
        stored = con.execute(
            "SELECT control_entity_2, fixed_at FROM recorded_fixes WHERE investigation_id = ?",
            [inv_id],
        ).fetchone()
    assert stored[0] == "FORD"
    assert str(stored[1]).startswith("2018-03-05")


def test_two_sources_rank_above_one_and_fixture_stays_quiet(reset_ops_db, tmp_path, monkeypatch):
    from src.data.warehouse import apply_domain_schema, domain_con
    from src.domains.source_ingest import load_demo_private_sources
    from src.frontline.fleet_scan import run_fleet_scan

    monkeypatch.setenv("DOMAIN_DB_PATH", str(tmp_path))
    monkeypatch.setenv("FRONTLINE_REPORTING_LAG_DAYS", "0")
    pack = "phase4_rank"
    start = datetime(2018, 1, 8)

    def _week(con, prefix, day, n, category, entity_2, source):
        for i in range(n):
            con.execute(
                """
                INSERT INTO records (
                    record_id, occurred_at, received_at, entity_2, category,
                    text, source, provenance
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'observed')
                """,
                [
                    f"{prefix}-{day:%Y%m%d}-{i}", day, day, entity_2, category,
                    f"{prefix} note {i}", source,
                ],
            )

    with domain_con(pack, read_only=False) as con:
        apply_domain_schema(con)
        for w in range(6):
            day = start + timedelta(weeks=w)
            n = 2 if w < 5 else 20
            _week(con, "nhtsa-brakes", day, n, "SERVICE BRAKES", "HONDA", "NHTSA")
            _week(con, "svc-brakes", day, n, "SERVICE BRAKES", "HONDA", "SERVICE")
            _week(con, "nhtsa-bags", day, 2, "AIR BAGS", "FORD", "NHTSA")
            _week(con, "svc-bags", day, n, "AIR BAGS", "FORD", "SERVICE")
    load_demo_private_sources(pack)
    report = run_fleet_scan(pack, rebuild_clusters=False, alert=False)
    assert not any(err.startswith("investigation:") for err in report["errors"])
    opened = report["investigations_opened"]
    assert opened
    assert opened[0]["category"] == "SERVICE BRAKES"
    assert set(opened[0]["sources_flagged"]) == {"NHTSA", "SERVICE"}
    assert {row["category"] for row in opened} == {"SERVICE BRAKES", "AIR BAGS"}
    bags = next(row for row in opened if row["category"] == "AIR BAGS")
    assert bags["sources_flagged"] == ["SERVICE"]
