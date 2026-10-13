"""Phase 3: a close records the fix date, then waits for eight event-weeks.

Synthetic rows. The close date is the one on the request, not the last
case timestamp. A later measure_fix pass appends history.
"""

from __future__ import annotations

from datetime import datetime, timedelta


def _insert_week(con, prefix, day, n, *, category, entity_2, entity_3, entity_key=None):
    for i in range(n):
        con.execute(
            """
            INSERT INTO records (
                record_id, occurred_at, received_at, entity_2, entity_3,
                category, text, source, entity_key, provenance
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'NHTSA', ?, 'observed')
            """,
            [
                f"{prefix}-{day:%Y%m%d}-{i}",
                day,
                day,
                entity_2,
                entity_3,
                category,
                f"{prefix} note {i}",
                entity_key if i == 0 else None,
            ],
        )


def test_close_waits_then_eight_event_weeks_conclude(reset_ops_db, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from src.api.main import app
    from src.data.warehouse import apply_domain_schema, domain_con, ops_con
    from src.enterprise.fix_effectiveness import measure_recorded_fixes
    from src.frontline.live_intercept import open_or_link_investigation
    from src.frontline.ops import ops_metrics

    monkeypatch.setenv("DOMAIN_DB_PATH", str(tmp_path))
    monkeypatch.delenv("FRONTLINE_API_KEY", raising=False)
    monkeypatch.setenv("FRONTLINE_ENABLED", "1")
    pack = "phase3_fix"
    pivot = datetime(2018, 3, 5)
    with domain_con(pack, read_only=False) as con:
        apply_domain_schema(con)
        for week in range(1, 9):
            day = pivot - timedelta(weeks=week)
            _insert_week(
                con, "before", day, 4,
                category="SERVICE BRAKES", entity_2="HONDA", entity_3="ACCORD",
                entity_key="VIN-RECUR" if week == 1 else None,
            )
            _insert_week(
                con, "ctrl", day, 2,
                category="SERVICE BRAKES", entity_2="FORD", entity_3="F-150",
            )
    inv_id = open_or_link_investigation(
        pack_id=pack,
        cluster_id=0,
        title="Service brakes Honda",
        category="SERVICE BRAKES",
        entity_2="HONDA",
        entity_3="ACCORD",
    )
    with ops_con() as con:
        con.execute(
            "UPDATE investigations SET last_case_at = ?, status = 'monitoring' WHERE investigation_id = ?",
            [datetime(2017, 1, 1), inv_id],
        )
    with TestClient(app) as client:
        rejected = client.patch(
            f"/api/frontline/investigations/{inv_id}",
            json={"status": "closed", "fixed_at": "not-a-date"},
        )
        assert rejected.status_code == 400
        with ops_con(read_only=True) as con:
            status = con.execute(
                "SELECT status FROM investigations WHERE investigation_id = ?",
                [inv_id],
            ).fetchone()[0]
        assert status == "monitoring"
        closed = client.patch(
            f"/api/frontline/investigations/{inv_id}",
            json={
                "status": "closed",
                "fixed_at": "2018-03-05",
                "note": "Hose batch 14 replaced",
                "control_entity_2": "FORD",
            },
        )
    assert closed.status_code == 200, closed.text
    body = closed.json()
    assert body["status"] == "closed"
    fix = body["fix_effectiveness"]["fix"]
    measured = body["fix_effectiveness"]["measured"]
    assert fix["fixed_at"].startswith("2018-03-05")
    assert not str(fix["fixed_at"]).startswith("2017")
    assert fix["note"] == "Hose batch 14 replaced"
    assert fix["control_entity_2"] == "FORD"
    assert measured["status"] == "waiting"
    assert measured["window_days"] == 56
    assert measured["event_weeks_after"] == 0
    assert measured["conclusive"] is False
    assert measured["diff_in_diff"] is not None

    with domain_con(pack, read_only=False) as con:
        for week in range(1, 9):
            day = pivot + timedelta(weeks=week)
            _insert_week(
                con, "after", day, 1,
                category="SERVICE BRAKES", entity_2="HONDA", entity_3="ACCORD",
                entity_key="VIN-RECUR" if week == 2 else None,
            )
            _insert_week(
                con, "ctrl-after", day, 2,
                category="SERVICE BRAKES", entity_2="FORD", entity_3="F-150",
            )
    again = measure_recorded_fixes(pack_id=pack)
    assert again["measured"] == 1
    result = again["results"][0]
    assert result["status"] == "improved"
    assert result["event_weeks_after"] >= 8
    assert result["before_count"] > result["after_count"]
    assert result["before_rate"] > result["after_rate"]
    assert result["window_days"] == 56
    assert result["diff_in_diff"] is not None
    assert result["diff_in_diff"] < 0
    with ops_con(read_only=True) as con:
        n = con.execute(
            "SELECT COUNT(*) FROM fix_measurements WHERE fix_id = ?",
            [fix["fix_id"]],
        ).fetchone()[0]
    assert n == 2
    loop = ops_metrics()["fix_loop"]
    assert {"fixes_recorded", "resolved", "reopened", "reopen_rate", "series"} <= set(loop)
    assert loop["reopened"] >= 1
    assert loop["reopen_rate"] > 0


def test_select_flagged_slices_skips_mix_only():
    from src.frontline.fleet_scan import select_flagged_slices

    combined = [{
        "iso_week": "2018-W10",
        "category": "ENGINE",
        "entity_2": "HONDA",
        "entity_3": "",
        "is_anomaly": True,
        "z_score": 4.0,
    }]
    by_source = {
        "NHTSA": [{**combined[0], "is_anomaly": False, "z_score": 1.0}],
        "SERVICE": [{**combined[0], "is_anomaly": False, "z_score": 1.2}],
    }
    assert select_flagged_slices(combined, by_source) == []
    fallback = select_flagged_slices(combined, None)
    assert len(fallback) == 1
    assert fallback[0]["sources_flagged"] == []
    agreed = select_flagged_slices(combined, {
        "NHTSA": [{**combined[0], "z_score": 3.0}],
        "SERVICE": [{**combined[0], "z_score": 5.0}],
        "FIELD": [{
            "iso_week": "2018-W10",
            "category": "AIR BAGS",
            "entity_2": "FORD",
            "entity_3": "",
            "is_anomaly": True,
            "z_score": 9.0,
        }],
    })
    assert [row["category"] for row in agreed] == ["ENGINE", "AIR BAGS"]
    assert set(agreed[0]["sources_flagged"]) == {"NHTSA", "SERVICE"}
