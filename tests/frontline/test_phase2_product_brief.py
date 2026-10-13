"""Phase 2 product brief: slice evidence, workspace reload, unclustered scans.

Synthetic rows. The committed NHTSA sample stays in the Phase 1 eval gate.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest


def _record(con, record_id, *, source, category, entity_2, entity_3, entity_key, text):
    con.execute(
        """
        INSERT INTO records (
            record_id, received_at, entity_2, entity_3, category, text,
            source, entity_key
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            record_id,
            datetime(2019, 1, 2),
            entity_2,
            entity_3,
            category,
            text,
            source,
            entity_key,
        ],
    )


def _insert_week(con, prefix, day, n, *, category, entity_2):
    for i in range(n):
        con.execute(
            """
            INSERT INTO records (
                record_id, occurred_at, received_at, entity_2, category, text, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                f"{prefix}-{day:%Y%m%d}-{i}",
                day,
                day,
                entity_2,
                category,
                f"{prefix} note {i}",
                "NHTSA",
            ],
        )


def test_brief_lists_every_source_and_recall_lead(reset_ops_db, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from src.api.main import app
    from src.data.warehouse import apply_domain_schema, domain_con
    from src.frontline.live_intercept import open_or_link_investigation

    monkeypatch.setenv("DOMAIN_DB_PATH", str(tmp_path))
    monkeypatch.delenv("FRONTLINE_API_KEY", raising=False)
    monkeypatch.setenv("FRONTLINE_ENABLED", "1")
    pack = "phase2_brief"
    with domain_con(pack, read_only=False) as con:
        apply_domain_schema(con)
        shared = "VIN123"
        for source in ("NHTSA", "FRONTLINE", "WARRANTY", "SERVICE"):
            _record(
                con,
                f"rec-{source.lower()}",
                source=source,
                category="SERVICE BRAKES",
                entity_2="HONDA",
                entity_3="ACCORD",
                entity_key=shared if source in ("NHTSA", "WARRANTY") else None,
                text=f"{source} hose leak",
            )
        _record(
            con,
            "rec-other",
            source="NHTSA",
            category="ENGINE",
            entity_2="TOYOTA",
            entity_3="CAMRY",
            entity_key=None,
            text="unrelated stall",
        )
        con.execute(
            """
            INSERT INTO weekly_anomalies (
                pack_id, iso_week, category, entity_2, entity_3, record_count,
                baseline_mean, baseline_std, z_score, is_anomaly, censored
            ) VALUES (?, '2019-W01', 'SERVICE BRAKES', 'HONDA', 'ACCORD',
                      20, 2.0, 0.5, 4.2, TRUE, FALSE)
            """,
            [pack],
        )
        con.execute(
            """
            INSERT INTO advisories (
                advisory_id, issued_at, scope_entity_2, scope_entity_3,
                scope_category, summary, remedy, url, source
            ) VALUES (?, ?, 'HONDA', 'ACCORD', 'SERVICE BRAKES',
                      'Hose leak.', 'Replace.', 'https://example.test/rcl', 'NHTSA')
            """,
            ["19V001|2019|HONDA|ACCORD|SERVICE BRAKES", datetime(2019, 1, 29)],
        )
    inv_id = open_or_link_investigation(
        pack_id=pack,
        cluster_id=0,
        title="Service brakes Honda",
        category="SERVICE BRAKES",
        entity_2="HONDA",
        entity_3="ACCORD",
    )
    with TestClient(app) as client:
        missing = client.get("/api/frontline/investigations/inv_missing/brief")
        assert missing.status_code == 404
        brief = client.get(f"/api/frontline/investigations/{inv_id}/brief")
        assert brief.status_code == 200, brief.text
        body = brief.json()
        by_source = body["records_by_source"]
        for source in ("NHTSA", "FRONTLINE", "WARRANTY", "SERVICE"):
            ids = {row["record_id"] for row in by_source[source]}
            assert f"rec-{source.lower()}" in ids
            assert by_source[source][0]["tag"] == source
        observed = [link for link in body["cross_source_links"] if link["basis"] == "observed"]
        assert observed
        assert observed[0]["entity_key"] == "VIN123"
        assert set(observed[0]["sources"]) >= {"NHTSA", "WARRANTY"}
        assert body["association_lift"] == pytest.approx(1.25)
        assert body["flagged_weeks"][0]["iso_week"] == "2019-W01"
        assert body["recall"]["advisory_id"].startswith("19V001")
        assert body["recall"]["lead_time_weeks"] == 4

        saved = client.patch(
            f"/api/frontline/investigations/{inv_id}/workspace",
            json={
                "assignee": "Lee Chen",
                "sla_due_at": "2026-10-15",
                "hypotheses": [{"body": "Hose batch 14", "status": "open"}],
            },
        )
        assert saved.status_code == 200, saved.text
        again = client.get(f"/api/frontline/investigations/{inv_id}/workspace")
        assert again.status_code == 200
        workspace = again.json()
        assert workspace["assignee"] == "Lee Chen"
        assert str(workspace["sla_due_at"]).startswith("2026-10-15")
        assert any(h["body"] == "Hose batch 14" for h in workspace["hypotheses"])


def test_unclustered_slices_stay_separate_and_carry_lead(reset_ops_db, tmp_path, monkeypatch):
    from src.data.warehouse import apply_domain_schema, domain_con, ops_con
    from src.frontline.fleet_scan import run_fleet_scan

    monkeypatch.setenv("DOMAIN_DB_PATH", str(tmp_path))
    monkeypatch.setenv("FRONTLINE_REPORTING_LAG_DAYS", "0")
    captured: list[tuple] = []

    async def _capture(cluster_id, live_count, lead_time_weeks, pack_id=None):
        captured.append((cluster_id, live_count, lead_time_weeks, pack_id))

    monkeypatch.setattr("src.frontline.alerts.alert_early_warning", _capture)
    pack = "phase2_scan"
    start = datetime(2018, 1, 8)
    with domain_con(pack, read_only=False) as con:
        apply_domain_schema(con)
        for w in range(6):
            day = start + timedelta(weeks=w)
            n = 2 if w < 5 else 20
            _insert_week(con, "brakes", day, n, category="SERVICE BRAKES", entity_2="HONDA")
            _insert_week(con, "bags", day, n, category="AIR BAGS", entity_2="FORD")
            _insert_week(con, "flat", day, 2, category="ENGINE", entity_2="HONDA")
        con.execute(
            """
            INSERT INTO advisories (
                advisory_id, issued_at, scope_entity_2, scope_category,
                summary, remedy, url, source
            ) VALUES (?, ?, 'HONDA', 'SERVICE BRAKES',
                      'Hose leak.', 'Replace.', 'https://example.test/rcl', 'NHTSA')
            """,
            ["18V100|2018|HONDA|SERVICE BRAKES", datetime(2018, 3, 12)],
        )
    first = run_fleet_scan(pack, rebuild_clusters=False, alert=True)
    assert not any(err.startswith("investigation:") for err in first["errors"])
    opened = first["investigations_opened"]
    assert {row["category"] for row in opened} == {"SERVICE BRAKES", "AIR BAGS"}
    assert {row["cluster_id"] for row in opened} == {0}
    assert len({row["investigation_id"] for row in opened}) == 2
    by_cat = {row["category"]: row for row in opened}
    assert by_cat["SERVICE BRAKES"]["lead_time_weeks"] == 4
    assert by_cat["AIR BAGS"]["lead_time_weeks"] is None
    leads = [row[2] for row in captured]
    assert 4 in leads
    assert None in leads
    second = run_fleet_scan(pack, rebuild_clusters=False, alert=True)
    assert second["investigations_opened"] == []
    assert {row["category"] for row in second["investigations_linked"]} == {
        "SERVICE BRAKES",
        "AIR BAGS",
    }
    with ops_con(read_only=True) as con:
        rows = con.execute(
            """
            SELECT cluster_id, slice_category
            FROM investigations WHERE pack_id = ? AND status = 'open'
            """,
            [pack],
        ).fetchall()
    assert len(rows) == 2
    assert {r[0] for r in rows} == {0}
    assert {r[1] for r in rows} == {"SERVICE BRAKES", "AIR BAGS"}
