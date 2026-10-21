"""Fault spread: region / month aggregation is exact.  [AI-LAYER]"""

from __future__ import annotations

from datetime import datetime

import pytest


@pytest.fixture
def sp_pack(tmp_path, monkeypatch):
    from src.data.warehouse import apply_domain_schema, domain_con

    monkeypatch.setenv("DOMAIN_DB_PATH", str(tmp_path))
    rows = [("CA", datetime(2024, 1, 5), "SERVICE BRAKES"), ("CA", datetime(2024, 2, 1), "SERVICE BRAKES"),
            ("TX", datetime(2024, 2, 9), "SERVICE BRAKES"), ("TX", datetime(2025, 3, 1), "SERVICE BRAKES, HYDRAULIC"),
            ("NY", datetime(2024, 1, 2), "ENGINE"), (None, datetime(2024, 1, 2), "SERVICE BRAKES")]
    with domain_con("sp_test", read_only=False) as con:
        apply_domain_schema(con)
        for i, (reg, d, cat) in enumerate(rows):
            con.execute("INSERT INTO records (record_id, occurred_at, received_at, entity_1, entity_2, entity_3, category, text, region, source) "
                        "VALUES (?, ?, ?, '2019', 'FORD', 'ESCAPE', ?, 't', ?, 'NHTSA')", [f"S{i}", d, d, cat, reg])
    return "sp_test"


def test_spread_counts(sp_pack):
    from src.ai_layer.spread import fault_spread

    d = fault_spread("ford", None, "brakes", pack_id=sp_pack)
    assert d["total_complaints"] == 4 and d["regions_reporting"] == 2
    assert d["by_region"] == [{"region": "CA", "count": 2, "first_seen": "2024-01-05"},
                              {"region": "TX", "count": 2, "first_seen": "2024-02-09"}]
    assert [m["month"] for m in d["by_month"]] == ["2024-01", "2024-02", "2025-03"]
    assert d["by_month"][1] == {"month": "2024-02", "count": 2, "regions": 2}
    assert d["region_year_heat"]["years"] == ["2024", "2025"]
    assert {r["region"]: r["counts"] for r in d["region_year_heat"]["rows"]} == {"CA": [2, 0], "TX": [1, 1]}
    assert d["first_seen_order"][0]["region"] == "CA"
    assert "not normalised" in d["note"]


def test_spread_empty_and_unknown(sp_pack):
    from src.ai_layer.spread import fault_spread

    d = fault_spread("nosuchmake", pack_id=sp_pack)
    assert d["total_complaints"] == 0 and d["by_region"] == [] and d["by_month"] == []
