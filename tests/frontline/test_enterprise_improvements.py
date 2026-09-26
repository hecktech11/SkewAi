"""Improvements to Enterprise Ops — recent catalog, copilot intents, risk history, scenario validate."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.data.warehouse import ops_con
from src.enterprise.catalog import list_recent_interactions
from src.enterprise.copilot import answer_supervisor_query
from src.enterprise.risk import list_risk_history, score_interaction_risk
from src.enterprise.scenarios import validate_scenario_steps
from src.ids import new_ulid


def _seed_ix(
    *,
    status: str = "completed",
    peak: float = 0.4,
    entity_2: str = "Toyota",
    entity_3: str = "Camry",
) -> str:
    iid = "int_" + new_ulid()
    now = datetime.now(timezone.utc)
    with ops_con() as con:
        con.execute(
            """
            INSERT INTO interactions (
                interaction_id, pack_id, pack_version, started_at, ended_at,
                channel, status, outcome, supervised, peak_frustration, last_frustration,
                entity_1, entity_2, entity_3, category
            ) VALUES (?, 'automotive_nhtsa', 't', ?, ?, 'sim', ?, 'case_created', FALSE, ?, ?,
                      '2019', ?, ?, 'brakes')
            """,
            [iid, now, now, status, peak, peak, entity_2, entity_3],
        )
        con.execute(
            """
            INSERT INTO cases (
                case_id, interaction_id, pack_id, created_at, category, description_summary,
                onset, severity, severity_source, priority, safety_flags, status, followup_draft
            ) VALUES (?, ?, 'automotive_nhtsa', ?, 'brakes', 'soft', ?, 'Critical', 'rules', 1, '{}', 'open', '')
            """,
            ["case_" + iid[-10:], iid, now, now],
        )
    return iid


@pytest.fixture
def client(reset_ops_db, monkeypatch):
    monkeypatch.delenv("FRONTLINE_API_KEY", raising=False)
    with TestClient(app) as c:
        yield c


def test_list_recent_interactions(reset_ops_db):
    a = _seed_ix(peak=0.2)
    b = _seed_ix(peak=0.9)
    rows = list_recent_interactions(limit=10)
    ids = [r["interaction_id"] for r in rows]
    assert a in ids and b in ids
    assert "status" in rows[0]
    assert "interaction_id" in rows[0]


def test_copilot_new_intents(reset_ops_db):
    _seed_ix(peak=0.95, status="active")
    # critical cases
    r = answer_supervisor_query("Show open critical cases")
    assert r["intent"] == "critical_cases"
    assert r["row_count"] >= 1
    # high risk
    r2 = answer_supervisor_query("Show high risk contacts")
    assert r2["intent"] == "high_risk"
    assert "suggestions" in r2 and len(r2["suggestions"]) >= 3
    # help returns suggestions
    h = answer_supervisor_query("help")
    assert h["intent"] == "help"
    assert h["suggestions"]


def test_risk_history_persist(reset_ops_db):
    iid = _seed_ix(status="active", peak=0.8)
    s1 = score_interaction_risk(iid, persist=True)
    assert s1.get("snapshot_id")
    s2 = score_interaction_risk(iid, persist=True)
    assert s2.get("snapshot_id") != s1.get("snapshot_id")
    hist = list_risk_history(iid)
    assert len(hist) >= 2
    assert hist[0]["escalation_prob"] >= 0


def test_validate_scenario_steps_unit():
    bad = validate_scenario_steps([], name="")
    assert bad["ok"] is False
    assert any(e["field"] == "steps" for e in bad["errors"])
    good = validate_scenario_steps(
        [{"text": "hello brakes"}, {"text": "still soft"}],
        name="soft brake",
        pack_id="automotive_nhtsa",
    )
    assert good["ok"] is True
    assert good["step_count"] == 2


def test_api_recent_risk_history_validate(client, reset_ops_db):
    iid = _seed_ix(status="active", peak=0.7)

    r = client.get("/api/frontline/enterprise/interactions/recent?limit=10")
    assert r.status_code == 200
    assert r.json()["count"] >= 1
    assert any(x["interaction_id"] == iid for x in r.json()["interactions"])

    r = client.get(f"/api/frontline/enterprise/risk/{iid}?persist=true")
    assert r.status_code == 200
    assert r.json().get("snapshot_id")

    r = client.get(f"/api/frontline/enterprise/risk/{iid}/history")
    assert r.status_code == 200
    assert r.json()["count"] >= 1

    r = client.post(
        "/api/frontline/enterprise/scenarios/validate",
        json={"name": "x", "pack_id": "automotive_nhtsa", "steps": []},
    )
    assert r.status_code == 200
    assert r.json()["ok"] is False

    r = client.post(
        "/api/frontline/enterprise/copilot",
        json={"query": "Show open critical cases"},
    )
    assert r.status_code == 200
    assert r.json()["intent"] == "critical_cases"


def test_dashboard_wires_improvements():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    page = (root / "dashboard/routes/EnterpriseOps.jsx").read_text(encoding="utf-8")
    assert "/api/frontline/enterprise/interactions/recent" in page
    assert "/api/frontline/enterprise/scenarios/validate" in page
    assert "risk/" in page and "history" in page
    assert "persist" in page
