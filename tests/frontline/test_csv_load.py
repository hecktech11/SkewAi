"""Full CSV load: every fixture row, then the lead-time report."""

from __future__ import annotations

import csv
import shutil

import pytest
from fastapi.testclient import TestClient

FIXTURE = (
    "eval/frontline/fixtures/recall_lead/complaints.csv"
)
RECALLS = "eval/frontline/fixtures/recall_lead/recalls.csv"
MAPPING = {
    "record_id": "RECORD_KEY",
    "received_at": "DATEA",
    "occurred_at": "DATEC",
    "entity_1": "MODEL_YR",
    "entity_2": "MAKETXT",
    "entity_3": "MODELTXT",
    "category": "COMPONENT",
    "subcategory": "COMPDESC",
    "text": "CDESC",
    "source": "NHTSA",
}


def _rows(path) -> int:
    with open(path, newline="", encoding="utf-8") as fh:
        return sum(1 for _ in csv.DictReader(fh))


def test_category_cannot_be_the_narrative_and_blank_stays_blank():
    from src.domains.csv_load import coerce_mapping
    from src.domains.mapping_ingest import map_row

    with pytest.raises(ValueError, match="narrative"):
        coerce_mapping({"record_id": "ID", "text": "CDESC", "category": "CDESC"})
    blank = map_row(
        {"ID": "1", "CDESC": "the brakes failed", "DATEA": "20150129"},
        {
            "record_id": "ID",
            "text": "CDESC",
            "received_at": "DATEA",
            "normalize_category": "nhtsa",
        },
    )
    assert blank["category"] == ""
    raw = map_row(
        {
            "ID": "2",
            "COMPDESC": "SERVICE BRAKES, HYDRAULIC:FOUNDATION COMPONENTS",
            "CDESC": "the brakes failed",
            "DATEA": "20150129",
        },
        {
            "record_id": "ID",
            "text": "CDESC",
            "category": "COMPDESC",
            "received_at": "DATEA",
            "normalize_category": "nhtsa",
        },
    )
    assert raw["category"] == "SERVICE BRAKES"
    assert raw["text"] == "the brakes failed"


def test_fixture_csv_loads_every_row_and_keeps_the_lead(reset_ops_db):
    from pathlib import Path

    from scripts.ingest_recalls import advisory_contract, insert_advisories
    from src.backtest.recall_lead import recall_lead_report
    from src.config import REPO_ROOT
    from src.domains.csv_load import load_csv_rows
    from src.ml_runtime.anomalies import current_spikes

    complaints = REPO_ROOT / FIXTURE
    report = load_csv_rows(
        "csv_lead_proof",
        complaints,
        mapping=MAPPING,
        source="nhtsa",
        provenance="observed",
        run_anomalies=True,
    )
    assert report["written"] == _rows(complaints)
    assert report["skipped"] == 0
    assert isinstance(report["anomalies_scored"], int)
    spikes = current_spikes("csv_lead_proof")
    assert set(spikes) >= {"spike_week", "spikes", "censored_weeks"}
    with (REPO_ROOT / RECALLS).open(newline="", encoding="utf-8") as fh:
        recalls = list(csv.DictReader(fh))
    insert_advisories(
        "csv_lead_proof",
        recalls,
        contract=advisory_contract("automotive_nhtsa"),
    )
    lead = recall_lead_report("csv_lead_proof", run_backtests=True)
    leads = sorted(int(row["lead_weeks"]) for row in lead["hits"])
    assert leads == [4, 12]
    assert lead["miss_count"] >= 1
    assert lead["unconfirmed_count"] >= 1
    assert Path(complaints).stat().st_size > 0


def test_http_ingest_writes_rows_and_rejects_bad_files(tmp_path, reset_ops_db, monkeypatch):
    from src.api.main import app
    from src.config import REPO_ROOT
    from src.data.warehouse import domain_con
    from src.jobs.queue import run_next

    monkeypatch.delenv("FRONTLINE_API_KEY", raising=False)
    monkeypatch.setenv("FRONTLINE_AUTH_REQUIRED", "1")
    monkeypatch.delenv("FRONTLINE_OPEN_MODE", raising=False)
    monkeypatch.setenv("FRONTLINE_ENABLED", "1")
    monkeypatch.setenv("BUILDER_UPLOAD_DIR", str(tmp_path))
    monkeypatch.setenv("SESSION_SECRET", "super-secret-key-for-test-32bytes")
    src = REPO_ROOT / FIXTURE
    dest = tmp_path / "complaints.csv"
    shutil.copyfile(src, dest)
    from src.api.rbac import issue_session
    admin_token = issue_session("admin_usr", "admin", issuer_role="admin")["token"]
    headers = {"X-Frontline-Session": admin_token}

    with TestClient(app) as client:
        ok = client.post(
            "/api/frontline/pack-builder/ingest",
            headers=headers,
            json={
                "pack_id": "csv_http_proof",
                "csv_path": str(dest),
                "mapping": MAPPING,
                "source": "nhtsa",
                "provenance": "observed",
            },
        )
        assert ok.status_code == 200, ok.text
        body = ok.json()
        assert body["written"] == _rows(dest)
        assert body["skipped"] == 0
        assert body["job_id"] is None
        notes = tmp_path / "notes.txt"
        notes.write_text("not a csv\n", encoding="utf-8")
        rejected = client.post(
            "/api/frontline/pack-builder/ingest",
            headers=headers,
            json={"pack_id": "csv_http_proof", "csv_path": str(notes), "mapping": MAPPING},
        )
        assert rejected.status_code == 400
        monkeypatch.setenv("FRONTLINE_CSV_MAX_BYTES", "32")
        oversized = client.post(
            "/api/frontline/pack-builder/ingest",
            headers=headers,
            json={"pack_id": "csv_http_proof", "csv_path": str(dest), "mapping": MAPPING},
        )
        assert oversized.status_code == 400
        monkeypatch.setenv("FRONTLINE_CSV_MAX_BYTES", str(dest.stat().st_size + 10))
        queued = client.post(
            "/api/frontline/pack-builder/ingest",
            headers=headers,
            json={
                "pack_id": "csv_http_job",
                "csv_path": str(dest),
                "mapping": MAPPING,
                "source": "nhtsa",
                "async": True,
                "run_anomalies": True,
            },
        )
        assert queued.status_code == 200, queued.text
        job_id = queued.json()["job_id"]
    done = None
    for _ in range(5):
        step = run_next()
        if step and step.get("job_id") == job_id:
            done = step
            break
    assert done is not None and done["status"] == "done"
    result = done["result"]
    assert result["upserted"] == _rows(dest)
    assert isinstance(result["anomalies_scored"], int)
    with domain_con("csv_http_proof") as con:
        count = con.execute("SELECT COUNT(*) FROM records").fetchone()[0]
    assert count == _rows(dest)


def test_ingest_route_is_closed_without_ops_write(tmp_path, reset_ops_db, monkeypatch):
    from src.api.main import app

    monkeypatch.setenv("FRONTLINE_AUTH_REQUIRED", "1")
    monkeypatch.setenv("FRONTLINE_API_KEY", "csv-load-test-key")
    monkeypatch.delenv("FRONTLINE_OPEN_MODE", raising=False)
    monkeypatch.setenv("FRONTLINE_ENABLED", "1")
    with TestClient(app) as client:
        anon = client.post("/api/frontline/pack-builder/ingest", json={})
        assert anon.status_code == 401
        denied = client.post(
            "/api/frontline/pack-builder/ingest",
            headers={"X-API-Key": "csv-load-test-key"},
            json={"pack_id": "csv_http_proof", "csv_path": str(tmp_path / "x.csv")},
        )
        assert denied.status_code == 403

    # Finding 4: In open mode, anonymous calls still receive 401 and agent credentials receive 403
    monkeypatch.setenv("FRONTLINE_OPEN_MODE", "1")
    monkeypatch.delenv("FRONTLINE_AUTH_REQUIRED", raising=False)
    with TestClient(app) as client:
        anon_open = client.post("/api/frontline/pack-builder/ingest", json={})
        assert anon_open.status_code == 401
        agent_open = client.post(
            "/api/frontline/pack-builder/ingest",
            headers={"X-API-Key": "csv-load-test-key"},
            json={"pack_id": "csv_http_proof", "csv_path": str(tmp_path / "x.csv")},
        )
        assert agent_open.status_code == 403


def test_csv_load_refuses_the_pilot_warehouse(monkeypatch):
    monkeypatch.setattr(
        "src.config.is_default_pilot_domain_path",
        lambda path=None: True,
    )
    from src.domains.csv_load import load_csv_rows

    with pytest.raises(ValueError, match="pilot warehouse"):
        load_csv_rows(
            "automotive_nhtsa",
            "eval/frontline/fixtures/recall_lead/complaints.csv",
            mapping=MAPPING,
        )


def test_sample_script_refuses_the_pilot_directory(monkeypatch):
    monkeypatch.setattr(
        "src.config.is_default_pilot_domain_path",
        lambda path=None: True,
    )
    from scripts.recall_lead_sample import main

    assert main([]) == 2


def test_cli_refuses_the_pilot_warehouse(tmp_path, monkeypatch, capsys):
    from scripts.ingest_source import main

    monkeypatch.setenv("DOMAIN_DB_PATH", str(tmp_path))
    monkeypatch.setattr(
        "src.config.is_default_pilot_domain_path",
        lambda path=None: True,
    )
    path = tmp_path / "one.csv"
    path.write_text(
        "ID,CDESC,DATEA\nr1,brake noise on the highway,20240102\n",
        encoding="utf-8",
    )
    mapping = tmp_path / "one.yaml"
    mapping.write_text(
        'records:\n  record_id: ID\n  text: CDESC\n  received_at: DATEA\n  source: "CSV"\n',
        encoding="utf-8",
    )
    argv = [
        "--pack", "automotive_nhtsa",
        "--source", "csv",
        "--csv", str(path),
        "--mapping", str(mapping),
    ]
    assert main(argv) == 1
    assert "pilot warehouse" in capsys.readouterr().out
    assert main([*argv, "--allow-pilot"]) == 0


def test_cli_run_anomalies_flag_is_accepted(tmp_path, monkeypatch, capsys):
    from scripts.ingest_source import main

    monkeypatch.setenv("DOMAIN_DB_PATH", str(tmp_path))
    path = tmp_path / "one.csv"
    path.write_text("ID,CDESC,DATEA\nr1,brake noise on the highway,20240102\n", encoding="utf-8")
    mapping = tmp_path / "one.yaml"
    mapping.write_text(
        "records:\n  record_id: ID\n  text: CDESC\n  received_at: DATEA\n  source: \"CSV\"\n",
        encoding="utf-8",
    )
    rc = main([
        "--pack", "csv_cli_proof",
        "--source", "csv",
        "--csv", str(path),
        "--mapping", str(mapping),
        "--run-anomalies",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "ingested 1/1" in out
    assert "anomalies scored" in out
