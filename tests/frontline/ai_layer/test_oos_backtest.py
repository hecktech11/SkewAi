"""Out-of-sample backtest: re-gate equality, freeze/refuse rules, period isolation.  [AI-LAYER]"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from scripts import ai_oos_backtest as oos


def test_prod_regate_equals_scorer_on_synthetic_series():
    from src.ml_runtime.anomalies import score_weekly_slices

    counts = {}
    start = datetime(2019, 1, 7)
    for w in range(40):
        wk = (start + timedelta(weeks=w)).isocalendar()
        label = f"{wk[0]}-W{wk[1]:02d}"
        counts[(label, "SERVICE BRAKES", "HONDA")] = 2 + (25 if w in (12, 30) else 0) + (w % 3)
        counts[(label, "ENGINE", "FORD")] = 10 + (9 if w == 20 else 0)
        if w >= 25:
            counts[(label, "STEERING", "KIA")] = 0 if w < 33 else 7
    rows = score_weekly_slices(counts, pack_id="t")
    assert any(r["is_anomaly"] for r in rows)
    assert all(oos.gate(r, oos.PROD) == bool(r["is_anomaly"]) for r in rows)
    looser = {"z": 2.0, "q": 0.05, "abs_lift": 3, "rel_lift": 0.5}
    stricter = {"z": 3.0, "q": 0.01, "abs_lift": 8, "rel_lift": 1.0}
    n = lambda c: sum(oos.gate(r, c) for r in rows)
    assert n(stricter) <= n(oos.PROD) <= n(looser)


def test_grid_contains_production_and_hash_is_canonical():
    assert oos.PROD in oos.grid_configs() and len(oos.grid_configs()) == 36
    assert oos.canonical_hash({"a": 1, "b": 2}) == oos.canonical_hash({"b": 2, "a": 1})
    assert oos.objective(0.5, 0.5) == 0.5 and oos.objective(0, 0) == 0.0


def test_tune_refuses_when_frozen_config_exists(tmp_path, monkeypatch):
    monkeypatch.setenv("DOMAIN_DB_PATH", str(tmp_path / "dom"))
    (tmp_path / "oos_config.json").write_text("{}")
    rc = oos.main(["--phase", "tune", "--out-dir", str(tmp_path)])
    assert rc == 3
    log = [json.loads(x) for x in (tmp_path / "run_log.jsonl").read_text().splitlines()]
    assert log[-1]["phase"] == "tune" and "refusing to re-tune" in log[-1]["refused"]
    oos.check_tune_allowed(tmp_path / "oos_config.json", True)  # explicit force is allowed


def test_test_phase_refuses_without_or_with_tampered_config_and_second_run(tmp_path, monkeypatch):
    monkeypatch.setenv("DOMAIN_DB_PATH", str(tmp_path / "dom"))
    assert oos.main(["--phase", "test", "--out-dir", str(tmp_path)]) == 3  # no config
    body = {"thresholds": oos.PROD, "test_window": ["2021-01-01", "2024-12-31"]}
    good = {"sha256": oos.canonical_hash(body), "body": body}
    (tmp_path / "oos_config.json").write_text(json.dumps({**good, "body": {**body, "thresholds": {**oos.PROD, "z": 9}}}))
    with pytest.raises(oos.Refused):
        oos.load_frozen(tmp_path / "oos_config.json")
    (tmp_path / "oos_config.json").write_text(json.dumps(good))
    assert oos.load_frozen(tmp_path / "oos_config.json")["sha256"] == good["sha256"]
    (tmp_path / "oos_test_results.json").write_text("{}")
    assert oos.main(["--phase", "test", "--out-dir", str(tmp_path)]) == 3  # second test run refused
    log = (tmp_path / "run_log.jsonl").read_text()
    assert "runs once" in log


def _insert_week(con, prefix, day, n, *, category, entity_2):
    for i in range(n):
        con.execute(
            "INSERT INTO records (record_id, occurred_at, received_at, entity_2, entity_3, category, text, source) "
            "VALUES (?, ?, ?, ?, '', ?, ?, 'NHTSA')",
            [f"{prefix}-{day:%Y%m%d}-{i}", day, day, entity_2, category, f"{prefix} note {i}"],
        )


def _tiny_pack(tmp_path, monkeypatch, pack):
    from scripts.ingest_recalls import advisory_contract, insert_advisories
    from src.data.warehouse import apply_domain_schema, domain_con

    monkeypatch.setenv("DOMAIN_DB_PATH", str(tmp_path / "dom"))
    with domain_con(pack, read_only=False) as con:
        apply_domain_schema(con)
        for base, spikes in ((datetime(2019, 1, 7), (6, 10)), (datetime(2022, 1, 3), (5, 9))):
            for w in range(14):
                day = base + timedelta(weeks=w)
                _insert_week(con, "b", day, 20 if w in spikes else 2, category="SERVICE BRAKES", entity_2="HONDA")
                _insert_week(con, "s", day, 2 if w < 8 else 16, category="STEERING", entity_2="HONDA")
    rows = []
    for key, rc, comp in [("19V100|2019|HONDA|ACCORD|SERVICE BRAKES", "20190401", "SERVICE BRAKES"),
                          ("19V200|2019|HONDA|ACCORD|ENGINE", "20190311", "ENGINE"),
                          ("22V100|2022|HONDA|CIVIC|SERVICE BRAKES", "20220321", "SERVICE BRAKES"),
                          ("22V200|2022|HONDA|CIVIC|STEERING", "20220404", "STEERING")]:
        rows.append({"ADVISORY_KEY": key, "RCDATE": rc, "YEARTXT": key.split("|")[1], "MAKETXT": "HONDA",
                     "MODELTXT": key.split("|")[3], "COMPONENT": comp, "DESC_DEFECT": "d.",
                     "CORRECTIVE_ACTION": "r.", "RCL_URL": "https://www.nhtsa.gov/recalls"})
    insert_advisories(pack, rows, contract=advisory_contract("automotive_nhtsa"))


def test_prod_rows_match_existing_lead_report_on_full_window(tmp_path, monkeypatch):
    """Same advisories + full data: OOS scoring with the production config == the existing report."""
    from src.backtest.recall_lead import recall_lead_report

    pack = "oos_eq"
    _tiny_pack(tmp_path, monkeypatch, pack)
    expected = recall_lead_report(pack, run_backtests=False)
    ev = oos.evaluate(pack, "2000-01-01", "2030-12-31", [oos.PROD], workers=1, progress=False)
    rep = oos.build_report(ev["advisories"], ev["scored"]["rows"], oos.cfg_id(oos.PROD), 0)
    got = {(r["advisory_id"], r["outcome"], r["lead_weeks"], r["first_flagged_week"]) for r in rep["recalls"]}
    exp = {(r["advisory_id"], r["outcome"], r["lead_weeks"], r["first_flagged_week"]) for r in expected["recalls"]}
    assert got == exp and ev["scored"]["regate_mismatches"] == 0
    assert any(r[1] == "hit" for r in got)


def test_tune_uses_only_train_window_then_test_once(tmp_path, monkeypatch):
    pack = "oos_e2e"
    _tiny_pack(tmp_path, monkeypatch, pack)
    out = tmp_path / "out"
    common = ["--pack", pack, "--out-dir", str(out), "--workers", "1"]
    assert oos.main(["--phase", "tune", *common]) == 0
    frozen = json.loads((out / "oos_config.json").read_text())
    assert oos.canonical_hash(frozen["body"]) == frozen["sha256"]
    tune = json.loads((out / "oos_tune_results.json").read_text())
    ins = tune["in_sample_frozen"]
    assert ins["advisories"] == 2  # only the 2019 recalls
    assert tune["regate_mismatches_vs_scorer"] == 0 and len(tune["grid_results"]) == 36
    assert "not computed" in ins["preventable_window_source"]  # no --zip: labelled, not invented
    # train records stop at 2020-12-31: none of the 2022 complaints were seen
    assert ins["records_received_by_period_end"] == sum(
        (20 if w in (6, 10) else 2) + (2 if w < 8 else 16) for w in range(14))
    assert oos.main(["--phase", "tune", *common]) == 3  # refuses to re-tune
    assert oos.main(["--phase", "test", *common]) == 0
    test = json.loads((out / "oos_test_results.json").read_text())
    assert test["advisories"] == 2 and test["config_sha256"] == frozen["sha256"]
    assert oos.main(["--phase", "test", *common]) == 3  # runs once
    md = (out / "OOS_BACKTEST.md").read_text()
    assert "In-sample 2015-2020" in md and "Out-of-sample 2021-2024" in md and "Right-censoring" in md
    phases = [json.loads(x)["phase"] for x in (out / "run_log.jsonl").read_text().splitlines()]
    assert phases == ["tune", "tune", "test_start", "test", "test"]
