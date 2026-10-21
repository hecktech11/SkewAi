"""Safety-prioritised early-warning board.  [AI-LAYER]"""

from datetime import datetime, timezone


def _inv(con, iid, cat, cases):
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    con.execute(
        "INSERT INTO investigations (investigation_id, pack_id, cluster_id, title, status, opened_at, case_count, "
        "slice_category, slice_entity_2, slice_entity_3) VALUES (?, 'automotive_nhtsa', 1, 't', 'open', ?, ?, ?, 'HONDA', '')",
        [iid, now, cases, cat])


def test_board_orders_by_safety_then_cases(reset_ops_db, monkeypatch, no_meta):
    from fastapi.testclient import TestClient

    from src.ai_layer.safety import score_records
    from src.api.main import app
    from src.data.warehouse import ops_con

    with ops_con() as con:
        _inv(con, "inv_a", "PAINT", 50)
        _inv(con, "inv_b", "SERVICE BRAKES", 3)
        _inv(con, "inv_c", "AIR BAGS", 9)
    texts = {"SERVICE BRAKES": [{"text": "crashed into a wall"}], "AIR BAGS": [{"text": "noise"}], "PAINT": [{"text": "peeling"}]}
    monkeypatch.setattr("src.ai_layer.safety_board.slice_safety",
                        lambda pack, c, e2, e3, **k: score_records(c, texts[c]))
    monkeypatch.delenv("FRONTLINE_API_KEY", raising=False)
    r = TestClient(app).get("/api/ai/early-warning/safety")
    assert r.status_code == 200
    body = r.json()
    assert [i["investigation_id"] for i in body["investigations"]] == ["inv_b", "inv_c", "inv_a"]
    assert body["counts_by_level"] == {"high": 1, "elevated": 1, "normal": 1}
    assert "not a crash prediction" in body["method"]


def test_backtest_headline_missing_file_is_null(monkeypatch, tmp_path):
    from src.ai_layer import safety_board

    monkeypatch.setattr(safety_board, "BACKTEST_JSON", tmp_path / "nope.json")
    assert safety_board.backtest_headline() is None
    (tmp_path / "b.json").write_text('{"headline": {"hits": 3}}')
    monkeypatch.setattr(safety_board, "BACKTEST_JSON", tmp_path / "b.json")
    assert safety_board.backtest_headline() == {"hits": 3}
