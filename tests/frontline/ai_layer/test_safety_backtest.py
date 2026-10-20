"""Preventable-window counting in scripts/ai_safety_backtest.py.  [AI-LAYER]"""

from datetime import date

from scripts.ai_safety_backtest import _monday, render, safety_rows, summarise, window_counts


def _line(odino, make, comp, datea, crash="N", fire="N", injured="0", deaths="0", year="2018"):
    p = [""] * 51
    p[1], p[3], p[5], p[6], p[8], p[9], p[10], p[11], p[15] = odino, make, year, crash, fire, injured, deaths, comp, datea
    return p


def test_monday():
    assert _monday("2024-W01") == date(2024, 1, 1)
    assert _monday("bad") is None


def test_window_counts_dedupes_odino_and_respects_bounds():
    rows = [
        {"odino": "1", "datea": date(2024, 1, 2), "crash": True, "fire": False, "injured": 0, "deaths": 0},
        {"odino": "1", "datea": date(2024, 1, 2), "crash": False, "fire": False, "injured": 2, "deaths": 0},
        {"odino": "2", "datea": date(2024, 1, 5), "crash": False, "fire": True, "injured": 0, "deaths": 0},
        {"odino": "3", "datea": date(2024, 2, 1), "crash": True, "fire": False, "injured": 1, "deaths": 0},  # == end
        {"odino": "4", "datea": date(2023, 12, 31), "crash": True, "fire": False, "injured": 0, "deaths": 0},
    ]
    c, ids = window_counts(rows, date(2024, 1, 1), date(2024, 2, 1))
    assert ids == {"1", "2"}
    assert c == {"complaints": 2, "crash": 1, "fire": 1, "injury_complaints": 1, "injured_persons_reported": 2,
                 "death_complaints": 0, "deaths_reported": 0, "crash_fire_or_injury": 2}


def test_safety_rows_filters_make_year_slice():
    lines = [
        _line("1", "HONDA", "SERVICE BRAKES, HYDRAULIC", "20240102", crash="Y"),
        _line("2", "BMW", "SERVICE BRAKES, HYDRAULIC", "20240102"),
        _line("3", "HONDA", "SERVICE BRAKES, HYDRAULIC", "20240102", year="2010"),
        _line("4", "HONDA", "ENGINE", "20240102"),
        _line("5", "HONDA", "SERVICE BRAKES", "bad"),
    ]
    out = safety_rows(lines, {"HONDA"}, 2015, 2024, {("SERVICE BRAKES", "HONDA")})
    assert [r["odino"] for r in out[("SERVICE BRAKES", "HONDA")]] == ["1"]
    assert out[("SERVICE BRAKES", "HONDA")][0]["crash"] is True


def test_summarise_and_render_no_invented_numbers():
    hit = {"advisory_id": "24V001", "issued_at": "2024-02-01", "scope_category": "SERVICE BRAKES",
           "scope_entity_2": "HONDA", "scope_entity_3": None, "first_flagged_week": "2024-W01", "lead_weeks": 4}
    miss = {**hit, "advisory_id": "24V002", "lead_weeks": 0, "first_flagged_week": None}
    report = {"hits": [hit], "misses": [miss], "unconfirmed_count": 3}
    safety = {("SERVICE BRAKES", "HONDA"): [
        {"odino": "1", "datea": date(2024, 1, 3), "crash": True, "fire": False, "injured": 1, "deaths": 0}]}
    s = summarise(report, safety)
    assert (s["hits"], s["misses"], s["recall_rate"], s["lead_weeks_median"]) == (1, 1, 0.5, 4)
    assert s["preventable_window_totals_dedup"]["crash"] == 1
    assert s["hero"]["advisory_id"] == "24V001"
    prec = {"records": 10, "advisories": 2, "flagged_slice_weeks": 4, "confirmed_slice_weeks": 1,
            "false_alarm_slice_weeks": 3, "precision_slice_weeks": 0.25, "flag_episodes": 2,
            "confirmed_episodes": 1, "precision_episodes": 0.5}
    md = render({"makes": "HONDA", "year_min": 2015, "year_max": 2024, "model_grain": False,
                 "elapsed_s": 1, "command": "x"}, s, prec)
    assert "correlation" in md.lower() and "| Recalls missed | 1 |" in md
    assert "lives" not in md.lower()


def test_summarise_empty():
    s = summarise({"hits": [], "misses": [], "unconfirmed_count": 0}, {})
    assert s["recall_rate"] is None and s["lead_weeks_median"] is None and s["hero"] is None


def test_campaign_level_rollup():
    from scripts.ai_safety_backtest import campaign_level

    rows = [
        {"advisory_id": "24V1|2019|HONDA|CR-V|SERVICE BRAKES", "outcome": "hit", "lead_weeks": 3},
        {"advisory_id": "24V1|2020|HONDA|CR-V|SERVICE BRAKES", "outcome": "hit", "lead_weeks": 7},
        {"advisory_id": "24V2|2019|FORD|ESCAPE|ENGINE", "outcome": "miss", "lead_weeks": 0},
    ]
    c = campaign_level(rows)
    assert (c["campaigns"], c["hits"], c["misses"], c["lead_weeks_median"]) == (2, 1, 1, 7)


def _insert_week(con, prefix, day, n, *, category, entity_2, entity_3=""):
    for i in range(n):
        con.execute(
            "INSERT INTO records (record_id, occurred_at, received_at, entity_2, entity_3, category, text, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 'NHTSA')",
            [f"{prefix}-{day:%Y%m%d}-{i}", day, day, entity_2, entity_3, category, f"{prefix} note {i}"],
        )


def test_streaming_lead_report_matches_existing_function(tmp_path, monkeypatch):
    """The memory-bounded runner must give exactly what recall_lead_report gives."""
    from datetime import datetime, timedelta

    from scripts.ai_safety_backtest import lead_report_streaming
    from scripts.ingest_recalls import advisory_contract, insert_advisories
    from src.backtest.recall_lead import recall_lead_report
    from src.data.warehouse import apply_domain_schema, domain_con

    monkeypatch.setenv("DOMAIN_DB_PATH", str(tmp_path))
    pack = "ai_stream_eq"
    start = datetime(2018, 1, 8)
    with domain_con(pack, read_only=False) as con:
        apply_domain_schema(con)
        for w in range(12):
            day = start + timedelta(weeks=w)
            _insert_week(con, "hit", day, 2 if w not in (5, 9) else 20, category="SERVICE BRAKES", entity_2="HONDA")
            _insert_week(con, "lone", day, 2 if w < 7 else 18, category="STEERING", entity_2="HONDA")
        _insert_week(con, "late", datetime(2018, 6, 4), 20, category="ENGINE", entity_2="HONDA")
    rows = []
    for key, rc, comp in [("18V100|2018|HONDA|ACCORD|SERVICE BRAKES", "20180312", "SERVICE BRAKES"),
                          ("18V150|2018|HONDA|CIVIC|SERVICE BRAKES", "20180416", "SERVICE BRAKES"),
                          ("18V200|2018|HONDA|ACCORD|ENGINE", "20180115", "ENGINE"),
                          ("18V201|2018|HONDA|CR-V|ENGINE", "20180115", "ENGINE")]:
        rows.append({"ADVISORY_KEY": key, "RCDATE": rc, "YEARTXT": "2018", "MAKETXT": "HONDA",
                     "MODELTXT": key.split("|")[3], "COMPONENT": comp, "DESC_DEFECT": "d.",
                     "CORRECTIVE_ACTION": "r.", "RCL_URL": "https://www.nhtsa.gov/recalls"})
    insert_advisories(pack, rows, contract=advisory_contract("automotive_nhtsa"))
    expected = recall_lead_report(pack, run_backtests=False)
    for workers in (1, 2):
        got = lead_report_streaming(pack, workers=workers)
        for k in ("recalls", "hits", "misses", "unconfirmed", "hit_count", "miss_count", "unconfirmed_count"):
            assert got[k] == expected[k], (workers, k)
    assert expected["hit_count"] >= 1 and expected["miss_count"] >= 1


def test_recent_episode_picks_latest_run_within_window():
    from datetime import datetime

    from scripts.ai_safety_backtest import _recent_episode

    adv = {"scope_category": "SERVICE BRAKES", "scope_entity_2": "HONDA", "scope_entity_3": "CR-V"}
    mk = lambda wk: {"iso_week": wk, "category": "SERVICE BRAKES", "entity_2": "HONDA", "entity_3": "", "is_anomaly": True}
    series = [mk("2015-W10"), mk("2018-W01"), mk("2018-W02"), mk("2018-W03"), mk("2018-W10")]
    other = {"iso_week": "2018-W09", "category": "ENGINE", "entity_2": "HONDA", "entity_3": "", "is_anomaly": True}
    ep = _recent_episode(series + [other], adv, datetime(2018, 4, 2), model_grain=False)
    assert ep["episode_start_week"] == "2018-W10" and ep["last_flag_week"] == "2018-W10"
    assert ep["lead_weeks"] >= 1
    ep2 = _recent_episode(series[:4], adv, datetime(2018, 4, 2), model_grain=False)
    assert ep2["episode_start_week"] == "2018-W01"
    assert _recent_episode(series[:1], adv, datetime(2018, 4, 2), model_grain=False) is None  # > 52 weeks old


def test_summarise_recent_block_and_render():
    from datetime import date

    from scripts.ai_safety_backtest import render, summarise

    hit = {"advisory_id": "18V1|2018|HONDA|CR-V|SERVICE BRAKES", "issued_at": "2018-04-02", "scope_category": "SERVICE BRAKES",
           "scope_entity_2": "HONDA", "scope_entity_3": "CR-V", "first_flagged_week": "2015-W10", "lead_weeks": 160, "outcome": "hit"}
    miss = {**hit, "advisory_id": "18V2|2018|HONDA|CR-V|SERVICE BRAKES", "first_flagged_week": None, "lead_weeks": 0, "outcome": "miss"}
    report = {"recalls": [hit, miss], "hits": [hit], "misses": [miss], "unconfirmed_count": 0,
              "recent_episode": [{"episode_start_week": "2018-W10", "last_flag_week": "2018-W10", "lead_weeks": 4}, None]}
    safety = {("SERVICE BRAKES", "HONDA"): [
        {"odino": "1", "datea": date(2018, 3, 10), "crash": True, "fire": False, "injured": 1, "deaths": 0},
        {"odino": "2", "datea": date(2016, 1, 1), "crash": True, "fire": False, "injured": 0, "deaths": 0},
    ]}
    s = summarise(report, safety)
    r = s["recent"]
    assert (r["hits"], r["misses"], r["lead_weeks_median"]) == (1, 1, 4)
    assert r["preventable_window_totals_dedup"]["crash"] == 1  # the 2016 complaint is outside the episode window
    assert s["preventable_window_totals_dedup"]["crash"] == 2  # first-flag window includes it
    prec = {"records": 2, "advisories": 2, "flagged_slice_weeks": 1, "confirmed_slice_weeks": 1, "false_alarm_slice_weeks": 0,
            "precision_slice_weeks": 1.0, "flag_episodes": 1, "confirmed_episodes": 1, "precision_episodes": 1.0}
    md = render({"makes": "HONDA", "year_min": 2015, "year_max": 2024, "model_grain": False, "elapsed_s": 1, "command": "x"}, s, prec)
    assert "Actionable early warning" in md and "upper bound" in md
