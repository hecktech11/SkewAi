"""Remediation regression tests — MEDIUM items 26-40."""

from __future__ import annotations

import pytest


# ── ITEM 26: embedding-fused cluster scoring ─────────────────────────────────


@pytest.mark.asyncio
async def test_item26_largest_cluster_does_not_auto_win(seed_automotive_pack):
    """A small entity-matching cluster must beat a large unrelated one."""
    from datetime import datetime, timedelta

    from src.agents.base import InteractionContext
    from src.agents.investigator import InvestigatorAgent
    from src.data.warehouse import domain_con
    from src.domains.loader import load_pack

    now = datetime(2024, 5, 1)
    with domain_con("automotive_nhtsa", read_only=False) as con:
        # Giant unrelated cluster (FORD electrical, 20 rows).
        con.execute(
            """INSERT INTO clusters (cluster_id, pack_id, top_terms, category,
               record_count, first_seen, last_seen)
               VALUES (1001, 'automotive_nhtsa', '["window", "regulator"]',
               'ELECTRICAL SYSTEM', 20, ?, ?)""",
            [now - timedelta(days=60), now - timedelta(days=1)],
        )
        for i in range(20):
            rid = f"GIANT-{i}"
            con.execute(
                """INSERT INTO records (record_id, occurred_at, received_at,
                   entity_2, entity_3, category, text, source)
                   VALUES (?, ?, ?, 'FORD', 'F-150', 'ELECTRICAL SYSTEM',
                   'power window regulator failure driver side', 'NHTSA')""",
                [rid, now - timedelta(days=30), now - timedelta(days=29)],
            )
            con.execute(
                "INSERT INTO cluster_assignments (record_id, cluster_id, distance)"
                " VALUES (?, 1001, 0.3)",
                [rid],
            )
        # Small matching cluster (HONDA brakes, 2 rows).
        con.execute(
            """INSERT INTO clusters (cluster_id, pack_id, top_terms, category,
               record_count, first_seen, last_seen)
               VALUES (1002, 'automotive_nhtsa', '["grinding", "brakes"]',
               'SERVICE BRAKES', 2, ?, ?)""",
            [now - timedelta(days=60), now - timedelta(days=1)],
        )
        for i in range(2):
            rid = f"SMALL-{i}"
            con.execute(
                """INSERT INTO records (record_id, occurred_at, received_at,
                   entity_2, entity_3, category, text, source)
                   VALUES (?, ?, ?, 'HONDA', 'CR-V', 'SERVICE BRAKES',
                   'front brake grinding noise when stopping', 'NHTSA')""",
                [rid, now - timedelta(days=30), now - timedelta(days=29)],
            )
            con.execute(
                "INSERT INTO cluster_assignments (record_id, cluster_id, distance)"
                " VALUES (?, 1002, 0.2)",
                [rid],
            )
    pack = load_pack("automotive_nhtsa", reload=True)
    ctx = InteractionContext(
        interaction_id="int_fused", pack=pack, channel="web_text"
    )
    ctx.slots.update({
        "category": "SERVICE BRAKES", "entity_2": "HONDA", "entity_3": "CR-V",
        "description": "front brake grinding noise when stopping my Honda",
    })
    res = await InvestigatorAgent(ctx).run()
    brief = ctx.investigation_brief
    assert brief and brief.get("cluster_id") != 1001, (
        f"largest cluster must not win on size alone: {brief.get('cluster_id')}"
    )
    assert brief.get("cluster_id") in (14, 1002), (
        f"entity+semantic fit must win: {brief.get('cluster_id')}"
    )


# ── ITEM 27: scoped live trend ───────────────────────────────────────────────


def test_item27_trend_scoped_to_category_and_entity(reset_ops_db, seed_automotive_pack):
    from datetime import timedelta

    from src.data.timeutil import utc_now
    from src.data.warehouse import domain_con, ops_con
    from src.qubot.retrievers import live_risk

    # Two live cases on different clusters/slices.
    with ops_con() as con:
        for cid, cat in (("case_t27_a", "SERVICE BRAKES"), ("case_t27_b", "AIR BAGS")):
            con.execute(
                """INSERT INTO cases (case_id, interaction_id, pack_id, created_at,
                   category, description_summary, onset, severity, severity_source,
                   priority, safety_flags, status, cluster_match_id)
                   VALUES (?, 'int_t27', 'automotive_nhtsa', ?, ?, 'd', ?, 'Medium',
                   'rules', 2, '{}', 'open', ?)""",
                [cid, utc_now(), cat, utc_now(), 14 if cat == "SERVICE BRAKES" else 22],
            )
    risks = live_risk(30)
    by_cid = {int(r["cluster_id"]): r for r in risks}
    assert 14 in by_cid and 22 in by_cid
    # Each cluster gets its OWN slice — never one global trend.
    assert by_cid[14]["trend_scope"] in ("category+entity_2", "category-only-fallback")
    assert by_cid[14].get("trend_entity_2") == "HONDA"
    weeks_a = {(w.get("iso_week"), w.get("category")) for w in by_cid[14]["weekly_trend"]}
    assert weeks_a, "expected trend rows"
    assert all(c == "SERVICE BRAKES" for _, c in weeks_a), (
        f"FORD/AIR rows must not leak into the HONDA trend: {weeks_a}"
    )
    # as_of caps the series for reproducible historical reads.
    capped = live_risk(30, as_of="2020-W01")
    for r in capped:
        for w in r["weekly_trend"]:
            assert str(w.get("iso_week")) <= "2020-W01"


# ── ITEM 28: sentiment ───────────────────────────────────────────────────────


def test_item28_adversarial_lexicon():
    from src.agents.sentiment import score_text

    # Substring false positive: 'media' must not fire inside 'immediate'.
    assert score_text("I need immediate help with my account") < 0.3
    # Intensifier alone is not sentiment.
    assert score_text("this is very very interesting") < 0.3
    # Intensifier multiplies a real hit (word-boundary).
    assert score_text("I am very furious about this") >= score_text("I am furious about this")
    # Profanity proxy: 2+ consecutive @/# only.
    assert score_text("my email is jane@example.com") < 0.3
    assert score_text("see case #123 for details") < 0.3
    assert score_text("what the @# is going on here") > score_text("what is going on here")
    # Peak is max, trigger is rolling (documented compat).
    assert score_text("I AM FURIOUS AND THIS IS UNACCEPTABLE!!!") >= 0.9


# ── ITEM 29: followup ordering (already correct — pin it) ────────────────────


@pytest.mark.asyncio
async def test_item29_case_id_first_and_honest_investigation_language(orchestrator_factory):
    orch, _ = orchestrator_factory()
    await orch.start()
    await orch.handle_customer_turn("My 2019 Honda CR-V grinds when I brake.")
    await orch._move_to_closing()
    assert orch.ctx.case_id and orch.ctx.case_id.startswith("case_")
    from src.data.warehouse import ops_con

    with ops_con(read_only=True) as con:
        row = con.execute(
            "SELECT followup_draft, investigation_id FROM cases WHERE case_id = ?",
            [orch.ctx.case_id],
        ).fetchone()
    assert row and orch.ctx.case_id in (row[0] or ""), "draft must cite the allocated id"
    if "under active investigation" in (row[0] or "").lower():
        assert row[1], "'under investigation' requires a linked investigation_id"


# ── ITEM 30: grounded NL ─────────────────────────────────────────────────────


def test_item30_grounding_adversarial(seed_automotive_pack):
    from src.enterprise.grounded_nl import _like_escape, grounded_query, plan_query

    # Wildcards are escaped, not stripped (stripping widens the match).
    assert _like_escape("100%_x") == "100\\%\\_x"
    # No hardcoded brake fallback: gibberish yields no conjured rows.
    out = grounded_query("zxqv kwyj plugh", pack_id="automotive_nhtsa")
    assert out["shown"] == [], "unparseable questions must not conjure brake rows"
    # %/_ in the question cannot widen into a match-all.
    out2 = grounded_query("%_%", pack_id="automotive_nhtsa")
    assert out2["shown"] == []
    # Real query grounds to real spans.
    out3 = grounded_query("brake grinding noise Honda", pack_id="automotive_nhtsa")
    assert out3["shown"], "expected grounded rows"
    for item in out3["shown"]:
        claim = item["claim"]["claim_text"]
        assert len(claim) >= 8
        assert claim.lower() in item["text"].lower()
    plan = plan_query("brake grinding")
    assert "brake" in (plan.get("tokens") or [])


# ── ITEM 31: distributed close claims ────────────────────────────────────────


def test_item31_close_claim_single_owner(reset_ops_db):
    from src.jobs.registry import acquire_close_claim, close_claim_owner

    assert acquire_close_claim("int_claim_1", "worker-a") is True
    assert acquire_close_claim("int_claim_1", "worker-b") is False, (
        "second worker must stand down, not open a duplicate case"
    )
    assert close_claim_owner("int_claim_1") == "worker-a"
    # Same owner re-acquiring is idempotent.
    assert acquire_close_claim("int_claim_1", "worker-a") is True


@pytest.mark.asyncio
async def test_item31_concurrent_close_single_case(orchestrator_factory):
    import asyncio

    from src.data.warehouse import ops_con

    orch, _ = orchestrator_factory()
    await orch.start()
    await orch.handle_customer_turn("My 2019 Honda CR-V grinds when I brake.")
    await asyncio.gather(*[orch._move_to_closing() for _ in range(5)])
    assert orch.ctx.case_id
    with ops_con(read_only=True) as con:
        n = con.execute(
            "SELECT COUNT(*) FROM cases WHERE interaction_id = ?",
            [orch.ctx.interaction_id],
        ).fetchone()[0]
    assert n == 1
    assert orch.ctx.state == "DONE"


# ── ITEM 32: postgres backend ────────────────────────────────────────────────


def test_item32_pg_ddl_and_fail_closed(monkeypatch):
    from src.data import postgres_backend as pg

    ddl = pg.PG_PRODUCTION_DDL
    assert "CREATE EXTENSION IF NOT EXISTS vector" in ddl
    assert "fts_main_records" in ddl
    assert "vector(512)" in ddl
    assert "weekly_rollups" in ddl
    assert "USING GIN (fts)" in ddl
    assert "USING ivfflat" in ddl
    # Migration file mirrors the embedded DDL.
    from pathlib import Path

    from src.config import REPO_ROOT

    mig = (REPO_ROOT / "migrations" / "002_pg_vector_fts.sql").read_text()
    assert "fts_main_records" in mig and "vector(512)" in mig
    # Semantic search validates dims loudly.
    import pytest

    class _FakeConn:
        pass

    with pytest.raises(ValueError, match="512"):
        pg.semantic_search_records(_FakeConn(), [0.1] * 64)
    with pytest.raises(ValueError, match="512"):
        pg.semantic_search_records(_FakeConn(), [])
    assert pg.fts_search_records(_FakeConn(), "   ") == []
    # Production-like + broken DSN must raise, never silently use DuckDB.
    monkeypatch.setenv("FRONTLINE_OPS_DSN", "postgresql://bogus:5432/x")
    monkeypatch.setenv("PILOT_HARDENED", "1")
    with pytest.raises(Exception):
        with pg.ops_connection():
            pass
    st = pg.backend_status()
    assert st["ops_backend"] == "postgres" and st["vector_dim"] == 512


def test_postgres_backend_parameter_bindings():
    from src.data import postgres_backend as pg

    calls = []

    class MockCursor:
        def __init__(self):
            self.description = [
                ("record_id",),
                ("category",),
                ("entity_2",),
                ("entity_3",),
                ("text",),
                ("sim",),
            ]

        def execute(self, sql, params):
            calls.append((sql, params))

        def fetchall(self):
            return []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class MockConn:
        def cursor(self):
            return MockCursor()

    # Test semantic_search_records without category
    vec = [0.1] * 512
    pg.semantic_search_records(MockConn(), vec, limit=5)
    _, params = calls[-1]
    assert params == [vec, vec, 5]

    # Test semantic_search_records with category
    pg.semantic_search_records(MockConn(), vec, category="safety", limit=10)
    _, params = calls[-1]
    assert params == [vec, "safety", vec, 10]

    # Test fts_search_records without category
    pg.fts_search_records(MockConn(), "fuel leak", limit=15)
    _, params = calls[-1]
    assert params == ["fuel leak", "fuel leak", 15]

    # Test fts_search_records with category
    pg.fts_search_records(MockConn(), "fuel leak", category="safety", limit=20)
    _, params = calls[-1]
    assert params == ["fuel leak", "fuel leak", "safety", 20]


# ── ITEM 33: forecast statistics ─────────────────────────────────────────────


def test_item33_ols_ci_and_gates():
    from src.frontline.analytics import _t_crit_95, project_next_week_volume

    # OLS slope over the whole series (not last-delta): [2,4,6,8] -> 2.0.
    p = project_next_week_volume([2, 4, 6, 8, 10])
    assert p["method"] == "ols"
    assert p["slope_per_week"] == pytest.approx(2.0)
    assert p["slope_se"] is not None
    assert p["t_crit_95"] == pytest.approx(3.182, rel=0.01)  # df = 5-2 = 3
    assert p["ci_low"] <= p["projected_next_week"] <= p["ci_high"]
    # Flat data: slope ~0, wide honest interval.
    f = project_next_week_volume([5, 5, 5, 5, 5])
    assert f["slope_per_week"] == pytest.approx(0.0)
    # N<4 refused.
    assert project_next_week_volume([1, 2, 3])["method"] == "insufficient_history"
    assert _t_crit_95(1) == pytest.approx(12.706, rel=0.01)
    assert _t_crit_95(100) == pytest.approx(1.96 + 1.0 / 400.0)


def test_item33_breach_first_and_pack_costs(reset_ops_db, pack):
    from datetime import timedelta

    from src.data.timeutil import utc_now
    from src.data.warehouse import ops_con
    from src.frontline.analytics import financial_impact, forecast_cluster_volume

    now = utc_now()
    for i, days in enumerate((21, 14, 7, 0)):
        cid = f"case_br_{i}"
        with ops_con() as con:
            con.execute(
                """INSERT INTO cases (case_id, interaction_id, pack_id, created_at,
                   category, description_summary, onset, severity, severity_source,
                   priority, safety_flags, status, cluster_match_id)
                   VALUES (?, 'int_br', ?, ?, 'SERVICE BRAKES', 'd', ?, 'Critical',
                   'rules', 1, '{}', 'open', 77)""",
                [cid, pack.id, now - timedelta(days=days), now - timedelta(days=days)],
            )
    fc = forecast_cluster_volume(pack_id=pack.id, window_days=28, threshold=1)
    row = next(r for r in fc["forecasts"] if int(r["cluster_id"]) == 77)
    assert row["status"] == "breached", "breached threshold is never 'flat'"
    assert row["projected_weeks_to_threshold"] == 0.0
    fin = financial_impact(pack_id=pack.id)
    assert fin["cost_source"] == "pack.yaml cost_model"
    assert fin["cost_per_case"] == 250.0
    assert "exposure_multiplier" in fin


# ── ITEM 34: cross-pack ontology ─────────────────────────────────────────────


def test_item34_concept_mapping(reset_ops_db, pack):
    from src.data.timeutil import utc_now
    from src.data.warehouse import ops_con
    from src.domains.concepts import concept_for, is_cross_pack_concept
    from src.frontline.analytics import cross_pack_patterns

    assert concept_for(
        "automotive_nhtsa", "AIR BAGS", {"groups": [{"name": "Safety equipment", "categories": ["AIR BAGS"]}]}
    ) == "safety_security"
    assert concept_for(
        "finance_cfpb", "Fraud or scam", {"groups": [{"name": "Fraud & Identity", "categories": ["Fraud or scam"]}]}
    ) == "safety_security"
    # Genuinely unrelated categories never merge.
    assert concept_for("automotive_nhtsa", "ENGINE", None) != concept_for(
        "finance_cfpb", "Incorrect charges", None
    )
    assert is_cross_pack_concept("safety_security") is True
    assert is_cross_pack_concept("automotive_nhtsa:engine") is False

    now = utc_now()
    with ops_con() as con:
        for i in range(2):
            con.execute(
                """INSERT INTO cases (case_id, interaction_id, pack_id, created_at,
                   category, description_summary, onset, severity, severity_source,
                   priority, safety_flags, status)
                   VALUES (?, 'int_cp', ?, ?, ?, 'd', ?, 'Medium', 'rules', 2, '{}', 'open')""",
                [f"case_cp_a{i}", "automotive_nhtsa", now, "AIR BAGS", now],
            )
            con.execute(
                """INSERT INTO cases (case_id, interaction_id, pack_id, created_at,
                   category, description_summary, onset, severity, severity_source,
                   priority, safety_flags, status)
                   VALUES (?, 'int_cp', ?, ?, ?, 'd', ?, 'Medium', 'rules', 2, '{}', 'open')""",
                [f"case_cp_f{i}", "finance_cfpb", now, "Fraud or scam", now],
            )
        con.execute(
            """INSERT INTO cases (case_id, interaction_id, pack_id, created_at,
               category, description_summary, onset, severity, severity_source,
               priority, safety_flags, status)
               VALUES ('case_cp_u', 'int_cp', 'automotive_nhtsa', ?, 'ENGINE', 'd',
               ?, 'Medium', 'rules', 2, '{}', 'open')""",
            [now, now],
        )
    out = cross_pack_patterns()
    assert out["count"] >= 1
    safety = next(p for p in out["patterns"] if p["concept"] == "safety_security")
    assert safety["pack_count"] == 2
    assert set(safety["packs"]) == {"automotive_nhtsa", "finance_cfpb"}


# ── ITEM 35: elicitation wiring ──────────────────────────────────────────────


def test_item35_answer_lifecycle(reset_ops_db, pack):
    from src.frontline.elicitation import (
        answers_for_interaction,
        next_question,
        record_elicitation_answer,
        register_question,
    )

    q = register_question(
        pack.id, "Does the grinding happen when cold?",
        category="SERVICE BRAKES",
    )
    first = next_question(
        pack.id, {"category": "SERVICE BRAKES"}, interaction_id="int_el1"
    )
    assert first and first["question_id"] == q["question_id"]
    ans = record_elicitation_answer(
        "int_el1", q["question_id"], "Yes, only on cold mornings",
        prompt=q["prompt"],
    )
    assert ans["answer"].startswith("Yes")
    # No duplicate questioning after an answer.
    assert next_question(
        pack.id, {"category": "SERVICE BRAKES"}, interaction_id="int_el1"
    ) is None or next_question(
        pack.id, {"category": "SERVICE BRAKES"}, interaction_id="int_el1"
    )["question_id"] != q["question_id"]
    stored = answers_for_interaction("int_el1")
    assert len(stored) == 1 and stored[0]["prompt"] == q["prompt"]


@pytest.mark.asyncio
async def test_item35_brief_and_hypotheses(orchestrator_factory):
    from src.frontline.elicitation import register_question

    orch, hooks = orchestrator_factory()
    pack_id = orch.ctx.pack.id
    q = register_question(pack_id, "Does the grinding happen when cold?")
    await orch.start()
    # Pre-fill slots and mark safety questions asked so the diagnostic path
    # triggers (mirrors the established elicitation test pattern).
    orch.ctx.slots.update({
        "entity_1": "2019", "entity_2": "HONDA", "entity_3": "CR-V",
        "category": "SERVICE BRAKES", "description": "grinding when braking",
    })
    n_safety = len(orch.ctx.pack.manifest.safety.safety_questions or [])
    orch.ctx.slots["__safety_questions_asked__"] = ",".join(
        str(i) for i in range(n_safety)
    )
    await orch.handle_customer_turn("still grinding this morning")
    # Answer the pending diagnostic question, then finish the contact.
    pending = (orch.ctx.slots or {}).get("__diag_qid")
    assert pending, "expected a diagnostic question to be asked"
    await orch.handle_customer_turn("Yes, only on cold mornings")
    for _ in range(14):
        if orch.ctx.state == "DONE":
            break
        await orch.handle_customer_turn("Additional detail for the record.")
    if orch.ctx.state != "DONE":
        await orch._move_to_closing()
    brief = orch.ctx.investigation_brief or {}
    answers = brief.get("elicitation_answers") or []
    assert any("cold mornings" in (a.get("answer") or "") for a in answers), (
        "brief must carry the elicitation answer"
    )
    # Console card surfaced.
    assert any(
        a.get("action_type") == "diagnostic_answered" for a in hooks.activities
    )


# ── ITEM 36: intercept threshold ─────────────────────────────────────────────


def test_item36_in_flight_tips_threshold(tmp_path, monkeypatch):
    from datetime import datetime, timedelta

    from src.data.warehouse import apply_domain_schema, domain_con
    from src.frontline.live_intercept import slice_is_anomalous
    from src.ml_runtime.anomalies import iso_week_label

    monkeypatch.setenv("DOMAIN_DB_PATH", str(tmp_path))
    pack = "intercept_pack"
    now = datetime.now()
    with domain_con(pack, read_only=False) as con:
        apply_domain_schema(con)
        # 4 quiet weeks (2 each) + current week with 4 pre-existing.
        for w in range(4):
            day = now - timedelta(weeks=4 - w)
            for i in range(2):
                con.execute(
                    """INSERT INTO records (record_id, occurred_at, received_at,
                       entity_2, category, text, source)
                       VALUES (?, ?, ?, 'HONDA', 'SERVICE BRAKES', 'q', 'NHTSA')""",
                    [f"Q-{w}-{i}", day, day],
                )
        for i in range(4):
            con.execute(
                """INSERT INTO records (record_id, occurred_at, received_at,
                   entity_2, category, text, source)
                   VALUES (?, ?, ?, 'HONDA', 'SERVICE BRAKES', 'q', 'NHTSA')""",
                [f"C-{i}", now, now],
            )
    db_only = slice_is_anomalous(pack, "SERVICE BRAKES", "HONDA", in_flight=False)
    assert db_only["in_flight_counted"] is False
    # DB-only must not persist phantom rows.
    with domain_con(pack) as con:
        n_db = con.execute("SELECT COUNT(*) FROM records").fetchone()[0]
    assert n_db == 12
    live = slice_is_anomalous(pack, "SERVICE BRAKES", "HONDA", in_flight=True)
    assert live["in_flight_counted"] is True
    with domain_con(pack) as con:
        assert con.execute("SELECT COUNT(*) FROM records").fetchone()[0] == n_db


# ── ITEM 37: fix loop ────────────────────────────────────────────────────────


def test_item37_fix_loop_metrics_and_board(reset_ops_db, pack):
    from datetime import timedelta

    from src.data.timeutil import utc_now
    from src.data.warehouse import ops_con
    from src.enterprise.fix_effectiveness import record_fix
    from src.frontline.ops import ops_metrics

    m0 = ops_metrics()
    assert "fix_loop" in m0
    assert {"fixes_recorded", "resolved", "reopened", "reopen_rate", "series"} <= set(
        m0["fix_loop"]
    )
    record_fix(
        pack_id=pack.id, fixed_at=utc_now() - timedelta(days=40),
        category="SERVICE BRAKES", note="test fix",
    )
    m1 = ops_metrics()
    assert m1["fix_loop"]["fixes_recorded"] == m0["fix_loop"]["fixes_recorded"] + 1
    # Board renders the fix loop distinctly from open/closed counts.
    from pathlib import Path

    from src.config import REPO_ROOT

    board = (REPO_ROOT / "dashboard" / "routes" / "EarlyWarningBoard.jsx").read_text()
    assert "Fix loop" in board and "fix_loop" in board and "before-after" in board


# ── ITEM 38: locker completeness ─────────────────────────────────────────────


def test_item38_airgapped_completeness(reset_ops_db, seed_automotive_pack, pack):
    import copy

    from src.data.warehouse import ops_con
    from src.ledger import AgentAction, record_action
    from src.qubot.locker import (
        build_locker_bundle,
        sign_locker_bundle,
        verify_bundle_completeness,
        verify_locker_bundle,
    )

    iid = "int_airgap"
    with ops_con() as con:
        con.execute(
            """INSERT INTO interactions (interaction_id, pack_id, pack_version,
               started_at, channel, status) VALUES (?, 'automotive_nhtsa', '1',
               CURRENT_TIMESTAMP, 'web_text', 'completed')""",
            [iid],
        )
    record_action(AgentAction(
        interaction_id=iid, agent="orchestrator",
        action_type="interaction_started", output_summary="created",
    ))
    bundle = sign_locker_bundle(build_locker_bundle(iid))
    # Pure offline check (no DB): complete bundle verifies.
    assert verify_bundle_completeness(bundle)["ok"] is True
    assert verify_locker_bundle(bundle)["ok"] is True
    # Missing file/section fails.
    slim = copy.deepcopy(bundle)
    del slim["snapshots"]
    slim.pop("signature", None)
    slim.pop("public_key_pem", None)
    from src.qubot.locker import sign_locker_bundle as _sign

    slim = _sign({k: v for k, v in slim.items()})
    assert verify_bundle_completeness(slim)["ok"] is False
    # Modified metadata fails.
    meta = copy.deepcopy(bundle)
    meta["merkle_proof"]["anchor_head"]["leaf_count"] += 3
    assert verify_bundle_completeness(meta)["ok"] is False
    # Invalid external root fails.
    badroot = copy.deepcopy(bundle)
    badroot["merkle_proof"]["anchor_head"]["root"] = "0" * 64
    assert verify_bundle_completeness(badroot)["ok"] is False


# ── ITEM 39: rate limits ─────────────────────────────────────────────────────


@pytest.mark.timeout(90)
def test_item39_rate_limits_present(reset_ops_db, seed_automotive_pack, monkeypatch):
    from fastapi.testclient import TestClient

    from src.api import main as main_mod
    from src.api.main import app

    src = (
        (main_mod.__file__ and open(main_mod.__file__.replace("main.py", "routes/frontline.py")).read())
        or ""
    )
    for path in ("/audits", "/audits/{interaction_id}", "/explain/{interaction_id}"):
        assert path in src
    # Endpoint-specific limits are declared (not just the global default).
    assert src.count("@limiter.limit") >= 8
    assert "60 per minute" in src and "10 per minute" in src
    # Interacions detail + explain are limited too.
    import src.api.routes.interactions as inter_mod

    isrc = open(inter_mod.__file__).read()
    assert "@limiter.limit" in isrc
    # A burst over the audits limit is rejected (429), not served.
    monkeypatch.delenv("FRONTLINE_API_KEY", raising=False)
    monkeypatch.delenv("FRONTLINE_OPEN_MODE", raising=False)
    with TestClient(app) as c:
        statuses = [c.get("/api/frontline/audits").status_code for _ in range(70)]
    assert 429 in statuses, "expected the per-endpoint limit to trip"


# ── ITEM 40: stub capabilities ───────────────────────────────────────────────


def test_item40_stubs_not_sellable(reset_ops_db):
    from src.frontline.billing import PLAN_TIERS, plan_includes, usage_dashboard
    from src.frontline.booking import offer_for_advisory
    from src.frontline.capabilities import (
        is_sellable,
        sellable_capabilities,
        stub_capabilities,
    )

    assert set(stub_capabilities()) == {"booking", "biometrics"}
    for plan in PLAN_TIERS:
        assert plan_includes(plan, "booking") is False
        assert plan_includes(plan, "biometrics") is False
        assert plan_includes(plan, "cases") is True
    assert "booking" not in sellable_capabilities()
    offer = offer_for_advisory(
        advisory_id="19V-12345", case_id="case_x",
        pack_id="automotive_nhtsa", interaction_id="int_x",
    )
    assert offer["availability"] == "stub"
    assert "specialist" in offer["prompt"], "offer must not read as live booking"
    dash = usage_dashboard("default")
    caps = {c["capability"]: c["availability"] for c in dash["capabilities"]}
    assert caps["booking"] == "stub" and caps["cases"] == "production"
