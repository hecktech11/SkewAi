from __future__ import annotations

import pytest

from src.data.warehouse import domain_con, ops_con
from src.eval.shadow_pilot import (
    ShadowContact,
    ShadowPilotEvaluator,
    cohen_kappa_with_ci,
    wilson_score_interval,
)
from src.ids import new_ulid
from src.ml_runtime.entity_resolution import clean_spoken_vin, identity_key, is_valid_vin


def test_wilson_score_interval_bounds():
    # 95 successes out of 100
    p, low, high = wilson_score_interval(95, 100)
    assert p == 0.95
    assert 0.88 <= low < 0.95
    assert 0.95 < high <= 0.99
    assert 0.0 <= low <= high <= 1.0

    # 0 out of 0 returns 0
    assert wilson_score_interval(0, 0) == (0.0, 0.0, 0.0)


def test_cohen_kappa_with_ci_perfect():
    rater_a = ["Critical", "Medium", "High", "Low"] * 25
    rater_b = ["Critical", "Medium", "High", "Low"] * 25
    kappa, low, high = cohen_kappa_with_ci(rater_a, rater_b)
    assert kappa == 1.0
    assert low == 1.0
    assert high == 1.0


def test_cohen_kappa_with_ci_disagreement():
    rater_a = ["Critical"] * 50 + ["Low"] * 50
    rater_b = ["Low"] * 50 + ["Critical"] * 50
    kappa, low, high = cohen_kappa_with_ci(rater_a, rater_b)
    assert kappa < 0.0  # Perfect inverse agreement


def test_shadow_evaluator_green_pipeline():
    # Generate 100 high-accuracy shadow contacts
    contacts = []
    for i in range(100):
        contacts.append(
            ShadowContact(
                contact_id=f"contact_{i}",
                ai_slots={"entity_1": "2019", "entity_2": "HONDA", "entity_3": "CR-V", "category": "ENGINE"},
                human_slots={"entity_1": "2019", "entity_2": "HONDA", "entity_3": "CR-V", "category": "ENGINE"},
                ai_kill_switch_triggered=(i < 5),
                human_kill_switch_needed=(i < 5),
                ai_severity="High" if i < 20 else "Medium",
                human_severity="High" if i < 20 else "Medium",
                ai_cluster_id=10,
                ai_top_3_clusters=[10, 11, 12],
                engineer_verified_cluster_id=10,
                cost_usd=0.012,
            )
        )

    evaluator = ShadowPilotEvaluator(contacts)
    report = evaluator.run_full_evaluation(target_cost_usd=0.05)

    assert report["total_contacts"] == 100
    assert report["overall_verdict"] == "green"
    assert report["metrics"]["slots"]["category"]["rating"] == "green"
    assert report["metrics"]["kill_switch"]["recall"]["rating"] == "green"
    assert report["metrics"]["severity_agreement"]["rating"] == "green"
    assert report["metrics"]["cluster_agreement"]["top_1"]["rating"] == "green"


def test_shadow_evaluator_red_on_missed_kill_switch():
    contacts = []
    for i in range(100):
        # AI missed 3 safety events out of 5
        human_safety = i < 5
        ai_safety = i < 2
        contacts.append(
            ShadowContact(
                contact_id=f"contact_{i}",
                ai_slots={"category": "ENGINE"},
                human_slots={"category": "ENGINE"},
                ai_kill_switch_triggered=ai_safety,
                human_kill_switch_needed=human_safety,
                ai_severity="Medium",
                human_severity="Medium",
                cost_usd=0.012,
            )
        )

    evaluator = ShadowPilotEvaluator(contacts)
    report = evaluator.run_full_evaluation()
    # Missed safety events force verdict to RED
    assert report["metrics"]["kill_switch"]["recall"]["rating"] == "red"
    assert report["overall_verdict"] == "red"


def test_clean_spoken_vin_homophones():
    # Spoken VIN with words and forbidden 'O'/'I'
    # Real Honda VIN: 1HGCR2F83HA000000
    spoken = "one HG CR 2 F 8 3 H A O O O O O O"
    cleaned = clean_spoken_vin(spoken)
    assert "O" not in cleaned
    assert len(cleaned) == 17
    assert cleaned.endswith("000000")


@pytest.mark.asyncio
async def test_idiomatic_kill_switch_suppression(pack):
    from src.agents.base import InteractionContext
    from src.agents.intake import IntakeAgent

    ctx = InteractionContext(interaction_id="int_test", pack=pack)
    agent = IntakeAgent(ctx)

    # Figurative hyperbole should NOT escalate
    res = await agent.run(customer_turn="This gas price is killing me, and I had a heart attack when I saw the bill.")
    assert res.get("kill_switch") is None
    assert ctx.safety_flags.get("escalation") is None or ctx.safety_flags.get("escalation") is False

    # Actual emergency MUST still escalate
    res_real = await agent.run(customer_turn="There is smoke and fire coming from the engine!")
    assert res_real.get("kill_switch") in ("fire", "smoke")
    assert ctx.safety_flags.get("escalation") is True


@pytest.mark.asyncio
async def test_multi_issue_secondary_category_captured(reset_ops_db, orchestrator_factory, pack):
    orch, hooks = orchestrator_factory()
    await orch.start()

    # Customer mentions both brakes and engine in one utterance
    turn_res = await orch.handle_customer_turn("My 2019 Honda CR-V has a brake problem and an engine misfire.")
    slots = orch.ctx.slots

    # Primary category extracted
    assert slots.get("category") in ("SERVICE BRAKES", "ENGINE")
    # Secondary category captured in slots
    sec = slots.get("secondary_categories")
    assert sec is not None
    assert len(sec) >= 1
    assert "ENGINE" in sec or "SERVICE BRAKES" in sec

    orch.ctx.slots["entity_1"] = "2019"
    orch.ctx.slots["entity_2"] = "HONDA"
    orch.ctx.slots["entity_3"] = "CR-V"
    orch.ctx.slots["description"] = "Brake and engine misfire"
    orch.ctx.case_id = "case_" + new_ulid()

    await orch.hangup()

    # Verify secondary issue persisted in records table subcategory column
    with domain_con(pack.id, read_only=True) as con:
        row = con.execute(
            "SELECT category, subcategory FROM records WHERE record_id = ?",
            [f"FRONTLINE-{orch.ctx.interaction_id}"],
        ).fetchone()
        assert row is not None
        assert row[0] in ("SERVICE BRAKES", "ENGINE")
        assert row[1] is not None
        assert "ENGINE" in row[1] or "SERVICE BRAKES" in row[1]


def test_fleet_scan_emerging_novel_clusters(reset_ops_db, pack):
    from src.frontline.fleet_scan import run_fleet_scan

    # Seed 4 open novel candidates with the same category and entity_2
    with ops_con() as con:
        for i in range(4):
            con.execute(
                """
                INSERT INTO novel_candidates
                (novel_id, interaction_id, pack_id, category, entity_2, entity_3, top_score, status, created_at)
                VALUES (?, ?, 'automotive_nhtsa', 'ELECTRICAL SYSTEM', 'TESLA', 'MODEL 3', 0.5, 'open', CURRENT_TIMESTAMP)
                """,
                [f"nvl_{i}", f"int_{i}"],
            )

    report = run_fleet_scan("automotive_nhtsa", rebuild_clusters=False, alert=False)
    assert report["novel_open"] >= 4
    emerging = report.get("novel_emerging_clusters", [])
    assert len(emerging) >= 1
    top_cluster = emerging[0]
    assert top_cluster["category"] == "ELECTRICAL SYSTEM"
    assert top_cluster["entity_2"] == "TESLA"
    assert top_cluster["count"] >= 4


def test_shadow_pilot_batch_runner(tmp_path):
    """Phase 0 honesty: the batch harness executes real intake code, so the
    synthetic cohort scores RED (category 0.76, cluster top-1 below gate).
    The old assertion pinned the rng-simulated GREEN baseline — that file is
    tombstoned, not re-blessed. Structure, CIs, determinism and the honest
    meta flags are what this test now guards."""
    from src.frontline.shadow_pilot import format_scorecard_table, run_shadow_pilot

    out_file = str(tmp_path / "shadow_test.json")
    report = run_shadow_pilot(mode="batch", sample_size=50, out_path=out_file, seed=42)

    assert report["total_contacts"] == 50
    assert report["overall_verdict"] == "red"
    assert report["meta"]["ai_source"] == "deterministic_intake_code"
    assert report["meta"]["human_labels"] == "synthetic_scenario_assignment"
    assert report["meta"]["synthetic_self_labels"] is True
    assert "slots" in report["metrics"]
    assert "kill_switch" in report["metrics"]
    assert "severity_agreement" in report["metrics"]
    assert "cluster_agreement" in report["metrics"]
    assert "cost_per_contact" in report["metrics"]

    # Deterministic: same seed, same honest numbers across dual streams.
    assert report["streams"]["synthetic_templated"]["metrics"]["slots"]["category"]["point_estimate"] == 0.76
    assert report["metrics"]["slots"]["category"]["point_estimate"] == 0.06

    # Assert 95% Wilson intervals are bounded
    for slot_name, slot_res in report["metrics"]["slots"].items():
        assert 0.0 <= slot_res["ci_lower"] <= slot_res["point_estimate"] <= slot_res["ci_upper"] <= 1.0

    table_text = format_scorecard_table(report)
    assert "SHADOW PILOT EVALUATION SCORECARD" in table_text
    assert "[RED]" in table_text

    import json
    with open(out_file, "r") as f:
        data = json.load(f)
    assert data["total_contacts"] == 50


@pytest.mark.asyncio
async def test_telephony_latency_benchmark():
    from src.voice.benchmark_latency import TelephonyLatencyBenchmarker, format_benchmark_table

    bench = TelephonyLatencyBenchmarker(trials=15, seed=42)
    res = await bench.run_full_benchmark()

    assert res["overall_status"] == "PASS"
    assert "total_ttfa" in res["stages"]
    ttfa_stage = res["stages"]["total_ttfa"]
    assert ttfa_stage["p95_ms"] <= 550.0
    assert ttfa_stage["passed"] is True

    table = format_benchmark_table(res)
    assert "TELEPHONY LATENCY BENCHMARK" in table
    assert "[PASS]" in table


def test_category_composition_is_diagnostic_not_a_gate():
    """Matched-subset sits beside aggregate; residual UNKNOWN is corpus mix."""
    contacts = []
    for i in range(10):
        human = "ENGINE" if i < 6 else "UNKNOWN OR OTHER"
        ai = human if i != 0 else "STEERING"
        contacts.append(
            ShadowContact(
                contact_id=f"c{i}",
                ai_slots={"category": ai},
                human_slots={"category": human},
                ai_kill_switch_triggered=False,
                human_kill_switch_needed=False,
                ai_severity="Medium",
                human_severity="Medium",
                ai_cluster_id=40,
                ai_top_3_clusters=[40, 14, 22],
                engineer_verified_cluster_id=40,
                cost_usd=0.01,
            )
        )
    report = ShadowPilotEvaluator(contacts).run_full_evaluation(target_cost_usd=0.45)
    comp = report["metrics"]["category_composition"]
    assert comp["human_specific"] == 6
    assert comp["human_unknown"] == 4
    assert comp["matched_subset"]["detail"].startswith("5/6")
    assert comp["residual_unknown"]["detail"].startswith("4/4")
    assert "not a gate" in comp["matched_subset"]["target"]
    # Composition must not be folded into the slot ratings that drive the verdict.
    assert "category_matched_subset" not in report["metrics"]["slots"]


def test_empirical_category_meets_gate_on_existing_human_labels():
    """Category gate vs corpus labels. Does not invent severity/cluster."""
    from src.frontline.shadow_pilot import load_empirical_cohort

    contacts = load_empirical_cohort("data/safety_eval_corpus.jsonl")
    assert len(contacts) == 150
    ok = sum(
        1
        for c in contacts
        if (c.ai_slots.get("category") or "").strip()
        == (c.human_slots.get("category") or "").strip()
    )
    assert ok / len(contacts) >= 0.90
    assert all(not (c.human_severity or "").strip() for c in contacts)


def test_shadow_pilot_empirical_runner(tmp_path):
    from src.frontline.shadow_pilot import format_scorecard_table, run_shadow_pilot

    corpus_path = "data/safety_eval_corpus.jsonl"
    out_file = str(tmp_path / "shadow_empirical_test.json")
    report = run_shadow_pilot(
        mode="batch",
        sample_size=30,
        out_path=out_file,
        input_path=corpus_path,
    )

    assert report["total_contacts"] == 30
    assert report["meta"]["source_type"] == "empirical_ground_truth"
    assert report["meta"]["data_source"] == corpus_path
    assert report["metrics"]["kill_switch"]["recall"]["rating"] == "green"

    table_text = format_scorecard_table(report)
    assert "EMPIRICAL_GROUND_TRUTH" in table_text
