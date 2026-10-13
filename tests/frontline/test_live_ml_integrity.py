"""Embedding identity, per-query retrieval metrics, and time-causal alerts."""

from __future__ import annotations

import os

import pytest

from src.agents.base import InteractionContext
from src.agents.intake import IntakeAgent
from src.agents.sentiment import score_signals, score_text
from src.data.warehouse import domain_con
from src.eval.embedding_gates import semantic_promotion_allowed
from src.ml_runtime.anomalies import score_weekly_slices
from src.ml_runtime.association import association_score, merge_candidate_rows
from src.ml_runtime.category_zeroshot import (
    ZeroShotCategoryResult,
    accepted_label_calibration,
    extract_category_with_mode,
    get_category_centroids,
    reset_category_centroids_cache,
)
from src.ml_runtime.embedding_eval import per_class_scorecard
from src.ml_runtime.embedding_match import rank_semantic
from src.ml_runtime.embedding_runtime import reset_embedding_runtime
from src.ml_runtime.embedding_space import compare_embeddings
from src.ml_runtime.embedding_store import get_embedding, upsert_embedding
from src.ml_runtime.embeddings import transformation_identity
from src.ml_runtime.onnx_embedder import ToySemanticEmbedder


def test_stored_vector_round_trip_and_semantic_retrieval(monkeypatch, pack):
    monkeypatch.setenv("FRONTLINE_EMBEDDING_TOY", "1")
    reset_embedding_runtime()
    embedder = ToySemanticEmbedder()
    text = "brake grinding smoke"
    written = embedder.embed(text)
    with domain_con(pack.id, read_only=True) as con:
        record_id = con.execute("SELECT record_id FROM records LIMIT 1").fetchone()[0]
    upsert_embedding(pack.id, str(record_id), written)
    loaded = get_embedding(pack.id, str(record_id), written.embedding_version)
    assert loaded is not None
    assert loaded.model_key == written.model_key
    assert loaded.padding_strategy == written.padding_strategy
    cosine = compare_embeddings(loaded, embedder.embed(text))
    assert cosine > 0.99
    ranked = rank_semantic(
        pack.id,
        text,
        [{"record_id": str(record_id), "text": text}],
    )
    assert ranked["status"] == "performed"
    assert ranked["ranked"]
    reset_embedding_runtime()


def test_query_level_scorecard_and_promotion_gate(monkeypatch, pack):
    embedder = ToySemanticEmbedder()
    pairs = [
        {
            "eval_class": "brakes",
            "query": "brakes failed",
            "candidate": "brakes failed badly",
            "label": "positive",
        },
        {
            "eval_class": "brakes",
            "query": "brakes failed",
            "candidate": "sunny day picnic",
            "label": "hard_negative",
        },
        {
            "eval_class": "brakes",
            "query": "engine stall",
            "candidate": "engine stalled cold",
            "label": "positive",
        },
        {
            "eval_class": "brakes",
            "query": "engine stall",
            "candidate": "sunny picnic afternoon",
            "label": "negative",
        },
    ]
    card = per_class_scorecard(embedder, pairs)["brakes"]
    assert card["query_level"] is True
    assert card["recall_at_5"] == 1.0
    assert card["ndcg_at_5"] is not None
    assert card["ndcg"] == card["ndcg_at_5"]
    assert card["hard_negative_performance"] == 1.0
    lo, hi = card["query_recall_ci95"]
    assert lo is not None and hi is not None
    assert lo <= card["recall_at_5"] <= hi

    version = embedder.version
    monkeypatch.setenv("FRONTLINE_EMBEDDING_TOY", "1")
    assert semantic_promotion_allowed(
        pack.id,
        version,
        dataset_id="pairs_v1",
        thresholds={"ndcg_at_5": 0.0, "recall_at_5": 0.0},
        model_key="toy-semantic-v1",
    ) is False
    from src.ml_runtime.cluster_builds import approve_retrieval_scorecard

    approve_retrieval_scorecard(
        pack.id,
        embedding_version=version,
        model_key="toy-semantic-v1",
        dataset_id="pairs_v1",
        thresholds={"ndcg_at_5": 0.0, "recall_at_5": 0.0},
        scorecard={"approved": True, "recall_at_5": 1.0, "ndcg_at_5": card["ndcg_at_5"]},
    )
    assert semantic_promotion_allowed(
        pack.id,
        version,
        dataset_id="pairs_v1",
        thresholds={"ndcg_at_5": 0.0, "recall_at_5": 0.0},
        model_key="toy-semantic-v1",
    ) is True


def test_symptom_text_changes_union_rank():
    lexical = [{"record_id": "lex", "text": "unrelated window rattle", "category": "VISIBILITY"}]
    vector = [{"record_id": "vec", "text": "brake grinding on cold mornings", "category": "BRAKES"}]
    entity = [{"record_id": "lex", "text": "duplicate", "category": "VISIBILITY"}]
    merged = merge_candidate_rows(entity, lexical, vector)
    assert [row["record_id"] for row in merged] == ["lex", "vec"]
    population = merged
    query = {"category": "BRAKES", "text": "brake grinding"}
    grinding = association_score(query, vector[0], population)
    window = association_score(query, lexical[0], population)
    assert grinding > window


def test_identity_is_never_the_string_unknown(pack):
    ident = transformation_identity(pack.id)
    assert ident
    assert "unknown" not in ident.lower()

    class _Gaz:
        values = ("BRAKES",)
        canonical = {"brakes": "BRAKES"}

    class _Pack:
        id = "centroid-probe"
        taxonomy = {"groups": []}

        def gazetteer_for_slot(self, name):
            return _Gaz()

    class _Emb:
        def embed(self, text):
            return ToySemanticEmbedder().embed(text or "brakes")

    reset_category_centroids_cache()
    centroids = get_category_centroids(_Pack(), _Emb())
    assert centroids.canonical_version == "unversioned-provider"
    assert centroids.canonical_version != "unknown"


def test_disagreement_is_not_recorded_as_model_error(monkeypatch, pack):
    monkeypatch.setenv("FRONTLINE_SLM_MODE", "live")
    captured = {}

    def _predict(text, pack_obj, **kwargs):
        return ZeroShotCategoryResult(
            category="ENGINE",
            top1_category="ENGINE",
            top1_score=0.9,
            top2_category="SERVICE BRAKES",
            top2_score=0.1,
            margin=0.8,
            floor=0.2,
            margin_gate=0.05,
            passed_floor=True,
            passed_margin=True,
            is_member=True,
            latency_ms=1.0,
            extraction_source="zeroshot",
        )

    def _record(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        "src.ml_runtime.category_zeroshot.predict_category_zero_shot", _predict
    )
    monkeypatch.setattr(
        "src.ml_runtime.category_zeroshot.record_category_shadow_comparison", _record
    )
    ctx = InteractionContext(interaction_id="int_cat", pack=pack)
    category, source, _meta = extract_category_with_mode(
        "the brakes are grinding", ctx, "SERVICE BRAKES"
    )
    assert category == "ENGINE"
    assert source == "zeroshot"
    assert captured["agreement"] is False
    assert captured["confident_wrong"] is False


def test_live_intake_keeps_category_abstention(monkeypatch, pack):
    def _abstain(text, ctx, rules_cat):
        return None, "none", {}

    monkeypatch.setattr(
        "src.ml_runtime.category_zeroshot.extract_category_with_mode", _abstain
    )
    ctx = InteractionContext(interaction_id="int_abstain", pack=pack)
    slot = next(item for item in pack.required_slots() if item.name == "category")
    value = IntakeAgent(ctx)._extract_slot(slot, "the brakes are grinding badly")
    assert value is None
    assert ctx.slots.get("__category_abstained__") == "1"
    assert ctx.slots.get("category") in (None, "")


def test_legal_words_are_not_anger_and_calibration_is_separate():
    for text in ("I have a complaint.", "CFPB", "I'm calling my lawyer."):
        signals = score_signals(text)
        assert score_text(text) < 0.65
        assert signals["emotion"] < 0.65
        assert signals["legal_intent"] > 0
        assert signals["urgency"] < 0.65
        assert signals["human_request"] == 0.0
        assert signals["frustration"] == signals["emotion"]
    urgent = score_signals("This feels dangerous.")
    assert urgent["urgency"] >= 0.85
    assert urgent["emotion"] < 0.65
    human = score_signals("I want to talk to a human.")
    assert human["human_request"] == 1.0
    assert human["emotion"] < 0.65
    report = accepted_label_calibration([
        {"accepted": True, "correct": True},
        {"accepted": True, "gold": "BRAKES", "predicted": "ENGINE"},
        {"accepted": False, "correct": True},
        {"accepted": False, "correct": False},
    ])
    assert report["precision"] == 0.5
    assert report["coverage"] == 0.5
    assert report["n"] == 4


def test_future_records_do_not_change_a_past_alert():
    early = {(f"2026-W{i:02d}", "BRAKES", "HONDA"): 5 for i in range(1, 9)}
    early[("2026-W09", "BRAKES", "HONDA")] = 65
    first = score_weekly_slices(early, pack_id="automotive_nhtsa")
    week9 = next(row for row in first if row["iso_week"] == "2026-W09")
    week1 = next(row for row in first if row["iso_week"] == "2026-W01")
    later = dict(early)
    later[("2026-W10", "BRAKES", "HONDA")] = 5
    later[("2026-W11", "BRAKES", "HONDA")] = 400
    second = score_weekly_slices(later, pack_id="automotive_nhtsa")
    again9 = next(row for row in second if row["iso_week"] == "2026-W09")
    again1 = next(row for row in second if row["iso_week"] == "2026-W01")
    assert again9["is_anomaly"] is week9["is_anomaly"] is True
    assert again1["is_anomaly"] is week1["is_anomaly"] is False
    assert again9["alert_available_week"] == "2026-W09"
    assert again9["model_identity"]
    assert "unknown" not in str(again9["model_identity"]).lower()
