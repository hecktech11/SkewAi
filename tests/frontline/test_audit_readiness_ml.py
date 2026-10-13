"""Audit findings: live intake category fallback and centroid deadline."""

from __future__ import annotations

import time


def test_live_intake_category_falls_back_when_artifacts_missing(
    monkeypatch, tmp_path
):
    """extract_pack_category is the extractor live intake calls."""
    from src.agents.intake import extract_pack_category
    from src.domains.loader import load_pack
    from src.ml_runtime.category_zeroshot import reset_category_centroids_cache
    from src.ml_runtime.onnx_embedder import reset_process_onnx_embedder

    monkeypatch.setenv("FRONTLINE_SLM_ARTIFACT_DIR", str(tmp_path / "missing-minilm"))
    monkeypatch.delenv("FRONTLINE_EMBEDDING_TOY", raising=False)
    monkeypatch.setenv("FRONTLINE_SLM_MODE", "live")
    reset_process_onnx_embedder()
    reset_category_centroids_cache()
    try:
        pack = load_pack("automotive_nhtsa", reload=True)
        ctx = type("Ctx", (), {"pack": pack, "interaction_id": "int_live_intake_fb"})()
        text = "the brakes grind when I stop"
        rules = extract_pack_category(text, ctx, rules_only=True)
        assert rules
        live = extract_pack_category(text, ctx)
        assert live == rules
    finally:
        reset_process_onnx_embedder()
        reset_category_centroids_cache()


def test_cold_centroid_construction_counts_inside_inference_deadline():
    """A cold centroid build that overruns the budget must not block the caller."""
    from src.ml_runtime.category_zeroshot import (
        predict_category_zero_shot,
        reset_category_centroids_cache,
    )

    class _Gaz:
        values = ("SERVICE BRAKES",)
        canonical: dict[str, str] = {}

    class _Pack:
        id = "cold_centroid_deadline"
        taxonomy: dict = {}

        def gazetteer_for_slot(self, name):
            return _Gaz() if name == "category" else None

    class _Slow:
        canonical_version = "cold-deadline-v1"

        def embed(self, text):
            time.sleep(0.4)

            class _Emb:
                values = [0.0] * 384

            return _Emb()

    reset_category_centroids_cache()
    started = time.monotonic()
    result = predict_category_zero_shot(
        "the brakes grind when I stop",
        _Pack(),
        embedder=_Slow(),
        timeout_ms=40.0,
    )
    elapsed = time.monotonic() - started
    assert result.category is None
    assert result.unavailable is True
    assert elapsed < 0.25, f"cold centroids ran outside the deadline ({elapsed:.2f}s)"
