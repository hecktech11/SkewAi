"""Versioned cosine, padding, and hash-wrapper compatibility."""

from __future__ import annotations

import math

import pytest

from src.ml_runtime.embedding_space import (
    HASH_EMBEDDING_VERSION,
    MINILM_NATIVE_DIMENSION,
    OUTPUT_DIMENSION,
    EmbeddingDimensionError,
    EmbeddingInvalidValuesError,
    EmbeddingVersionMismatch,
    as_embedded,
    compare_embeddings,
    native_prefix,
    pad_native_to_output,
)
from src.ml_runtime.embeddings import cosine, embed_text
from src.ml_runtime.hash_embedder import HashEmbedder

PHRASE = "front brake grinding on honda cr-v at low speed"


def test_hash_wrapper_matches_embed_text():
    h = HashEmbedder()
    raw = embed_text(PHRASE)
    wrapped = h.embed(PHRASE)
    assert wrapped.embedding_version == HASH_EMBEDDING_VERSION
    assert list(wrapped.values) == raw
    assert wrapped.output_dimension == OUTPUT_DIMENSION
    assert wrapped.native_dimension == OUTPUT_DIMENSION


def test_hash_wrapper_batch_matches_single():
    h = HashEmbedder()
    texts = [PHRASE, "airbag light on", ""]
    many = h.embed_many(texts)
    singles = [h.embed(t) for t in texts]
    assert [list(v.values) for v in many] == [list(v.values) for v in singles]


def test_compare_rejects_version_mismatch():
    a = as_embedded([1.0] + [0.0] * 511, HASH_EMBEDDING_VERSION)
    b = as_embedded([1.0] + [0.0] * 511, "all-MiniLM-L6-v2@x:onnx-int8:mean-pool:l2:zeropad512:v1:modelsha256=ab")
    with pytest.raises(EmbeddingVersionMismatch):
        compare_embeddings(a, b)


def test_compare_rejects_same_dim_different_version():
    a = as_embedded([0.0] * 512, "rev-a:dim512")
    b = as_embedded([0.0] * 512, "rev-b:dim512")
    with pytest.raises(EmbeddingVersionMismatch):
        compare_embeddings(a, b)


def test_compare_same_version_succeeds():
    a = as_embedded([1.0] + [0.0] * 511, HASH_EMBEDDING_VERSION)
    b = as_embedded([1.0] + [0.0] * 511, HASH_EMBEDDING_VERSION)
    assert compare_embeddings(a, b) == pytest.approx(1.0)


def test_compare_rejects_nan_and_inf():
    a = as_embedded([1.0] + [0.0] * 511, HASH_EMBEDDING_VERSION)
    with pytest.raises(EmbeddingInvalidValuesError):
        compare_embeddings(a, as_embedded([float("nan")] + [0.0] * 511, HASH_EMBEDDING_VERSION))
    with pytest.raises(EmbeddingInvalidValuesError):
        compare_embeddings(a, as_embedded([float("inf")] + [0.0] * 511, HASH_EMBEDDING_VERSION))


def test_compare_does_not_pad_or_truncate():
    a = as_embedded([1.0, 0.0], "v:dim2")
    with pytest.raises(EmbeddingDimensionError):
        as_embedded([1.0, 0.0, 0.0], "v:dim2", output_dimension=2)


def test_zero_vector_policy():
    a = as_embedded([0.0] * 512, HASH_EMBEDDING_VERSION)
    b = as_embedded([1.0] + [0.0] * 511, HASH_EMBEDDING_VERSION)
    assert compare_embeddings(a, b) == 0.0
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_legacy_numeric_cosine_unchanged():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_zero_padding_preserves_cosine():
    a384 = [0.6, 0.8] + [0.0] * (MINILM_NATIVE_DIMENSION - 2)
    b384 = [0.8, 0.6] + [0.0] * (MINILM_NATIVE_DIMENSION - 2)
    n = math.sqrt(sum(x * x for x in a384)) or 1.0
    a384 = [x / n for x in a384]
    n = math.sqrt(sum(x * x for x in b384)) or 1.0
    b384 = [x / n for x in b384]
    a512 = pad_native_to_output(a384, native_dimension=384, output_dimension=512)
    b512 = pad_native_to_output(b384, native_dimension=384, output_dimension=512)
    assert len(a512) == 512
    assert a512[384:] == [0.0] * 128
    native = cosine(a384, b384)
    padded = cosine(a512, b512)
    assert padded == pytest.approx(native, abs=1e-12)
    va = as_embedded(a512, "pad-test", native_dimension=384, output_dimension=512)
    vb = as_embedded(b512, "pad-test", native_dimension=384, output_dimension=512)
    assert compare_embeddings(va, vb) == pytest.approx(native, abs=1e-12)
    assert native_prefix(a512, 384) == a384
    assert abs(math.sqrt(sum(x * x for x in a512)) - 1.0) < 1e-9


def test_pack_scoped_idf_isolation_and_version_digest():
    """R12: pack-scoped IDF fitting does not alter other packs and reflects in version digest."""
    from src.ml_runtime.embeddings import fit_idf, reset_idf, embed_text, idf_digest
    from src.ml_runtime.hash_embedder import HashEmbedder
    from src.ml_runtime.embedding_space import compare_embeddings, EmbeddingVersionMismatch

    reset_idf()
    try:
        # Uniform version before fit
        h_uniform = HashEmbedder()
        assert idf_digest() == "uniform"
        assert h_uniform.version == HASH_EMBEDDING_VERSION

        # Fit pack A
        texts_a = ["brake pedal failure emergency stopping", "antilock brake system hydraulic leak"]
        fit_idf(texts_a, pack_id="pack_a")
        h_a = HashEmbedder(pack_id="pack_a")
        assert h_a.version != HASH_EMBEDDING_VERSION
        assert f":idf-{idf_digest('pack_a')}" in h_a.version

        # Vector for pack A before pack B is touched
        vec_a_1 = embed_text("brake pedal failure", pack_id="pack_a")

        # Fit pack B with completely different corpus
        texts_b = ["credit card dispute billing interest rate charges", "overdraft fee checking account balance"]
        fit_idf(texts_b, pack_id="pack_b")
        h_b = HashEmbedder(pack_id="pack_b")
        assert h_b.version != h_a.version

        # Fitting pack B must not mutate pack A's vector or version
        vec_a_2 = embed_text("brake pedal failure", pack_id="pack_a")
        assert vec_a_1 == vec_a_2

        # Comparing embeddings across pack A and pack B raises EmbeddingVersionMismatch
        emb_a = h_a.embed("brake pedal failure")
        emb_b = h_b.embed("credit card dispute")
        with pytest.raises(EmbeddingVersionMismatch):
            compare_embeddings(emb_a, emb_b)
    finally:
        reset_idf()
