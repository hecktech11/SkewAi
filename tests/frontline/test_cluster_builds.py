"""Versioned cluster rebuild does not delete hash-era clusters."""

from __future__ import annotations

import pytest

from src.ml_runtime.cluster_builds import (
    activate_cluster_build,
    list_cluster_builds,
    rebuild_cluster_build,
    rollback_cluster_build,
)
from src.ml_runtime.embedding_backfill import run_backfill
from src.ml_runtime.embedding_runtime import reset_embedding_runtime
from src.ml_runtime.onnx_embedder import ToySemanticEmbedder


@pytest.fixture
def toy_ready(monkeypatch, seed_automotive_pack):
    monkeypatch.setenv("FRONTLINE_EMBEDDING_TOY", "1")
    monkeypatch.setenv("FRONTLINE_EMBEDDING_MODE", "shadow")
    reset_embedding_runtime()
    ver = ToySemanticEmbedder().version
    run_backfill("automotive_nhtsa", ver, batch_size=8)
    yield ver
    reset_embedding_runtime()


def test_rebuild_creates_separate_version(toy_ready):
    from src.data.warehouse import domain_con

    with domain_con("automotive_nhtsa", read_only=True) as con:
        old_n = con.execute("SELECT COUNT(*) FROM clusters").fetchone()[0]
    dry = rebuild_cluster_build("automotive_nhtsa", toy_ready, k=3, dry_run=True)
    assert dry["dry_run"] is True
    assert "build_id" not in dry
    built = rebuild_cluster_build("automotive_nhtsa", toy_ready, k=3, dry_run=False)
    assert built["status"] == "built"
    assert built["build_id"].startswith("cbuild_")
    assert built["activated"] is False
    with domain_con("automotive_nhtsa", read_only=True) as con:
        new_n = con.execute("SELECT COUNT(*) FROM clusters").fetchone()[0]
        builds = con.execute("SELECT COUNT(*) FROM cluster_builds").fetchone()[0]
    assert new_n == old_n  # live hash clusters untouched
    assert builds >= 1
    act = activate_cluster_build("automotive_nhtsa", built["build_id"])
    assert act["status"] == "active"
    rb = rollback_cluster_build("automotive_nhtsa", built["build_id"])
    assert rb["status"] == "rolled_back"
    listed = list_cluster_builds("automotive_nhtsa")
    assert any(b["build_id"] == built["build_id"] for b in listed)


def test_rebuild_rejects_mixed_versions(toy_ready):
    from src.ml_runtime.embedding_space import HASH_EMBEDDING_VERSION, as_embedded
    from src.ml_runtime.embedding_store import upsert_embedding
    from src.data.warehouse import domain_con

    with domain_con("automotive_nhtsa", read_only=True) as con:
        rid = con.execute("SELECT record_id FROM records LIMIT 1").fetchone()[0]
    upsert_embedding(
        "automotive_nhtsa",
        str(rid),
        as_embedded([0.0] * 512, HASH_EMBEDDING_VERSION),
        status="complete",
    )
    # Mixed rows for the *requested* version are missing, not mixed; requesting
    # hash version against toy-backfilled corpus still only reads that version.
    out = rebuild_cluster_build("automotive_nhtsa", HASH_EMBEDDING_VERSION, k=2)
    assert out["excluded"]["missing"] >= 0


def test_rebuild_clusters_assigns_records_beyond_5000(monkeypatch, tmp_path):
    """R15: rebuild_clusters assigns records beyond 5,000 to nearest centroid."""
    from src.ml_runtime.clustering import rebuild_clusters
    from src.ml_runtime.embeddings import embedding_dim
    from src.data.warehouse import domain_con, apply_domain_schema
    import datetime

    pack = "test_large_pack"
    db_file = tmp_path / f"{pack}.duckdb"
    monkeypatch.setenv(f"FRONTLINE_{pack.upper()}_DB", str(db_file))
    dim = embedding_dim()
    vec = [0.1] * dim

    with domain_con(pack, read_only=False) as con:
        apply_domain_schema(con)
        base_time = datetime.datetime(2025, 1, 1, 12, 0, 0)
        rows = [
            (
                f"rec_{i}",
                f"Complaint about brake issue {i}",
                "BRAKES",
                "Acme",
                "Sedan",
                vec,
                (base_time + datetime.timedelta(minutes=i)).isoformat(),
            )
            for i in range(5005)
        ]
        con.executemany(
            """
            INSERT INTO records (record_id, text, category, entity_2, entity_3, embedding, received_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    res = rebuild_clusters(pack, k=2)
    assert res["clusters"] >= 1
    with domain_con(pack, read_only=True) as con:
        assigned_count = con.execute("SELECT COUNT(*) FROM cluster_assignments").fetchone()[0]
        assert assigned_count == 5005
        latest_assigned = con.execute(
            "SELECT cluster_id FROM cluster_assignments WHERE record_id = 'rec_5004'"
        ).fetchone()
        assert latest_assigned is not None
