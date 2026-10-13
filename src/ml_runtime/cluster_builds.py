"""Versioned cluster rebuilds that do not delete the active hash-era clusters.

A semantic query may only match a build whose embedding_version equals the
query vector's version. Cluster IDs from different embedding spaces are not
equivalent; cross-space correspondence is member-Jaccard only.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections import Counter, defaultdict
from typing import Any

from src.data.warehouse import _rollback_quietly, apply_domain_schema, domain_con
from src.ids import new_ulid
from src.ml_runtime.clustering import (
    _kmeans,
    _tokens,
    cluster_signature,
    select_k,
)
from src.ml_runtime.embedding_space import (
    EmbeddedVector,
    EmbeddingVersionMismatch,
    compare_embeddings,
)
from src.ml_runtime.embedding_store import ensure_record_embeddings, get_embeddings_for_records

logger = logging.getLogger(__name__)

CLUSTER_ALGORITHM_VERSION = "spherical-kmeans-v1:farthest-seed:rounds8"

CLUSTER_BUILDS_DDL = """
CREATE TABLE IF NOT EXISTS cluster_builds (
    build_id                  VARCHAR PRIMARY KEY,
    pack_id                   VARCHAR NOT NULL,
    embedding_version         VARCHAR NOT NULL,
    cluster_algorithm_version VARCHAR NOT NULL,
    parameters_json           VARCHAR NOT NULL,
    source_cutoff             TIMESTAMP,
    status                    VARCHAR NOT NULL,
    coverage_json             VARCHAR,
    created_at                TIMESTAMP DEFAULT current_timestamp,
    activated_at              TIMESTAMP
)
"""

CLUSTER_BUILD_CLUSTERS_DDL = """
CREATE TABLE IF NOT EXISTS cluster_build_clusters (
    build_id           VARCHAR NOT NULL,
    cluster_id         INTEGER NOT NULL,
    cluster_uid        VARCHAR NOT NULL,
    embedding_version  VARCHAR NOT NULL,
    centroid           FLOAT[],
    top_terms          VARCHAR,
    category           VARCHAR,
    record_count       INTEGER NOT NULL,
    first_seen         TIMESTAMP,
    last_seen          TIMESTAMP,
    signature          VARCHAR,
    PRIMARY KEY (build_id, cluster_id)
)
"""

CLUSTER_BUILD_ASSIGNMENTS_DDL = """
CREATE TABLE IF NOT EXISTS cluster_build_assignments (
    build_id           VARCHAR NOT NULL,
    record_id          VARCHAR NOT NULL,
    cluster_id         INTEGER NOT NULL,
    embedding_version  VARCHAR NOT NULL,
    distance           DOUBLE,
    PRIMARY KEY (build_id, record_id)
)
"""


def ensure_cluster_build_tables(con) -> None:
    con.execute(CLUSTER_BUILDS_DDL)
    con.execute(CLUSTER_BUILD_CLUSTERS_DDL)
    con.execute(CLUSTER_BUILD_ASSIGNMENTS_DDL)
    try:
        con.execute(
            "ALTER TABLE cluster_lineage ADD COLUMN correspondence_kind VARCHAR"
        )
    except Exception:
        _rollback_quietly(con)
    for col_ddl in (
        "ALTER TABLE cluster_builds ADD COLUMN embedding_model_key VARCHAR",
        "ALTER TABLE cluster_builds ADD COLUMN mean_cosine_intra DOUBLE",
        "ALTER TABLE cluster_builds ADD COLUMN mean_cosine_inter DOUBLE",
        "ALTER TABLE cluster_builds ADD COLUMN record_count INTEGER",
    ):
        try:
            con.execute(col_ddl)
        except Exception:
            _rollback_quietly(con)
    for ddl in (
        "CREATE INDEX IF NOT EXISTS idx_cluster_builds_pack "
        "ON cluster_builds(pack_id, embedding_version, status)",
        "CREATE INDEX IF NOT EXISTS idx_cluster_build_clusters_uid "
        "ON cluster_build_clusters(cluster_uid)",
        "CREATE INDEX IF NOT EXISTS idx_cluster_build_assign_cluster "
        "ON cluster_build_assignments(build_id, cluster_id)",
    ):
        try:
            con.execute(ddl)
        except Exception:
            _rollback_quietly(con)


def load_cluster_build(pack_id: str, build_id: str) -> dict[str, Any] | None:
    with domain_con(pack_id, read_only=True) as con:
        try:
            row = con.execute(
                """
                SELECT build_id, pack_id, embedding_version, cluster_algorithm_version,
                       parameters_json, source_cutoff, status, coverage_json,
                       created_at, activated_at
                FROM cluster_builds WHERE build_id = ? AND pack_id = ?
                """,
                [build_id, pack_id],
            ).fetchone()
        except Exception:
            return None
    if not row:
        return None
    keys = [
        "build_id",
        "pack_id",
        "embedding_version",
        "cluster_algorithm_version",
        "parameters_json",
        "source_cutoff",
        "status",
        "coverage_json",
        "created_at",
        "activated_at",
    ]
    data = dict(zip(keys, row))
    try:
        data["parameters"] = json.loads(data["parameters_json"] or "{}")
    except json.JSONDecodeError:
        data["parameters"] = {}
    try:
        data["coverage"] = json.loads(data["coverage_json"] or "{}")
    except json.JSONDecodeError:
        data["coverage"] = {}
    return data


def list_cluster_builds(pack_id: str) -> list[dict[str, Any]]:
    with domain_con(pack_id, read_only=True) as con:
        try:
            rows = con.execute(
                """
                SELECT build_id, embedding_version, status, created_at
                FROM cluster_builds WHERE pack_id = ? ORDER BY created_at
                """,
                [pack_id],
            ).fetchall()
        except Exception:
            return []
    return [
        {
            "build_id": r[0],
            "embedding_version": r[1],
            "status": r[2],
            "created_at": r[3],
        }
        for r in rows
    ]


def _load_versioned_vectors(
    pack_id: str, embedding_version: str
) -> tuple[list[dict[str, Any]], list[EmbeddedVector], dict[str, int]]:
    with domain_con(pack_id, read_only=True) as con:
        try:
            rows = con.execute(
            """
            SELECT r.record_id, r.text, r.category, r.entity_2, r.entity_3, r.received_at
            FROM records r
            ORDER BY r.record_id
            """
            ).fetchall()
        except Exception:
            rows = []
    records = [
        {
            "record_id": str(r[0]),
            "text": r[1] or "",
            "category": r[2],
            "entity_2": r[3],
            "entity_3": r[4],
            "received_at": r[5],
        }
        for r in rows
    ]
    rids = [r["record_id"] for r in records]
    found = get_embeddings_for_records(pack_id, rids, embedding_version)
    included: list[dict[str, Any]] = []
    vectors: list[EmbeddedVector] = []
    excluded = {"missing": 0, "wrong_version": 0}
    for rec in records:
        vec = found.get(rec["record_id"])
        if vec is None:
            excluded["missing"] += 1
            continue
        if vec.embedding_version != embedding_version:
            excluded["wrong_version"] += 1
            continue
        included.append(rec)
        vectors.append(vec)
    return included, vectors, excluded


def rebuild_cluster_build(
    pack_id: str,
    embedding_version: str,
    *,
    k: int = 5,
    dry_run: bool = False,
    max_distance: float | None = None,
) -> dict[str, Any]:
    """Build an immutable cluster version from one embedding version only."""
    started = time.perf_counter()
    with domain_con(pack_id, read_only=False) as con:
        apply_domain_schema(con)
        ensure_cluster_build_tables(con)
        ensure_record_embeddings(con)
    records, vectors, excluded = _load_versioned_vectors(pack_id, embedding_version)
    mixed = excluded.get("wrong_version") or 0
    if mixed:
        raise EmbeddingVersionMismatch(
            f"refusing mixed embedding versions in cluster rebuild ({mixed} rows)"
        )
    eligible_embedded = len(records)
    if not records:
        return {
            "pack_id": pack_id,
            "embedding_version": embedding_version,
            "clusters": 0,
            "assignments": 0,
            "excluded": excluded,
            "dry_run": dry_run,
        }
    # Homogeneous version: k-means uses numeric values after this check.
    numeric = [list(v.values) for v in vectors]
    selected_k, sil_width, k_method = select_k(numeric, k_max=min(k, len(numeric)))
    labels = _kmeans(numeric, max(1, selected_k))
    by_lab: dict[int, list[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        by_lab[lab].append(i)
    if max_distance is None:
        import os

        try:
            max_distance = float(os.getenv("FRONTLINE_CLUSTER_MAX_DISTANCE", "0.85"))
        except ValueError:
            max_distance = 0.85
    centroids: dict[int, EmbeddedVector] = {}
    for lab, idxs in by_lab.items():
        dim = len(numeric[idxs[0]])
        avg = [0.0] * dim
        for i in idxs:
            for d in range(dim):
                avg[d] += numeric[i][d]
        n = float(len(idxs))
        c = [x / n for x in avg]
        norm = math.sqrt(sum(x * x for x in c)) or 1.0
        c = [x / norm for x in c]
        centroids[lab] = EmbeddedVector(
            values=tuple(c),
            embedding_version=embedding_version,
            native_dimension=vectors[0].native_dimension,
            output_dimension=vectors[0].output_dimension,
            model_key=vectors[0].model_key,
            padding_strategy=vectors[0].padding_strategy,
        )
    sizes = sorted(len(v) for v in by_lab.values())
    intra: list[float] = []
    inter: list[float] = []
    for lab, idxs in by_lab.items():
        for i in idxs:
            intra.append(compare_embeddings(vectors[i], centroids[lab]))
        for other, c2 in centroids.items():
            if other == lab:
                continue
            inter.append(compare_embeddings(centroids[lab], c2))
    mean_intra = round(sum(intra) / len(intra), 4) if intra else 0.0
    mean_inter = round(sum(inter) / len(inter), 4) if inter else 0.0
    assignments_preview = 0
    novel_preview = 0
    for lab, idxs in by_lab.items():
        for i in idxs:
            dist = max(
                0.0,
                min(2.0, 1.0 - compare_embeddings(vectors[i], centroids[lab])),
            )
            if dist > max_distance:
                novel_preview += 1
            else:
                assignments_preview += 1
    report = {
        "pack_id": pack_id,
        "embedding_version": embedding_version,
        "cluster_algorithm_version": CLUSTER_ALGORITHM_VERSION,
        "eligible_embedded": eligible_embedded,
        "included": eligible_embedded,
        "excluded": excluded,
        "k": min(k, len(numeric)),
        "k_selected": selected_k,
        "silhouette": round(sil_width, 4),
        "k_method": k_method,
        "clusters": len(by_lab),
        "size_distribution": {
            "min": sizes[0] if sizes else 0,
            "max": sizes[-1] if sizes else 0,
            "median": sizes[len(sizes) // 2] if sizes else 0,
        },
        "assignments": assignments_preview,
        "novel_candidates": novel_preview,
        "dry_run": bool(dry_run),
        "duration_s": round(time.perf_counter() - started, 4),
        "status": "dry_run" if dry_run else "built",
        "activated": False,
        "mean_cosine_intra": mean_intra,
        "mean_cosine_inter": mean_inter,
    }
    if dry_run:
        return report

    build_id = "cbuild_" + new_ulid()
    params = {
        "k_max": k,
        "k_selected": selected_k,
        "max_distance": max_distance,
        "algorithm": CLUSTER_ALGORITHM_VERSION,
    }
    cutoff = max((r["received_at"] for r in records if r["received_at"]), default=None)
    with domain_con(pack_id, read_only=False) as con:
        apply_domain_schema(con)
        ensure_cluster_build_tables(con)
        # Snapshot current live (hash-era) assignments for correspondence.
        old_members: dict[str, set[str]] = {}
        try:
            for cid, uid in con.execute(
                "SELECT cluster_id, cluster_uid FROM clusters WHERE pack_id = ?",
                [pack_id],
            ).fetchall():
                if not uid:
                    continue
                members = {
                    str(r[0])
                    for r in con.execute(
                        "SELECT record_id FROM cluster_assignments WHERE cluster_id = ?",
                        [int(cid)],
                    ).fetchall()
                }
                old_members[str(uid)] = members
        except Exception:
            old_members = {}
        written = 0
        for lab, idxs in sorted(by_lab.items()):
            cid = 1000 + lab
            uid = "clu_" + new_ulid()
            texts = [records[i]["text"] for i in idxs]
            tok_counts: Counter[str] = Counter()
            df: Counter[str] = Counter()
            for t in texts:
                toks = _tokens(t)
                tok_counts.update(toks)
                df.update(set(toks))
            ndocs = max(1, len(idxs))
            scored_terms = [
                (c * math.log((ndocs + 1) / (df[w] + 1)) + c * 0.1, w)
                for w, c in tok_counts.items()
            ]
            scored_terms.sort(reverse=True)
            top = [w for _, w in scored_terms[:5]] or [w for w, _ in tok_counts.most_common(5)]
            cats = Counter(records[i]["category"] for i in idxs if records[i]["category"])
            category = cats.most_common(1)[0][0] if cats else None
            times = [records[i]["received_at"] for i in idxs if records[i]["received_at"]]
            e2_counts = Counter(
                str(records[i]["entity_2"])
                for i in idxs
                if records[i].get("entity_2")
            )
            modal_e2 = e2_counts.most_common(1)[0][0] if e2_counts else None
            sig = cluster_signature(category, modal_e2, top)
            con.execute(
                """
                INSERT INTO cluster_build_clusters (
                    build_id, cluster_id, cluster_uid, embedding_version, centroid,
                    top_terms, category, record_count, first_seen, last_seen, signature
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    build_id,
                    cid,
                    uid,
                    embedding_version,
                    list(centroids[lab].values),
                    json.dumps(top),
                    category,
                    len(idxs),
                    min(times) if times else None,
                    max(times) if times else None,
                    sig,
                ],
            )
            members = {records[i]["record_id"] for i in idxs}
            best_uid, best_ov = None, 0.0
            for ouid, omembers in old_members.items():
                union = members | omembers
                ov = (len(members & omembers) / len(union)) if union else 0.0
                if ov > best_ov:
                    best_uid, best_ov = ouid, ov
            if best_uid and best_ov >= 0.5:
                try:
                    con.execute(
                        """
                        INSERT INTO cluster_lineage
                        (old_cluster_uid, new_cluster_uid, pack_id, overlap, correspondence_kind)
                        VALUES (?, ?, ?, ?, 'cross_embedding')
                        """,
                        [best_uid, uid, pack_id, round(best_ov, 4)],
                    )
                except Exception:
                    try:
                        con.execute(
                            """
                            INSERT INTO cluster_lineage
                            (old_cluster_uid, new_cluster_uid, pack_id, overlap)
                            VALUES (?, ?, ?, ?)
                            """,
                            [best_uid, uid, pack_id, round(best_ov, 4)],
                        )
                    except Exception as e:
                        logger.warning("cross-embedding lineage write failed: %s", e)
            for i in idxs:
                dist = max(
                    0.0,
                    min(2.0, 1.0 - compare_embeddings(vectors[i], centroids[lab])),
                )
                if dist > max_distance:
                    continue
                con.execute(
                    """
                    INSERT INTO cluster_build_assignments
                    (build_id, record_id, cluster_id, embedding_version, distance)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    [build_id, records[i]["record_id"], cid, embedding_version, dist],
                )
                written += 1
        coverage = {
            "included": eligible_embedded,
            "excluded": excluded,
            "assignments": written,
            "novel_unassigned": novel_preview,
            "silhouette": round(sil_width, 4),
            "mean_cosine_intra": mean_intra,
            "mean_cosine_inter": mean_inter,
            "embedding_model_key": vectors[0].model_key if vectors else "",
        }
        con.execute(
            """
            INSERT INTO cluster_builds (
                build_id, pack_id, embedding_version, cluster_algorithm_version,
                parameters_json, source_cutoff, status, coverage_json
            ) VALUES (?, ?, ?, ?, ?, ?, 'built', ?)
            """,
            [
                build_id,
                pack_id,
                embedding_version,
                CLUSTER_ALGORITHM_VERSION,
                json.dumps(params),
                cutoff,
                json.dumps(coverage),
            ],
        )
    report.update(
        {
            "build_id": build_id,
            "assignments": written,
            "status": "built",
            "duration_s": round(time.perf_counter() - started, 4),
            "mean_cosine_intra": mean_intra,
            "mean_cosine_inter": mean_inter,
        }
    )
    return report


def activate_cluster_build(pack_id: str, build_id: str) -> dict[str, Any]:
    """Mark a built version active without deleting prior builds."""
    from src.data.timeutil import utc_now

    with domain_con(pack_id, read_only=False) as con:
        ensure_cluster_build_tables(con)
        row = con.execute(
            "SELECT status, embedding_version FROM cluster_builds WHERE build_id = ? AND pack_id = ?",
            [build_id, pack_id],
        ).fetchone()
        if not row:
            raise ValueError(f"cluster build {build_id} not found")
        if row[0] not in {"built", "active", "superseded", "rolled_back"}:
            raise ValueError(f"cannot activate cluster build in status {row[0]!r}")
        now = utc_now().replace(tzinfo=None)
        con.execute(
            """
            UPDATE cluster_builds SET status = 'superseded'
            WHERE pack_id = ? AND status = 'active' AND build_id != ?
            """,
            [pack_id, build_id],
        )
        con.execute(
            """
            UPDATE cluster_builds
            SET status = 'active', activated_at = ?
            WHERE build_id = ? AND pack_id = ?
            """,
            [now, build_id, pack_id],
        )
    return {"build_id": build_id, "status": "active", "embedding_version": row[1]}


def rollback_cluster_build(pack_id: str, build_id: str) -> dict[str, Any]:
    """Deactivate a build. Hash-era ``clusters`` table is untouched."""
    with domain_con(pack_id, read_only=False) as con:
        ensure_cluster_build_tables(con)
        con.execute(
            """
            UPDATE cluster_builds SET status = 'rolled_back'
            WHERE build_id = ? AND pack_id = ? AND status = 'active'
            """,
            [build_id, pack_id],
        )
    return {"build_id": build_id, "status": "rolled_back"}


def clusters_for_build(pack_id: str, build_id: str) -> list[dict[str, Any]]:
    with domain_con(pack_id, read_only=True) as con:
        rows = con.execute(
            """
            SELECT cluster_id, cluster_uid, embedding_version, centroid, top_terms,
                   category, record_count, signature
            FROM cluster_build_clusters WHERE build_id = ?
            ORDER BY record_count DESC
            """,
            [build_id],
        ).fetchall()
    out = []
    for r in rows:
        top = r[4]
        if isinstance(top, str):
            try:
                top = json.loads(top)
            except json.JSONDecodeError:
                top = [top]
        out.append(
            {
                "cluster_id": r[0],
                "cluster_uid": r[1],
                "embedding_version": r[2],
                "centroid": list(r[3]) if r[3] is not None else None,
                "top_terms": top,
                "category": r[5],
                "record_count": r[6],
                "signature": r[7],
            }
        )
    return out


__all__ = [
    "CLUSTER_ALGORITHM_VERSION",
    "ensure_cluster_build_tables",
    "load_cluster_build",
    "list_cluster_builds",
    "rebuild_cluster_build",
    "activate_cluster_build",
    "rollback_cluster_build",
    "clusters_for_build",
]
