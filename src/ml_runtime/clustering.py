"""Per-pack cluster rebuild from embeddings (feature #5).

Uses bag-of-hash embeddings + simple k-means-style assignment (no sklearn).
Writes ``clusters`` and ``cluster_assignments`` tables agents already read.
"""

from __future__ import annotations

import json
import logging
import math
import re
import warnings
from collections import Counter, defaultdict
from typing import Any

from src.data.warehouse import apply_domain_schema, domain_con
from src.ids import new_ulid
from src.ml_runtime.embeddings import cosine, embed_text, embedding_dim, fit_idf

logger = logging.getLogger(__name__)


# ── Cosine-distance semantics (item 5) ─────────────────────────────────────
# cluster_assignments.distance = 1 - cosine_similarity over L2-normalized
# bag-of-hash vectors (see src/ml_runtime/embeddings.cosine):
#   0.0  identical direction (same content)
#   1.0  orthogonal (no shared hashed features)
#   2.0  opposite direction (clamped max)
# Embeddings are non-negative hashed counts, so real distances fall in
# [0, 1]; values above 1 only occur with mixed-sign vectors. Assignment is
# nearest-centroid (argmax cosine) with NO distance cutoff: every record is
# assigned and weak fits stay visible as large distances instead of being
# silently dropped. Zero/empty vectors yield cosine 0.0 (distance 1.0):
# maximally dissimilar, never a false-identical 0.0.
DISTANCE_IDENTICAL = 0.0
DISTANCE_ORTHOGONAL = 1.0
DISTANCE_OPPOSITE = 2.0

_TOKEN = re.compile(r"[a-z0-9]+", re.I)
_STOP = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "when",
    "with",
    "from",
    "that",
    "this",
    "for",
    "to",
    "of",
    "in",
    "on",
    "is",
    "are",
    "was",
    "at",
}


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text or "") if t.lower() not in _STOP and len(t) > 2]


# ── Model selection (item 22) ──────────────────────────────────────────────
# k is chosen by mean silhouette width over cosine distance — a justified
# criterion, not a hardcoded cap. Small datasets are NOT forced to k=1: any
# k in [2, min(k_max, n-1)] with positive separation wins over the
# single-cluster null. O(n^2) pairwise distances: fine for pilot corpora;
# sample the series for very large n (future work, see entity blocking).


def _pairwise_cosine_dist(vectors: list[list[float]]) -> list[list[float]]:
    n = len(vectors)
    d = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            try:
                dist = max(0.0, min(2.0, 1.0 - cosine(vectors[i], vectors[j])))
            except ValueError:
                dist = 1.0
            d[i][j] = d[j][i] = dist
    return d


def silhouette_score(vectors: list[list[float]], labels: list[int]) -> float:
    """Mean silhouette width in [−1, 1] (cosine distance). Higher = better."""
    n = len(vectors)
    if n < 3 or len(set(labels)) < 2:
        return 0.0
    d = _pairwise_cosine_dist(vectors)
    total = 0.0
    for i in range(n):
        same = [d[i][j] for j in range(n) if j != i and labels[j] == labels[i]]
        if not same:
            continue  # singleton: s undefined; contributes 0
        a = sum(same) / len(same)
        b = min(
            (
                sum(d[i][j] for j in range(n) if labels[j] == lab) /
                max(1, sum(1 for j in range(n) if labels[j] == lab))
                for lab in set(labels)
                if lab != labels[i]
            ),
            default=a,
        )
        total += (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return total / n


def select_k(
    vectors: list[list[float]], k_max: int = 5
) -> tuple[int, float, str]:
    """Choose k by silhouette; returns (k, best_score, method).

    - n == 0 → (0, ...); n <= 2 → distinct points get k=n (never forced 1).
    - Otherwise argmax silhouette over k in [2, min(k_max, n-1)]; ties go to
      the smaller k. A non-positive best still returns k=1 ("single_blob") —
      no structure is an honest answer, not a failure.
    """
    n = len(vectors)
    if n == 0:
        return 0, 0.0, "empty"
    if n <= 2:
        if n == 2:
            try:
                distinct = cosine(vectors[0], vectors[1]) < 0.999
            except ValueError:
                distinct = True
            return (2, 1.0, "pairwise") if distinct else (1, 0.0, "identical_pair")
        return 1, 0.0, "singleton"
    hi = max(2, min(int(k_max), n - 1))
    best_k, best_s = 1, 0.0
    for k in range(2, hi + 1):
        try:
            labels = _kmeans(vectors, k)
        except Exception:
            continue
        if len(set(labels)) < 2:
            continue
        s = silhouette_score(vectors, labels)
        if s > best_s + 1e-9:
            best_k, best_s = k, s
    if best_k == 1:
        return 1, 0.0, "single_blob"
    return best_k, best_s, "silhouette"


def _kmeans(vectors: list[list[float]], k: int, rounds: int = 8) -> list[int]:
    if not vectors:
        return []
    k = max(1, min(k, len(vectors)))
    # Deterministic k-means++-style seeding: first centroid = first vector,
    # each next = farthest from chosen set (stable, covers semantic spread,
    # not temporal slices). Seed is deterministic so rebuilds reproduce.
    centroids: list[list[float]] = [vectors[0][:]]
    while len(centroids) < k:
        best_idx, best_d = 0, -1.0
        for i, v in enumerate(vectors):
            dmin = min(1.0 - cosine(v, c) for c in centroids)
            if dmin > best_d:
                best_d, best_idx = dmin, i
        centroids.append(vectors[best_idx][:])
    labels = [0] * len(vectors)
    for _ in range(rounds):
        # assign
        for i, v in enumerate(vectors):
            best, bi = -2.0, 0
            for j, c in enumerate(centroids):
                try:
                    s = cosine(v, c)
                except ValueError:
                    s = -1.0
                if s > best:
                    best, bi = s, j
            labels[i] = bi
        # update
        buckets: list[list[list[float]]] = [[] for _ in range(k)]
        for i, lab in enumerate(labels):
            buckets[lab].append(vectors[i])
        for j in range(k):
            if not buckets[j]:
                # Revive dead centroid at farthest point from live centroids
                far_idx, far_d = 0, -1.0
                for i, v in enumerate(vectors):
                    dmin = min(cosine(v, centroids[t]) for t in range(k) if t != j or True)
                    d = 1.0 - dmin
                    if d > far_d:
                        far_d, far_idx = d, i
                centroids[j] = vectors[far_idx][:]
                continue
            dim = len(buckets[j][0])
            avg = [0.0] * dim
            for v in buckets[j]:
                for d in range(dim):
                    avg[d] += v[d]
            n = float(len(buckets[j]))
            centroids[j] = [x / n for x in avg]
            norm = math.sqrt(sum(x * x for x in centroids[j])) or 1.0
            centroids[j] = [x / norm for x in centroids[j]]
    return labels


def cluster_signature(category: str | None, entity_2: str | None, top_terms: list[str]) -> str:
    """Stable content signature (board #5).

    sha256 over ``category | entity_2 | sorted(top_terms)`` (upper-cased).
    Unlike legacy integer ids (rebuild-local) and even ULIDs (unique per
    rebuild), the signature is STABLE across rebuilds for the same content,
    so investigations and eval pins resolve through rebuilds via the lineage
    map. Returns "" when there is nothing to sign.
    """
    import hashlib as _hashlib

    terms = sorted({str(t or "").strip().upper() for t in (top_terms or []) if str(t or "").strip()})
    if not terms and not (category or "").strip() and not (entity_2 or "").strip():
        return ""
    body = "|".join([
        (category or "").strip().upper(),
        (entity_2 or "").strip().upper(),
        ",".join(terms),
    ])
    return _hashlib.sha256(body.encode("utf-8")).hexdigest()[:32]


def resolve_cluster_lineage(pack_id: str, cluster_uid: str) -> list[dict[str, Any]]:
    """Successor chain for a cluster uid across rebuilds (board #5)."""
    with domain_con(pack_id, read_only=True) as con:
        try:
            apply_domain_schema(con)
        except Exception:
            pass
        try:
            cur = con.execute(
                """
                SELECT old_cluster_uid, new_cluster_uid, overlap, created_at
                FROM cluster_lineage WHERE pack_id = ? AND old_cluster_uid = ?
                ORDER BY created_at
                """,
                [pack_id, cluster_uid],
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]
        except Exception:
            return []
    """Emit a compatibility warning when a legacy integer cluster id is used.

    Legacy ``1000+lab`` ids are stable only within a single rebuild; the
    authoritative identity is ``cluster_uid`` (``clu_`` + ULID). Callers that
    join historical records (``cases.cluster_match_id``, ``backtest_results``)
    should resolve the version via :func:`resolve_cluster_version`.
    """
    try:
        int(cluster_id)
    except (TypeError, ValueError):
        return
    warnings.warn(
        f"legacy integer cluster_id={cluster_id} is rebuild-local; "
        "resolve the historical version via cluster_uid "
        "(src.ml_runtime.clustering.resolve_cluster_version)",
        DeprecationWarning,
        stacklevel=3,
    )


def resolve_cluster_version(
    pack_id: str,
    cluster_id: int,
    *,
    as_of: Any = None,
) -> dict[str, Any] | None:
    """Resolve the ``cluster_versions`` row for a legacy id.

    With ``as_of`` (timestamp or ISO string), returns the latest version
    created at or before that time — so a historical case keeps pointing at
    the original cluster identity/version. Without ``as_of``, returns the
    latest version.
    """
    with domain_con(pack_id, read_only=True) as con:
        try:
            apply_domain_schema(con)
        except Exception:
            pass
        try:
            if as_of is not None:
                row = con.execute(
                    """
                    SELECT cluster_uid, cluster_id, pack_id, version,
                           supersedes_uid, member_count, top_terms, category,
                           created_at
                    FROM cluster_versions
                    WHERE pack_id = ? AND cluster_id = ? AND created_at <= ?
                    ORDER BY version DESC LIMIT 1
                    """,
                    [pack_id, int(cluster_id), as_of],
                ).fetchone()
            else:
                row = con.execute(
                    """
                    SELECT cluster_uid, cluster_id, pack_id, version,
                           supersedes_uid, member_count, top_terms, category,
                           created_at
                    FROM cluster_versions
                    WHERE pack_id = ? AND cluster_id = ?
                    ORDER BY version DESC LIMIT 1
                    """,
                    [pack_id, int(cluster_id)],
                ).fetchone()
        except Exception:
            return None
        if not row:
            return None
        cols = [d[0] for d in con.description]
        return dict(zip(cols, row))


def rebuild_clusters(pack_id: str, *, k: int = 5) -> dict[str, Any]:
    """Rebuild clusters for pack from records text/embeddings."""
    with domain_con(pack_id, read_only=False) as con:
        apply_domain_schema(con)
        rows = con.execute(
            """
            SELECT record_id, text, category, embedding, received_at
            FROM records
            ORDER BY received_at
            LIMIT 5000
            """
        ).fetchall()
        if not rows:
            return {"pack_id": pack_id, "clusters": 0, "assignments": 0}

        records = []
        vectors = []
        # Fit corpus IDF before embedding so frequent terms don't dominate
        # (item 21). Deterministic for a fixed corpus.
        try:
            fit_idf([str(t or "") for _, t, _, _, _ in rows], pack_id=pack_id)
        except Exception:
            pass
        current_dim = embedding_dim()
        for rid, text, cat, emb, rec_at in rows:
            if emb is None or len(list(emb)) != current_dim:
                # Missing OR stale-dimension embedding (pre-512 migration):
                # recompute from text rather than comparing across dims.
                emb = embed_text(text or "", pack_id=pack_id)
                try:
                    con.execute(
                        "UPDATE records SET embedding = ? WHERE record_id = ?",
                        [list(emb), rid],
                    )
                except Exception:
                    pass
            else:
                emb = list(emb)
            records.append(
                {
                    "record_id": rid,
                    "text": text or "",
                    "category": cat,
                    "received_at": rec_at,
                }
            )
            vectors.append(emb)

        kk = min(k, len(records))
        selected_k, sil_width, k_method = select_k(vectors, k_max=kk)
        labels = _kmeans(vectors, max(1, selected_k))

        # wipe old clusters for this pack — but first snapshot the outgoing
        # identity/version map so history is preserved (item 4).
        old_uids: dict[int, str] = {}
        try:
            for cid_old, uid_old in con.execute(
                "SELECT cluster_id, cluster_uid FROM clusters WHERE pack_id = ?",
                [pack_id],
            ).fetchall():
                if uid_old:
                    old_uids[int(cid_old)] = str(uid_old)
        except Exception:
            pass
        # Lineage inputs (board #5): outgoing uid → member record set +
        # signature, captured BEFORE the wipe so the new clusters can link.
        old_members: dict[str, set[str]] = {}
        old_sigs: dict[str, str] = {}
        try:
            for cid_old, uid_old, sig_old in con.execute(
                "SELECT cluster_id, cluster_uid, signature FROM clusters WHERE pack_id = ?",
                [pack_id],
            ).fetchall():
                if not uid_old:
                    continue
                try:
                    members = {
                        str(r[0])
                        for r in con.execute(
                            "SELECT record_id FROM cluster_assignments WHERE cluster_id = ?",
                            [int(cid_old)],
                        ).fetchall()
                    }
                except Exception:
                    members = set()
                old_members[str(uid_old)] = members
                if sig_old:
                    old_sigs[str(uid_old)] = str(sig_old)
        except Exception:
            pass
        prior_versions: dict[int, tuple[int, str | None]] = {}
        try:
            for cid_old, ver_old, uid_old in con.execute(
                """
                SELECT cluster_id, MAX(version), MAX(cluster_uid)
                FROM cluster_versions WHERE pack_id = ?
                GROUP BY cluster_id
                """,
                [pack_id],
            ).fetchall():
                prior_versions[int(cid_old)] = (int(ver_old or 0), uid_old)
        except Exception:
            pass
        con.execute(
            "DELETE FROM cluster_assignments WHERE record_id IN (SELECT record_id FROM records)"
        )
        # delete clusters for pack
        old = con.execute(
            "SELECT cluster_id FROM clusters WHERE pack_id = ?", [pack_id]
        ).fetchall()
        for (cid,) in old:
            con.execute("DELETE FROM clusters WHERE cluster_id = ?", [cid])

        by_lab: dict[int, list[int]] = defaultdict(list)
        for i, lab in enumerate(labels):
            by_lab[lab].append(i)

        # Cluster identity (item 4): every rebuild mints a globally unique
        # cluster_uid ('clu_' + ULID) per cluster and appends a
        # cluster_versions row (versioned, supersedes-chain). The legacy
        # integer ids (1000+lab) are kept ONLY for backward compat with
        # cases.cluster_match_id / backtest_results / pinned eval fixtures —
        # they are rebuild-local and MUST NOT be treated as long-lived
        # identity. Reuse is explicit (returned + logged), never silent.
        written = 0
        cluster_uids: dict[int, str] = {}
        legacy_ids_reused: list[int] = []
        # Modal entity per record for stable signatures (board #5).
        _ent2_by_rid: dict[str, str] = {}
        try:
            _rids = [rec["record_id"] for rec in records]
            for _chunk in [_rids[i:i + 500] for i in range(0, len(_rids), 500)] or [[]]:
                if not _chunk:
                    continue
                _ph = ",".join("?" for _ in _chunk)
                for _rid, _e2 in con.execute(
                    f"SELECT record_id, entity_2 FROM records WHERE record_id IN ({_ph})",
                    _chunk,
                ).fetchall():
                    if _e2:
                        _ent2_by_rid[str(_rid)] = str(_e2)
        except Exception:
            pass
        # Precompute centroids for real distances (1 - cosine)
        centroids: dict[int, list[float]] = {}
        for lab, idxs in sorted(by_lab.items()):
            dim = len(vectors[idxs[0]])
            avg = [0.0] * dim
            for i in idxs:
                for d in range(dim):
                    avg[d] += vectors[i][d]
            n = float(len(idxs))
            c = [x / n for x in avg]
            norm = math.sqrt(sum(x * x for x in c)) or 1.0
            centroids[lab] = [x / norm for x in c]
        for lab, idxs in sorted(by_lab.items()):
            cid = 1000 + lab
            # Authoritative identity: fresh ULID every rebuild, never reused.
            uid = "clu_" + new_ulid()
            cluster_uids[cid] = uid
            prev_ver, prev_latest_uid = prior_versions.get(cid, (0, None))
            supersedes = prev_latest_uid or old_uids.get(cid)
            if prev_ver or cid in old_uids:
                legacy_ids_reused.append(cid)
            texts = [records[i]["text"] for i in idxs]
            tok_counts: Counter[str] = Counter()
            for t in texts:
                tok_counts.update(_tokens(t))
            # Discriminative top terms: downweight corpus-frequent tokens
            # (TF-IDF-lite so "ford/brake" doesn't top every cluster).
            import math as _math

            df: Counter[str] = Counter()
            for t in texts:
                for w in set(_tokens(t)):
                    df[w] += 1
            ndocs = max(1, len(idxs))
            scored_terms = [
                (c * _math.log((ndocs + 1) / (df[w] + 1)) + c * 0.1, w)
                for w, c in tok_counts.items()
            ]
            scored_terms.sort(reverse=True)
            top = [w for _, w in scored_terms[:5]] or [w for w, _ in tok_counts.most_common(5)]
            cats = Counter(records[i]["category"] for i in idxs if records[i]["category"])
            category = cats.most_common(1)[0][0] if cats else None
            times = [records[i]["received_at"] for i in idxs if records[i]["received_at"]]
            first_seen = min(times) if times else None
            last_seen = max(times) if times else None
            _e2_counts: Counter[str] = Counter(
                _ent2_by_rid[records[i]["record_id"]]
                for i in idxs
                if records[i]["record_id"] in _ent2_by_rid
            )
            _modal_e2 = _e2_counts.most_common(1)[0][0] if _e2_counts else None
            sig = cluster_signature(category, _modal_e2, top)
            try:
                con.execute(
                    """
                    INSERT INTO clusters
                    (cluster_id, pack_id, cluster_uid, signature, top_terms, category, record_count, first_seen, last_seen)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        cid,
                        pack_id,
                        uid,
                        sig,
                        json.dumps(top),
                        category,
                        len(idxs),
                        first_seen,
                        last_seen,
                    ],
                )
            except Exception:
                # Pre-signature schema.
                con.execute(
                    """
                    INSERT INTO clusters
                    (cluster_id, pack_id, cluster_uid, top_terms, category, record_count, first_seen, last_seen)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        cid,
                        pack_id,
                        uid,
                        json.dumps(top),
                        category,
                        len(idxs),
                        first_seen,
                        last_seen,
                    ],
                )
            # Lineage (board #5): signature hit wins (overlap 1.0), else best
            # member Jaccard ≥ 0.5; below that the cluster is genuinely new.
            try:
                _members = {records[i]["record_id"] for i in idxs}
                _best_uid, _best_ov = None, 0.0
                for _ouid, _omembers in old_members.items():
                    if sig and old_sigs.get(_ouid) == sig:
                        _best_uid, _best_ov = _ouid, 1.0
                        break
                    _union = _members | _omembers
                    _ov = (len(_members & _omembers) / len(_union)) if _union else 0.0
                    if _ov > _best_ov:
                        _best_uid, _best_ov = _ouid, _ov
                if _best_uid and _best_ov >= 0.5:
                    con.execute(
                        """
                        INSERT INTO cluster_lineage
                        (old_cluster_uid, new_cluster_uid, pack_id, overlap)
                        VALUES (?, ?, ?, ?)
                        """,
                        [_best_uid, uid, pack_id, round(_best_ov, 4)],
                    )
            except Exception as e:
                logger.warning("lineage write failed for %s: %s", uid, e)
            try:
                con.execute(
                    """
                    INSERT INTO cluster_versions
                    (cluster_uid, cluster_id, pack_id, version, supersedes_uid,
                     member_count, top_terms, category)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        uid,
                        cid,
                        pack_id,
                        prev_ver + 1,
                        supersedes,
                        len(idxs),
                        json.dumps(top),
                        category,
                    ],
                )
            except Exception as e:
                logger.warning("cluster version write failed for %s: %s", uid, e)
            for i in idxs:
                try:
                    dist = max(0.0, min(2.0, 1.0 - cosine(vectors[i], centroids[lab])))
                except ValueError:
                    dist = 1.0
                # 1.3: distance cutoff — far points become novel, not forced members
                import os as _os
                try:
                    _cut = float(_os.getenv("FRONTLINE_CLUSTER_MAX_DISTANCE", "0.85"))
                except ValueError:
                    _cut = 0.85
                if dist > _cut:
                    try:
                        con.execute(
                            "INSERT INTO novel_candidates (novel_id, interaction_id, pack_id, "
                            "category, entity_2, entity_3, top_score, status, created_at) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, 'open', current_timestamp)",
                            [f"nov_{records[i]['record_id']}", "", pack_id,
                             records[i].get("category"), records[i].get("entity_2"),
                             records[i].get("entity_3"), round(1.0 - dist, 3)],
                        )
                    except Exception:
                        pass
                con.execute(
                    """
                    INSERT INTO cluster_assignments (record_id, cluster_id, distance)
                    VALUES (?, ?, ?)
                    """,
                    [records[i]["record_id"], cid, dist],
                )
                written += 1
            # 1.3: absorb open novels whose signature now matches this cluster
            try:
                con.execute(
                    "UPDATE novel_candidates SET status='clustered' "
                    "WHERE pack_id=? AND status='open' AND category=?",
                    [pack_id, category],
                )
            except Exception:
                pass

        # R15: Assign all remaining records in the pack so records > 5,000 are not excluded
        sample_rids = {rec["record_id"] for rec in records}
        remaining_rows = con.execute(
            """
            SELECT record_id, text, category, embedding, received_at, entity_2, entity_3
            FROM records
            WHERE record_id NOT IN (
                SELECT record_id FROM cluster_assignments
            )
            ORDER BY received_at
            """
        ).fetchall()

        if remaining_rows and centroids:
            import os as _os

            try:
                _cut = float(_os.getenv("FRONTLINE_CLUSTER_MAX_DISTANCE", "0.85"))
            except ValueError:
                _cut = 0.85

            for rem_rid, rem_text, rem_cat, rem_emb, rem_rec_at, rem_e2, rem_e3 in remaining_rows:
                if rem_rid in sample_rids:
                    # Already handled during initial sampling loop
                    continue
                if rem_emb is None or len(list(rem_emb)) != current_dim:
                    rem_emb = embed_text(rem_text or "", pack_id=pack_id)
                    try:
                        con.execute(
                            "UPDATE records SET embedding = ? WHERE record_id = ?",
                            [list(rem_emb), rem_rid],
                        )
                    except Exception:
                        pass
                else:
                    rem_emb = list(rem_emb)

                # Find closest centroid
                best_lab = None
                best_dist = float("inf")
                for lab, c_vec in centroids.items():
                    try:
                        d = max(0.0, min(2.0, 1.0 - cosine(rem_emb, c_vec)))
                    except ValueError:
                        d = 1.0
                    if d < best_dist:
                        best_dist = d
                        best_lab = lab

                if best_lab is not None:
                    cid = 1000 + best_lab
                    con.execute(
                        """
                        INSERT INTO cluster_assignments (record_id, cluster_id, distance)
                        VALUES (?, ?, ?)
                        """,
                        [rem_rid, cid, best_dist],
                    )
                    written += 1
                    if best_dist > _cut:
                        try:
                            con.execute(
                                "INSERT INTO novel_candidates (novel_id, interaction_id, pack_id, "
                                "category, entity_2, entity_3, top_score, status, created_at) "
                                "VALUES (?, ?, ?, ?, ?, ?, ?, 'open', current_timestamp)",
                                [f"nov_{rem_rid}", "", pack_id,
                                 rem_cat, rem_e2, rem_e3, round(1.0 - best_dist, 3)],
                            )
                        except Exception:
                            pass

            # Update cluster and version member counts with the newly assigned records
            for lab in by_lab:
                cid = 1000 + lab
                row_stats = con.execute(
                    """
                    SELECT COUNT(*), MIN(r.received_at), MAX(r.received_at)
                    FROM cluster_assignments ca
                    JOIN records r ON ca.record_id = r.record_id
                    WHERE ca.cluster_id = ?
                    """,
                    [cid],
                ).fetchone()
                if row_stats and row_stats[0] > 0:
                    cnt, f_seen, l_seen = row_stats
                    con.execute(
                        "UPDATE clusters SET record_count = ?, first_seen = COALESCE(?, first_seen), last_seen = COALESCE(?, last_seen) WHERE cluster_id = ? AND pack_id = ?",
                        [cnt, f_seen, l_seen, cid, pack_id],
                    )
                    con.execute(
                        "UPDATE cluster_versions SET member_count = ? WHERE cluster_id = ? AND pack_id = ? AND version = (SELECT MAX(version) FROM cluster_versions WHERE cluster_id = ? AND pack_id = ?)",
                        [cnt, cid, pack_id, cid, pack_id],
                    )

        if legacy_ids_reused:
            msg = (
                "rebuild reused legacy integer cluster_ids "
                f"{sorted(legacy_ids_reused)} for pack {pack_id}; "
                "these ids are rebuild-local — historical references resolve "
                "via cluster_uid/cluster_versions"
            )
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            logger.warning(msg)

        return {
            "pack_id": pack_id,
            "clusters": len(by_lab),
            "assignments": written,
            "k": kk,
            "k_selected": selected_k,
            "silhouette": round(sil_width, 4),
            "k_method": k_method,
            "cluster_uids": cluster_uids,
            "legacy_ids_reused": sorted(legacy_ids_reused),
        }


def warn_on_legacy_cluster_id(cluster_id: Any, pack_id: str = "default") -> None:
    msg = (
        f"legacy integer cluster_id {cluster_id} is rebuild-local — "
        "historical references resolve via cluster_uid/cluster_versions"
    )
    warnings.warn(msg, DeprecationWarning, stacklevel=2)
    logger.warning(msg)



__all__ = [
    "rebuild_clusters",
    "resolve_cluster_version",
    "warn_on_legacy_cluster_id",
    "select_k",
    "silhouette_score",
    "cluster_signature",
    "resolve_cluster_lineage",
]
