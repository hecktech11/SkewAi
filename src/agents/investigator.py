"""Investigator Agent — corpus RCA.

Runs once category + description are filled:
  1. similar_records retriever (v1 similar_complaints generalized) with
     description + entity filters → top-k with scores.
  2. cluster_themes → dominant cluster.
  3. spikes_in_window for category/entity.
  4. backtest_proof → historical lead time for the matched cluster.

Output: InvestigationBrief — all numbers straight from SQL. LLM writes only the
two-sentence console narration (citation-verified).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from src.agents.base import Agent
from src.data.warehouse import domain_con
from src.ledger import record_action


# ── SQL templates (canonical schema, pack-agnostic) ──────────────────────
# DuckDB's Python client binds `?` placeholders positionally, so every
# statement takes a flat list of parameters (repeated where a value is used
# in more than one placeholder).

_SIMILAR_SQL = """
SELECT r.record_id, r.category, r.severity_label, r.received_at,
       r.text, r.entity_1, r.entity_2, r.entity_3, r.source, r.entity_key
FROM records r
WHERE (? IS NULL OR r.category = ?)
  AND (? IS NULL OR r.entity_2 = ?)
  AND (? IS NULL OR r.entity_3 = ?)
  AND (? IS NULL OR r.text ILIKE '%' || ? || '%')
LIMIT ?
"""

_CLUSTER_SQL = """
SELECT c.cluster_id, c.top_terms, c.category, c.record_count,
       c.first_seen, c.last_seen
FROM clusters c
WHERE c.pack_id = ?
  AND (c.category = ? OR ? IS NULL)
ORDER BY c.record_count DESC
LIMIT 5
"""

_SPIKE_SQL = """
SELECT iso_week, record_count, z_score, is_anomaly
FROM weekly_anomalies
WHERE pack_id = ?
  AND (category = ? OR ? IS NULL)
ORDER BY iso_week DESC
LIMIT 4
"""

_BACKTEST_SQL = """
SELECT b.cluster_id, b.advisory_id, b.lead_time_weeks, b.matched
FROM backtest_results b
WHERE b.cluster_id = ? AND b.matched = TRUE
LIMIT 1
"""


def _source_mix(similar: list[dict[str, Any]]) -> dict[str, int]:
    """Per-source count over similar records (Axion slice).

    Shows the engineer which origins corroborate the match (e.g. NHTSA: 3,
    WARRANTY: 2) instead of a sourceless list.
    """
    mix: dict[str, int] = {}
    for s in similar or []:
        src = str(s.get("source") or "unknown").upper()
        mix[src] = mix.get(src, 0) + 1
    return mix


def _cross_source_links(
    similar: list[dict[str, Any]], *, limit: int = 10
) -> list[dict[str, Any]]:
    """Same-entity records appearing in DIFFERENT sources (Axion slice).

    Two tiers, labeled honestly:
    - ``observed``: rows share the canonical ``entity_key`` (declared per
      source in mapping.yaml, e.g. VIN/model-year or account/product/issue).
      This is evidence an engineer can act on.
    - ``inferred``: rows share only the normalized (category, entity_2,
      entity_3) triple with no declared key — a heuristic, not evidence.
    """
    def _norm(v: Any) -> str:
        return str(v or "").strip().upper()

    by_key: dict[str, list[dict[str, Any]]] = {}
    by_triple: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for s in similar or []:
        ek = _norm(s.get("entity_key"))
        if ek:
            by_key.setdefault(ek, []).append(s)
        triple = (_norm(s.get("category")), _norm(s.get("entity_2")), _norm(s.get("entity_3")))
        if any(triple):
            by_triple.setdefault(triple, []).append(s)
    links: list[dict[str, Any]] = []
    for ek, rows in by_key.items():
        sources = sorted({_norm(r.get("source")) or "UNKNOWN" for r in rows})
        if len(sources) < 2:
            continue
        links.append(
            {
                "entity_key": ek,
                "basis": "observed",
                "sources": sources,
                "record_ids": [str(r.get("record_id")) for r in rows],
            }
        )
        if len(links) >= limit:
            return links
    for (cat, e2, e3), rows in by_triple.items():
        sources = sorted({_norm(r.get("source")) or "UNKNOWN" for r in rows})
        if len(sources) < 2:
            continue
        if any(str(r.get("entity_key") or "") for r in rows):
            continue  # keyed rows already linked above as observed
        links.append(
            {
                "category": cat,
                "entity_2": e2,
                "entity_3": e3,
                "basis": "inferred",
                "sources": sources,
                "record_ids": [str(r.get("record_id")) for r in rows],
            }
        )
        if len(links) >= limit:
            break
    return links


def _record_novel_candidate(
    ctx: Any, category: str | None, entity_2: str | None, entity_3: str | None,
    top_score: float,
) -> None:
    """Queue a below-threshold contact as a novel failure-mode candidate.

    Stores structured fields only (no free text): nightly clustering jobs
    sweep ``status='open'`` rows and absorb them into real clusters.
    """
    try:
        from src.data.warehouse import ops_con as _ops_con
        from src.ids import new_ulid as _ulid

        with _ops_con() as _con:
            _con.execute(
                """
                INSERT INTO novel_candidates
                (novel_id, interaction_id, pack_id, category, entity_2,
                 entity_3, top_score, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'open', CURRENT_TIMESTAMP)
                """,
                [
                    "nvl_" + _ulid(), ctx.interaction_id, ctx.pack.id,
                    category, entity_2, entity_3, float(top_score),
                ],
            )
    except Exception:
        pass


class InvestigatorAgent(Agent):
    """Corpus RCA: similar records, cluster, spike, lead-time. No LLM."""

    name = "investigator"

    async def run(self, **kwargs: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._run_sync, **kwargs)

    def _run_sync(self, **kwargs: Any) -> dict[str, Any]:
        ctx = self.ctx
        try:
            from src.ops.pilot import agent_enabled
            from src.ledger import record_action as _ra
            if not agent_enabled("investigator"):
                _ra(self._action(action_type="investigation_briefed",
                                 input_summary="flag disabled",
                                 output_summary="investigator disabled; skipped"))
                return {"skipped": True, "reason": "disabled_by_flag"}
        except Exception:
            pass
        entity_2 = ctx.slots.get("entity_2")
        entity_3 = ctx.slots.get("entity_3")
        category = ctx.slots.get("category")
        description = ctx.slots.get("description") or ""
        try:
            from src.ml_runtime.embedding_runtime import customer_visible_mode
        except Exception:
            def customer_visible_mode() -> str:  # type: ignore[misc]
                return "hash"

        # Extract a keyword from the description for the ILIKE filter.
        keyword = self._extract_keyword(description)

        # ── Similar records (semantic rank over candidate pool + ILIKE fallback)
        retrieval_mode = "ilike"
        try:
            with domain_con(ctx.pack.id) as con:
                # Broad candidate pool for semantic re-rank (entity/category filter).
                candidate_rows = con.execute(
                    _SIMILAR_SQL,
                    [
                        category, category,
                        entity_2, entity_2,
                        entity_3, entity_3,
                        None, None,
                        40,
                    ],
                ).fetchall()
                similar_cols = [d[0] for d in con.description]
                candidates = [dict(zip(similar_cols, r)) for r in candidate_rows]
                pool: list[dict[str, Any]] = []

                # Full-corpus population for honest lift denominators
                # (item 23): category/entity_2 projection over ALL records —
                # never the 40-row candidate shortlist.
                try:
                    from src.ml_runtime.association import get_pack_population_index

                    population = get_pack_population_index(con, pack_id=pack_id)
                except Exception:
                    population = candidates

                if candidates:
                    try:
                        from src.ml_runtime.association import rank_by_association

                        query = {
                            "category": category,
                            "entity_2": entity_2,
                            "entity_3": entity_3,
                            "text": description,
                        }
                        similar = rank_by_association(
                            query, candidates, population, top_k=5
                        )
                        retrieval_mode = "association"
                    except Exception:
                        similar = []
                    if description:
                        try:
                            from src.ml_runtime.embeddings import rank_by_similarity

                            ranked = rank_by_similarity(
                                description, similar or candidates, top_k=5
                            )
                            if ranked and (ranked[0].get("sim_score") or 0) > 0.05:
                                # Keep association order when scores are close; prefer
                                # assoc-ranked list as the primary similar set.
                                if similar:
                                    retrieval_mode = "association"
                                else:
                                    similar = ranked
                                    retrieval_mode = "semantic"
                        except Exception:
                            pass
                else:
                    similar = []

                if not similar:
                    # Wider ILIKE pool, then association rank (not recency LIMIT 5).
                    similar_rows = con.execute(
                        _SIMILAR_SQL,
                        [
                            category, category,
                            entity_2, entity_2,
                            entity_3, entity_3,
                            keyword, keyword,
                            40,
                        ],
                    ).fetchall()
                    if not similar_rows and keyword:
                        similar_rows = con.execute(
                            _SIMILAR_SQL,
                            [
                                category, category,
                                entity_2, entity_2,
                                entity_3, entity_3,
                                None, None,
                                40,
                            ],
                        ).fetchall()
                    similar_cols = [d[0] for d in con.description]
                    pool = [dict(zip(similar_cols, r)) for r in similar_rows]
                    from src.ml_runtime.association import rank_by_association

                    query = {
                        "category": category,
                        "entity_2": entity_2,
                        "entity_3": entity_3,
                        "text": description,
                    }
                    similar = rank_by_association(query, pool, population, top_k=5)
                    retrieval_mode = "association" if similar else "ilike"

                # Full candidate pool for source-mix corroboration (Axion
                # slice): whichever pool the ranking actually drew from.
                search_pool = list(candidates) if candidates else list(pool)

                cluster_rows = con.execute(
                    _CLUSTER_SQL, [ctx.pack.id, category, category]
                ).fetchall()
                cluster_cols = [d[0] for d in con.description]
                # Fused cluster relevance (item 26): category fit + term
                # overlap + ENTITY overlap (modal member values) + semantic
                # similarity (description vs top terms). record_count is only
                # a final tie-break — the largest cluster must not win on
                # size alone. Tie-breaks are fully deterministic
                # (score desc, cluster_id asc).
                cluster_row = None
                if cluster_rows:
                    try:
                        _cids = []
                        for _r in cluster_rows:
                            try:
                                _cids.append(int(_r[0]))
                            except (TypeError, ValueError):
                                continue
                        _member_ent: dict[int, tuple[str, str]] = {}
                        if _cids:
                            _ph = ",".join("?" * len(_cids))
                            for _er in con.execute(
                                f"""
                                SELECT a.cluster_id AS cid, r.entity_2 AS e2,
                                       r.entity_3 AS e3, COUNT(*) AS n
                                FROM cluster_assignments a
                                JOIN records r ON r.record_id = a.record_id
                                WHERE a.cluster_id IN ({_ph})
                                GROUP BY a.cluster_id, r.entity_2, r.entity_3
                                """,
                                _cids,
                            ).fetchall():
                                try:
                                    _cc = int(_er[0])
                                except (TypeError, ValueError):
                                    continue
                                _prev = _member_ent.get(_cc)
                                if _prev is None or int(_er[3] or 0) > int(_prev[2] or 0):
                                    _member_ent[_cc] = (
                                        str(_er[1] or ""), str(_er[2] or ""), _er[3],
                                    )
                    except Exception:
                        _member_ent = {}
                    try:
                        from src.ml_runtime.embeddings import cosine, embed_text

                        _qvec = embed_text(description or "")
                        _qok = any(_qvec)
                    except Exception:
                        _qvec, _qok = [], False

                    def _cscore(r: tuple) -> tuple:
                        d = dict(zip(cluster_cols, r))
                        try:
                            _cid = int(d.get("cluster_id"))
                        except (TypeError, ValueError):
                            _cid = -1
                        cat_s = 0.0
                        if category and str(d.get("category") or "").upper() == str(category).upper():
                            cat_s = 2.0
                        tt = str(d.get("top_terms") or "").upper()
                        desc = str(description or "").upper()
                        overlap = sum(1 for w in tt.replace(",", " ").split() if w and w in desc)
                        term_s = min(float(overlap), 5.0)
                        ent_s = 0.0
                        _me = _member_ent.get(_cid)
                        if _me:
                            if entity_2 and _me[0] and str(entity_2).upper() == _me[0].upper():
                                ent_s += 3.0
                            if entity_3 and _me[1] and str(entity_3).upper() == _me[1].upper():
                                ent_s += 2.0
                        sem_s = 0.0
                        if _qok:
                            try:
                                _raw_terms = d.get("top_terms")
                                if isinstance(_raw_terms, str):
                                    try:
                                        _parsed = json.loads(_raw_terms)
                                        _terms = (
                                            " ".join(str(w) for w in _parsed)
                                            if isinstance(_parsed, list)
                                            else _raw_terms
                                        )
                                    except (json.JSONDecodeError, TypeError):
                                        _terms = _raw_terms
                                elif isinstance(_raw_terms, list):
                                    _terms = " ".join(str(w) for w in _raw_terms)
                                else:
                                    _terms = ""
                                if _terms.strip():
                                    _cvec = embed_text(_terms)
                                    sem_s = max(0.0, cosine(_qvec, _cvec)) * 2.0
                            except Exception:
                                sem_s = 0.0
                        import math as _math

                        try:
                            size_s = _math.log1p(float(d.get("record_count") or 0)) * 0.1
                        except (TypeError, ValueError):
                            size_s = 0.0
                        total = cat_s + term_s + ent_s + sem_s + size_s
                        return (round(total, 6), -_cid if _cid >= 0 else 0)

                    cluster_rows = sorted(
                        cluster_rows,
                        key=lambda r: (_cscore(r), r[3] or 0),
                        reverse=True,
                    )
                    # Novelty gate (board #3): a forced assignment hides
                    # genuinely NEW failure modes — the most valuable RCA
                    # signal. Below threshold, no cluster is claimed; the
                    # contact is queued as a novel candidate instead.
                    import os as _os

                    try:
                        _nov_min = float(_os.getenv("FRONTLINE_NOVELTY_MIN_SCORE", "3.0"))
                    except ValueError:
                        _nov_min = 3.0
                    _best = _cscore(cluster_rows[0])[0] if cluster_rows else 0.0
                    if _best < _nov_min:
                        cluster_row = None
                        if customer_visible_mode() != "semantic":
                            _record_novel_candidate(
                                ctx, category, entity_2, entity_3, _best
                            )
                    else:
                        cluster_row = cluster_rows[0]
                if cluster_row is None and (category or description):
                    # Empty candidate set is also novelty (no score computed
                    # above): queue it once with score 0.0.
                    if customer_visible_mode() != "semantic":
                        _record_novel_candidate(ctx, category, entity_2, entity_3, 0.0)

                # Representation-layer overlay. Hash results above remain the
                # customer-visible default (legacy/shadow/rollback). Semantic
                # mode replaces visible ranking only when the sidecar+build
                # are compatible; missing vectors never become novelty.
                _emb_match = "performed"
                _emb_q = "blake2b-512-v1:feature-hash:tok2.0-bi0.5:sublinear-tf:idf-optional:l2:dim512"
                _emb_c = _emb_q
                _emb_build = None
                _skip_novel_missing = False
                try:
                    from src.ml_runtime.embedding_match import (
                        format_match_meta,
                        maybe_run_shadow,
                        rank_semantic,
                    )
                    from src.ml_runtime.embedding_runtime import (
                        active_cluster_build_id,
                        customer_visible_mode,
                    )
                    from src.ml_runtime.embedding_space import HASH_EMBEDDING_VERSION

                    _emb_q = HASH_EMBEDDING_VERSION
                    _emb_c = HASH_EMBEDDING_VERSION
                    maybe_run_shadow(
                        interaction_id=ctx.interaction_id,
                        pack_id=ctx.pack.id,
                        query=description,
                        candidates=search_pool or candidates,
                        hash_ranked=similar,
                        hash_cluster_id=int(cluster_row[0]) if cluster_row is not None else None,
                        hash_novel=cluster_row is None,
                    )
                    if customer_visible_mode() == "semantic":
                        sem = rank_semantic(
                            ctx.pack.id, description, search_pool or candidates, top_k=5
                        )
                        _emb_match = str(sem.get("status") or "skipped")
                        _emb_q = str(sem.get("query_version") or "")
                        _emb_c = str(sem.get("candidate_version") or "")
                        _emb_build = active_cluster_build_id() or None
                        if sem.get("status") == "performed":
                            similar = sem.get("ranked") or []
                            retrieval_mode = "semantic"
                            try:
                                from src.ml_runtime.cluster_builds import clusters_for_build
                                from src.ml_runtime.embedding_match import semantic_cluster_score

                                _build_clusters = clusters_for_build(ctx.pack.id, _emb_build or "")
                            except Exception:
                                _build_clusters = []
                            _best_c, _best_s = None, -1.0
                            for _bc in _build_clusters:
                                _sc = semantic_cluster_score(ctx.pack.id, description, _bc)
                                if _sc.get("status") != "performed":
                                    continue
                                if float(_sc.get("score") or 0) > _best_s:
                                    _best_s = float(_sc["score"])
                                    _best_c = _bc
                            if _best_c is None:
                                cluster_row = None
                                cluster_cols = [
                                    "cluster_id",
                                    "top_terms",
                                    "category",
                                    "record_count",
                                    "first_seen",
                                    "last_seen",
                                ]
                                if _emb_build and _build_clusters:
                                    _record_novel_candidate(
                                        ctx, category, entity_2, entity_3, max(_best_s, 0.0)
                                    )
                                else:
                                    _skip_novel_missing = True
                            else:
                                cluster_cols = [
                                    "cluster_id",
                                    "top_terms",
                                    "category",
                                    "record_count",
                                    "first_seen",
                                    "last_seen",
                                ]
                                cluster_row = (
                                    _best_c.get("cluster_id"),
                                    json.dumps(_best_c.get("top_terms") or []),
                                    _best_c.get("category"),
                                    _best_c.get("record_count") or 0,
                                    None,
                                    None,
                                )
                        else:
                            similar = []
                            cluster_row = None
                            retrieval_mode = "semantic_skipped"
                            _skip_novel_missing = True
                except Exception:
                    pass
                _emb_meta = (
                    f"emb_q={_emb_q} emb_c={_emb_c} match={_emb_match}"
                    + (f" cluster_build={_emb_build}" if _emb_build else "")
                )

                spike_rows = con.execute(
                    _SPIKE_SQL, [ctx.pack.id, category, category]
                ).fetchall()
                spike_cols = [d[0] for d in con.description]

                backtest_row = None
                backtest_cols: list[str] = []
                if cluster_row:
                    cluster_id = cluster_row[0]
                    backtest_row = con.execute(
                        _BACKTEST_SQL, [cluster_id]
                    ).fetchone()
                    backtest_cols = [d[0] for d in con.description]
        except FileNotFoundError:
            # Domain warehouse not built yet — degrade gracefully.
            record_action(self._action(
                action_type="similar_search",
                input_summary="domain warehouse not found",
                output_summary="investigator skipped; no corpus available",
                ok=False,
                error="domain warehouse missing",
            ))
            return {"skipped": True, "reason": "domain warehouse not built"}
        except Exception as e:
            # Any other DB/ML failure must degrade to skipped — never strand
            # the contact in ENRICHING (item 8). The orchestrator preserves
            # sibling agents' partial results.
            record_action(self._action(
                action_type="similar_search",
                input_summary=(
                    f"keyword='{keyword}', category={category}, "
                    f"entity_2={entity_2}, entity_3={entity_3}"
                ),
                output_summary="investigator skipped after failure; partial results kept",
                ok=False,
                error=f"{type(e).__name__}: {e}"[:500],
            ))
            return {"skipped": True, "reason": f"investigator failed: {type(e).__name__}"}

        # ── Ledger: similar_search ───────────────────────────────────────
        similar_ids = [s["record_id"] for s in similar if s.get("record_id")]
        from src.qubot.claims import bound_claims_from_records

        claims = bound_claims_from_records(similar, prefer=keyword)
        record_action(self._action(
            action_type="similar_search",
            input_summary=(
                f"keyword='{keyword}', category={category}, entity_2={entity_2}, "
                f"entity_3={entity_3}, mode={retrieval_mode} {_emb_meta}"
            )[:500],
            output_summary=f"found {len(similar)} similar records ({retrieval_mode})",
            evidence_ids=similar_ids,
            claims=claims,
        ))

        # ── Cluster match ─────────────────────────────────────────────────
        cluster_id = None
        cluster_top_terms: list = []
        cluster_count = 0
        if cluster_row:
            cd = dict(zip(cluster_cols, cluster_row))
            cluster_id = cd.get("cluster_id")
            cluster_count = cd.get("record_count", 0) or 0
            tt = cd.get("top_terms")
            if isinstance(tt, str):
                try:
                    cluster_top_terms = json.loads(tt)
                except json.JSONDecodeError:
                    cluster_top_terms = [tt]
            elif isinstance(tt, list):
                cluster_top_terms = tt
            ctx.investigation_brief = {
                "cluster_id": cluster_id,
                "cluster_count": cluster_count,
                "cluster_top_terms": cluster_top_terms,
            }
            record_action(self._action(
                action_type="cluster_matched",
                input_summary=f"category={category} {_emb_meta}"[:500],
                output_summary=f"cluster_id={cluster_id}, count={cluster_count}",
                evidence_ids=[str(cluster_id)] if cluster_id is not None else [],
            ))

        # ── Spike check ───────────────────────────────────────────────────
        spikes = [dict(zip(spike_cols, r)) for r in spike_rows]
        record_action(self._action(
            action_type="spike_checked",
            input_summary=f"category={category}",
            output_summary=f"recent weeks: {[(s.get('iso_week'), s.get('z_score'), s.get('is_anomaly')) for s in spikes]}",
        ))

        # ── Backtest lead-time + conclusion brief ─────────────────────────
        lead_time_weeks = None
        advisory_id = None
        if backtest_row:
            bd = dict(zip(backtest_cols, backtest_row))
            lead_time_weeks = bd.get("lead_time_weeks")
            advisory_id = bd.get("advisory_id")
        brief_ids = list(similar_ids)
        if advisory_id:
            brief_ids.append(str(advisory_id))
        record_action(self._action(
            action_type="brief_written",
            input_summary=f"cluster_id={cluster_id}",
            output_summary=(
                f"lead_time_weeks={lead_time_weeks}, matched_advisory={advisory_id}"
            ),
            evidence_ids=brief_ids,
            claims=claims,
        ))

        # ── InvestigationBrief ────────────────────────────────────────────
        # Elicitation answers attach here (item 35) with provenance, so the
        # brief — and any hypotheses derived from it — cite what the customer
        # actually said, not just slot state.
        try:
            from src.frontline.elicitation import answers_for_interaction

            elicitation_answers = answers_for_interaction(ctx.interaction_id)
        except Exception:
            elicitation_answers = []
        brief = {
            "similar_records": similar[:5],
            "similar_record_count": len(similar),
            "cluster_id": cluster_id,
            "cluster_top_terms": cluster_top_terms,
            "cluster_count": cluster_count,
            "spikes": spikes,
            "lead_time_weeks": lead_time_weeks,
            "elicitation_answers": elicitation_answers,
            "sources": _source_mix(similar),
            # Pool-level corroboration (Axion slice): the cited top-5 is
            # what the brief claims; the pool mix + cross-source links show
            # how the FULL candidate set — every origin — corroborates it.
            "candidate_sources": _source_mix(search_pool),
            "candidate_count": len(search_pool),
            "cross_source_links": _cross_source_links(search_pool),
            "novel_candidate": cluster_id is None,
            "brief_version": 1,
        }
        ctx.investigation_brief = brief

        # Two-sentence narration — deterministic in v2 (no LLM by default).
        narration = self._narrate(brief)
        return {
            "brief": brief,
            "narration": narration,
            "evidence_ids": similar_ids + ([str(cluster_id)] if cluster_id is not None else []),
        }

    # ── Helpers ────────────────────────────────────────────────────────────
    def _extract_keyword(self, description: str) -> str | None:
        """Pick the most informative word from the description for the ILIKE
        filter. Falls back to None (no text filter)."""
        if not description:
            return None
        # Strip common stopwords + verbs; pick the longest content word.
        stopwords = {
            "the", "a", "an", "my", "is", "are", "was", "were", "when", "i",
            "to", "on", "in", "at", "and", "or", "but", "of", "with", "for",
            "it", "that", "this", "there", "here",
        }
        words = [w.strip(".,!?;:'\"()[]").lower() for w in description.split()]
        cands = [w for w in words if w and w not in stopwords and len(w) >= 4]
        if not cands:
            return None
        cands.sort(key=len, reverse=True)
        return cands[0]

    def _narrate(self, brief: dict[str, Any]) -> str:
        """Two-sentence narration; LLM optional with deterministic fallback."""
        count = brief.get("similar_record_count", 0)
        cluster = brief.get("cluster_id")
        cluster_count = brief.get("cluster_count", 0)
        lead = brief.get("lead_time_weeks")
        evidence_ids = [
            s.get("record_id")
            for s in (brief.get("similar_records") or [])
            if s.get("record_id")
        ]

        s1 = f"Found {count} similar historical record(s)"
        if cluster is not None:
            s1 += f"; strongest match is cluster #{cluster} ({cluster_count} records)"
        s1 += "."

        s2 = ""
        if lead is not None:
            s2 = f"Historically, similar clusters preceded an advisory by {lead} weeks."
        elif brief.get("spikes"):
            recent = brief["spikes"][0]
            if recent.get("is_anomaly"):
                s2 = f"Volume spiked in {recent.get('iso_week')} (z-score {recent.get('z_score'):.1f})."
        fallback = (s1 + " " + s2).strip()

        try:
            from src.ai.narration import phrase_investigation_brief

            res = phrase_investigation_brief(
                similar_count=count,
                cluster_id=cluster,
                lead_time_weeks=lead,
                keyword=self._extract_keyword(self.ctx.slots.get("description") or "") or "",
                evidence_ids=[str(e) for e in evidence_ids[:5]],
            )
            return res.text
        except Exception:
            return fallback
