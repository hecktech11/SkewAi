"""Co-occurrence / lift ranking for similar records and similar contacts.

Recency is a tie-break only. A co-occurring sibling outranks a newer
unrelated row.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable, Sequence


ASSOC_KEYS = ("category", "entity_2", "entity_3", "source")


def _norm(val: Any) -> str:
    return str(val or "").strip().upper()


def _ts(val: Any) -> float:
    if val is None:
        return 0.0
    if isinstance(val, datetime):
        return val.timestamp()
    if isinstance(val, (int, float)):
        return float(val)
    try:
        return datetime.fromisoformat(str(val).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


class PopulationIndex:
    """Pre-indexed support counts for O(1) lift calculations."""

    def __init__(self, population: Sequence[dict[str, Any]]) -> None:
        self.rows = list(population)
        self.n = len(self.rows) or 1
        self.n_cat: dict[str, int] = {}
        self.n_ent: dict[str, int] = {}
        self.n_both: dict[tuple[str, str], int] = {}
        for row in self.rows:
            cat = _norm(row.get("category"))
            ent = _norm(row.get("entity_2"))
            if cat:
                self.n_cat[cat] = self.n_cat.get(cat, 0) + 1
            if ent:
                self.n_ent[ent] = self.n_ent.get(ent, 0) + 1
            if cat and ent:
                self.n_both[(cat, ent)] = self.n_both.get((cat, ent), 0) + 1

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    def get_counts(self, c_cat: str, c_ent: str) -> tuple[int, int, int, int]:
        return (
            self.n,
            self.n_both.get((c_cat, c_ent), 0),
            self.n_cat.get(c_cat, 0),
            self.n_ent.get(c_ent, 0),
        )


import time

_POPULATION_CACHE: dict[str, tuple[float, PopulationIndex]] = {}


def get_pack_population_index(con: Any, pack_id: str = "") -> PopulationIndex:
    """Return precomputed or cached PopulationIndex directly using grouped SQL counts.

    Avoids O(N) full-corpus row fetches per contact (R21).
    """
    now = time.monotonic()
    if pack_id and pack_id in _POPULATION_CACHE:
        ts, p_idx = _POPULATION_CACHE[pack_id]
        if now - ts < 300.0:
            return p_idx

    n_cat: dict[str, int] = {}
    n_ent: dict[str, int] = {}
    n_both: dict[tuple[str, str], int] = {}
    total_n = 0
    try:
        rows = con.execute(
            "SELECT category, entity_2, COUNT(*) FROM records GROUP BY category, entity_2"
        ).fetchall()
        for cat_raw, ent_raw, count in rows:
            c = _norm(cat_raw)
            e = _norm(ent_raw)
            cnt = int(count)
            total_n += cnt
            if c:
                n_cat[c] = n_cat.get(c, 0) + cnt
            if e:
                n_ent[e] = n_ent.get(e, 0) + cnt
            if c and e:
                n_both[(c, e)] = n_both.get((c, e), 0) + cnt
    except Exception:
        total_n = 1

    idx = PopulationIndex([])
    idx.n = max(1, total_n)
    idx.n_cat = n_cat
    idx.n_ent = n_ent
    idx.n_both = n_both
    if pack_id:
        _POPULATION_CACHE[pack_id] = (now, idx)
    return idx


def lift_for_pair(
    query: dict[str, Any],
    candidate: dict[str, Any],
    population: Sequence[dict[str, Any]],
) -> float:
    """P(cat∧ent) / (P(cat)P(ent)) using the CANDIDATE's cat+entity_2.

    ``population`` MUST be the full corpus (or the full in-scope corpus),
    never the candidate shortlist: support counts (n, n_cat, n_ent, n_both)
    are population denominators, while the pair under test comes from the
    candidate side. Mixing the two (candidate-side denominators) inflates
    lift arbitrarily. See :func:`population_support` for the distinct
    population-side semantics.
    """
    c_cat, c_ent = _norm(candidate.get("category")), _norm(candidate.get("entity_2"))
    if not c_cat or not c_ent:
        return 0.0
    # Candidate must share at least the pair to earn lift (else 0)
    q_cat, q_ent = _norm(query.get("category")), _norm(query.get("entity_2"))
    if q_cat and c_cat != q_cat:
        return 0.0
    if q_ent and c_ent != q_ent:
        return 0.0
    if isinstance(population, PopulationIndex):
        n, n_both, n_cat, n_ent = population.get_counts(c_cat, c_ent)
    else:
        n = len(population) or 1
        n_both = n_cat = n_ent = 0
        for row in population:
            cat, ent = _norm(row.get("category")), _norm(row.get("entity_2"))
            if cat == c_cat:
                n_cat += 1
            if ent == c_ent:
                n_ent += 1
            if cat == c_cat and ent == c_ent:
                n_both += 1
    p_both = n_both / n
    p_cat = n_cat / n
    p_ent = n_ent / n
    if p_cat <= 0 or p_ent <= 0:
        return 0.0
    return p_both / (p_cat * p_ent)


def association_score(
    query: dict[str, Any],
    candidate: dict[str, Any],
    population: Sequence[dict[str, Any]],
) -> float:
    import math as _math

    shared = 0
    for k in ASSOC_KEYS:
        qv, cv = _norm(query.get(k)), _norm(candidate.get(k))
        if qv and qv == cv:
            shared += 1
    lift = lift_for_pair(query, candidate, population)
    # Cap unbounded lift: log1p keeps rare-pair signal without dwarfing
    # shared-key count (previously lift=1000 beat shared=3).
    lift_capped = _math.log1p(max(0.0, lift))
    # Recency intentionally EXCLUDED from relevance (new != related).
    # Callers sort ties by received_at explicitly; see rank_by_association.
    return shared * 10.0 + min(lift_capped, 5.0)


def population_support(
    population: Sequence[dict[str, Any]],
    *,
    category: str,
    entity_2: str,
) -> dict[str, float]:
    """Population-side support counts for a (category, entity_2) pair.

    Returns ``{n, n_cat, n_ent, n_both, support, p_cat, p_ent}`` — the
    denominators lift is computed against. Distinct from candidate-side
    values (the pair under test); callers must not substitute one for
    the other.
    """
    c_cat, c_ent = _norm(category), _norm(entity_2)
    n = len(population) or 1
    n_both = n_cat = n_ent = 0
    for row in population:
        cat, ent = _norm(row.get("category")), _norm(row.get("entity_2"))
        if cat == c_cat:
            n_cat += 1
        if ent == c_ent:
            n_ent += 1
        if cat == c_cat and ent == c_ent:
            n_both += 1
    return {
        "n": float(len(population)),
        "n_cat": float(n_cat),
        "n_ent": float(n_ent),
        "n_both": float(n_both),
        "support": n_both / n,
        "p_cat": n_cat / n,
        "p_ent": n_ent / n,
    }


def rank_by_association(
    query: dict[str, Any],
    candidates: Iterable[dict[str, Any]],
    population: Sequence[dict[str, Any]] | None = None,
    *,
    top_k: int = 5,
    id_key: str = "record_id",
) -> list[dict[str, Any]]:
    """Return candidates sorted by association score (adds ``assoc_score``).

    Ties broken by recency explicitly (newest first) — recency is NOT mixed
    into the relevance score.

    ``population`` should be the FULL corpus for honest lift denominators;
    when omitted it falls back to ``candidates`` and each scored row is
    marked ``assoc_population='candidates-fallback'`` (vs ``'full-corpus'``)
    so candidate-side lift can never masquerade as corpus lift.
    """
    raw_pop = list(population) if population is not None else list(candidates)
    pop = PopulationIndex(raw_pop)
    pop_source = "full-corpus" if population is not None else "candidates-fallback"
    qid = _norm(query.get(id_key) or query.get("interaction_id"))
    scored: list[tuple[float, float, dict[str, Any]]] = []
    for c in candidates:
        cid = _norm(c.get(id_key) or c.get("interaction_id"))
        if qid and cid and qid == cid:
            continue
        row = dict(c)
        row["assoc_score"] = association_score(query, c, pop)
        row["assoc_population"] = pop_source
        scored.append((row["assoc_score"], _ts(c.get("received_at") or c.get("started_at")), row))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [r for _, _, r in scored[:top_k]]


__all__ = [
    "ASSOC_KEYS",
    "lift_for_pair",
    "population_support",
    "association_score",
    "rank_by_association",
]
