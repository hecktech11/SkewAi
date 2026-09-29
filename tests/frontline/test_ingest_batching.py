"""The bulk-load path: batched writes, streamed rows, preserved semantics.

A 500k-row NHTSA load used to take 4+ hours, and almost none of it was the
embedding: the cost was one DuckDB commit per row in the domain warehouse
plus one fresh ops connection per row in the trust layer. Batching both is
a behaviour change hiding inside a performance change, so these tests pin
the semantics that must survive it — above all which row wins when a batch
contains the same record_id twice.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

import src.data.trust as trust_mod
import src.domains.mapping_ingest as mi
from src.data.timeutil import utc_now
from src.data.trust import iter_trusted, persist_trust_many
from src.data.warehouse import domain_con, ops_con
from src.domains.mapping_ingest import ingest_mapped_csv, upsert_records

PACK = "automotive_nhtsa"
MAPPING = Path("domains/automotive_nhtsa/data/mapping.yaml")


@pytest.fixture
def domain_tmp(tmp_path, monkeypatch):
    """Point the domain warehouse at a scratch dir, not the fixture pack."""
    monkeypatch.setenv("DOMAIN_DB_PATH", str(tmp_path))
    return tmp_path


def _rec(rid: str, **overrides):
    row = {
        "record_id": rid,
        "received_at": utc_now(),
        "occurred_at": utc_now() - timedelta(days=1),
        "entity_1": "2019",
        "entity_2": "HONDA",
        "entity_3": "CR-V",
        "category": "SERVICE BRAKES",
        "subcategory": "SERVICE BRAKES, HYDRAULIC:DISC",
        "text": f"narrative for {rid}",
        "severity_label": "",
        "region": "CA",
        "source": "NHTSA",
        "entity_key": "2019|HONDA|CR-V",
    }
    row.update(overrides)
    return row


def _stored(cols="record_id, text"):
    with domain_con(PACK) as con:
        return con.execute(f"SELECT {cols} FROM records ORDER BY record_id").fetchall()


# ── The trap: which duplicate wins ───────────────────────────────────────────


def test_duplicate_record_ids_in_one_batch_keep_the_last(domain_tmp):
    """A per-row loop lets the last write win. A set-based insert does not.

    ``INSERT OR REPLACE ... SELECT`` over a staged batch keeps the FIRST row
    for a duplicated key and silently drops the rest — no error, and the
    returned count is still the full input length, so nothing signals it.
    The batch must therefore be deduplicated in Python, keep-last, to match
    the semantics every existing caller was written against.
    """
    n = upsert_records(PACK, [_rec("DUP", text="first version"), _rec("DUP", text="last version")])
    assert n == 2, "the count reports rows consumed, not rows resident"
    assert _stored() == [("DUP", "last version")]


def test_duplicate_record_ids_across_a_batch_boundary_keep_the_last(domain_tmp, monkeypatch):
    """The winner must not depend on where the batch happens to be cut."""
    monkeypatch.setattr(mi, "_UPSERT_BATCH_ROWS", 2)
    rows = [_rec("DUP", text="first"), _rec("A"), _rec("B"), _rec("DUP", text="last")]
    assert upsert_records(PACK, rows) == 4
    stored = dict(_stored())
    assert stored["DUP"] == "last"
    assert set(stored) == {"A", "B", "DUP"}


def test_duplicate_collapse_does_not_lose_unrelated_rows(domain_tmp, monkeypatch):
    """Collapsing a duplicate shortens the batch; the rest must still land."""
    monkeypatch.setattr(mi, "_UPSERT_BATCH_ROWS", 3)
    rows = [_rec("A"), _rec("A"), _rec("B"), _rec("C"), _rec("D")]
    assert upsert_records(PACK, rows) == 5
    assert [r[0] for r in _stored()] == ["A", "B", "C", "D"]


# ── Batch mechanics ──────────────────────────────────────────────────────────


def test_rows_are_written_in_batches_not_one_statement_per_row(domain_tmp, monkeypatch):
    """The whole point of the change: writes scale with batches, not rows."""
    monkeypatch.setattr(mi, "_UPSERT_BATCH_ROWS", 10)
    flushes = []
    real_flush = mi._flush_batch

    def spy(con, batch):
        flushes.append(len(batch))
        return real_flush(con, batch)

    monkeypatch.setattr(mi, "_flush_batch", spy)
    assert upsert_records(PACK, [_rec(f"R{i:03d}") for i in range(25)]) == 25
    # 10 + 10 + 5, and the trailing partial batch is not lost.
    assert [f for f in flushes if f] == [10, 10, 5]
    assert len(_stored()) == 25


def test_every_row_survives_a_load_that_spans_many_batches(domain_tmp, monkeypatch):
    monkeypatch.setattr(mi, "_UPSERT_BATCH_ROWS", 7)
    ids = [f"R{i:04d}" for i in range(100)]
    assert upsert_records(PACK, [_rec(r) for r in ids]) == 100
    assert [r[0] for r in _stored()] == ids


def test_empty_batch_returns_zero_without_raising(domain_tmp):
    """DuckDB's executemany raises on an empty parameter list; this must not.

    A load whose every row is rejected by the trust contract reaches the
    writer with nothing, and that is a normal outcome, not an error.
    """
    assert upsert_records(PACK, []) == 0
    assert _stored() == []


def test_reingest_replaces_rather_than_duplicating(domain_tmp):
    """Idempotence: loading the same rows twice leaves one row each."""
    rows = [_rec("A", text="original"), _rec("B")]
    assert upsert_records(PACK, rows) == 2
    assert upsert_records(PACK, [_rec("A", text="revised"), _rec("B")]) == 2
    stored = dict(_stored())
    assert len(stored) == 2
    assert stored["A"] == "revised"


# ── Arrow typing: columns that are entirely NULL ─────────────────────────────


def test_all_null_optional_columns_load(domain_tmp):
    """A staged column with no non-null value has no type to infer.

    occurred_at is absent from sources with no incident date and entity_key
    from sources with no entity grain, so a real batch can be uniformly
    NULL in either. Arrow infers such a column as `null`, which will not
    cast into a TIMESTAMP or VARCHAR column unless the schema is explicit.
    """
    rows = [
        _rec("N1", occurred_at=None, entity_key="", region="", subcategory=""),
        _rec("N2", occurred_at=None, entity_key="", region="", subcategory=""),
    ]
    assert upsert_records(PACK, rows) == 2
    with domain_con(PACK) as con:
        got = con.execute(
            "SELECT record_id, occurred_at, entity_key, region, subcategory "
            "FROM records ORDER BY record_id"
        ).fetchall()
    assert [r[0] for r in got] == ["N1", "N2"]
    for _rid, occurred, ekey, region, sub in got:
        assert occurred is None
        assert ekey is None and region is None and sub is None


def test_embedding_round_trips_through_the_staged_batch(domain_tmp):
    """The embedding is a FLOAT[] column, the one non-scalar in the batch."""
    from src.ml_runtime.embeddings import cosine, embed_text

    upsert_records(PACK, [_rec("E1", text="grinding noise when braking")])
    with domain_con(PACK) as con:
        emb = con.execute("SELECT embedding FROM records WHERE record_id = 'E1'").fetchone()[0]
    assert cosine(list(emb), embed_text("grinding noise when braking")) > 0.999


def test_provenance_defaults_to_observed_and_is_honoured_when_given(domain_tmp):
    rows = [_rec("P1"), _rec("P2", provenance="inferred")]
    upsert_records(PACK, rows)
    with domain_con(PACK) as con:
        got = dict(
            con.execute("SELECT record_id, provenance FROM records ORDER BY record_id").fetchall()
        )
    assert got == {"P1": "observed", "P2": "inferred"}


# ── Trust layer batching ─────────────────────────────────────────────────────


def test_persist_trust_many_opens_one_connection_for_the_batch(reset_ops_db, monkeypatch):
    """Per-row persist spent 11.7 ms of every 20.2 ms on connect-and-close."""
    calls = []
    real = trust_mod.ops_con

    def spy(*a, **kw):
        calls.append(1)
        return real(*a, **kw)

    monkeypatch.setattr(trust_mod, "ops_con", spy)
    evs = [
        {"record_id": f"T{i}", "trusted": True, "badge": {}, "checks": {}, "content_hash": f"h{i}"}
        for i in range(50)
    ]
    tids = persist_trust_many("automotive_nhtsa", evs)
    assert len(tids) == 50
    assert len(set(tids)) == 50, "each evaluation needs its own trust_id"
    assert len(calls) == 1
    with ops_con(read_only=True) as con:
        n = con.execute(
            "SELECT COUNT(*) FROM record_trust WHERE pack_id = 'automotive_nhtsa'"
        ).fetchone()[0]
    assert n == 50


def test_persist_trust_many_on_an_empty_batch_opens_nothing(reset_ops_db, monkeypatch):
    calls = []
    monkeypatch.setattr(trust_mod, "ops_con", lambda *a, **kw: calls.append(1))
    assert persist_trust_many("automotive_nhtsa", []) == []
    assert calls == []


# ── Streaming ────────────────────────────────────────────────────────────────


def test_iter_trusted_does_not_drain_its_source_up_front(reset_ops_db, domain_tmp):
    """Constant memory is the point: the CSV must not be materialized."""
    pulled = []

    def source():
        for i in range(10):
            pulled.append(i)
            yield _rec(f"S{i}")

    stats, stream = iter_trusted(PACK, source(), persist=False)
    assert pulled == [], "nothing may be read before the stream is consumed"
    next(stream)
    assert len(pulled) == 1
    assert stats["evaluated"] == 1


def test_iter_trusted_seeds_dedup_before_the_stream_is_stepped(reset_ops_db, domain_tmp):
    """Load order here prevents a deadlock, so it is pinned.

    load_seen opens the pack's domain warehouse. The caller hands the stream
    straight to upsert_records, which holds a write connection to that same
    warehouse for the whole iteration — and domain_con takes a per-pack lock
    that is not reentrant. Seeding lazily, from inside the generator body,
    would run under the writer and hang.
    """
    seeded = []
    real_load_seen = trust_mod.load_seen

    def spy(pack_id):
        seeded.append(pack_id)
        return real_load_seen(pack_id)

    import unittest.mock as mock

    with mock.patch.object(trust_mod, "load_seen", spy):
        _stats, stream = iter_trusted(PACK, iter([_rec("Z1")]), persist=False)
        assert seeded == [PACK], "seeding must happen at call time, not on first next()"
        list(stream)


def test_iter_trusted_counts_are_complete_once_the_stream_is_exhausted(reset_ops_db, domain_tmp):
    rows = [_rec("A"), _rec("A"), _rec("B")]  # second A is a content duplicate
    stats, stream = iter_trusted(PACK, iter(rows), persist=False)
    kept = list(stream)
    assert stats["evaluated"] == 3
    assert stats["trusted"] + stats["rejected"] == 3
    assert len(kept) == stats["trusted"]


def test_keep_untrusted_yields_every_row(reset_ops_db, domain_tmp):
    """The historical-backfill path writes rows the contract rejected."""
    stale = _rec("OLD", received_at=utc_now() - timedelta(days=400))
    strict_stats, strict = iter_trusted(PACK, iter([_rec("NEW"), stale]), persist=False)
    assert len(list(strict)) == 1
    assert strict_stats["rejected"] == 1

    all_stats, loose = iter_trusted(
        PACK, iter([_rec("NEW2"), _rec("OLD2", received_at=utc_now() - timedelta(days=400))]),
        persist=False,
        keep_untrusted=True,
    )
    assert len(list(loose)) == 2
    assert all_stats["rejected"] == 1, "a kept row is still reported as rejected"


# ── End to end through mapping.yaml ──────────────────────────────────────────


def test_ingest_mapped_csv_reports_streamed_counts(domain_tmp, reset_ops_db):
    """mapped/trusted/rejected must stay correct without a materialized list."""
    now = utc_now().strftime("%Y%m%d")
    csv_path = domain_tmp / "load.csv"
    header = (
        "RECORD_KEY,ODINO,CMPLID,DATEA,DATEC,MODEL_YR,MAKETXT,MODELTXT,"
        "COMPONENT,COMPDESC,CDESC,NUM_CYLS,STATE\n"
    )
    lines = [
        f"K{i}|2019|HONDA|CR-V|SERVICE BRAKES,1155{i},{i},{now},{now},2019,HONDA,CR-V,"
        f"SERVICE BRAKES,SB:DISC,narrative number {i} about the brakes,4,CA"
        for i in range(30)
    ]
    csv_path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
    result = ingest_mapped_csv(PACK, csv_path, mapping_path=MAPPING)
    assert result["mapped"] == 30
    assert result["trusted"] == 30
    assert result["rejected"] == 0
    assert result["upserted"] == 30
    assert len(_stored()) == 30


def test_ingest_mapped_csv_spanning_many_batches_loads_every_row(
    domain_tmp, reset_ops_db, monkeypatch
):
    """A real load is many batches; the boundary must not drop or merge rows."""
    monkeypatch.setattr(mi, "_UPSERT_BATCH_ROWS", 9)
    monkeypatch.setattr(trust_mod, "_TRUST_BATCH_ROWS", 4)
    now = utc_now().strftime("%Y%m%d")
    csv_path = domain_tmp / "many.csv"
    header = (
        "RECORD_KEY,ODINO,CMPLID,DATEA,DATEC,MODEL_YR,MAKETXT,MODELTXT,"
        "COMPONENT,COMPDESC,CDESC,NUM_CYLS,STATE\n"
    )
    lines = [
        f"KEY{i:03d},1155{i},{i},{now},{now},2019,HONDA,CR-V,"
        f"SERVICE BRAKES,SB:DISC,distinct narrative {i} describing a brake fault,4,CA"
        for i in range(50)
    ]
    csv_path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
    result = ingest_mapped_csv(PACK, csv_path, mapping_path=MAPPING)
    assert result["mapped"] == 50
    assert result["upserted"] == 50
    assert len(_stored()) == 50
    with ops_con(read_only=True) as con:
        n = con.execute(
            "SELECT COUNT(*) FROM record_trust WHERE pack_id = ?", [PACK]
        ).fetchone()[0]
    assert n == 50, "every evaluated row needs a trust row, batching notwithstanding"
