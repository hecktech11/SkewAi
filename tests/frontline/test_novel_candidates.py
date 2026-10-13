"""A cluster rebuild's novel candidates have to reach the ops warehouse.

``novel_candidates`` is declared in OPS_SCHEMA_SQL, but ``rebuild_clusters``
runs inside a writer on the pack's *domain* warehouse, and its three
statements against the table were wrapped in ``except Exception: pass``.
Every one of them failed with "table does not exist" and was swallowed, so a
record too far from every centroid to belong anywhere was detected and then
thrown away. The Early Warning board reads that table.

Fixing the connection alone is not enough, and these tests pin the rest: the
same rebuild used to mark its own findings 'clustered', the id was derived
per-record but inserted plainly so the second night collided on the primary
key, and the sampled path never carried entity_2 — the column the board
groups by.

On the cutoff: FRONTLINE_CLUSTER_MAX_DISTANCE defaults to 0.85, and the
sampled path is a k-means assignment, which minimises that exact distance.
Measured on this fixture the spread is 0.0 / 0.08 / 0.38 — nothing comes
close to 0.85, and a lone outlier is worse still because k-means gives it
its own cluster at distance 0. The tests that need a flagged row therefore
lower the cutoff explicitly. That is the product's own knob, not a test
trick, but it does mean the default only really bites on the second path,
where records past the 5,000-row sample are assigned to an existing
centroid with no re-fit to soften the distance.
"""

from __future__ import annotations

import pytest

from src.data.timeutil import utc_now
from src.data.warehouse import domain_con, ops_con
from src.ml_runtime.clustering import (
    novel_candidate_id,
    persist_novel_candidates,
    rebuild_clusters,
)

PACK = "novel_pack"

#: Two tight groups and a third, smaller one. Forced to k=2 the steering
#: records are absorbed into the brake cluster at a measurably larger
#: distance (0.3783) than its real members (0.0799).
SEED = (
    [(f"b{i}", "front brake pads grinding worn down to the backing plate metal",
      "SERVICE BRAKES", "HONDA", "CR-V") for i in range(6)]
    + [(f"a{i}", "airbag warning light illuminated on the instrument cluster panel",
        "AIR BAGS", "TOYOTA", "CAMRY") for i in range(6)]
    + [(f"s{i}", "steering wheel vibration at highway speed pulling to the right",
        "STEERING", "FORD", "F-150") for i in range(3)]
)

#: Between the two (0.0799 < 0.30 < 0.3783), so exactly the steering records
#: are novel.
CUTOFF = "0.30"


@pytest.fixture
def warehouses(tmp_path, monkeypatch, reset_ops_db):
    """Fresh domain warehouse, freshly recreated ops warehouse.

    The ops path cannot simply be repointed: ``ops_con`` applies the schema
    once behind a module-level ``_ops_initialized`` flag, so a later path
    change gets a connection to a file that has no tables. ``reset_ops_db``
    is the fixture that handles this properly.
    """
    monkeypatch.setenv("DOMAIN_DB_PATH", str(tmp_path))
    return tmp_path


@pytest.fixture
def seeded(warehouses, monkeypatch):
    from src.ml_runtime.embeddings import embed_text

    monkeypatch.setenv("FRONTLINE_CLUSTER_MAX_DISTANCE", CUTOFF)
    with domain_con(PACK, read_only=False) as con:
        for rid, text, cat, e2, e3 in SEED:
            con.execute(
                """
                INSERT OR REPLACE INTO records
                (record_id, received_at, occurred_at, entity_1, entity_2,
                 entity_3, category, text, source, embedding, entity_key,
                 provenance)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'NHTSA', ?, ?, 'observed')
                """,
                [
                    rid, utc_now(), utc_now(), "2019", e2, e3, cat, text,
                    embed_text(text), f"2019|{e2}|{e3}",
                ],
            )
    return warehouses


def _rows():
    with ops_con(read_only=True) as con:
        return con.execute(
            """
            SELECT novel_id, category, entity_2, entity_3, status, interaction_id
            FROM novel_candidates WHERE pack_id = ? ORDER BY novel_id
            """,
            [PACK],
        ).fetchall()


def _status_by_id():
    return {r[0]: r[4] for r in _rows()}


# ── The id ───────────────────────────────────────────────────────────────────


def test_the_id_is_stable_across_rebuilds():
    """A fresh ULID per rebuild would manufacture an emerging slice.

    The board calls a (category, entity_2) slice emerging at three open
    candidates. One stubborn outlier re-flagged nightly crosses that line in
    three days without a second vehicle ever complaining.
    """
    assert novel_candidate_id(PACK, "s1") == novel_candidate_id(PACK, "s1")


def test_the_id_is_scoped_per_pack():
    """novel_candidates is ops-wide, so two packs can share a record_id."""
    assert novel_candidate_id("pack_a", "s1") != novel_candidate_id("pack_b", "s1")


# ── The write lands at all ───────────────────────────────────────────────────


def test_a_rebuild_writes_its_novel_candidates_to_ops(seeded):
    out = rebuild_clusters(PACK, k=2)
    assert out["novel"]["staged"] == 3, "the steering records should be flagged"
    rows = _rows()
    assert rows, "novel_candidates is empty: the write went to the wrong warehouse"
    assert out["novel"]["open"] == len([r for r in rows if r[4] == "open"])


def test_a_flagged_candidate_keeps_its_fleet_slice(seeded):
    """The board groups by (category, entity_2). NULL collapses every slice.

    The sampled path selected five columns and read seven: entity_2 and
    entity_3 were never in the record dict, so every candidate it wrote
    landed as 'unknown'.
    """
    rebuild_clusters(PACK, k=2)
    open_rows = [r for r in _rows() if r[4] == "open"]
    assert open_rows
    for _nid, category, entity_2, entity_3, _status, _iid in open_rows:
        assert category == "STEERING"
        assert entity_2 == "FORD", "entity_2 is NULL — the sampled path dropped it"
        assert entity_3 == "F-150"


def test_a_rebuild_does_not_absorb_what_it_just_flagged(seeded):
    """The absorb swept the same pass's findings and blanked the feature.

    It matched on category alone and ran inside the per-cluster loop, right
    after the insert: flag a record as too far from every centroid, then
    mark it 'clustered' because some cluster shares its category.
    """
    rebuild_clusters(PACK, k=2)
    assert "open" in _status_by_id().values(), (
        "every candidate was absorbed by the rebuild that flagged it"
    )


def test_a_rebuild_has_no_contact_behind_it(seeded):
    """interaction_id distinguishes these from the investigator's own writes."""
    rebuild_clusters(PACK, k=2)
    assert _rows()
    assert all(r[5] == "" for r in _rows())


# ── Idempotence and triage ───────────────────────────────────────────────────


def test_a_second_rebuild_does_not_duplicate_or_raise(seeded):
    """The id was derived per record but inserted plainly.

    Once the connection is right, a plain INSERT collides on the primary key
    the second night — and that collision was swallowed too.
    """
    rebuild_clusters(PACK, k=2)
    first = {r[0] for r in _rows()}
    rebuild_clusters(PACK, k=2)
    second = {r[0] for r in _rows()}
    assert first and second == first


def test_a_dismissed_candidate_is_not_reopened(seeded):
    """Engineer triage must survive the next nightly rebuild.

    This is why the insert is ON CONFLICT DO NOTHING rather than an upsert:
    an upsert would helpfully refresh top_score and quietly undo the triage.
    """
    rebuild_clusters(PACK, k=2)
    target = next(nid for nid, st in _status_by_id().items() if st == "open")
    with ops_con() as con:
        con.execute(
            "UPDATE novel_candidates SET status='dismissed' WHERE novel_id = ?",
            [target],
        )
    rebuild_clusters(PACK, k=2)
    assert _status_by_id()[target] == "dismissed"


# ── The absorb, on its own ───────────────────────────────────────────────────


def _insert_open(novel_id, category, entity_2="HONDA"):
    with ops_con() as con:
        con.execute(
            """
            INSERT INTO novel_candidates
            (novel_id, interaction_id, pack_id, category, entity_2, entity_3,
             top_score, status, created_at)
            VALUES (?, '', ?, ?, ?, 'CR-V', 0.1, 'open', CURRENT_TIMESTAMP)
            """,
            [novel_id, PACK, category, entity_2],
        )


def _staged(nid, category="SERVICE BRAKES"):
    return {nid: {
        "novel_id": nid, "interaction_id": "", "pack_id": PACK,
        "category": category, "entity_2": "HONDA", "entity_3": "CR-V",
        "top_score": 0.1,
    }}


def test_absorb_clears_a_pre_existing_open_whose_category_now_clusters(warehouses):
    _insert_open("nvl_old", "SERVICE BRAKES")
    out = persist_novel_candidates(PACK, {}, {"SERVICE BRAKES"})
    assert out["absorbed"] == 1
    assert _status_by_id()["nvl_old"] == "clustered"


def test_absorb_leaves_other_categories_alone(warehouses):
    _insert_open("nvl_old", "AIR BAGS")
    out = persist_novel_candidates(PACK, {}, {"SERVICE BRAKES"})
    assert out["absorbed"] == 0
    assert _status_by_id()["nvl_old"] == "open"


def test_absorb_skips_a_candidate_this_run_re_flagged(warehouses):
    """Still too far from every centroid means still open, not absorbed."""
    nid = novel_candidate_id(PACK, "s0")
    _insert_open(nid, "SERVICE BRAKES")
    out = persist_novel_candidates(PACK, _staged(nid), {"SERVICE BRAKES"})
    assert out["absorbed"] == 0
    assert _status_by_id()[nid] == "open"


def test_counts_describe_this_run_not_the_pack_total(warehouses):
    """absorbed is what this call moved, so a second call reports zero."""
    _insert_open("nvl_a", "SERVICE BRAKES")
    _insert_open("nvl_b", "SERVICE BRAKES")
    assert persist_novel_candidates(PACK, {}, {"SERVICE BRAKES"})["absorbed"] == 2
    assert persist_novel_candidates(PACK, {}, {"SERVICE BRAKES"})["absorbed"] == 0


# ── Failure modes ────────────────────────────────────────────────────────────


def test_nothing_to_do_opens_no_connection(warehouses, monkeypatch):
    import src.data.warehouse as wh

    def boom(*a, **kw):  # pragma: no cover - reaching this is the failure
        raise AssertionError("opened an ops connection for an empty write")

    monkeypatch.setattr(wh, "ops_con", boom)
    assert persist_novel_candidates(PACK, {}, set()) == {
        "staged": 0, "absorbed": 0, "open": 0,
    }


def test_an_unavailable_ops_warehouse_warns_instead_of_failing(warehouses, monkeypatch, caplog):
    """A rebuild is domain work; it should survive — but it must say so.

    Silence is what hid the broken write in the first place, so the warning
    is the part worth pinning.
    """
    import src.data.warehouse as wh

    def boom(*a, **kw):
        raise RuntimeError("ops warehouse locked")

    monkeypatch.setattr(wh, "ops_con", boom)
    with caplog.at_level("WARNING"):
        out = persist_novel_candidates(PACK, _staged("nvl_x"), set())
    assert out["open"] == 0
    assert "novel candidate write failed" in caplog.text


def test_a_rebuild_survives_an_unavailable_ops_warehouse(seeded, monkeypatch):
    import src.data.warehouse as wh

    def boom(*a, **kw):
        raise RuntimeError("ops warehouse locked")

    monkeypatch.setattr(wh, "ops_con", boom)
    out = rebuild_clusters(PACK, k=2)
    assert out["clusters"] == 2, "clustering itself must still complete"
    assert out["novel"] == {"staged": 3, "absorbed": 0, "open": 0}
