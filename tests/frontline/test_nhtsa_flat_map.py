"""NHTSA FLAT line → mapping.yaml CSV row (no network).

The conversion is where the free-text complaint narrative stops being used
as the record category. These tests pin that split: COMPONENT carries a
gazetteer value, COMPDESC carries the raw hierarchy, CDESC carries only the
narrative.
"""

from __future__ import annotations

import csv
import os

import pytest

from scripts.ingest_nhtsa import (
    CSV_HEADERS,
    _FLAT_FIELDS,
    flat_line_to_mapping_row,
    preflight_local_csv,
    record_key,
    write_mapping_csv_from_flat,
)
from src.domains.nhtsa_components import UNKNOWN, category_values

_IDX = {n: i for i, n in enumerate(_FLAT_FIELDS)}


def _flat_line(**fields: str) -> str:
    """Build one FLAT_CMPL line with the named fields populated."""
    parts = [""] * len(_FLAT_FIELDS)
    for name, value in fields.items():
        parts[_IDX[name]] = value
    return "\t".join(parts)


def _complete_line(**overrides: str) -> str:
    base = {
        "CMPLID": "16123456",
        "ODINO": "11555001",
        "MAKETXT": "HONDA",
        "MODELTXT": "CR-V",
        "YEARTXT": "2019",
        "FAILDATE": "20240110",
        "DATEA": "20240115",
        "COMPDESC": "SERVICE BRAKES, HYDRAULIC:FOUNDATION COMPONENTS:DISC:ROTOR",
        "CDESCR": "BRAKES GRIND AT LOW SPEED",
        "STATE": "CA",
        "NUM_CYLS": "4",
    }
    base.update(overrides)
    return _flat_line(**base)


def _convert(lines, dest, *, limit=1000):
    """write_mapping_csv_from_flat takes a factory: it reads the source twice."""
    return write_mapping_csv_from_flat(lambda: list(lines), dest, limit=limit)


def _rows(dest):
    with dest.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ── Field layout ─────────────────────────────────────────────────────────────


def test_flat_field_list_matches_the_live_corpus_width():
    """NHTSA's FLAT_CMPL is 51 tab-separated fields.

    The parser zips this list against the split line positionally, so a short
    list silently discards trailing columns and, worse, would mis-name every
    field after an insertion. Measured against the real FLAT_CMPL.zip: all
    2,250,304 data lines have exactly 51 fields.
    """
    assert len(_FLAT_FIELDS) == 51
    assert _FLAT_FIELDS[-2:] == ["STATE_OF_INCIDENT", "VEHICLE_OPERATOR"]
    assert len(set(_FLAT_FIELDS)) == 51, "duplicate name would shadow a real column"


# ── One line → one row ───────────────────────────────────────────────────────


def test_flat_tab_line_maps_to_mapping_headers():
    row = flat_line_to_mapping_row(_complete_line())
    assert row is not None
    assert set(row) == set(CSV_HEADERS)
    assert row["CMPLID"] == "16123456"
    assert row["ODINO"] == "11555001"
    assert row["MODEL_YR"] == "2019"
    assert row["CDESC"] == "BRAKES GRIND AT LOW SPEED"
    assert row["DATEC"] == "20240110"


def test_component_is_normalized_to_a_gazetteer_value():
    """The whole point of the conversion: COMPONENT is a closed vocabulary."""
    row = flat_line_to_mapping_row(_complete_line())
    assert row["COMPONENT"] == "SERVICE BRAKES"
    assert row["COMPONENT"] in category_values()


def test_raw_hierarchy_is_preserved_in_compdesc():
    """Normalizing must not throw away the detail an investigator needs."""
    row = flat_line_to_mapping_row(_complete_line())
    assert row["COMPDESC"] == "SERVICE BRAKES, HYDRAULIC:FOUNDATION COMPONENTS:DISC:ROTOR"


def test_narrative_is_never_used_as_the_component():
    """CDESC is free text and must not leak into COMPONENT.

    Before the split, mapping.yaml read `category: CDESC`, so the category
    column held a ~770-distinct-value narrative. Every consumer matches
    category by exact string equality, so this returned nothing rather than
    failing.
    """
    row = flat_line_to_mapping_row(_complete_line())
    assert row["COMPONENT"] != row["CDESC"]
    assert row["COMPONENT"] in category_values()


def test_unrecognized_component_becomes_unknown_not_free_text():
    row = flat_line_to_mapping_row(_complete_line(COMPDESC="SOMETHING NHTSA INVENTED"))
    assert row["COMPONENT"] == UNKNOWN
    assert row["COMPDESC"] == "SOMETHING NHTSA INVENTED"


def test_blank_compdesc_becomes_unknown():
    """149 corpus rows have a blank COMPDESC; they are kept, not dropped."""
    row = flat_line_to_mapping_row(_complete_line(COMPDESC=""))
    assert row is not None
    assert row["COMPONENT"] == UNKNOWN


# ── Rows that cannot be used ─────────────────────────────────────────────────


def test_blank_narrative_is_dropped_not_backfilled_from_compdesc():
    """The old parser fell back to COMPDESC when CDESCR was empty.

    That put a part name — "SERVICE BRAKES, HYDRAULIC" — into `records.text`,
    where it becomes a degenerate cluster of identical short strings that
    outranks real complaints. 39 corpus rows are affected; dropping them is
    correct.
    """
    assert flat_line_to_mapping_row(_complete_line(CDESCR="")) is None


def test_blank_odino_is_dropped():
    """ODINO is the complaint key; without it the record cannot be addressed."""
    assert flat_line_to_mapping_row(_complete_line(ODINO="")) is None


def test_short_line_is_dropped():
    assert flat_line_to_mapping_row("only\tthree\tfields") is None


# ── Conversion report ────────────────────────────────────────────────────────


def test_conversion_report_accounts_for_every_line(tmp_path):
    lines = [
        _complete_line(),
        "",  # blank
        _complete_line(ODINO="11555002", CDESCR=""),  # unparseable
        _complete_line(ODINO="11555003", COMPDESC="NOT A REAL COMPONENT"),
        "too\tshort",  # unparseable
    ]
    dest = tmp_path / "out.csv"
    report = _convert(lines, dest)
    assert report["read"] == 5
    assert report["written"] == 2
    assert report["skipped_blank_line"] == 1
    assert report["skipped_unparseable"] == 2
    assert report["unknown_component"] == 1
    assert report["collapsed"] == 0
    assert (
        report["written"]
        + report["skipped_blank_line"]
        + report["skipped_unparseable"]
        + report["collapsed"]
        == report["read"]
    ), "every line read must be accounted for by exactly one outcome"


def test_written_csv_has_the_declared_headers(tmp_path):
    dest = tmp_path / "out.csv"
    _convert([_complete_line()], dest)
    with dest.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == CSV_HEADERS
        rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["COMPONENT"] == "SERVICE BRAKES"


def test_limit_stops_the_conversion(tmp_path):
    lines = [_complete_line(ODINO=str(11555000 + i)) for i in range(10)]
    report = _convert(lines, tmp_path / "out.csv", limit=3)
    assert report["written"] == 3
    assert report["distinct_keys"] == 10
    assert report["truncated_by_limit"] == 7, "a bounded load must say what it left out"


# ── Composite record key: one row per complaint/vehicle/component ────────────


def test_record_key_is_odino_vehicle_component():
    row = flat_line_to_mapping_row(_complete_line())
    assert row["RECORD_KEY"] == "11555001|2019|HONDA|CR-V|SERVICE BRAKES"
    assert record_key(row) == row["RECORD_KEY"]


def test_one_complaint_naming_two_components_becomes_two_rows(tmp_path):
    """NHTSA emits one line per component; both are real, distinct failures."""
    lines = [
        _complete_line(CMPLID="1", COMPDESC="SERVICE BRAKES, HYDRAULIC"),
        _complete_line(CMPLID="2", COMPDESC="ENGINE AND ENGINE COOLING"),
    ]
    dest = tmp_path / "out.csv"
    report = _convert(lines, dest)
    assert report["written"] == 2
    assert {r["COMPONENT"] for r in _rows(dest)} == {"SERVICE BRAKES", "ENGINE"}


def test_two_lines_for_the_same_pair_collapse_to_one_row(tmp_path):
    """The old CMPLID key made these two records, double-counting one failure.

    NHTSA's COMPDESC hierarchy is finer than the pack's category vocabulary,
    so two sub-components of the same top-level component arrive as two lines
    that normalize to the same category.
    """
    lines = [
        _complete_line(CMPLID="1", COMPDESC="SERVICE BRAKES, HYDRAULIC:DISC:ROTOR"),
        _complete_line(CMPLID="2", COMPDESC="SERVICE BRAKES, HYDRAULIC:LINES:HOSE"),
    ]
    dest = tmp_path / "out.csv"
    report = _convert(lines, dest)
    assert report["written"] == 1
    assert report["collapsed"] == 1
    assert report["distinct_keys"] == 1


def test_collapse_keeps_the_longest_narrative(tmp_path):
    lines = [
        _complete_line(CMPLID="1", CDESCR="SHORT"),
        _complete_line(CMPLID="2", CDESCR="A MUCH LONGER AND MORE USEFUL NARRATIVE"),
        _complete_line(CMPLID="3", CDESCR="MID LENGTH"),
    ]
    dest = tmp_path / "out.csv"
    _convert(lines, dest)
    rows = _rows(dest)
    assert len(rows) == 1
    assert rows[0]["CDESC"] == "A MUCH LONGER AND MORE USEFUL NARRATIVE"
    assert rows[0]["CMPLID"] == "2"


def test_collapse_is_order_independent(tmp_path):
    """The winner must not depend on where in a 2.25M-line file it appears.

    22,532 ODINOs reappear non-contiguously in the corpus, so a key's best
    line can be millions of lines after its first — which is why the
    conversion takes two passes instead of collapsing adjacent runs.
    """
    long_text = "A MUCH LONGER AND MORE USEFUL NARRATIVE"
    first = [
        _complete_line(CMPLID="2", CDESCR=long_text),
        _complete_line(CMPLID="1", CDESCR="SHORT"),
    ]
    last = [
        _complete_line(CMPLID="1", CDESCR="SHORT"),
        _complete_line(CMPLID="2", CDESCR=long_text),
    ]
    for i, lines in enumerate((first, last)):
        dest = tmp_path / f"out{i}.csv"
        _convert(lines, dest)
        assert _rows(dest)[0]["CDESC"] == long_text


def test_collapse_reports_discarded_narratives(tmp_path):
    """The merge loses text; the count must be honest about how much."""
    lines = [
        _complete_line(CMPLID="1", CDESCR="AAA"),
        _complete_line(CMPLID="2", CDESCR="BBBB"),  # different text, discarded
        _complete_line(ODINO="11555099", CMPLID="3", CDESCR="CCC"),
        _complete_line(ODINO="11555099", CMPLID="4", CDESCR="CCC"),  # same text
    ]
    report = _convert(lines, tmp_path / "out.csv")
    assert report["collapsed"] == 2
    assert report["conflicting_narrative"] == 1, "identical duplicates lose nothing"


def test_same_component_on_different_vehicles_stays_separate(tmp_path):
    """41,616 ODINOs span more than one year/make/model.

    Dropping the vehicle from the key would merge distinct vehicles into one
    record — and the vehicle is exactly the grain every anomaly baseline is
    computed over.
    """
    lines = [
        _complete_line(CMPLID="1", MODELTXT="CR-V"),
        _complete_line(CMPLID="2", MODELTXT="ACCORD"),
        _complete_line(CMPLID="3", YEARTXT="2020"),
    ]
    dest = tmp_path / "out.csv"
    report = _convert(lines, dest)
    assert report["written"] == 3
    assert len({r["RECORD_KEY"] for r in _rows(dest)}) == 3


def test_record_keys_in_the_output_are_unique(tmp_path):
    """The invariant upsert_records depends on: it is INSERT OR REPLACE, so a
    duplicate key is silently overwritten rather than reported."""
    lines = [
        _complete_line(CMPLID=str(i), COMPDESC="SERVICE BRAKES, HYDRAULIC:PART%d" % (i % 3))
        for i in range(12)
    ] + [_complete_line(CMPLID="90", ODINO="11555002")]
    dest = tmp_path / "out.csv"
    _convert(lines, dest)
    keys = [r["RECORD_KEY"] for r in _rows(dest)]
    assert len(keys) == len(set(keys))


def test_key_separator_is_absent_from_the_component_vocabulary():
    """A separator occurring inside a field would make the key ambiguous."""
    for value in category_values():
        assert "|" not in value


# ── Local-CSV preflight ──────────────────────────────────────────────────────


def _write_csv(path, rows, headers=None):
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=headers or CSV_HEADERS)
        w.writeheader()
        w.writerows(rows)
    return path


def _valid_local_row(**overrides):
    row = {h: "" for h in CSV_HEADERS}
    row.update(
        {
            "RECORD_KEY": "11555001|2019|HONDA|CR-V|SERVICE BRAKES",
            "ODINO": "11555001",
            "CMPLID": "16123456",
            "DATEA": "20240115",
            "MODEL_YR": "2019",
            "MAKETXT": "HONDA",
            "MODELTXT": "CR-V",
            "COMPONENT": "SERVICE BRAKES",
            "CDESC": "grinding",
        }
    )
    row.update(overrides)
    return row


def _keyed_row(n, **overrides):
    """A distinct row whose RECORD_KEY matches its ODINO."""
    odino = str(11555000 + n)
    return _valid_local_row(
        ODINO=odino,
        RECORD_KEY=f"{odino}|2019|HONDA|CR-V|SERVICE BRAKES",
        **overrides,
    )


def test_converted_work_dir_is_readable_only_during_its_ingest(tmp_path, monkeypatch):
    """A download lands outside the repo. The jail must allow that file, then close."""
    from src.security.identifiers import InvalidIdentifier, safe_csv_path
    from scripts.ingest_nhtsa import authorize_converted_work

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("FRONTLINE_TEST_ISOLATION", raising=False)
    monkeypatch.setenv("FRONTLINE_INGEST_ROOT", str(tmp_path / "kept"))
    csv_path = tmp_path / "work" / "nhtsa_complaints_mapped.csv"
    csv_path.parent.mkdir()
    csv_path.write_text("a\n", encoding="utf-8")
    with pytest.raises(InvalidIdentifier):
        safe_csv_path(csv_path)
    with authorize_converted_work(csv_path.parent):
        assert safe_csv_path(csv_path) == csv_path.resolve()
    assert os.environ["FRONTLINE_INGEST_ROOT"] == str(tmp_path / "kept")
    with pytest.raises(InvalidIdentifier):
        safe_csv_path(csv_path)


def test_preflight_accepts_a_normalized_local_csv(tmp_path):
    path = _write_csv(tmp_path / "ok.csv", [_valid_local_row()])
    report = preflight_local_csv(path, limit=100)
    assert report["checked"] == 1


def test_preflight_rejects_free_text_categories(tmp_path):
    """--csv bypasses the FLAT conversion, so it must be checked here.

    Without this gate, `--csv` reintroduces exactly the free-text category
    vocabulary the conversion exists to eliminate, and nothing downstream
    reports an error — the joins just return nothing.
    """
    path = _write_csv(
        tmp_path / "bad.csv",
        [_valid_local_row(COMPONENT="grinding noise when braking")],
    )
    with pytest.raises(ValueError, match="categories gazetteer"):
        preflight_local_csv(path, limit=100)


def test_preflight_rejects_a_csv_missing_the_component_column(tmp_path):
    headers = [h for h in CSV_HEADERS if h != "COMPONENT"]
    row = {h: v for h, v in _valid_local_row().items() if h != "COMPONENT"}
    path = _write_csv(tmp_path / "nocol.csv", [row], headers=headers)
    with pytest.raises(ValueError, match="COMPONENT"):
        preflight_local_csv(path, limit=100)


def test_preflight_only_checks_rows_that_will_be_loaded(tmp_path):
    """--limit bounds the load, so it bounds the check too."""
    rows = [_keyed_row(i) for i in range(3)]
    rows.append(_keyed_row(9, COMPONENT="not a category"))
    path = _write_csv(tmp_path / "tail.csv", rows)
    assert preflight_local_csv(path, limit=3)["checked"] == 3
    with pytest.raises(ValueError):
        preflight_local_csv(path, limit=100)


def test_preflight_rejects_duplicate_record_keys(tmp_path):
    """upsert_records is INSERT OR REPLACE: a duplicate is silently dropped.

    The row count then comes up short of `mapped` with no error, which reads
    as a partial load rather than a malformed input.
    """
    path = _write_csv(tmp_path / "dupe.csv", [_keyed_row(1), _keyed_row(1)])
    with pytest.raises(ValueError, match="duplicate RECORD_KEY"):
        preflight_local_csv(path, limit=100)


def test_preflight_rejects_a_csv_missing_the_record_key_column(tmp_path):
    headers = [h for h in CSV_HEADERS if h != "RECORD_KEY"]
    row = {h: v for h, v in _valid_local_row().items() if h != "RECORD_KEY"}
    path = _write_csv(tmp_path / "nokey.csv", [row], headers=headers)
    with pytest.raises(ValueError, match="RECORD_KEY"):
        preflight_local_csv(path, limit=100)


# ── The invariant the pack depends on ────────────────────────────────────────


def test_mapping_yaml_points_category_at_the_normalized_column():
    """Guards the pairing between the converter and the mapping.

    CSV_HEADERS and mapping.yaml have to agree; a rename on one side alone
    makes `_cell` resolve to empty and every record lands with a blank
    category.
    """
    import yaml

    from src.config import settings

    mapping = yaml.safe_load(
        (settings.pack_dir("automotive_nhtsa") / "data" / "mapping.yaml").read_text()
    )
    records = mapping["records"]
    assert records["record_id"] == "RECORD_KEY"
    assert records["category"] == "COMPONENT"
    assert records["subcategory"] == "COMPDESC"
    assert records["text"] == "CDESC"
    for field in ("record_id", "received_at", "category", "subcategory", "text"):
        assert records[field] in CSV_HEADERS, (
            f"mapping.yaml reads column {records[field]!r} for {field}, "
            f"which the converter never writes"
        )


def test_every_component_the_converter_emits_is_a_gazetteer_value():
    """The Phase 0 done-when condition, checked at the converter boundary."""
    vocabulary = category_values()
    samples = [
        "SERVICE BRAKES, HYDRAULIC:FOUNDATION COMPONENTS",
        "ENGINE AND ENGINE COOLING:ENGINE:GASOLINE",
        "FUEL SYSTEM, GASOLINE:DELIVERY:FUEL PUMP",
        "ELECTRICAL SYSTEM:WIRING",
        "BACK OVER PREVENTION:SENSING SYSTEM:CAMERA",
        "OTHER/I AM NOT SURE",
        "",
        "TOTALLY UNSEEN COMPONENT NAME",
    ]
    for raw in samples:
        row = flat_line_to_mapping_row(_complete_line(COMPDESC=raw))
        assert row is not None
        assert row["COMPONENT"] in vocabulary, f"{raw!r} → {row['COMPONENT']!r}"


# ── The same condition, one hop further: after the rows are in records ───────


def test_every_category_in_the_warehouse_came_from_the_gazetteer(tmp_path, monkeypatch):
    """The Phase 0 done-when condition, checked where it is actually claimed.

    The converter boundary is pinned above and mapping.yaml is pinned by
    test_mapping_yaml_points_category_at_the_normalized_column, but neither
    proves the value survives the load. A mapping key that silently resolves
    to nothing writes a NULL category and both of those tests still pass.
    """
    from datetime import timedelta

    from src.data.timeutil import utc_now
    from src.data.warehouse import domain_con
    from src.domains.mapping_ingest import ingest_mapped_csv

    monkeypatch.setenv("DOMAIN_DB_PATH", str(tmp_path))
    # The real pack, so the real mapping.yaml is the one under test. Only the
    # warehouse is redirected.
    pack = "automotive_nhtsa"
    # Inside the freshness SLA, or the trust contract rejects every row as
    # stale and the assertion below passes over an empty table.
    recent = (utc_now() - timedelta(days=3)).strftime("%Y%m%d")
    raws = [
        "SERVICE BRAKES, HYDRAULIC:FOUNDATION COMPONENTS:DISC:ROTOR",
        "ENGINE AND ENGINE COOLING:ENGINE:GASOLINE",
        "ELECTRICAL SYSTEM:WIRING",
        "FORWARD COLLISION AVOIDANCE: AUTOMATIC EMERGENCY BRAKING",
        "Other/I am not sure",
        "",
        "SOMETHING NHTSA HAS NEVER EMITTED",
    ]
    lines = [
        _complete_line(
            CMPLID=str(16123456 + i), ODINO=str(11555001 + i),
            COMPDESC=raw, DATEA=recent, FAILDATE=recent,
            CDESCR=f"narrative {i} describing the failure in the owner's words",
        )
        for i, raw in enumerate(raws)
    ]
    dest = tmp_path / "flat_mapped.csv"
    _convert(lines, dest)
    report = ingest_mapped_csv(pack, dest)
    assert report["upserted"] == len(raws)

    with domain_con(pack) as con:
        categories = [
            r[0] for r in con.execute(
                "SELECT DISTINCT category FROM records"
            ).fetchall()
        ]
    vocabulary = category_values()
    assert categories
    assert None not in categories, "a NULL category means the mapping key missed"
    unknown = sorted(c for c in categories if c not in vocabulary)
    assert not unknown, f"categories not in categories.csv: {unknown}"
    assert UNKNOWN in categories, "the unmatched and blank values need a bucket"
