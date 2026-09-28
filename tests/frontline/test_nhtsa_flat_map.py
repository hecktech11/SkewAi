"""NHTSA FLAT line → mapping.yaml CSV row (no network).

The conversion is where the free-text complaint narrative stops being used
as the record category. These tests pin that split: COMPONENT carries a
gazetteer value, COMPDESC carries the raw hierarchy, CDESC carries only the
narrative.
"""

from __future__ import annotations

import csv

import pytest

from scripts.ingest_nhtsa import (
    CSV_HEADERS,
    _FLAT_FIELDS,
    flat_line_to_mapping_row,
    preflight_local_csv,
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
    report = write_mapping_csv_from_flat(lines, dest, limit=1000)
    assert report["read"] == 5
    assert report["written"] == 2
    assert report["skipped_blank_line"] == 1
    assert report["skipped_unparseable"] == 2
    assert report["unknown_component"] == 1
    assert (
        report["written"] + report["skipped_blank_line"] + report["skipped_unparseable"]
        == report["read"]
    ), "every line read must be accounted for by exactly one outcome"


def test_written_csv_has_the_declared_headers(tmp_path):
    dest = tmp_path / "out.csv"
    write_mapping_csv_from_flat([_complete_line()], dest, limit=10)
    with dest.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == CSV_HEADERS
        rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["COMPONENT"] == "SERVICE BRAKES"


def test_limit_stops_the_conversion(tmp_path):
    lines = [_complete_line(ODINO=str(11555000 + i)) for i in range(10)]
    report = write_mapping_csv_from_flat(lines, tmp_path / "out.csv", limit=3)
    assert report["written"] == 3


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
            "ODINO": "11555001",
            "CMPLID": "16123456",
            "DATEA": "20240115",
            "COMPONENT": "SERVICE BRAKES",
            "CDESC": "grinding",
        }
    )
    row.update(overrides)
    return row


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
    rows = [_valid_local_row(ODINO=str(11555000 + i)) for i in range(3)]
    rows.append(_valid_local_row(ODINO="99", COMPONENT="not a category"))
    path = _write_csv(tmp_path / "tail.csv", rows)
    assert preflight_local_csv(path, limit=3)["checked"] == 3
    with pytest.raises(ValueError):
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
