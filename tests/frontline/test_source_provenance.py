"""Provenance is set by the loader, never inferred from the file.

``records.provenance`` decides whether a row counts as evidence. The anomaly
scan reads ``observed`` rows and leaves ``inferred`` ones out, so a demo CSV
loaded with no tag becomes indistinguishable from a real failure signal the
moment it lands. Before this there was no way to say otherwise: every source
row went in as ``observed`` because that is the column default.

These tests pin that the tag is explicit, validated at the edge, and carried
from the CLI and the job queue all the way down to the column.
"""

from __future__ import annotations

import pytest

from src.config import REPO_ROOT
from src.data.timeutil import utc_now
from src.data.trust import schema_contract_ok
from src.data.warehouse import domain_con
from src.domains.source_ingest import (
    DEFAULT_PROVENANCE,
    PROVENANCE_VALUES,
    ingest_source,
    normalize_provenance,
)

PACK = "prov_pack"

WARRANTY_CSV = """CLAIM_ID,CLAIM_DATE,REPAIR_DATE,MODEL_YR,MAKETXT,MODELTXT,COMPNAME,FAIL_CODE,TECH_NOTES,CLAIM_SEVERITY,STATE
200001,2024-03-01,2024-02-20,2019,HONDA,CR-V,SERVICE BRAKES,PAD-WEAR,front pads worn to the backing plate grinding,Medium,CA
200002,2024-03-05,2024-02-28,2019,HONDA,CR-V,SERVICE BRAKES,PAD-WEAR,grinding noise from the front brakes on braking,Medium,TX
200003,2024-03-06,2024-03-01,2018,TOYOTA,CAMRY,AIR BAGS,,airbag warning light on the dash,,WA
"""

WARRANTY_MAPPING = REPO_ROOT / "domains" / "automotive_nhtsa" / "data" / "mapping.warranty.yaml"


@pytest.fixture
def csv_path(tmp_path, monkeypatch):
    monkeypatch.setenv("DOMAIN_DB_PATH", str(tmp_path))
    path = tmp_path / "warranty.csv"
    path.write_text(WARRANTY_CSV, encoding="utf-8")
    return path


def _load(csv_path, **kw):
    return ingest_source(
        PACK, "warranty", csv_path, mapping_path=WARRANTY_MAPPING, **kw
    )


def _provenances():
    with domain_con(PACK) as con:
        rows = con.execute(
            "SELECT record_id, provenance FROM records ORDER BY record_id"
        ).fetchall()
    return {r[0]: r[1] for r in rows}


# ── Validation ───────────────────────────────────────────────────────────────


def test_unset_provenance_defaults_to_observed():
    assert normalize_provenance(None) == DEFAULT_PROVENANCE
    assert normalize_provenance("") == DEFAULT_PROVENANCE
    assert normalize_provenance("   ") == DEFAULT_PROVENANCE
    assert DEFAULT_PROVENANCE == "observed", "must match the records column default"


def test_provenance_is_case_insensitive_and_trimmed():
    assert normalize_provenance("  FIXTURE ") == "fixture"
    assert normalize_provenance("Inferred") == "inferred"


def test_unknown_provenance_is_rejected_not_downgraded():
    """Defaulting a typo to ``observed`` would load a demo file as evidence.

    The failure mode is quiet: the rows land, the scan counts them, and the
    only symptom is an early-warning alert with no vehicle behind it.
    """
    with pytest.raises(ValueError, match="unknown provenance"):
        normalize_provenance("demo")
    with pytest.raises(ValueError, match="observed"):
        normalize_provenance("trusted")


def test_the_three_values_are_the_ones_the_column_documents():
    assert PROVENANCE_VALUES == ("observed", "inferred", "fixture")


# ── The trust contract has to let the key through ────────────────────────────


def test_provenance_does_not_break_the_schema_contract():
    """schema_contract_ok rejects any key that is not a mapped field.

    It checks against CANONICAL_RECORD_FIELDS, which is the set of fields
    mapping.yaml can name — and provenance is not one, because the loader
    sets it rather than reading it from a source column. Any row carrying
    the tag would otherwise fail as "unknown field provenance", which reads
    like a malformed CSV rather than the loader's own doing.
    """
    ok, why = schema_contract_ok({
        "record_id": "WARRANTY-200001",
        "text": "front pads worn to the backing plate",
        "source": "WARRANTY",
        "received_at": utc_now(),
        "provenance": "fixture",
    })
    assert ok, why


def test_schema_contract_still_rejects_a_genuinely_unknown_field():
    ok, why = schema_contract_ok({
        "record_id": "WARRANTY-200001", "text": "t", "source": "WARRANTY",
        "received_at": utc_now(), "not_a_column": "x",
    })
    assert not ok
    assert "not_a_column" in why


# ── End to end, down to the column ───────────────────────────────────────────


def test_a_load_with_no_tag_is_observed(csv_path):
    report = _load(csv_path)
    assert report["provenance"] == "observed"
    assert report["upserted"] == 3
    assert set(_provenances().values()) == {"observed"}


def test_a_demo_file_can_be_loaded_as_a_fixture(csv_path):
    """The step this exists for: demo data must be loadable as demo data."""
    report = _load(csv_path, provenance="fixture")
    assert report["provenance"] == "fixture"
    got = _provenances()
    assert set(got) == {"WARRANTY-200001", "WARRANTY-200002", "WARRANTY-200003"}
    assert set(got.values()) == {"fixture"}


def test_inferred_is_loadable_too(csv_path):
    _load(csv_path, provenance="inferred")
    assert set(_provenances().values()) == {"inferred"}


def test_a_bad_tag_is_rejected_before_the_file_is_read(tmp_path, monkeypatch):
    """A rejected tag fails the same way whether or not the CSV exists.

    Validating after the filesystem work would report FileNotFoundError for
    a missing file and only mention the real problem on the retry.
    """
    monkeypatch.setenv("DOMAIN_DB_PATH", str(tmp_path))
    with pytest.raises(ValueError, match="unknown provenance"):
        ingest_source(
            PACK, "warranty", tmp_path / "does_not_exist.csv",
            mapping_path=WARRANTY_MAPPING, provenance="nonsense",
        )


def test_a_later_load_can_promote_fixture_rows_to_observed(csv_path):
    """The tag travels with the row, so a re-ingest must actually retag it."""
    _load(csv_path, provenance="fixture")
    assert set(_provenances().values()) == {"fixture"}
    _load(csv_path, provenance="observed")
    assert set(_provenances().values()) == {"observed"}


def test_limit_applies_with_a_tag(csv_path):
    report = _load(csv_path, limit=2, provenance="fixture")
    assert report["mapped"] == 2
    assert len(_provenances()) == 2


# ── The job queue ────────────────────────────────────────────────────────────


def test_job_payload_rejects_an_unknown_provenance_at_enqueue(csv_path):
    """Failing in the worker is too late — the caller is long gone by then."""
    from src.jobs.queue import validate_job_payload

    with pytest.raises(ValueError, match="unknown provenance"):
        validate_job_payload("ingest_source", {
            "pack_id": PACK, "source": "warranty",
            "csv_path": str(csv_path), "provenance": "demo",
        })


def test_job_payload_normalizes_a_valid_provenance(csv_path):
    from src.jobs.queue import validate_job_payload

    out = validate_job_payload("ingest_source", {
        "pack_id": PACK, "source": "warranty",
        "csv_path": str(csv_path), "provenance": "  FIXTURE ",
    })
    assert out["provenance"] == "fixture"


def test_job_payload_leaves_an_absent_provenance_absent(csv_path):
    """An omitted key must not arrive downstream as the string 'None'."""
    from src.jobs.queue import validate_job_payload

    out = validate_job_payload("ingest_source", {
        "pack_id": PACK, "source": "warranty", "csv_path": str(csv_path),
    })
    assert out.get("provenance") in (None, "")


def test_the_handler_forwards_provenance_and_limit(csv_path, monkeypatch):
    """Both were dropped on the floor: the job ran, but not as asked.

    ``limit`` was never forwarded at all, so a job enqueued with a bound
    quietly loaded the whole file. _default_handler swallows exceptions into
    {"ok": False}, so the return value is asserted too — otherwise a
    signature mismatch in the spy would read as a pass.
    """
    import src.domains.source_ingest as si
    from src.jobs.queue import _default_handler

    seen: dict[str, object] = {}

    def spy(pack_id, source, csv_path, **kw):
        seen.update({"pack_id": pack_id, "source": source, "csv_path": csv_path})
        seen.update(kw)
        return {"ok": True, "spied": True}

    monkeypatch.setattr(si, "ingest_source", spy)
    out = _default_handler("ingest_source", {
        "pack_id": PACK, "source": "warranty", "csv_path": str(csv_path),
        "limit": 5, "provenance": "fixture",
    })
    assert out == {"ok": True, "spied": True}, "handler swallowed an error"
    assert seen["limit"] == 5
    assert seen["provenance"] == "fixture"


def test_the_handler_passes_no_provenance_when_the_payload_has_none(csv_path, monkeypatch):
    import src.domains.source_ingest as si
    from src.jobs.queue import _default_handler

    seen: dict[str, object] = {}

    def spy(pack_id, source, csv_path, **kw):
        seen.update(kw)
        return {"ok": True}

    monkeypatch.setattr(si, "ingest_source", spy)
    _default_handler("ingest_source", {
        "pack_id": PACK, "source": "warranty", "csv_path": str(csv_path),
    })
    assert seen["provenance"] is None, "None means 'use the default', not 'invalid'"


# ── The CLI ──────────────────────────────────────────────────────────────────


def test_cli_loads_a_fixture_and_says_so(tmp_path, monkeypatch, capsys):
    """The report line has to name the tag: a silent fixture load is the bug."""
    from scripts.ingest_source import main

    monkeypatch.setenv("DOMAIN_DB_PATH", str(tmp_path))
    path = tmp_path / "warranty.csv"
    path.write_text(WARRANTY_CSV, encoding="utf-8")
    rc = main([
        "--pack", PACK, "--source", "warranty", "--csv", str(path),
        "--mapping", str(WARRANTY_MAPPING), "--provenance", "fixture",
    ])
    assert rc == 0
    assert "as fixture" in capsys.readouterr().out
    assert set(_provenances().values()) == {"fixture"}


def test_cli_rejects_an_unknown_provenance(tmp_path, monkeypatch):
    """argparse choices means a typo never reaches the warehouse at all."""
    from scripts.ingest_source import main

    monkeypatch.setenv("DOMAIN_DB_PATH", str(tmp_path))
    path = tmp_path / "warranty.csv"
    path.write_text(WARRANTY_CSV, encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        main([
            "--pack", PACK, "--source", "warranty", "--csv", str(path),
            "--mapping", str(WARRANTY_MAPPING), "--provenance", "demo",
        ])
    assert exc.value.code == 2
