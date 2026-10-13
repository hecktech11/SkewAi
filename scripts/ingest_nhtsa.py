"""Ingest real NHTSA complaints through the pack mapping.yaml loader.

This is **not** ``ingest_scale`` (synthetic clones of the 10-row fixture).

Usage
-----
    python -m scripts.ingest_nhtsa --limit 50000
    python -m scripts.ingest_nhtsa --csv /path/to/complaints.csv --limit 50000
    NHTSA_COMPLAINTS_CSV=/path/to/file.csv python -m scripts.ingest_nhtsa

Download (when ``--csv`` / env is unset):
    Official ODI flat file ZIP: https://static.nhtsa.gov/odi/ffdd/cmpl/FLAT_CMPL.zip
    Converted to a headered CSV whose columns match
    ``domains/automotive_nhtsa/data/mapping.yaml``.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import REPO_ROOT
from src.domains.mapping_ingest import ingest_mapped_csv
from src.domains.nhtsa_components import (
    UNKNOWN,
    category_values,
    normalize_component,
    validate_alias_table,
)

PACK_ID = "automotive_nhtsa"
DEFAULT_LIMIT = 50_000
NHTSA_FLAT_URL = "https://static.nhtsa.gov/odi/ffdd/cmpl/FLAT_CMPL.zip"

#: A re-iterable line source. The conversion needs two passes, so it takes a
#: factory rather than an iterator.
LineSource = Callable[[], Iterable[str]]

# Official FLAT_CMPL column order (NHTSA CMPL.txt). Positional: the row is
# split on tabs and zipped against this list, so order matters and length
# must keep up with NHTSA's additions.
_FLAT_FIELDS = [
    "CMPLID",
    "ODINO",
    "MFR_NAME",
    "MAKETXT",
    "MODELTXT",
    "YEARTXT",
    "CRASH",
    "FAILDATE",
    "FIRE",
    "INJURED",
    "DEATHS",
    "COMPDESC",
    "CITY",
    "STATE",
    "VIN",
    "DATEA",
    "LDATE",
    "MILES",
    "OCCURENCES",
    "CDESCR",
    "CMPL_TYPE",
    "POLICE_RPT_YN",
    "PURCH_DT",
    "ORIG_OWNER_YN",
    "ANTI_BRAKES_YN",
    "CRUISE_CONT_YN",
    "NUM_CYLS",
    "DRIVE_TRAIN",
    "FUEL_SYS",
    "FUEL_TYPE",
    "TRANS_TYPE",
    "VEH_SPEED",
    "DOT",
    "TIRE_SIZE",
    "LOC_OF_TIRE",
    "TIRE_FAIL_TYPE",
    "ORIG_EQUIP_YN",
    "MANUF_DT",
    "SEAT_TYPE",
    "RESTRAINT_TYPE",
    "DEALER_NAME",
    "DEALER_TEL",
    "DEALER_CITY",
    "DEALER_STATE",
    "DEALER_ZIP",
    "PROD_TYPE",
    "REPAIRED_YN",
    "MEDICAL_ATTN",
    "VEHICLES_TOWED_YN",
    # Added by NHTSA on 2026-04-30. Every line in the current corpus has
    # exactly 51 tab-separated fields; without these two the positional
    # enumerate() below silently drops them.
    "STATE_OF_INCIDENT",
    "VEHICLE_OPERATOR",
]

#: Columns of the intermediate CSV that ``mapping.yaml`` reads.
#:
#: COMPONENT is the normalized component and is what reaches
#: ``records.category``; COMPDESC keeps the full raw hierarchy and reaches
#: ``records.subcategory``; CDESC is the narrative and reaches ``records.text``.
#: Before this split, ``category: CDESC`` put the narrative into the category
#: column, giving a ~770-value free-text vocabulary where every consumer
#: matches by exact string equality.
CSV_HEADERS = [
    "RECORD_KEY",
    "ODINO",
    "CMPLID",
    "DATEA",
    "DATEC",
    "MODEL_YR",
    "MAKETXT",
    "MODELTXT",
    "COMPONENT",
    "COMPDESC",
    "CDESC",
    "NUM_CYLS",
    "STATE",
]

#: A line must have at least this many fields to carry a narrative.
#: CDESCR is index 19, so a usable line has >= 20 fields. (COMPDESC is 11,
#: ODINO is 1.) The live corpus has exactly 51 fields on every line.
_MIN_FLAT_FIELDS = 20


def flat_line_to_mapping_row(line: str) -> dict[str, str] | None:
    """Parse one tab-delimited FLAT_CMPL line into :data:`CSV_HEADERS`.

    Returns None when the line cannot yield a usable record: too few fields,
    no ODINO, or no narrative. Each FLAT line carries exactly one component,
    so this stays strictly one-line-in / one-row-out — a complaint that names
    several components arrives as several lines sharing an ODINO, and
    collapsing those is :func:`write_mapping_csv_from_flat`'s job.
    """
    parts = line.rstrip("\n\r").split("\t")
    if len(parts) < _MIN_FLAT_FIELDS:
        parts = line.rstrip("\n\r").split("|")
    if len(parts) < 16:
        return None
    raw = {name: (parts[i].strip() if i < len(parts) else "") for i, name in enumerate(_FLAT_FIELDS)}
    # `text` is the narrative only. COMPDESC is no longer an acceptable
    # fallback: it names a part, and a part name embedded as a complaint
    # narrative produces a degenerate cluster. 39 corpus rows have a blank
    # CDESCR and are dropped here rather than silently later.
    text = raw.get("CDESCR") or ""
    compdesc = raw.get("COMPDESC") or ""
    if not raw.get("ODINO") or not text:
        return None
    row = {
        "RECORD_KEY": "",
        "ODINO": raw["ODINO"],
        "CMPLID": raw.get("CMPLID") or "",
        "DATEA": raw.get("DATEA") or "",
        "DATEC": raw.get("FAILDATE") or "",
        "MODEL_YR": raw.get("YEARTXT") or "",
        "MAKETXT": raw.get("MAKETXT") or "",
        "MODELTXT": raw.get("MODELTXT") or "",
        "COMPONENT": normalize_component(compdesc),
        "COMPDESC": compdesc,
        "CDESC": text,
        "NUM_CYLS": raw.get("NUM_CYLS") or "",
        "STATE": raw.get("STATE") or "",
    }
    row["RECORD_KEY"] = record_key(row)
    return row


#: Separator inside the composite record key. Verified absent from every
#: COMPDESC, MAKETXT and MODELTXT value in the corpus, and already the
#: separator `entity_key` uses ("2019|HONDA|CR-V").
KEY_SEP = "|"


def record_key(row: dict[str, str]) -> str:
    """Build the composite ``record_id`` for a converted row.

    One NHTSA complaint (ODINO) can name several components and, less
    obviously, can span several vehicles: 41,616 ODINOs cover more than one
    year/make/model. Keying on ODINO alone would collapse genuinely different
    (vehicle, component) failures into one record; keying on ODINO|component
    alone would collapse the same component across different vehicles, which
    is exactly the grouping every anomaly baseline is computed over.

    Measured on the live corpus: 2,250,304 lines collapse to 2,174,511 keys,
    so this merges 75,793 rows (3.4%) rather than fanning out. CMPLID — the
    old key — is unique per line, which is why it produced one record per
    complaint/component line with no way to count a complaint once.
    """
    return KEY_SEP.join(
        (
            row["ODINO"],
            row["MODEL_YR"],
            row["MAKETXT"],
            row["MODELTXT"],
            row["COMPONENT"],
        )
    )


def _select_winners(
    line_source: LineSource,
    *,
    keep: Callable[[dict[str, str]], bool] | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Pass 1: pick one line per composite key.

    Returns ``(key -> winning CMPLID, tally)``. The winner is the row with
    the longest narrative, ties going to the first occurrence, which is
    deterministic for a given input order.

    Only the winning CMPLID is retained, not the row — holding 2.17M full
    rows would cost >1 GB, most of it narrative text that pass 2 re-reads
    for free. State is kept in one dict rather than three so the corpus
    costs one hash table, not three.

    This cannot be a streaming single pass: 22,532 ODINOs reappear
    non-contiguously in the corpus, so a key's last line can be millions of
    lines after its first.
    """
    # key -> (best_narrative_len, winning_cmplid, hash of the first narrative)
    state: dict[str, tuple[int, str, int]] = {}
    conflicted: set[str] = set()
    tally: dict[str, Any] = {
        "read": 0,
        "skipped_blank_line": 0,
        "skipped_unparseable": 0,
        "skipped_out_of_bounds": 0,
        "collapsed": 0,
    }
    for line in line_source():
        tally["read"] += 1
        if not line.strip():
            tally["skipped_blank_line"] += 1
            continue
        row = flat_line_to_mapping_row(line)
        if row is None:
            tally["skipped_unparseable"] += 1
            continue
        if keep is not None and not keep(row):
            tally["skipped_out_of_bounds"] += 1
            continue
        key = row["RECORD_KEY"]
        text = row["CDESC"]
        prior = state.get(key)
        if prior is None:
            state[key] = (len(text), row["CMPLID"], hash(text))
            continue
        tally["collapsed"] += 1
        best_len, _, first_hash = prior
        if hash(text) != first_hash:
            # The merge is about to discard a narrative that differs from
            # the one it keeps. Counted per key, not per line, because that
            # is the number of records with text loss.
            conflicted.add(key)
        if len(text) > best_len:
            state[key] = (len(text), row["CMPLID"], first_hash)
    tally["conflicting_narrative"] = len(conflicted)
    return {k: v[1] for k, v in state.items()}, tally


def write_mapping_csv_from_flat(
    line_source: LineSource,
    dest: Path,
    *,
    limit: int,
    keep: Callable[[dict[str, str]], bool] | None = None,
) -> dict[str, int]:
    """Convert FLAT_CMPL lines to the mapping CSV; return a conversion report.

    ``line_source`` is a zero-argument callable returning an iterable of
    lines. It is called twice — see :func:`_select_winners` for why one pass
    is not enough.

    Each complaint/component/vehicle pair is written exactly once. NHTSA
    emits one line per component, so a complaint naming several components
    arrives as several lines sharing an ODINO; the old CMPLID key made each
    of those a separate record, so a single complaint could be counted up to
    17 times.

    The report accounts for every line read, so a short load is diagnosable
    instead of merely surprising. ``collapsed`` is the number of lines that
    lost to a duplicate; ``conflicting_narrative`` is the number of keys
    where the merge discarded a different narrative (7,455, 0.34%, on the
    live corpus).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    winner, tally = _select_winners(line_source, keep=keep)

    report = {
        "read": tally["read"],
        "written": 0,
        "skipped_blank_line": tally["skipped_blank_line"],
        "skipped_unparseable": tally["skipped_unparseable"],
        "skipped_out_of_bounds": tally["skipped_out_of_bounds"],
        "collapsed": tally["collapsed"],
        "conflicting_narrative": tally["conflicting_narrative"],
        "unknown_component": 0,
        "distinct_keys": len(winner),
        "truncated_by_limit": 0,
    }

    written_keys: set[str] = set()
    with dest.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=CSV_HEADERS)
        writer.writeheader()
        for line in line_source():
            if not line.strip():
                continue
            row = flat_line_to_mapping_row(line)
            if row is None:
                continue
            if keep is not None and not keep(row):
                continue
            key = row["RECORD_KEY"]
            if winner.get(key) != row["CMPLID"] or key in written_keys:
                continue
            if report["written"] >= limit:
                report["truncated_by_limit"] = len(winner) - report["written"]
                break
            written_keys.add(key)
            if row["COMPONENT"] == UNKNOWN:
                report["unknown_component"] += 1
            writer.writerow(row)
            report["written"] += 1
    return report


def _download_flat_zip(url: str, dest_zip: Path) -> None:
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "skewai-nhtsa-ingest/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, dest_zip.open("wb") as fh:
        while True:
            chunk = resp.read(1024 * 256)
            if not chunk:
                break
            fh.write(chunk)


def _iter_flat_from_zip(zip_path: Path) -> Iterable[str]:
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        if not names:
            raise RuntimeError(f"empty zip: {zip_path}")
        with zf.open(names[0]) as raw:
            wrapper = io.TextIOWrapper(raw, encoding="latin-1", errors="replace")
            yield from wrapper


#: Header columns a hand-supplied ``--csv`` must carry for the preflight to
#: be able to check it. These are the ones that reach a non-free-text column
#: of ``records``.
_REQUIRED_LOCAL_COLUMNS = ("RECORD_KEY", "ODINO", "COMPONENT", "CDESC", "DATEA")


def preflight_local_csv(path: Path, *, limit: int, pack_id: str = PACK_ID) -> dict[str, int]:
    """Verify a hand-supplied CSV carries an already-normalized COMPONENT.

    ``--csv`` / ``NHTSA_COMPLAINTS_CSV`` skips the FLAT conversion, which is
    where normalization happens. Without this gate a local file silently
    reintroduces the free-text category vocabulary that the conversion exists
    to eliminate — the failure mode is invisible because every consumer
    matches categories by string equality and simply returns nothing.

    Raises ValueError on a missing column or an out-of-vocabulary COMPONENT.
    """
    vocabulary = category_values(pack_id)
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        missing = [c for c in _REQUIRED_LOCAL_COLUMNS if c not in header]
        if missing:
            raise ValueError(
                f"{path.name} is missing required column(s) {missing}. A local CSV must "
                f"already be in the converted layout ({', '.join(CSV_HEADERS)}); run "
                f"without --csv to build it from FLAT_CMPL."
            )
        checked = 0
        offenders: dict[str, int] = {}
        seen_keys: set[str] = set()
        duplicate_keys = 0
        for row in reader:
            if checked >= limit:
                break
            checked += 1
            value = (row.get("COMPONENT") or "").strip()
            if value not in vocabulary:
                offenders[value] = offenders.get(value, 0) + 1
            key = (row.get("RECORD_KEY") or "").strip()
            if key in seen_keys:
                duplicate_keys += 1
            seen_keys.add(key)
    if offenders:
        worst = sorted(offenders.items(), key=lambda kv: -kv[1])[:5]
        raise ValueError(
            f"{path.name} has {sum(offenders.values())} row(s) across "
            f"{len(offenders)} COMPONENT value(s) that are not in the pack's "
            f"categories gazetteer, e.g. {worst}. Normalize with "
            f"src.domains.nhtsa_components.normalize_component before loading."
        )
    if duplicate_keys:
        # upsert_records is INSERT OR REPLACE, so duplicates do not error —
        # the last one silently wins and the row count is short of `mapped`.
        raise ValueError(
            f"{path.name} has {duplicate_keys} duplicate RECORD_KEY value(s). "
            f"Each complaint/vehicle/component pair must appear once; "
            f"duplicates are silently overwritten at upsert."
        )
    return {"checked": checked, "written": checked}


def bounds_keep(
    *,
    makes: list[str] | None,
    year_min: int | None,
    year_max: int | None,
) -> Callable[[dict[str, str]], bool] | None:
    """Row filter for a bounded make/year load. None keeps every parsed row."""
    make_set = {m.strip().upper() for m in makes if m.strip()} if makes else set()
    if not make_set and year_min is None and year_max is None:
        return None

    def keep(row: dict[str, str]) -> bool:
        if make_set and (row.get("MAKETXT") or "").strip().upper() not in make_set:
            return False
        if year_min is None and year_max is None:
            return True
        try:
            year = int((row.get("MODEL_YR") or "").strip())
        except ValueError:
            return False
        if year_min is not None and year < year_min:
            return False
        if year_max is not None and year > year_max:
            return False
        return True

    return keep


def resolve_complaints_csv(
    *,
    csv_arg: str | None,
    limit: int,
    work_dir: Path,
    keep: Callable[[dict[str, str]], bool] | None = None,
) -> tuple[Path, str, dict[str, int]]:
    """Return (csv_path, provenance, report).

    provenance is 'local' or 'nhtsa_flat_download'.
    """
    local = csv_arg or os.environ.get("NHTSA_COMPLAINTS_CSV", "").strip()
    if local:
        # The jail runs before the first open. preflight_local_csv reads the
        # file, so checking only inside ingest_mapped_csv would be too late.
        p = safe_csv_path(local)
        if not p.exists():
            raise FileNotFoundError(f"NHTSA CSV not found: {p}")
        return p, "local", preflight_local_csv(p, limit=limit)

    zip_path = work_dir / "FLAT_CMPL.zip"
    csv_path = work_dir / "nhtsa_complaints_mapped.csv"
    _download_flat_zip(NHTSA_FLAT_URL, zip_path)
    report = write_mapping_csv_from_flat(
        lambda: _iter_flat_from_zip(zip_path), csv_path, limit=limit, keep=keep
    )
    if report["written"] == 0:
        raise RuntimeError(f"FLAT_CMPL produced 0 mapped rows (report={report})")
    return csv_path, "nhtsa_flat_download", report


@contextmanager
def authorize_converted_work(work_dir: Path):
    """Let this process read the CSV the flat-file converter just wrote.

    ``safe_csv_path`` allows the repo and ``FRONTLINE_INGEST_ROOT`` only.
    The converter's temp directory is outside the repo. The previous root
    is restored when the ingest call returns.
    """
    key = "FRONTLINE_INGEST_ROOT"
    prior = os.environ.get(key)
    os.environ[key] = str(work_dir)
    try:
        yield
    finally:
        if prior is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prior


def run(
    *,
    pack_id: str = PACK_ID,
    csv_path: str | None = None,
    limit: int = DEFAULT_LIMIT,
    purge_first: bool = False,
    makes: list[str] | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
) -> dict:
    # Fail before the download rather than after it: a stale alias table
    # would send real components to UNKNOWN OR OTHER for the whole load.
    validate_alias_table(pack_id)
    work = Path(tempfile.mkdtemp(prefix="skewai-nhtsa-"))
    try:
        src, provenance, convert = resolve_complaints_csv(
            csv_arg=csv_path,
            limit=limit,
            work_dir=work,
            keep=bounds_keep(makes=makes, year_min=year_min, year_max=year_max),
        )
        purged = None
        if purge_first:
            # Required whenever the record_id scheme has changed. Without it
            # the old-key rows survive the INSERT OR REPLACE and the corpus
            # doubles with no error.
            from src.domains.purge import purge_source

            purged = purge_source(pack_id, "NHTSA")
        def _load():
            return ingest_mapped_csv(
                pack_id,
                src,
                limit=limit,
                allow_untrusted_historical_backfill=True,
            )

        if provenance == "nhtsa_flat_download":
            with authorize_converted_work(work):
                result = _load()
        else:
            result = _load()
        result["provenance"] = provenance
        result["convert"] = convert
        result["purged"] = purged
        result["limit"] = limit
        result["ok"] = True
        result["error"] = None
        return result
    except (
        urllib.error.URLError,
        TimeoutError,
        OSError,
        RuntimeError,
        ValueError,
        FileNotFoundError,
    ) as e:
        return {
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "provenance": "failed",
            "limit": limit,
            "pack_id": pack_id,
            "upserted": 0,
        }
    finally:
        # Keep converted CSV if download succeeded is unnecessary; tmp cleaned by OS.
        pass


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Ingest real NHTSA complaints via mapping.yaml")
    p.add_argument("--pack", default=PACK_ID)
    p.add_argument("--csv", default=None, help="Local headered CSV (skips download)")
    p.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    p.add_argument("--makes", default="", help="Comma-separated MAKETXT values")
    p.add_argument("--year-min", type=int, default=None)
    p.add_argument("--year-max", type=int, default=None)
    p.add_argument(
        "--purge-first",
        action="store_true",
        help="delete existing NHTSA records before loading (required after a "
        "record_id scheme change, or the corpus doubles)",
    )
    args = p.parse_args(argv)
    result = run(
        pack_id=args.pack,
        csv_path=args.csv,
        limit=args.limit,
        purge_first=args.purge_first,
        makes=[m for m in args.makes.split(",") if m.strip()] or None,
        year_min=args.year_min,
        year_max=args.year_max,
    )
    print(result)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
