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
from pathlib import Path
from typing import Iterable

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
    return {
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


def write_mapping_csv_from_flat(
    flat_lines: Iterable[str],
    dest: Path,
    *,
    limit: int,
) -> dict[str, int]:
    """Convert FLAT_CMPL lines to the mapping CSV; return a conversion report.

    The report carries a reason for every line that did not become a row, so
    a short load is diagnosable instead of merely surprising. Keys:
    ``read``, ``written``, ``skipped_blank_line``, ``skipped_unparseable``,
    ``unknown_component``.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "read": 0,
        "written": 0,
        "skipped_blank_line": 0,
        "skipped_unparseable": 0,
        "unknown_component": 0,
    }
    with dest.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=CSV_HEADERS)
        writer.writeheader()
        for line in flat_lines:
            report["read"] += 1
            if not line.strip():
                report["skipped_blank_line"] += 1
                continue
            row = flat_line_to_mapping_row(line)
            if row is None:
                report["skipped_unparseable"] += 1
                continue
            if row["COMPONENT"] == UNKNOWN:
                report["unknown_component"] += 1
            writer.writerow(row)
            report["written"] += 1
            if report["written"] >= limit:
                break
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
_REQUIRED_LOCAL_COLUMNS = ("ODINO", "COMPONENT", "CDESC", "DATEA")


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
        for row in reader:
            if checked >= limit:
                break
            checked += 1
            value = (row.get("COMPONENT") or "").strip()
            if value not in vocabulary:
                offenders[value] = offenders.get(value, 0) + 1
    if offenders:
        worst = sorted(offenders.items(), key=lambda kv: -kv[1])[:5]
        raise ValueError(
            f"{path.name} has {sum(offenders.values())} row(s) across "
            f"{len(offenders)} COMPONENT value(s) that are not in the pack's "
            f"categories gazetteer, e.g. {worst}. Normalize with "
            f"src.domains.nhtsa_components.normalize_component before loading."
        )
    return {"checked": checked, "written": checked}


def resolve_complaints_csv(
    *, csv_arg: str | None, limit: int, work_dir: Path
) -> tuple[Path, str, dict[str, int]]:
    """Return (csv_path, provenance, report).

    provenance is 'local' or 'nhtsa_flat_download'.
    """
    local = csv_arg or os.environ.get("NHTSA_COMPLAINTS_CSV", "").strip()
    if local:
        p = Path(local).expanduser()
        if not p.is_absolute():
            p = (REPO_ROOT / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"NHTSA CSV not found: {p}")
        return p, "local", preflight_local_csv(p, limit=limit)

    zip_path = work_dir / "FLAT_CMPL.zip"
    csv_path = work_dir / "nhtsa_complaints_mapped.csv"
    _download_flat_zip(NHTSA_FLAT_URL, zip_path)
    report = write_mapping_csv_from_flat(_iter_flat_from_zip(zip_path), csv_path, limit=limit)
    if report["written"] == 0:
        raise RuntimeError(f"FLAT_CMPL produced 0 mapped rows (report={report})")
    return csv_path, "nhtsa_flat_download", report


def run(
    *,
    pack_id: str = PACK_ID,
    csv_path: str | None = None,
    limit: int = DEFAULT_LIMIT,
) -> dict:
    # Fail before the download rather than after it: a stale alias table
    # would send real components to UNKNOWN OR OTHER for the whole load.
    validate_alias_table(pack_id)
    work = Path(tempfile.mkdtemp(prefix="skewai-nhtsa-"))
    try:
        src, provenance, convert = resolve_complaints_csv(
            csv_arg=csv_path, limit=limit, work_dir=work
        )
        result = ingest_mapped_csv(
            pack_id,
            src,
            limit=limit,
            allow_untrusted_historical_backfill=True,
        )
        result["provenance"] = provenance
        result["convert"] = convert
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
    args = p.parse_args(argv)
    result = run(pack_id=args.pack, csv_path=args.csv, limit=args.limit)
    print(result)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
