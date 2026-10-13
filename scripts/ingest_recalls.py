"""Ingest NHTSA recall flat file FLAT_RCL into ``advisories``.

The column contract is the ``advisories:`` block in the pack mapping.yaml.
One campaign covers many models, so ``advisory_id`` is

    CAMPNO|YEARTXT|MAKETXT|MODELTXT|COMPONENT

``issued_at`` is RCDATE (the Part 573 report-received date), not ODATE and
not DATEA. ``scope_category`` is COMPNAME run through the same
``normalize_component`` complaints use.

Usage
-----
    python -m scripts.ingest_recalls
    python -m scripts.ingest_recalls --makes HONDA,TOYOTA,FORD,HYUNDAI,KIA \\
        --year-min 2015 --year-max 2024 --rcl-type V
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import REPO_ROOT
from src.data.warehouse import apply_domain_schema, domain_con
from src.domains.mapping_ingest import _cell, _parse_ts
from src.domains.nhtsa_components import normalize_component, validate_alias_table

PACK_ID = "automotive_nhtsa"
# NHTSA retired the single FLAT_RCL.zip (404 as of 2026-09). The post-2010
# flat file is the same field layout and covers the 2015-2024 window.
NHTSA_RECALL_URL = "https://static.nhtsa.gov/odi/ffdd/rcl/FLAT_RCL_POST_2010.zip"
RECALL_PAGE = "https://www.nhtsa.gov/recalls?nhtsaId="

# Official FLAT_RCL order (NHTSA RCL.txt, May 2025). Positional: tabs are
# zipped against this list. RCDATE is field 16 and stays put when NHTSA
# appends columns after it (DO_NOT_DRIVE, PARK_OUTSIDE).
FLAT_RCL_FIELDS = [
    "RECORD_ID",
    "CAMPNO",
    "MAKETXT",
    "MODELTXT",
    "YEARTXT",
    "MFGCAMPNO",
    "COMPNAME",
    "MFGNAME",
    "BGMAN",
    "ENDMAN",
    "RCLTYPECD",
    "POTAFF",
    "ODATE",
    "INFLUENCED_BY",
    "MFGTXT",
    "RCDATE",
    "DATEA",
    "RPNO",
    "FMVSS",
    "DESC_DEFECT",
    "CONEQUENCE_DEFECT",
    "CORRECTIVE_ACTION",
    "NOTES",
    "RCL_CMPT_ID",
    "MFR_COMP_NAME",
    "MFR_COMP_DESC",
    "MFR_COMP_PTNO",
    "DO_NOT_DRIVE",
    "PARK_OUTSIDE",
]

# RCDATE is index 15, so a usable line has at least 16 fields.
_MIN_FLAT_FIELDS = 16
KEY_SEP = "|"

ADVISORY_FIELDS = (
    "advisory_id",
    "issued_at",
    "scope_entity_1",
    "scope_entity_2",
    "scope_entity_3",
    "scope_category",
    "summary",
    "remedy",
    "url",
    "source",
)


def advisory_contract(pack_id: str = PACK_ID) -> dict[str, str]:
    """Return the mapping.yaml advisories block. Missing keys are an error."""
    path = REPO_ROOT / "domains" / pack_id / "data" / "mapping.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    block = raw.get("advisories")
    if not isinstance(block, dict):
        raise ValueError(f"advisories block missing in {path}")
    missing = [k for k in ADVISORY_FIELDS if k not in block]
    if missing:
        raise ValueError(f"advisories block missing {missing} in {path}")
    return {k: "" if block[k] is None else str(block[k]) for k in ADVISORY_FIELDS}


def flat_line_to_advisory_row(line: str) -> dict[str, str] | None:
    """Parse one tab-delimited FLAT_RCL line into the mapping.csv columns.

    Returns None when the line has no campaign number or no report-received
    date. An empty defect description falls back to the component name so
    ``summary`` can satisfy NOT NULL.
    """
    parts = line.rstrip("\n\r").split("\t")
    if len(parts) < _MIN_FLAT_FIELDS:
        parts = line.rstrip("\n\r").split("|")
    if len(parts) < _MIN_FLAT_FIELDS:
        return None
    raw = {
        name: (parts[i].strip() if i < len(parts) else "")
        for i, name in enumerate(FLAT_RCL_FIELDS)
    }
    camp = raw.get("CAMPNO") or ""
    rcdate = raw.get("RCDATE") or ""
    if not camp or _parse_ts(rcdate) is None:
        return None
    component = normalize_component(raw.get("COMPNAME") or "")
    year = (raw.get("YEARTXT") or "").strip()
    make = (raw.get("MAKETXT") or "").strip().upper()
    model = (raw.get("MODELTXT") or "").strip().upper()
    summary = (raw.get("DESC_DEFECT") or "").strip() or (raw.get("COMPNAME") or "").strip() or component
    key = KEY_SEP.join((camp.strip().upper(), year, make, model, component))
    return {
        "ADVISORY_KEY": key,
        "RCDATE": rcdate,
        "YEARTXT": year,
        "MAKETXT": make,
        "MODELTXT": model,
        "COMPONENT": component,
        "DESC_DEFECT": summary,
        "CORRECTIVE_ACTION": (raw.get("CORRECTIVE_ACTION") or "").strip(),
        "RCL_URL": RECALL_PAGE + camp.strip(),
        "CAMPNO": camp.strip(),
        "RCLTYPECD": (raw.get("RCLTYPECD") or "").strip().upper(),
    }


def _in_bounds(
    row: dict[str, str],
    *,
    makes: set[str] | None,
    year_min: int | None,
    year_max: int | None,
    rcl_type: str | None,
) -> bool:
    if rcl_type and row.get("RCLTYPECD") != rcl_type:
        return False
    if makes is not None and row.get("MAKETXT") not in makes:
        return False
    if year_min is None and year_max is None:
        return True
    try:
        year = int(row.get("YEARTXT") or "")
    except ValueError:
        return False
    if year_min is not None and year < year_min:
        return False
    if year_max is not None and year > year_max:
        return False
    return True


def iter_advisory_rows(
    lines: Iterable[str],
    *,
    makes: set[str] | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    rcl_type: str | None = None,
) -> Iterator[tuple[dict[str, str] | None, str]]:
    """Yield ``(row or None, skip_reason)`` for every non-empty line."""
    for line in lines:
        if not line.strip():
            yield None, "blank"
            continue
        row = flat_line_to_advisory_row(line)
        if row is None:
            yield None, "unparseable"
            continue
        if not _in_bounds(
            row, makes=makes, year_min=year_min, year_max=year_max, rcl_type=rcl_type
        ):
            yield None, "out_of_bounds"
            continue
        yield row, ""


def _iter_flat_from_zip(zip_path: Path) -> Iterator[str]:
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        if not names:
            raise RuntimeError(f"empty zip: {zip_path}")
        with zf.open(names[0]) as raw:
            import io

            wrapper = io.TextIOWrapper(raw, encoding="latin-1", errors="replace")
            yield from wrapper


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "skewai-nhtsa-ingest/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, dest.open("wb") as fh:
        while True:
            chunk = resp.read(1024 * 256)
            if not chunk:
                break
            fh.write(chunk)


def insert_advisories(
    pack_id: str,
    rows: Iterable[dict[str, str]],
    *,
    contract: dict[str, str] | None = None,
) -> dict[str, int]:
    """Insert mapping rows into ``advisories``. Last row wins on a repeated id."""
    spec = contract or advisory_contract(pack_id)
    written = 0
    skipped = 0
    with domain_con(pack_id, read_only=False) as con:
        apply_domain_schema(con)
        for row in rows:
            headers = set(row)
            issued = _parse_ts(_cell(row, spec["issued_at"], headers, "issued_at"))
            summary = _cell(row, spec["summary"], headers, "summary")
            advisory_id = _cell(row, spec["advisory_id"], headers, "advisory_id")
            if not advisory_id or issued is None or not summary:
                skipped += 1
                continue
            con.execute(
                """
                INSERT OR REPLACE INTO advisories (
                    advisory_id, issued_at, scope_entity_1, scope_entity_2,
                    scope_entity_3, scope_category, summary, remedy, url, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    advisory_id,
                    issued,
                    _cell(row, spec["scope_entity_1"], headers, "scope_entity_1"),
                    _cell(row, spec["scope_entity_2"], headers, "scope_entity_2"),
                    _cell(row, spec["scope_entity_3"], headers, "scope_entity_3"),
                    _cell(row, spec["scope_category"], headers, "scope_category"),
                    summary,
                    _cell(row, spec["remedy"], headers, "remedy"),
                    _cell(row, spec["url"], headers, "url"),
                    _cell(row, spec["source"], headers, "source") or "NHTSA",
                ],
            )
            written += 1
    return {"written": written, "skipped": skipped}


def run(
    *,
    pack_id: str = PACK_ID,
    zip_path: str | None = None,
    makes: list[str] | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    rcl_type: str | None = None,
) -> dict[str, Any]:
    """Download (or read) FLAT_RCL and insert the bounded advisory set."""
    validate_alias_table(pack_id)
    make_set = {m.strip().upper() for m in makes if m.strip()} if makes else None
    type_code = (rcl_type or "").strip().upper() or None
    work = Path(tempfile.mkdtemp(prefix="skewai-rcl-"))
    src = Path(zip_path).expanduser() if zip_path else work / "FLAT_RCL.zip"
    if zip_path and not src.is_file():
        return {"ok": False, "error": f"zip not found: {src}", "written": 0}
    try:
        if not zip_path:
            _download(NHTSA_RECALL_URL, src)
        rows = []
        tally = {"read": 0, "blank": 0, "unparseable": 0, "out_of_bounds": 0}
        for row, reason in iter_advisory_rows(
            _iter_flat_from_zip(src),
            makes=make_set,
            year_min=year_min,
            year_max=year_max,
            rcl_type=type_code,
        ):
            tally["read"] += 1
            if reason:
                tally[reason] = tally.get(reason, 0) + 1
                continue
            rows.append(row)
        inserted = insert_advisories(pack_id, rows)
        return {
            "ok": True,
            "error": None,
            "pack_id": pack_id,
            "written": inserted["written"],
            "skipped": inserted["skipped"],
            "rows": len(rows),
            "tally": tally,
        }
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
            "pack_id": pack_id,
            "written": 0,
        }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Ingest NHTSA FLAT_RCL into advisories")
    p.add_argument("--pack", default=PACK_ID)
    p.add_argument("--zip", default=None, help="Local FLAT_RCL.zip (skips download)")
    p.add_argument("--makes", default="", help="Comma-separated MAKETXT values")
    p.add_argument("--year-min", type=int, default=None)
    p.add_argument("--year-max", type=int, default=None)
    p.add_argument("--rcl-type", default="", help="RCLTYPECD filter, e.g. V for vehicles")
    args = p.parse_args(argv)
    makes = [m for m in args.makes.split(",") if m.strip()] or None
    result = run(
        pack_id=args.pack,
        zip_path=args.zip,
        makes=makes,
        year_min=args.year_min,
        year_max=args.year_max,
        rcl_type=args.rcl_type or None,
    )
    print(result)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
