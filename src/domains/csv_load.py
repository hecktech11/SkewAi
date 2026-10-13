"""Load a whole CSV into a pack warehouse.

Pack Builder profiles a file and seeds a short sample. This path writes every
accepted row, then can recompute weekly anomalies. Large files go through the
existing ``ingest_source`` job so the HTTP request does not sit open.
"""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import Any

from src.config import REPO_ROOT
from src.domains.mapping_ingest import load_mapping
from src.domains.source_ingest import ingest_source, normalize_provenance

_SOURCE_TAG_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
_PILOT_PACKS = frozenset({"automotive_nhtsa", "finance_cfpb"})
_RECORD_FIELDS = frozenset({
    "record_id",
    "text",
    "category",
    "subcategory",
    "received_at",
    "occurred_at",
    "entity_1",
    "entity_2",
    "entity_3",
    "entity_key",
    "severity_label",
    "region",
    "source",
    "event_count",
    "normalize_category",
})
DEFAULT_CSV_MAX_BYTES = 5 * 1024 * 1024
DEFAULT_INLINE_BYTES = 1_000_000


def as_bool(value: Any) -> bool:
    """True only for a real opt-in. The string ``false`` stays false."""
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def builder_csv_max_bytes() -> int:
    """HTTP upload and ingest cap. ``FRONTLINE_CSV_MAX_BYTES`` raises it."""
    return _positive_int("FRONTLINE_CSV_MAX_BYTES", DEFAULT_CSV_MAX_BYTES)


def csv_inline_max_bytes() -> int:
    """Larger than this, the HTTP ingest enqueues a job instead of waiting."""
    return _positive_int("FRONTLINE_CSV_INLINE_BYTES", DEFAULT_INLINE_BYTES)


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        n = int(raw)
    except ValueError:
        return default
    if n < 1:
        return default
    return n


def assert_not_pilot_warehouse(pack_id: str) -> None:
    """Block a load that would write the seeded pilot files.

    A side ``DOMAIN_DB_PATH`` is the way to score a real window. The CLI
    ``python -m scripts.ingest_source`` calls this unless ``--allow-pilot``
    is set. Job workers call ``ingest_source()`` directly and stay unchanged.
    """
    from src.config import is_default_pilot_domain_path

    if pack_id in _PILOT_PACKS and is_default_pilot_domain_path():
        raise ValueError(
            "refusing to load CSV into the pilot warehouse; "
            "set DOMAIN_DB_PATH to a side directory or choose another pack_id"
        )


def _spec_column(spec: Any) -> str:
    if not isinstance(spec, str):
        return ""
    key = spec.strip()
    if len(key) >= 2 and key[0] == key[-1] and key[0] in {'"', "'"}:
        return ""
    return key.lower()


def coerce_mapping(mapping: Any, *, normalize_components: bool = False) -> dict[str, Any]:
    """Accept a pack-builder field map or a full ``{records: ...}`` document."""
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("mapping must be an object")
    if isinstance(mapping.get("records"), dict):
        records = dict(mapping["records"])
        doc = dict(mapping)
    else:
        unknown = sorted(set(mapping) - _RECORD_FIELDS)
        if unknown:
            raise ValueError(f"unknown mapping field {unknown[0]}")
        records = dict(mapping)
        doc = {"records": records}
    if not records.get("record_id") or not records.get("text"):
        raise ValueError("mapping needs record_id and text")
    text_col = _spec_column(records.get("text"))
    cat_col = _spec_column(records.get("category"))
    if text_col and cat_col and text_col == cat_col:
        raise ValueError("category must not be the narrative column")
    if normalize_components:
        records["normalize_category"] = "nhtsa"
    doc["records"] = records
    return doc


def _mapping_dest(csv_path: Path) -> Path:
    from secrets import token_hex

    raw = os.getenv("BUILDER_UPLOAD_DIR", "").strip()
    if raw:
        root = Path(raw).expanduser()
    elif os.getenv("PYTEST_CURRENT_TEST") or os.getenv("FRONTLINE_TEST_ISOLATION"):
        import tempfile

        root = Path(tempfile.gettempdir())
    else:
        root = REPO_ROOT / "data" / "uploads"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{csv_path.stem}_{token_hex(4)}.mapping.yaml"


def _write_mapping(csv_path: Path, doc: dict[str, Any]) -> Path:
    import yaml

    dest = _mapping_dest(csv_path)
    dest.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return dest


def materialize_mapping(
    csv_path: str | Path,
    *,
    mapping: Any = None,
    mapping_path: str | Path | None = None,
    normalize_components: bool = False,
) -> Path:
    """Return a yaml path the ingest job can read. Never edits a shipped mapping."""
    from src.security.identifiers import safe_mapping_path

    if mapping_path not in (None, ""):
        src = safe_mapping_path(mapping_path)
        if not src.is_file():
            raise FileNotFoundError(f"mapping not found: {src}")
        if not normalize_components:
            return src
        doc = load_mapping(src)
        records = dict(doc["records"])
        records["normalize_category"] = "nhtsa"
        doc = dict(doc)
        doc["records"] = records
        return _write_mapping(Path(csv_path), doc)
    doc = coerce_mapping(mapping, normalize_components=normalize_components)
    return _write_mapping(Path(csv_path), doc)


def _data_rows(csv_path: Path) -> int:
    with csv_path.open(newline="", encoding="utf-8", errors="replace") as fh:
        return sum(1 for _ in csv.DictReader(fh))


def load_csv_rows(
    pack_id: str,
    csv_path: str | Path,
    *,
    mapping: Any = None,
    mapping_path: str | Path | None = None,
    source: str = "csv",
    provenance: str | None = None,
    run_anomalies: bool = False,
    normalize_components: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    """Ingest every mapped row. Returns written, skipped, and anomaly count."""
    from src.security.identifiers import safe_pack_id

    pack_id = str(safe_pack_id(pack_id))
    assert_not_pilot_warehouse(pack_id)
    tag = (source or "").strip().lower()
    if not _SOURCE_TAG_RE.match(tag):
        raise ValueError("source must be a short alphanumeric tag")
    prov = normalize_provenance(provenance)
    mpath = materialize_mapping(
        csv_path,
        mapping=mapping,
        mapping_path=mapping_path,
        normalize_components=normalize_components,
    )
    report = ingest_source(
        pack_id,
        tag,
        csv_path,
        mapping_path=mpath,
        limit=limit,
        provenance=prov,
    )
    written = int(report.get("upserted") or 0)
    skipped = int(report.get("skipped") or 0)
    if limit is None:
        skipped = max(skipped, _data_rows(Path(report["csv_path"])) - written)
    anomalies_scored = None
    if run_anomalies:
        from src.ml_runtime.anomalies import recompute_weekly_anomalies

        anomalies_scored = len(recompute_weekly_anomalies(pack_id))
    return {
        "written": written,
        "skipped": skipped,
        "job_id": None,
        "anomalies_scored": anomalies_scored,
        "pack_id": pack_id,
        "mapping_path": str(mpath),
        "upserted": written,
        "provenance": report.get("provenance"),
    }


__all__ = [
    "as_bool",
    "assert_not_pilot_warehouse",
    "builder_csv_max_bytes",
    "coerce_mapping",
    "csv_inline_max_bytes",
    "load_csv_rows",
    "materialize_mapping",
]
