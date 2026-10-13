"""Execute pack ``data/mapping.yaml``: source CSV → canonical ``records``.

Mapping parse is pure (no DuckDB). Warehouse I/O is a separate step so tests
can feed a tiny CSV without touching the pilot domain file.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml

from src.config import REPO_ROOT
from src.data.timeutil import utc_now
from src.data.warehouse import domain_con
from src.ml_runtime.embeddings import embed_text

CANONICAL_RECORD_FIELDS = (
    "record_id",
    "received_at",
    "occurred_at",
    "entity_1",
    "entity_2",
    "entity_3",
    "category",
    "subcategory",
    "text",
    "severity_label",
    "region",
    "source",
    "entity_key",
)


def default_mapping_path(pack_id: str) -> Path:
    return REPO_ROOT / "domains" / pack_id / "data" / "mapping.yaml"


def _norm_entity_key(val: Any) -> str:
    """UPPER|separated canonical form; '' when nothing meaningful resolved."""
    parts = [p.strip().upper() for p in str(val or "").replace(":", "|").split("|")]
    parts = [p for p in parts if p]
    return "|".join(parts)


def load_mapping(path: str | Path) -> dict[str, Any]:
    """Load and validate a mapping.yaml. No I/O beyond reading the file."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"mapping must be a mapping document: {path}")
    records = raw.get("records")
    if not isinstance(records, dict) or not records:
        raise ValueError(f"mapping.records missing in {path}")
    if "record_id" not in records or "text" not in records:
        raise ValueError(f"mapping.records must include record_id and text: {path}")
    return raw


def _parse_ts(val: Any) -> datetime | None:
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    if s.isdigit() and len(s) == 8:
        try:
            return datetime(int(s[0:4]), int(s[4:6]), int(s[6:8]))
        except ValueError:
            return None
    # 1. ISO-8601 with or without timezone (handles T and space, timezone offsets, Z)
    try:
        iso_str = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(iso_str)
        if dt.tzinfo is not None:
            return dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except ValueError:
        pass
    # 2. Strptime formats with timezone offset
    for fmt in ("%Y-%m-%d %H:%M:%S%z", "%Y-%m-%d %H:%M:%S %z"):
        try:
            dt = datetime.strptime(s, fmt)
            if dt.tzinfo is not None:
                return dt.astimezone(timezone.utc).replace(tzinfo=None)
            return dt
        except ValueError:
            continue
    # 3. Naive standard formats
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%m/%d/%Y", "%Y%m%d"):
        try:
            return datetime.strptime(s[:19], fmt)
        except ValueError:
            continue
    return None


def _cell(row: dict[str, str], spec: Any, headers: set[str], field: str = "") -> str:
    """Resolve a mapping spec.

    - CSV column name → cell value (case-insensitive fallback);
    - ``"quoted"`` or ``'quoted'`` → literal constant;
    - ``{COL_A}|{COL_B}`` template → joined column values (used for the
      canonical cross-source ``entity_key``);
    - ``source`` field → literal string when not a header;
    - any other field where spec is not in headers → "" (do not fabricate values from column names).
    """
    if spec is None:
        return ""
    if not isinstance(spec, str):
        return str(spec).strip()
    key = spec.strip()
    if (key.startswith('"') and key.endswith('"') and len(key) >= 2) or (
        key.startswith("'") and key.endswith("'") and len(key) >= 2
    ):
        return key[1:-1]
    if "{" in key and "}" in key:
        import re as _re

        header_ci = {h.lower(): h for h in headers}

        def _sub(m: "_re.Match[str]") -> str:
            col = m.group(1).strip()
            if col in headers:
                return (row.get(col) or "").strip()
            if col.lower() in header_ci:
                return (row.get(header_ci[col.lower()]) or "").strip()
            return ""

        return _re.sub(r"\{([^{}]+)\}", _sub, key).strip()
    if key in headers:
        return (row.get(key) or "").strip()
    header_ci = {h.lower(): h for h in headers}
    if key.lower() in header_ci:
        return (row.get(header_ci[key.lower()]) or "").strip()
    # Literal (e.g. source: NHTSA) only allowed for source field or quoted strings
    if field == "source":
        return key
    return ""


def map_row(row: dict[str, Any], records_map: dict[str, Any]) -> dict[str, Any] | None:
    """Map one source dict to a canonical record. None if unusable."""
    headers = {str(k) for k in row.keys()}
    raw_row = {str(k): "" if v is None else str(v) for k, v in row.items()}
    out: dict[str, Any] = {}
    for field in CANONICAL_RECORD_FIELDS:
        spec = records_map.get(field)
        if spec is None:
            out[field] = ""
            continue
        out[field] = _cell(raw_row, spec, headers, field=field)
    rid = (out.get("record_id") or "").strip()
    text = (out.get("text") or "").strip()
    if not rid or not text:
        return None
    out["record_id"] = rid
    out["text"] = text

    # Validate received_at: if mapping declared a received_at column, require valid parsed timestamp
    raw_rec = out.get("received_at")
    declared_rec = records_map.get("received_at")
    if declared_rec:
        parsed_rec = _parse_ts(raw_rec)
        if parsed_rec is None:
            return None
        out["received_at"] = parsed_rec
    else:
        out["received_at"] = _parse_ts(raw_rec) or utc_now()

    out["occurred_at"] = _parse_ts(out.get("occurred_at"))
    out["entity_key"] = _norm_entity_key(out.get("entity_key"))
    spec = records_map.get("event_count")
    raw_count = _cell(raw_row, spec, headers, field="event_count") if spec is not None else ""
    try:
        parsed_count = int(float(raw_count)) if str(raw_count).strip() else 1
    except (TypeError, ValueError):
        parsed_count = 1
    out["event_count"] = parsed_count if parsed_count >= 1 else 1
    # Optional. A blank category stays blank: normalize_component would
    # otherwise fill it with UNKNOWN OR OTHER, which then matches nothing
    # and looks like a real component.
    if str(records_map.get("normalize_category") or "").strip().lower() == "nhtsa":
        raw_cat = str(out.get("category") or "").strip()
        if raw_cat:
            from src.domains.nhtsa_components import normalize_component

            out["category"] = normalize_component(raw_cat)
    return out


def iter_mapped_rows(
    csv_path: str | Path,
    mapping: dict[str, Any],
    *,
    limit: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield canonical records from a headered CSV. No warehouse writes."""
    records_map = mapping["records"]
    n = 0
    with Path(csv_path).open(newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rec = map_row(row, records_map)
            if rec is None:
                continue
            yield rec
            n += 1
            if limit is not None and n >= limit:
                return


#: Rows per warehouse round-trip. DuckDB's Python client is in autocommit,
#: so a per-row INSERT costs one commit per row — measured at 3.5 ms rising
#: to 17.5 ms over 8k rows (superlinear), which is 4+ hours for a 500k load.
#: Staging a batch as an Arrow table and inserting it with one statement
#: measures 116 µs/row and stays flat across batches.
_UPSERT_BATCH_ROWS = 10_000


def _arrow_schema():
    """Staging schema for one batch.

    Every column is typed explicitly. A batch where one column is entirely
    NULL — occurred_at on a source with no incident date, entity_key on a
    source with no entities — would otherwise infer as Arrow ``null`` and
    fail to cast into its DuckDB column.
    """
    import pyarrow as pa

    return pa.schema(
        [
            ("record_id", pa.string()),
            ("occurred_at", pa.timestamp("us")),
            ("received_at", pa.timestamp("us")),
            ("entity_1", pa.string()),
            ("entity_2", pa.string()),
            ("entity_3", pa.string()),
            ("category", pa.string()),
            ("subcategory", pa.string()),
            ("text", pa.string()),
            ("severity_label", pa.string()),
            ("region", pa.string()),
            ("source", pa.string()),
            ("embedding", pa.list_(pa.float32())),
            ("entity_key", pa.string()),
            ("provenance", pa.string()),
            ("event_count", pa.int64()),
        ]
    )


def _staged_row(rec: dict[str, Any]) -> dict[str, Any]:
    """One canonical record as a column-ordered dict for Arrow staging."""
    return {
        "record_id": rec["record_id"],
        "occurred_at": rec.get("occurred_at"),
        "received_at": rec["received_at"],
        "entity_1": rec.get("entity_1") or None,
        "entity_2": rec.get("entity_2") or None,
        "entity_3": rec.get("entity_3") or None,
        "category": rec.get("category") or None,
        "subcategory": rec.get("subcategory") or None,
        "text": rec["text"],
        "severity_label": rec.get("severity_label") or None,
        "region": rec.get("region") or None,
        "source": rec.get("source") or None,
        "embedding": embed_text(rec["text"]),
        "entity_key": rec.get("entity_key") or None,
        "provenance": rec.get("provenance") or "observed",
        "event_count": int(rec.get("event_count") or 1),
    }


def _flush_batch(con, batch: list[dict[str, Any]]) -> None:
    """Insert one staged batch with a single statement."""
    if not batch:
        return
    import pyarrow as pa

    table = pa.Table.from_pylist(batch, schema=_arrow_schema())
    cols = ", ".join(table.schema.names)
    con.register("_upsert_batch", table)
    try:
        con.execute(f"INSERT OR REPLACE INTO records ({cols}) SELECT {cols} FROM _upsert_batch")
    finally:
        con.unregister("_upsert_batch")


def upsert_records(pack_id: str, records: Iterable[dict[str, Any]]) -> int:
    """Insert-or-replace canonical rows into the pack domain warehouse.

    Returns the number of records consumed, which is not the same as rows
    changed: a re-ingest of the same CSV replaces every row and still
    reports the full count.

    Duplicate record_ids inside one batch are resolved keep-last, in Python,
    before staging. This matters and is easy to get wrong: a per-row loop
    and ``executemany`` both let the LAST write win, but a single-statement
    ``INSERT OR REPLACE ... SELECT`` keeps the FIRST and drops the rest
    silently, with no error and no short row count to notice.
    """
    count = 0
    batch: list[dict[str, Any]] = []
    index: dict[str, int] = {}  # record_id -> position in `batch`
    with domain_con(pack_id, read_only=False) as con:
        # domain_con already applied the schema on a writable connection.
        for rec in records:
            count += 1
            staged = _staged_row(rec)
            rid = staged["record_id"]
            prior = index.get(rid)
            if prior is not None:
                batch[prior] = staged  # keep-last, matching the old per-row loop
            else:
                index[rid] = len(batch)
                batch.append(staged)
            if len(batch) >= _UPSERT_BATCH_ROWS:
                _flush_batch(con, batch)
                batch = []
                index = {}
        _flush_batch(con, batch)
    return count


def ingest_mapped_csv(
    pack_id: str,
    csv_path: str | Path,
    *,
    mapping_path: str | Path | None = None,
    limit: int | None = None,
    enforce_trust: bool = True,
    allow_untrusted_historical_backfill: bool = False,
    sla_days: int | None = None,
) -> dict[str, Any]:
    """Execute mapping.yaml against a CSV and upsert trusted rows.

    Untrusted writes require ``allow_untrusted_historical_backfill=True``
    (NHTSA historical FLAT). ``enforce_trust=False`` alone is rejected so a
    copier cannot silently bypass the contract.
    """
    from src.data.trust import DEFAULT_FRESHNESS_SLA_DAYS, iter_trusted
    from src.security.identifiers import safe_csv_path, safe_pack_id

    pack_id = str(safe_pack_id(pack_id))
    csv_path = safe_csv_path(csv_path)
    if enforce_trust is False and not allow_untrusted_historical_backfill:
        raise ValueError(
            "untrusted ingest requires allow_untrusted_historical_backfill=True"
        )
    mpath = Path(mapping_path) if mapping_path else default_mapping_path(pack_id)
    mapping = load_mapping(mpath)
    # One pass: map → trust → upsert, streamed. Nothing here holds the CSV,
    # so the peak cost of a load is the two 10k-row batch buffers rather
    # than the row count.
    stats, stream = iter_trusted(
        pack_id,
        iter_mapped_rows(csv_path, mapping, limit=limit),
        persist=True,
        sla_days=DEFAULT_FRESHNESS_SLA_DAYS if sla_days is None else sla_days,
        keep_untrusted=allow_untrusted_historical_backfill,
    )
    written = upsert_records(pack_id, stream)
    return {
        "pack_id": pack_id,
        "csv_path": str(csv_path),
        "mapping_path": str(mpath),
        "mapped": stats["evaluated"],
        "upserted": written,
        "trusted": stats["trusted"],
        "rejected": stats["rejected"],
        "source": mapping.get("source"),
        "enforce_trust": not allow_untrusted_historical_backfill,
        "allow_untrusted_historical_backfill": allow_untrusted_historical_backfill,
    }


__all__ = [
    "CANONICAL_RECORD_FIELDS",
    "default_mapping_path",
    "load_mapping",
    "map_row",
    "iter_mapped_rows",
    "upsert_records",
    "ingest_mapped_csv",
]
