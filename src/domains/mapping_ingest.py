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
from src.data.warehouse import apply_domain_schema, domain_con
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


def upsert_records(pack_id: str, records: Iterable[dict[str, Any]]) -> int:
    """Insert-or-replace canonical rows into the pack domain warehouse."""
    count = 0
    with domain_con(pack_id, read_only=False) as con:
        apply_domain_schema(con)
        for rec in records:
            emb = embed_text(rec["text"])
            con.execute(
                """
                INSERT OR REPLACE INTO records (
                    record_id, occurred_at, received_at, entity_1, entity_2, entity_3,
                    category, subcategory, text, severity_label, region, source, embedding,
                    entity_key, provenance
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    rec["record_id"],
                    rec.get("occurred_at"),
                    rec["received_at"],
                    rec.get("entity_1") or None,
                    rec.get("entity_2") or None,
                    rec.get("entity_3") or None,
                    rec.get("category") or None,
                    rec.get("subcategory") or None,
                    rec["text"],
                    rec.get("severity_label") or None,
                    rec.get("region") or None,
                    rec.get("source") or None,
                    emb,
                    rec.get("entity_key") or None,
                    rec.get("provenance") or "observed",
                ],
            )
            count += 1
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
    from src.data.trust import DEFAULT_FRESHNESS_SLA_DAYS, evaluate_ingest
    from src.security.identifiers import safe_csv_path, safe_pack_id

    pack_id = str(safe_pack_id(pack_id))
    csv_path = safe_csv_path(csv_path)
    if enforce_trust is False and not allow_untrusted_historical_backfill:
        raise ValueError(
            "untrusted ingest requires allow_untrusted_historical_backfill=True"
        )
    mpath = Path(mapping_path) if mapping_path else default_mapping_path(pack_id)
    mapping = load_mapping(mpath)
    rows = list(iter_mapped_rows(csv_path, mapping, limit=limit))
    trust = evaluate_ingest(
        pack_id,
        rows,
        persist=True,
        sla_days=DEFAULT_FRESHNESS_SLA_DAYS if sla_days is None else sla_days,
    )
    to_write = rows if allow_untrusted_historical_backfill else trust["trusted_rows"]
    written = upsert_records(pack_id, to_write)
    return {
        "pack_id": pack_id,
        "csv_path": str(csv_path),
        "mapping_path": str(mpath),
        "mapped": len(rows),
        "upserted": written,
        "trusted": trust.get("trusted"),
        "rejected": trust.get("rejected"),
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
