"""Second/third-source inbound connectors (Axion slice, Phase 1).

Warranty claims, service/repair records, and bank service logs land in the
pack's canonical ``records`` via source mapping files
(``domains/<pack>/data/mapping.<source>.yaml``), alongside complaints and
call transcripts. See ``scripts/ingest_source.py`` for the CLI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config import REPO_ROOT
from src.domains.mapping_ingest import (
    iter_mapped_rows,
    load_mapping,
    upsert_records,
)


#: Values ``records.provenance`` accepts. ``observed`` is something that
#: happened and was recorded; ``inferred`` is something the system derived
#: and the anomaly scan leaves out; ``fixture`` is demo or seed data that
#: must never be counted as evidence of a real failure.
PROVENANCE_VALUES = ("observed", "inferred", "fixture")
DEFAULT_PROVENANCE = "observed"


def normalize_provenance(value: str | None) -> str:
    """Validate a provenance tag, defaulting when unset.

    Raises rather than silently downgrading: a demo file loaded as
    ``observed`` is indistinguishable from real evidence once it is in
    ``records``, and the anomaly scan would treat it as a live signal.
    """
    tag = (value or "").strip().lower() or DEFAULT_PROVENANCE
    if tag not in PROVENANCE_VALUES:
        raise ValueError(
            f"unknown provenance {value!r}; expected one of {', '.join(PROVENANCE_VALUES)}"
        )
    return tag


def default_source_mapping(pack_id: str, source: str) -> Path:
    return REPO_ROOT / "domains" / pack_id / "data" / f"mapping.{source}.yaml"


def namespace_id(source_tag: str, rid: str) -> str:
    """Namespace record_ids per source so a claim can never collide with a
    complaint (data integrity: no cross-source overwrites)."""
    rid = (rid or "").strip()
    if not rid:
        return rid
    if rid.upper().startswith(source_tag.upper() + "-") or rid.upper().startswith(
        source_tag.upper() + ":"
    ):
        return rid
    return f"{source_tag.upper()}-{rid}"


def aggregate_daily(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse telematics events to one row per day, cohort, and fault code.

    ``event_count`` is the number of events in that bucket. A file that is
    already one row per day stays one row; its count is the sum of the
    rows that shared the bucket.
    """
    grouped: dict[tuple, dict[str, Any]] = {}
    for rec in rows:
        when = rec.get("occurred_at") or rec.get("received_at")
        day = when.date().isoformat() if hasattr(when, "date") else ""
        cohort = str(rec.get("entity_1") or "")
        fault = str(rec.get("subcategory") or rec.get("category") or "")
        key = (
            day,
            cohort,
            fault,
            str(rec.get("entity_2") or ""),
            str(rec.get("entity_3") or ""),
            str(rec.get("category") or ""),
        )
        try:
            weight = int(rec.get("event_count") or 1)
        except (TypeError, ValueError):
            weight = 1
        if weight < 1:
            weight = 1
        slot = grouped.get(key)
        if slot is None:
            slot = dict(rec)
            slot["event_count"] = weight
            slot["record_id"] = f"{cohort}|{fault}|{day}" or rec.get("record_id")
            if not str(slot.get("text") or "").strip():
                slot["text"] = fault or str(rec.get("category") or "telematics")
            grouped[key] = slot
        else:
            slot["event_count"] = int(slot["event_count"]) + weight
    return list(grouped.values())


def ingest_source(
    pack_id: str,
    source: str,
    csv_path: str | Path,
    *,
    mapping_path: str | Path | None = None,
    limit: int | None = None,
    provenance: str | None = None,
) -> dict[str, Any]:
    """Ingest one source CSV into canonical records. Returns the run report.

    ``provenance`` tags every row written by this run. Demo and sample files
    should load as ``fixture`` so the anomaly scan and the early-warning
    board can exclude them; see ``PROVENANCE_VALUES``.
    """
    from src.security.identifiers import safe_csv_path, safe_mapping_path

    tag = (source or "").strip().lower()
    if not tag:
        raise ValueError("source tag required (e.g. warranty, service)")
    # Validate before touching the filesystem: a rejected tag should fail
    # the same way whether or not the CSV happens to exist.
    prov = normalize_provenance(provenance)
    # Jail caller-supplied paths here too: this is reachable from the job
    # queue, not only from the upload route that jails its own input (R01).
    csv_path = safe_csv_path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"source CSV not found: {csv_path}")
    # Audit 0.5: refuse to write into a behind-HEAD warehouse.
    try:
        from scripts.migrate import require_schema_current

        from src.config import settings as _settings

        require_schema_current(_settings.domain_db_path(pack_id), target="domain")
    except RuntimeError:
        raise
    except Exception:
        pass
    if mapping_path:
        mpath = safe_mapping_path(mapping_path)
        if not mpath.is_file():
            raise FileNotFoundError(f"source mapping not found: {mpath}")
    else:
        mpath = default_source_mapping(pack_id, tag)
        if not mpath.is_file():
            # Fall back to the pack's default mapping (source still forced).
            mpath = REPO_ROOT / "domains" / pack_id / "data" / "mapping.yaml"
        if not mpath.is_file():
            raise ValueError(
                f"no source mapping for pack={pack_id} source={tag}: "
                f"expected {default_source_mapping(pack_id, tag)} "
                "or pass mapping_path explicitly"
            )
    mapping = load_mapping(mpath)
    mapped = skipped = 0
    staged: list[dict[str, Any]] = []
    for rec in iter_mapped_rows(csv_path, mapping, limit=limit):
        mapped += 1
        rec["source"] = tag.upper()
        rec["provenance"] = prov
        if not rec["record_id"] or not (rec.get("text") or "").strip():
            skipped += 1
            continue
        staged.append(rec)
    if str(mapping.get("aggregate") or "").strip().lower() == "day":
        staged = aggregate_daily(staged)
    for rec in staged:
        rec["record_id"] = namespace_id(tag, str(rec.get("record_id") or ""))
    written = upsert_records(pack_id, [rec for rec in staged if rec.get("record_id")])
    return {
        "pack_id": pack_id,
        "source": tag.upper(),
        "csv_path": str(csv_path),
        "mapping_path": str(mpath),
        "mapped": mapped,
        "upserted": written,
        "skipped": skipped,
        "provenance": prov,
    }


def load_demo_private_sources(
    pack_id: str,
    *,
    provenance: str | None = "fixture",
    mapping_pack: str = "automotive_nhtsa",
) -> dict[str, Any]:
    """Load the shipped field, telematics, and exposure samples.

    The default tag is ``fixture``. Those rows stay out of the anomaly
    scan until a partner file is loaded as ``observed``.
    """
    from src.data.vio_pipeline import ingest_vio_csv

    prov = normalize_provenance(provenance)
    base = REPO_ROOT / "domains" / mapping_pack / "data"
    demo = base / "demo"
    field = ingest_source(
        pack_id,
        "field",
        demo / "field_returns.csv",
        mapping_path=base / "mapping.field.yaml",
        provenance=prov,
    )
    telematics = ingest_source(
        pack_id,
        "telematics",
        demo / "telematics_events.csv",
        mapping_path=base / "mapping.telematics.yaml",
        provenance=prov,
    )
    exposure = ingest_vio_csv(
        pack_id,
        demo / "exposure.csv",
        source="demo_vio",
        unit_type="vehicles_in_operation",
    )
    return {
        "provenance": prov,
        "field": field,
        "telematics": telematics,
        "exposure": exposure,
    }


__all__ = [
    "PROVENANCE_VALUES",
    "DEFAULT_PROVENANCE",
    "normalize_provenance",
    "default_source_mapping",
    "namespace_id",
    "aggregate_daily",
    "ingest_source",
    "load_demo_private_sources",
]
