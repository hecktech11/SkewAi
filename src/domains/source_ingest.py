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


def ingest_source(
    pack_id: str,
    source: str,
    csv_path: str | Path,
    *,
    mapping_path: str | Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Ingest one source CSV into canonical records. Returns the run report."""
    from src.security.identifiers import safe_csv_path, safe_mapping_path

    tag = (source or "").strip().lower()
    if not tag:
        raise ValueError("source tag required (e.g. warranty, service)")
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
        rec["record_id"] = namespace_id(tag, rec["record_id"])
        if not rec["record_id"] or not (rec.get("text") or "").strip():
            skipped += 1
            continue
        staged.append(rec)
    written = upsert_records(pack_id, staged)
    return {
        "pack_id": pack_id,
        "source": tag.upper(),
        "csv_path": str(csv_path),
        "mapping_path": str(mpath),
        "mapped": mapped,
        "upserted": written,
        "skipped": skipped,
    }


__all__ = ["default_source_mapping", "namespace_id", "ingest_source"]
