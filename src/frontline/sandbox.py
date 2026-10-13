"""Self-serve trial sandbox: sample data → first insight, no sales step."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from src.config import REPO_ROOT
from src.data.timeutil import utc_now
from src.data.trust import evaluate_ingest
from src.domains.mapping_ingest import ingest_mapped_csv, map_row
from src.ids import new_ulid

SAMPLE_ROWS = [
    {
        "odi_number": "SBX-100001",
        "date": "2026-06-01",
        "make": "HONDA",
        "model": "CR-V",
        "year": "2019",
        "component": "SERVICE BRAKES",
        "complaint": "grinding noise when braking at low speed",
    },
    {
        "odi_number": "SBX-100002",
        "date": "2026-06-08",
        "make": "HONDA",
        "model": "CR-V",
        "year": "2019",
        "component": "SERVICE BRAKES",
        "complaint": "brake pedal sinks to the floor after rain",
    },
    {
        "odi_number": "SBX-100003",
        "date": "2026-06-15",
        "make": "TOYOTA",
        "model": "CAMRY",
        "year": "2020",
        "component": "ELECTRICAL SYSTEM",
        "complaint": "window rattle and warning light on dash",
    },
]


def write_sample_csv(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["odi_number", "date", "make", "model", "year", "component", "complaint"],
        )
        w.writeheader()
        w.writerows(SAMPLE_ROWS)
    return dest


def first_insight_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cats: dict[str, int] = {}
    for r in rows:
        c = str(r.get("category") or r.get("component") or "unknown")
        cats[c] = cats.get(c, 0) + 1
    top = max(cats.items(), key=lambda kv: kv[1]) if cats else ("none", 0)
    return {
        "insight_id": "ins_" + new_ulid()[:10],
        "headline": f"{top[0]} is the leading theme ({top[1]} of {len(rows)} records)",
        "theme": top[0],
        "count": top[1],
        "n": len(rows),
        "as_of": utc_now().isoformat(),
    }


def boot_sandbox(*, pack_id: str = "automotive_nhtsa", dest: Path | None = None) -> dict[str, Any]:
    """Load sample rows, trust-evaluate, ingest trusted, return first insight."""
    path = dest or (REPO_ROOT / "data" / "sandbox" / "sample_complaints.csv")
    write_sample_csv(path)
    mapping = {
        "source": "sandbox",
        "records": {
            "record_id": "odi_number",
            "received_at": "date",
            "entity_1": "year",
            "entity_2": "make",
            "entity_3": "model",
            "category": "component",
            "text": "complaint",
            "source": '"sandbox"',
        },
    }
    mapped: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for raw in csv.DictReader(fh):
            rec = map_row(raw, mapping["records"])
            if rec:
                mapped.append(rec)
    trust = evaluate_ingest(pack_id, mapped, persist=True)
    ingested = None
    try:
        ingested = ingest_mapped_csv(pack_id, path, mapping_path=None)
    except Exception as e:
        ingested = {"error": type(e).__name__, "detail": str(e)}
    insight = first_insight_from_rows(trust["trusted_rows"] or mapped)
    return {
        "sandbox": True,
        "pack_id": pack_id,
        "sample_csv": str(path),
        "mapped": len(mapped),
        "trusted": trust["trusted"],
        "insight": insight,
        "ingest": ingested,
    }


__all__ = ["boot_sandbox", "write_sample_csv", "first_insight_from_rows", "SAMPLE_ROWS"]
