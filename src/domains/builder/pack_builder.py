"""Pack Builder MVP — CSV profile → draft domain pack scaffold.

Limitations (documented): not full magic mapping; human confirms gazetteers.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

from src.config import REPO_ROOT

_SAFE_ID = re.compile(r"^[a-z][a-z0-9_]{1,48}$")


def profile_csv(path: Path, *, max_rows: int = 500) -> dict[str, Any]:
    """Read CSV and propose column → pack field mapping."""
    path = Path(path)
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("csv has no header")
        cols = list(reader.fieldnames)
        sample: list[dict[str, str]] = []
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            sample.append({k: (v or "") for k, v in row.items()})
    mapping = _propose_mapping(cols)
    return {
        "path": str(path),
        "columns": cols,
        "row_sample_count": len(sample),
        "proposed_mapping": mapping,
        "sample": sample[:5],
    }


def _propose_mapping(cols: list[str]) -> dict[str, str]:
    """Heuristic map of CSV column name → pack field."""
    lower = {c: c.lower().strip() for c in cols}
    inv = {v: k for k, v in lower.items()}
    out: dict[str, str] = {}
    aliases = {
        "text": ["text", "complaint", "description", "desc", "narrative", "comment", "body"],
        "entity_1": ["year", "entity_1", "model_year", "product_year"],
        "entity_2": ["make", "entity_2", "company", "manufacturer", "brand"],
        "entity_3": ["model", "entity_3", "product", "sub_product"],
        "category": ["category", "component", "issue", "product_category"],
        "received_at": ["date", "received", "received_at", "filed", "created"],
        "record_id": ["id", "record_id", "complaint_id", "odi_number"],
        "severity_label": ["severity", "severity_label", "priority"],
        "region": ["state", "region", "geo"],
    }
    used: set[str] = set()
    for field, names in aliases.items():
        for n in names:
            if n in inv and inv[n] not in used:
                out[field] = inv[n]
                used.add(inv[n])
                break
    return out


def build_draft_pack(
    *,
    pack_id: str,
    display_name: str,
    csv_path: Path,
    mapping: dict[str, str] | None = None,
    out_root: Path | None = None,
    seed_warehouse: bool = False,
) -> dict[str, Any]:
    """Create domains/<pack_id>/ scaffold + seed sample records from CSV."""
    if not _SAFE_ID.match(pack_id):
        raise ValueError("pack_id must be snake_case [a-z][a-z0-9_]{1,48}")
    if pack_id in ("automotive_nhtsa", "finance_cfpb"):
        raise ValueError(f"Cannot overwrite protected runtime pack: {pack_id}")
    root = out_root or (REPO_ROOT / "domains")
    pack_dir = root / pack_id
    if pack_dir.exists() and any(pack_dir.iterdir()):
        # Allow rebuild of builder-generated packs only when pack.yaml has marker
        py = pack_dir / "pack.yaml"
        if py.exists() and "builder_mvp: true" not in py.read_text(encoding="utf-8"):
            raise FileExistsError(f"pack already exists (not builder-generated): {pack_dir}")

    profile = profile_csv(Path(csv_path))
    map_ = mapping or profile["proposed_mapping"]
    if "text" not in map_:
        raise ValueError("mapping must include text column")

    pack_dir.mkdir(parents=True, exist_ok=True)
    (pack_dir / "gazetteers").mkdir(exist_ok=True)
    (pack_dir / "data").mkdir(exist_ok=True)
    (pack_dir / "demo").mkdir(exist_ok=True)

    # Sample rows for seed JSON
    samples: list[dict[str, Any]] = []
    with Path(csv_path).open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= 50:
                break
            rec = {
                "record_id": row.get(map_.get("record_id", ""), f"{pack_id.upper()}-{i+1:05d}")
                or f"{pack_id.upper()}-{i+1:05d}",
                "entity_1": row.get(map_.get("entity_1", ""), "") or None,
                "entity_2": (row.get(map_.get("entity_2", ""), "") or "").upper() or None,
                "entity_3": (row.get(map_.get("entity_3", ""), "") or "").upper() or None,
                "category": (row.get(map_.get("category", ""), "") or "GENERAL").upper(),
                "text": row.get(map_["text"], "") or "",
                "severity_label": row.get(map_.get("severity_label", ""), "") or "Medium",
                "region": row.get(map_.get("region", ""), "") or None,
            }
            if rec["text"]:
                samples.append(rec)

    (pack_dir / "data" / "mapping.yaml").write_text(
        "# Builder MVP mapping\n"
        + "\n".join(f"{k}: {v}" for k, v in map_.items())
        + "\n",
        encoding="utf-8",
    )
    (pack_dir / "data" / "seed_sample.json").write_text(
        json.dumps(samples, indent=2) + "\n", encoding="utf-8"
    )

    # Gazetteer stubs from unique values
    for field, gfile in (
        ("entity_2", "entity_2.csv"),
        ("entity_3", "entity_3.csv"),
        ("category", "categories.csv"),
    ):
        vals = sorted({(s.get(field) or "").strip() for s in samples if s.get(field)})
        lines = ["value\n"] + [f"{v}\n" for v in vals if v]
        (pack_dir / "gazetteers" / gfile).write_text("".join(lines), encoding="utf-8")

    pack_yaml = f"""# Generated by Pack Builder MVP — review before production use
# Schema-conformant with src/domains/schema.py::PackManifest.
builder_mvp: true
id: {pack_id}
display_name: {display_name}
greeting: "Thanks for contacting us about {display_name}. How can I help?"
goodbye: "We've opened case {{case_id}}. Someone will follow up."
refusal_topics:
  - legal advice
entities:
  entity_1: "Entity 1"
  entity_2: "Entity 2"
  entity_3: "Entity 3"
slot_frame:
  - name: entity_2
    label: "Entity 2"
    prompt: "Which company, brand, or entity is this about?"
    required: true
    max_re_asks: 2
    validation: gazetteer
    gazetteer: entity_2
  - name: category
    label: "Category"
    prompt: "What category best describes the issue?"
    required: true
    max_re_asks: 2
    validation: gazetteer
    gazetteer: categories
  - name: description
    label: "Description"
    prompt: "Please describe what happened."
    required: true
    max_re_asks: 1
    validation: free-text
safety:
  escalation_lexicon:
    - emergency
    - urgent
    - dangerous
  safety_questions:
    - "Is anyone in immediate danger?"
    - "Are you in a safe location right now?"
  escalation_script: |
    I want to flag this immediately for a specialist. A specialist will reach
    out within the hour. If you're in immediate danger, please contact
    emergency services.
advisory_match:
  sql_template: |
    SELECT advisory_id, issued_at,
           scope_entity_2 || ' ' || scope_entity_3 || ' ' || scope_category AS scope_summary,
           summary, remedy, url, source
    FROM advisories
    WHERE (scope_entity_1 = $entity_1 OR scope_entity_1 IS NULL)
      AND (scope_entity_2 = $entity_2 OR scope_entity_2 IS NULL)
      AND (scope_entity_3 = $entity_3 OR scope_entity_3 IS NULL)
      AND (scope_category = $category OR scope_category IS NULL)
    ORDER BY issued_at DESC
    LIMIT 5
  readback_fields: [advisory_id, scope_summary, remedy, url]
severity:
  rules:
    - when: safety.any
      severity: Critical
      reason: "Safety flag present"
    - when: default
      severity: Medium
      reason: "Default medium"
  priority_matrix:
    safety: 1
    Critical: 1
    Medium: 2
    Low: 3
taxonomy_ref: taxonomy.yaml
"""
    (pack_dir / "pack.yaml").write_text(pack_yaml, encoding="utf-8")
    (pack_dir / "taxonomy.yaml").write_text(
        "categories:\n  - GENERAL\n", encoding="utf-8"
    )
    (pack_dir / "demo" / "demo_script.md").write_text(
        f"# Demo — {display_name}\n\nBuilder MVP pack from `{csv_path}`.\n",
        encoding="utf-8",
    )

    # ── Seed the domain warehouse so `make contact` works immediately ───────
    # Build records from the CSV sample (up to 50 rows) + an empty advisories
    # table so the pack is immediately usable. The user can re-run with a
    # larger CSV + advisories CSV to build the full warehouse.
    if seed_warehouse:
        try:
            _seed_domain_warehouse(pack_id, samples)
        except Exception as e:
            # Don't fail the build if seeding fails — the pack.yaml is still valid
            # and the user can run a separate ingest later.
            print(f"   (note) domain warehouse seed skipped: {e}", file=sys.stderr)

    return {
        "pack_id": pack_id,
        "pack_dir": str(pack_dir),
        "sample_records": len(samples),
        "mapping": map_,
        "lint_hint": f"make pack-lint PACK={pack_id}",
        "contact_hint": f"DOMAIN_PACK={pack_id} make contact",
    }


# ── Domain warehouse seeding ────────────────────────────────────────────────


def _seed_domain_warehouse(pack_id: str, samples: list[dict[str, Any]]) -> None:
    """Seed data/domains/<pack_id>.duckdb with the built sample records.

    Creates the canonical `records` table (and empty `advisories`, `clusters`,
    `cluster_assignments`, `weekly_anomalies`, `backtest_results`). Enough for
    `make contact` to work; the user can run a real ingest for full data.
    """
    import sys
    from datetime import datetime, timezone
    from src.data.warehouse import domain_con

    _now = datetime.now(timezone.utc)
    with domain_con(pack_id, read_only=False) as con:
        # Insert records
        for r in samples:
            con.execute(
                """
                INSERT INTO records
                (record_id, occurred_at, received_at, entity_1, entity_2, entity_3,
                 category, subcategory, text, severity_label, region, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?)
                """,
                [
                    r["record_id"],
                    _now,
                    _now,
                    r.get("entity_1"),
                    r.get("entity_2"),
                    r.get("entity_3"),
                    r.get("category"),
                    r.get("text"),
                    r.get("severity_label"),
                    r.get("region"),
                    pack_id.upper(),
                ],
            )
        print(f"   ✓ domain warehouse seeded: {len(samples)} records", file=sys.stderr)


def lint_mapping(mapping: dict[str, str], columns: list[str]) -> dict[str, Any]:
    """Lint a proposed CSV→canonical mapping. Fail if text or record_id missing."""
    cols = set(columns)
    errors: list[str] = []
    warnings: list[str] = []
    if "text" not in mapping:
        errors.append("mapping must include text")
    elif mapping["text"] not in cols:
        errors.append(f"text column {mapping['text']!r} not in CSV")
    if "record_id" not in mapping:
        warnings.append("no record_id column; builder will synthesize ids")
    elif mapping["record_id"] not in cols:
        errors.append(f"record_id column {mapping['record_id']!r} not in CSV")
    unused = [c for c in columns if c not in mapping.values()]
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "unused_columns": unused,
        "mapping": mapping,
    }


def first_insight_from_csv(
    csv_path: Path,
    *,
    mapping: dict[str, str] | None = None,
    pack_id: str = "builder_preview",
    display_name: str = "Preview pack",
    out_root: Path | None = None,
) -> dict[str, Any]:
    """CSV → profile → lint → draft pack → first insight (theme count)."""
    from src.frontline.sandbox import first_insight_from_rows

    profile = profile_csv(Path(csv_path))
    map_ = mapping or profile["proposed_mapping"]
    lint = lint_mapping(map_, profile["columns"])
    if not lint["ok"]:
        return {"ok": False, "lint": lint, "insight": None}
    built = build_draft_pack(
        pack_id=pack_id,
        display_name=display_name,
        csv_path=Path(csv_path),
        mapping=map_,
        out_root=out_root,
        seed_warehouse=False,
    )
    samples = []
    seed = Path(built["pack_dir"]) / "data" / "seed_sample.json"
    if seed.is_file():
        samples = json.loads(seed.read_text(encoding="utf-8"))
    insight = first_insight_from_rows(samples)
    return {"ok": True, "lint": lint, "pack": built, "insight": insight}


__all__ = ["profile_csv", "build_draft_pack", "lint_mapping", "first_insight_from_csv"]
