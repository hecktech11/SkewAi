"""Domain Pack loader.

Loads `domains/<pack_id>/pack.yaml`, validates against the pydantic schema,
resolves gazetteers, compiles slot frame + lexicons, and computes a
`pack_version` hash stamped on every interaction, case, and audit report.

Usage
-----
    from src.domains.loader import load_pack
    pack = load_pack("automotive_nhtsa")

CLI (lint)
----------
    python -m src.domains.loader --lint automotive_nhtsa
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.config import settings
from src.domains.schema import PackManifest, SlotSpec

# ── Loaded pack ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Gazetteer:
    """A frequency-ranked entity value list.

    `values` is ordered most-frequent first; `value_set` is the O(1) lookup.
    `canonical` maps lowercase forms to the canonical (display) form, so
    "honda" / "HONDA" / "Honda" all resolve to "HONDA" if that's how the
    gazetteer lists it.
    """

    name: str
    values: tuple[str, ...]
    value_set: frozenset[str]
    canonical: dict[str, str]

    @classmethod
    def from_csv(cls, path: Path) -> "Gazetteer":
        values: list[str] = []
        # CSV format: header row, then one value per row.
        # Column 1 = canonical value (required).
        # Column 2 (optional) = alias. Multiple rows may share a canonical value
        # with different aliases (e.g. "JPMORGAN CHASE & CO.,Chase").
        # The gazetteer is frequency-ranked from the top; aliases inherit the
        # rank of their canonical value's first occurrence.
        canonical: dict[str, str] = {}
        aliases: dict[str, str] = {}  # alias (lowercase) -> canonical value
        seen_canonical: set[str] = set()
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if not row:
                    continue
                value = row[0].strip()
                if not value or value.startswith("#"):
                    continue
                if value not in seen_canonical:
                    values.append(value)
                    seen_canonical.add(value)
                canonical[value.lower()] = value
                # Optional alias column
                if len(row) >= 2 and row[1].strip():
                    alias = row[1].strip()
                    aliases[alias.lower()] = value
        value_set = frozenset(canonical.keys()) | frozenset(aliases.keys())
        # Merge canonical + aliases into one lookup map.
        merged = dict(canonical)
        merged.update(aliases)
        return cls(
            name=path.stem,
            values=tuple(values),
            value_set=value_set,
            canonical=merged,
        )

    def lookup(self, raw: str) -> str | None:
        """Return the canonical form of `raw` if it's in the gazetteer, else None.

        Handles aliases: lookup('Chase') returns 'JPMORGAN CHASE & CO.' if the
        gazetteer has an alias row for it.
        """
        return self.canonical.get(raw.strip().lower())

    def match_substring(self, text: str) -> str | None:
        """Find the first gazetteer value OR alias that appears as a whole-word
        substring of `text`. Used by intake extraction for free-form utterances.
        """
        lowered = text.lower()
        # Try every key (canonical + aliases), longest first so 'JP Morgan Chase'
        # wins over 'Chase'.
        keys = sorted(self.canonical.keys(), key=len, reverse=True)
        for k in keys:
            if k in lowered:
                # crude whole-word check: surrounding chars are non-alpha
                idx = lowered.find(k)
                before = lowered[idx - 1] if idx > 0 else " "
                after = lowered[idx + len(k)] if idx + len(k) < len(lowered) else " "
                if before.isalnum() or after.isalnum():
                    continue
                # Short tokens ("IS"): reject a fully-lowercase span so English
                # "is" does not match the Lexus IS gazetteer entry.
                if len(k) <= 3:
                    span = text[idx : idx + len(k)]
                    if span == span.lower():
                        continue
                return self.canonical[k]
        return None


@dataclass(frozen=True)
class LoadedPack:
    """Runtime-ready pack: validated manifest + resolved gazetteers + version hash."""

    manifest: PackManifest
    pack_dir: Path
    gazetteers: dict[str, Gazetteer] = field(default_factory=dict)
    pack_version: str = ""
    taxonomy: dict[str, Any] = field(default_factory=dict)

    # ── Convenience accessors ──────────────────────────────────────────────
    @property
    def id(self) -> str:
        return self.manifest.id

    @property
    def display_name(self) -> str:
        return self.manifest.display_name

    @property
    def locale(self) -> str:
        return getattr(self.manifest, "locale", None) or "en-US"

    @property
    def greeting(self) -> str:
        return self.manifest.greeting

    @property
    def goodbye(self) -> str:
        return self.manifest.goodbye

    @property
    def entity_labels(self) -> dict[str, str]:
        e = self.manifest.entities
        return {"entity_1": e.entity_1, "entity_2": e.entity_2, "entity_3": e.entity_3}

    def slots_by_name(self) -> dict[str, SlotSpec]:
        return {s.name: s for s in self.manifest.slot_frame}

    def required_slots(self) -> list[SlotSpec]:
        return [s for s in self.manifest.slot_frame if s.required]

    def domain_db_path(self) -> Path:
        if self.manifest.domain_db:
            return self.pack_dir / self.manifest.domain_db
        return settings.domain_db_path(self.id)

    def gazetteer_for_slot(self, slot_name: str) -> Gazetteer | None:
        spec = self.slots_by_name().get(slot_name)
        if not spec or spec.validation != "gazetteer" or not spec.gazetteer:
            return None
        return self.gazetteers.get(spec.gazetteer)


# ── Loader ──────────────────────────────────────────────────────────────────

# Cache by pack id — packs are immutable at runtime.
_pack_cache: dict[str, LoadedPack] = {}


def load_pack(pack_id: str, *, reload: bool = False) -> LoadedPack:
    """Load and validate a pack. Cached by pack_id.

    Raises FileNotFoundError if the pack directory or pack.yaml is missing.
    Raises pydantic.ValidationError if the manifest is invalid.
    Raises ValueError on referential integrity failures (missing gazetteers).
    """
    from src.security.identifiers import safe_pack_id

    pack_id = str(safe_pack_id(pack_id))
    if not reload and pack_id in _pack_cache:
        return _pack_cache[pack_id]

    pack_dir = settings.pack_dir(pack_id)
    manifest_path = pack_dir / "pack.yaml"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Pack manifest not found: {manifest_path}")

    with manifest_path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    manifest = PackManifest.model_validate(raw)

    # ── Resolve gazetteers ─────────────────────────────────────────────────
    gazetteers: dict[str, Gazetteer] = {}
    gaz_dir = pack_dir / "gazetteers"
    for slot in manifest.slot_frame:
        if slot.validation == "gazetteer":
            if not slot.gazetteer:
                raise ValueError(
                    f"slot '{slot.name}' has validation=gazetteer but no `gazetteer` field"
                )
            if slot.gazetteer in gazetteers:
                continue
            gpath = gaz_dir / f"{slot.gazetteer}.csv"
            if not gpath.is_file():
                raise FileNotFoundError(
                    f"Gazetteer for slot '{slot.name}' not found: {gpath}"
                )
            gazetteers[slot.gazetteer] = Gazetteer.from_csv(gpath)

    # ── Load taxonomy (optional) ───────────────────────────────────────────
    taxonomy: dict[str, Any] = {}
    tax_path = pack_dir / manifest.taxonomy_ref
    if tax_path.is_file():
        with tax_path.open(encoding="utf-8") as f:
            taxonomy = yaml.safe_load(f) or {}

    # ── Compute pack_version hash ──────────────────────────────────────────
    # Hash covers: manifest (canonical JSON) + gazetteer contents + taxonomy.
    # Same discipline as context_version in v1.
    pack_version = _compute_pack_version(manifest_path, gazetteers, tax_path)

    pack = LoadedPack(
        manifest=manifest,
        pack_dir=pack_dir,
        gazetteers=gazetteers,
        pack_version=pack_version,
        taxonomy=taxonomy,
    )
    _pack_cache[pack_id] = pack
    return pack


def _compute_pack_version(
    manifest_path: Path,
    gazetteers: dict[str, Gazetteer],
    taxonomy_path: Path,
) -> str:
    h = hashlib.sha256()
    # manifest contents
    h.update(manifest_path.read_bytes())
    h.update(b"\x00manifest\x00")
    # gazetteers (sorted by name for determinism)
    for name in sorted(gazetteers):
        h.update(name.encode())
        h.update(b"\x00")
        for v in gazetteers[name].values:
            h.update(v.encode())
            h.update(b"\x00")
    h.update(b"\x00taxonomy\x00")
    if taxonomy_path.is_file():
        h.update(taxonomy_path.read_bytes())
    return h.hexdigest()[:12]


# ── Lint ────────────────────────────────────────────────────────────────────


def lint_pack(pack_id: str) -> list[str]:
    """Run schema + referential checks. Returns list of error strings
    (empty list = pack is clean).
    """
    errors: list[str] = []
    try:
        pack = load_pack(pack_id, reload=True)
    except FileNotFoundError as e:
        return [str(e)]
    except Exception as e:
        return [f"Schema validation failed: {e}"]

    # Referential checks
    for slot in pack.manifest.slot_frame:
        if slot.validation == "gazetteer" and slot.gazetteer:
            g = pack.gazetteers.get(slot.gazetteer)
            if g is None or len(g.values) == 0:
                errors.append(
                    f"slot '{slot.name}' references empty/missing gazetteer '{slot.gazetteer}'"
                )
        if slot.validation == "regex" and not slot.regex:
            errors.append(f"slot '{slot.name}' has validation=regex but no `regex` field")

    # Domain DB path: warn if parent missing (packs are lintable before their
    # warehouse is built via ingestion).
    db_path = pack.domain_db_path()
    if not db_path.parent.exists():
        # Soft warning — not an error. Print to stderr.
        import sys as _sys
        print(f"   (note) domain_db dir does not exist yet: {db_path.parent}", file=_sys.stderr)

    # At least one severity path is configured
    sev = pack.manifest.severity
    if not sev.model_artifact and not sev.rules:
        errors.append("severity: must define either model_artifact or rules")

    return errors


def list_packs() -> list[str]:
    """Return all pack ids (subdirectories of domains/ that contain pack.yaml)."""
    packs_root = settings.packs_root
    if not packs_root.is_dir():
        return []
    out: list[str] = []
    for child in sorted(packs_root.iterdir()):
        if child.is_dir() and (child / "pack.yaml").is_file():
            out.append(child.name)
    return out


# ── CLI ─────────────────────────────────────────────────────────────────────


def _main() -> int:
    parser = argparse.ArgumentParser(description="Skew AI pack loader + lint")
    parser.add_argument("--lint", metavar="PACK", help="Lint a pack and print errors")
    parser.add_argument("--list", action="store_true", help="List available packs")
    args = parser.parse_args()

    if args.list:
        for pid in list_packs():
            print(pid)
        return 0

    if args.lint:
        errors = lint_pack(args.lint)
        if errors:
            print(f"❌ Pack '{args.lint}' has {len(errors)} error(s):")
            for e in errors:
                print(f"  - {e}")
            return 1
        try:
            pack = load_pack(args.lint)
            print(f"✅ Pack '{args.lint}' is clean.")
            print(f"   display_name: {pack.display_name}")
            print(f"   pack_version: {pack.pack_version}")
            print(f"   slots: {[s.name for s in pack.manifest.slot_frame]}")
            print(f"   gazetteers: {sorted(pack.gazetteers)}")
        except Exception as e:
            print(f"❌ Failed to load clean pack: {e}")
            return 1
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(_main())
