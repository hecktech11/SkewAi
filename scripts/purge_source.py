"""Delete a source's records so a corpus can be reloaded under a new key.

Usage
-----
    python -m scripts.purge_source --source NHTSA --dry-run
    python -m scripts.purge_source --source NHTSA --yes

Without ``--yes`` this only reports. Deleting records is not recoverable
from inside the product: there is no undo and no soft-delete column.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.domains.purge import purge_source

PACK_ID = "automotive_nhtsa"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Delete a source's records and their side tables")
    p.add_argument("--pack", default=PACK_ID)
    p.add_argument("--source", required=True, help="records.source tag, e.g. NHTSA")
    p.add_argument("--dry-run", action="store_true", help="report only (the default)")
    p.add_argument("--yes", action="store_true", help="actually delete")
    args = p.parse_args(argv)

    dry = args.dry_run or not args.yes
    report = purge_source(args.pack, args.source, dry_run=dry)
    verb = "would delete" if dry else "deleted"
    print(f"{verb} from pack {report['pack_id']} source {report['source']}:")
    for key, value in report.items():
        if key in ("pack_id", "source", "dry_run"):
            continue
        print(f"  {key}: {value}")
    if dry:
        print("\nnothing was written. re-run with --yes to delete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
