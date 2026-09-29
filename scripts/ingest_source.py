"""Inbound source connector CLI (Axion slice, Phase 1).

    python -m scripts.ingest_source --pack automotive_nhtsa \\
        --source warranty --csv data/inbound/warranty_2026-08.csv
    python -m scripts.ingest_source --pack automotive_nhtsa \\
        --source service --csv data/inbound/service_2026-08.csv

Core logic lives in src.domains.source_ingest (importable by workers).
"""

from __future__ import annotations

import argparse
import sys

from src.domains.source_ingest import (
    DEFAULT_PROVENANCE,
    PROVENANCE_VALUES,
    ingest_source,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest a second/third source")
    parser.add_argument("--pack", default="automotive_nhtsa")
    parser.add_argument("--source", required=True, help="warranty | service | ...")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--mapping", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--provenance",
        default=None,
        choices=list(PROVENANCE_VALUES),
        help=(
            "how the rows should be treated downstream (default: "
            f"{DEFAULT_PROVENANCE}). Use 'fixture' for demo and sample files "
            "so the anomaly scan does not read them as real failures."
        ),
    )
    args = parser.parse_args(argv)
    try:
        report = ingest_source(
            args.pack, args.source, args.csv,
            mapping_path=args.mapping, limit=args.limit,
            provenance=args.provenance,
        )
    except (ValueError, FileNotFoundError) as e:
        print(f"ingest_source failed: {e}")
        return 1
    print(
        f"ingested {report['upserted']}/{report['mapped']} rows "
        f"(skipped {report['skipped']}) from {report['source']} "
        f"as {report['provenance']} "
        f"into {report['pack_id']} via {report['mapping_path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
