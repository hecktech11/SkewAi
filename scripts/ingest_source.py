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
    parser.add_argument(
        "--run-anomalies",
        action="store_true",
        help="recompute weekly anomalies after the rows are written",
    )
    parser.add_argument(
        "--allow-pilot",
        action="store_true",
        help="write a pilot pack under the default data/domains directory",
    )
    args = parser.parse_args(argv)
    if not args.allow_pilot:
        try:
            from src.domains.csv_load import assert_not_pilot_warehouse

            assert_not_pilot_warehouse(args.pack)
        except ValueError as e:
            print(f"ingest_source failed: {e}")
            return 1
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
    if args.run_anomalies:
        try:
            from src.ml_runtime.anomalies import recompute_weekly_anomalies

            scored = recompute_weekly_anomalies(args.pack)
        except Exception as e:
            print(f"anomaly recompute failed: {e}")
            return 1
        print(f"anomalies scored {len(scored)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
