"""Regenerate the committed Honda/Ford lead-time sample.

Refuses the default ``data/domains`` directory. The pilot warehouse stays
where it is.

    DOMAIN_DB_PATH=/tmp/skew-recall-lead python -m scripts.recall_lead_sample
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main(argv: list[str] | None = None) -> int:
    del argv
    from src.backtest.recall_lead import write_recall_lead_report
    from src.config import REPO_ROOT, is_default_pilot_domain_path
    from src.domains.mapping_ingest import default_mapping_path, ingest_mapped_csv

    if is_default_pilot_domain_path():
        print(
            "refusing the pilot data/domains directory; "
            "set DOMAIN_DB_PATH to a side directory",
            file=sys.stderr,
        )
        return 2
    fixture = REPO_ROOT / "eval" / "frontline" / "fixtures" / "recall_lead"
    manifest = json.loads((fixture / "manifest.json").read_text(encoding="utf-8"))
    from scripts.ingest_recalls import advisory_contract, insert_advisories

    pack = "recall_lead_sample"
    complaints = fixture / str(manifest["complaints"])
    recalls = fixture / str(manifest["recalls"])
    ingest_mapped_csv(
        pack,
        complaints,
        mapping_path=default_mapping_path("automotive_nhtsa"),
        allow_untrusted_historical_backfill=True,
    )
    with recalls.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    insert_advisories(pack, rows, contract=advisory_contract("automotive_nhtsa"))
    dest = REPO_ROOT / "reports" / "nhtsa" / "recall_lead_time.md"
    report = write_recall_lead_report(
        pack,
        dest,
        model_grain=bool(manifest.get("model_grain")),
    )
    print(
        f"wrote {report['path']}: "
        f"hits={report['hit_count']} misses={report['miss_count']} "
        f"unconfirmed={report['unconfirmed_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
