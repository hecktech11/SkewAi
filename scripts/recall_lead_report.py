"""Write the per-recall lead-time report for one pack.

Scoring walks every recall against every complaint and replaces
``weekly_anomalies``, so this refuses the default ``data/domains`` directory.
Point ``DOMAIN_DB_PATH`` at a side directory first.

    DOMAIN_DB_PATH=/tmp/skew-nhtsa python -m scripts.recall_lead_report \\
        --pack automotive_nhtsa --out reports/nhtsa/recall_lead_time_2015_2024.md

``reports/nhtsa/recall_lead_time.md`` is the committed Honda/Ford sample.
Only ``scripts.recall_lead_sample`` (pack ``recall_lead_sample``) writes it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest.recall_lead import write_recall_lead_report
from src.config import REPO_ROOT, is_default_pilot_domain_path

SAMPLE_REPORT = REPO_ROOT / "reports" / "nhtsa" / "recall_lead_time.md"
SAMPLE_PACK = "recall_lead_sample"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Recall lead-time report")
    p.add_argument("--pack", default="automotive_nhtsa")
    p.add_argument("--out", default=str(SAMPLE_REPORT))
    p.add_argument("--model-grain", action="store_true")
    args = p.parse_args(argv)
    if is_default_pilot_domain_path():
        print(
            "refusing the pilot data/domains directory; "
            "set DOMAIN_DB_PATH to a side directory",
            file=sys.stderr,
        )
        return 2
    out = Path(args.out).expanduser()
    if not out.is_absolute():
        out = REPO_ROOT / out
    if out.resolve() == SAMPLE_REPORT.resolve() and args.pack != SAMPLE_PACK:
        print(
            f"refusing to overwrite the committed sample {SAMPLE_REPORT} with pack "
            f"{args.pack!r}; run `python -m scripts.recall_lead_sample` for the "
            "sample, or pass --out with another path",
            file=sys.stderr,
        )
        return 2
    report = write_recall_lead_report(
        args.pack, out, model_grain=args.model_grain
    )
    print(
        f"wrote {report['path']}: "
        f"hits={report['hit_count']} misses={report['miss_count']} "
        f"unconfirmed={report['unconfirmed_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
