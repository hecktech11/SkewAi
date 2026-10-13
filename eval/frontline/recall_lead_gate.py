"""CI gate: a committed sample of real complaints leads a known recall.

The sample is scored in its own domain file so the seeded fixture warehouse
is left alone. Finance packs do not run this gate. A miss and an unconfirmed
flag have to stay visible; the pass condition is a positive lead on the
known-positive advisories only.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from eval.frontline.report import GateResult
from src.config import REPO_ROOT

FIXTURE_DIR = REPO_ROOT / "eval" / "frontline" / "fixtures" / "recall_lead"
SAMPLE_PACK = "recall_lead_sample"


def recall_lead_gate() -> GateResult:
    """Load the committed sample and require a positive lead on known recalls."""
    manifest_path = FIXTURE_DIR / "manifest.json"
    if not manifest_path.is_file():
        return GateResult(
            name="recall_lead_time",
            pack_id="automotive_nhtsa",
            passed=False,
            expected="positive lead",
            actual="missing fixture",
            detail=f"no manifest at {manifest_path}",
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    complaints = FIXTURE_DIR / str(manifest["complaints"])
    recalls = FIXTURE_DIR / str(manifest["recalls"])
    positives = [str(a) for a in manifest.get("known_positive") or []]
    if not complaints.is_file() or not recalls.is_file() or not positives:
        return GateResult(
            name="recall_lead_time",
            pack_id="automotive_nhtsa",
            passed=False,
            expected="positive lead",
            actual="incomplete fixture",
            detail="manifest, complaints, recalls, and known_positive are required",
        )
    try:
        from scripts.ingest_recalls import advisory_contract, insert_advisories
        from src.backtest.recall_lead import recall_lead_report
        from src.domains.mapping_ingest import default_mapping_path, ingest_mapped_csv

        ingest_mapped_csv(
            SAMPLE_PACK,
            complaints,
            mapping_path=default_mapping_path("automotive_nhtsa"),
            allow_untrusted_historical_backfill=True,
        )
        with recalls.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        insert_advisories(
            SAMPLE_PACK, rows, contract=advisory_contract("automotive_nhtsa")
        )
        report = recall_lead_report(
            SAMPLE_PACK,
            model_grain=bool(manifest.get("model_grain")),
            run_backtests=True,
        )
    except Exception as e:
        return GateResult(
            name="recall_lead_time",
            pack_id="automotive_nhtsa",
            passed=False,
            expected="positive lead",
            actual="error",
            detail=f"{type(e).__name__}: {e}",
        )
    by_id = {r["advisory_id"]: r for r in report["recalls"]}
    leads = []
    missing = []
    for advisory_id in positives:
        row = by_id.get(advisory_id)
        if row is None or int(row.get("lead_weeks") or 0) <= 0:
            missing.append(advisory_id)
        else:
            leads.append(int(row["lead_weeks"]))
    miss_ok = (not manifest.get("require_miss")) or report["miss_count"] >= 1
    unconfirmed_ok = (
        (not manifest.get("require_unconfirmed")) or report["unconfirmed_count"] >= 1
    )
    passed = not missing and miss_ok and unconfirmed_ok and bool(leads)
    return GateResult(
        name="recall_lead_time",
        pack_id="automotive_nhtsa",
        passed=passed,
        expected="lead_weeks > 0 on known_positive",
        actual=leads or missing,
        detail=(
            f"hits={report['hit_count']} misses={report['miss_count']} "
            f"unconfirmed={report['unconfirmed_count']} "
            f"failed={missing or 'none'}"
        ),
    )
