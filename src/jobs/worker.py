"""Standalone job worker — same loop the API lifespan runs in-process.

    python -m src.jobs.worker

Compose starts this as `frontline-worker` / `frontline-hardened-worker` so
enqueued jobs and the weekly erasure drill run even if the API process is
replaced. The in-process lifespan worker remains for single-container deploys;
lease claiming makes two workers safe.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL_S = 5.0
FAIRNESS_EVERY_S = 60.0
DEFAULT_SCAN_INTERVAL_S = 86400.0

# Process-local cadence. A restart enqueues again; the idempotency key
# stops a second pending scan for the same pack and bucket.
_last_scan_mono = 0.0
_last_measure_mono = 0.0
DEFAULT_MEASURE_INTERVAL_S = 7 * 86400.0


def scan_interval_s() -> float:
    """Seconds between scheduled_scan enqueues.

    Unset means daily. ``0`` disables. Under ``FRONTLINE_TEST_ISOLATION``
    the unset value is off, so a test tick does not leave a fleet scan
    sitting in front of the job the test just queued. Set the env to opt in.
    """
    raw = os.getenv("FRONTLINE_SCAN_INTERVAL_S", "").strip()
    if not raw:
        if os.getenv("FRONTLINE_TEST_ISOLATION", "").strip() == "1":
            return 0.0
        return DEFAULT_SCAN_INTERVAL_S
    try:
        return max(0.0, float(raw))
    except ValueError:
        return DEFAULT_SCAN_INTERVAL_S


def maybe_enqueue_scheduled_scans() -> list[dict[str, Any]]:
    """Enqueue one ``scheduled_scan`` per pack when the cadence is due."""
    global _last_scan_mono
    interval = scan_interval_s()
    if interval <= 0:
        return []
    now = time.monotonic()
    if _last_scan_mono and (now - _last_scan_mono) < interval:
        return []
    from datetime import datetime, timezone

    from src.data.warehouse import ops_con
    from src.domains.loader import list_packs
    from src.jobs.queue import enqueue

    if interval >= DEFAULT_SCAN_INTERVAL_S:
        bucket = datetime.now(timezone.utc).date().isoformat()
    else:
        bucket = str(int(time.time() // interval))
    enqueued: list[dict[str, Any]] = []
    for pack in list_packs():
        if pack.startswith("_"):
            continue
        key = f"scheduled_scan:{pack}:{bucket}"
        try:
            with ops_con() as con:
                existing = con.execute(
                    "SELECT status FROM job_queue WHERE idempotency_key = ?",
                    [key],
                ).fetchone()
        except Exception:
            existing = None
        if existing and existing[0] in ("pending", "running", "done"):
            continue
        try:
            enqueued.append(
                enqueue(
                    "scheduled_scan",
                    {"pack_id": pack, "alert": True},
                    idempotency_key=key,
                )
            )
        except Exception:
            logger.exception("scheduled_scan_enqueue_failed pack=%s", pack)
    _last_scan_mono = now
    return enqueued


def measure_interval_s() -> float:
    """Seconds between measure_fix enqueues.

    Unset means weekly. ``0`` disables. Under test isolation the unset
    value is off, same as the fleet scan.
    """
    raw = os.getenv("FRONTLINE_MEASURE_INTERVAL_S", "").strip()
    if not raw:
        if os.getenv("FRONTLINE_TEST_ISOLATION", "").strip() == "1":
            return 0.0
        return DEFAULT_MEASURE_INTERVAL_S
    try:
        return max(0.0, float(raw))
    except ValueError:
        return DEFAULT_MEASURE_INTERVAL_S


def maybe_enqueue_measure_fixes() -> list[dict[str, Any]]:
    """Enqueue one ``measure_fix`` per pack when the weekly cadence is due."""
    global _last_measure_mono
    interval = measure_interval_s()
    if interval <= 0:
        return []
    now = time.monotonic()
    if _last_measure_mono and (now - _last_measure_mono) < interval:
        return []
    from datetime import datetime, timezone

    from src.data.warehouse import ops_con
    from src.domains.loader import list_packs
    from src.jobs.queue import enqueue

    bucket = datetime.now(timezone.utc).date().isoformat()
    if interval < DEFAULT_MEASURE_INTERVAL_S:
        bucket = str(int(time.time() // interval))
    enqueued: list[dict[str, Any]] = []
    for pack in list_packs():
        if pack.startswith("_"):
            continue
        key = f"measure_fix:{pack}:{bucket}"
        try:
            with ops_con() as con:
                existing = con.execute(
                    "SELECT status FROM job_queue WHERE idempotency_key = ?",
                    [key],
                ).fetchone()
        except Exception:
            existing = None
        if existing and existing[0] in ("pending", "running", "done"):
            continue
        try:
            enqueued.append(
                enqueue(
                    "measure_fix",
                    {"pack_id": pack},
                    idempotency_key=key,
                )
            )
        except Exception:
            logger.exception("measure_fix_enqueue_failed pack=%s", pack)
    _last_measure_mono = now
    return enqueued


def worker_tick(*, run_fairness: bool = False) -> dict[str, Any]:
    """One pass: claim a job, maybe run the weekly drill, optionally fairness.

    The scan is enqueued after ``run_next`` so this tick does not steal a
    job the caller just queued. The following tick claims it.
    """
    out: dict[str, Any] = {
        "job": None, "drill": None, "fairness": None, "scan": None, "measure": None,
    }
    try:
        from src.jobs.queue import run_next

        out["job"] = run_next()
    except Exception:
        logger.exception("job_worker_run_next_failed")
    try:
        out["scan"] = maybe_enqueue_scheduled_scans()
    except Exception:
        logger.exception("job_worker_scan_enqueue_failed")
    try:
        out["measure"] = maybe_enqueue_measure_fixes()
    except Exception:
        logger.exception("job_worker_measure_enqueue_failed")
    try:
        from src.compliance.erasure_drill import maybe_run_weekly_drill

        out["drill"] = maybe_run_weekly_drill()
    except Exception:
        logger.exception("job_worker_erasure_drill_failed")
    if run_fairness:
        try:
            from src.frontline.analytics import evaluate_fairness_circuit_breaker

            out["fairness"] = evaluate_fairness_circuit_breaker()
        except Exception:
            logger.exception("job_worker_fairness_failed")
    return out


def run_forever(*, interval_s: float | None = None) -> None:
    raw = os.getenv("FRONTLINE_WORKER_INTERVAL_S", "").strip()
    delay = float(interval_s) if interval_s is not None else (
        float(raw) if raw else DEFAULT_INTERVAL_S
    )
    delay = max(1.0, delay)
    last_fairness = 0.0
    while True:
        now = time.monotonic()
        due = (now - last_fairness) >= FAIRNESS_EVERY_S
        worker_tick(run_fairness=due)
        if due:
            last_fairness = now
        time.sleep(delay)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logger.info("frontline_job_worker_start")
    run_forever()


if __name__ == "__main__":
    main()
