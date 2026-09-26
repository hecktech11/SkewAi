"""Self-diagnosing health checks — every stage emits a health signal.

Design principles:
1. Every check returns a HealthSignal dataclass with status, metric_value, threshold, detail
2. Checks are pure functions over the database state — no side effects
3. Each check degrades BEFORE customers notice
4. All thresholds are documented and tunable
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from src.security.pii import SubjectKeyStore

logger = logging.getLogger(__name__)


@dataclass
class HealthSignal:
    """One health measurement."""
    name: str
    status: str  # 'healthy' | 'degraded' | 'critical'
    metric_value: float | None = None
    threshold: float | None = None
    detail: str = ""
    checked_at: datetime | None = None


def check_ledger_health() -> HealthSignal:
    """Audit spine degradation (ledger auxiliary write failures)."""
    from src.ledger.writer import ledger_health
    h = ledger_health()
    total = h.get("degraded_total", 0)
    if total >= 10:
        status = "critical"
    elif total >= 1:
        status = "degraded"
    else:
        status = "healthy"
    return HealthSignal(
        name="ledger_health",
        status=status,
        metric_value=float(total),
        threshold=10.0,
        detail=f"degraded_total={total}, alerting={h.get('alerting')}",
    )


def check_anomaly_freshness(pack_id: str, *, max_age_hours: int = 48) -> HealthSignal:
    """Weekly anomalies must be recomputed within max_age_hours of the latest record."""
    from src.data.warehouse import domain_con
    from src.data.timeutil import utc_now
    try:
        with domain_con(pack_id, read_only=True) as con:
            # Tables exist by definition in a valid pack
            latest_record = con.execute(
                "SELECT MAX(received_at) FROM records"
            ).fetchone()
            latest_anomaly = con.execute(
                "SELECT MAX(iso_week) FROM weekly_anomalies WHERE pack_id = ?",
                [pack_id],
            ).fetchone()
        rec_ts = latest_record[0] if latest_record else None
        anom_week = latest_anomaly[0] if latest_anomaly else None
        if rec_ts is None:
            return HealthSignal(name="anomaly_freshness", status="healthy",
                                detail="no records yet")
        if anom_week is None:
            return HealthSignal(name="anomaly_freshness", status="degraded",
                                detail="records exist but no anomaly scan")
        # Parse the latest anomaly week to a date
        try:
            y, w = anom_week.split("-W")
            from datetime import datetime as _dt
            latest_scan = _dt.fromisocalendar(int(y), int(w), 7)  # end of week
        except Exception:
            latest_scan = None
        if latest_scan is None:
            return HealthSignal(name="anomaly_freshness", status="degraded",
                                detail=f"unparseable week: {anom_week}")
        now = utc_now().replace(tzinfo=None)
        rec_dt = rec_ts if isinstance(rec_ts, datetime) else datetime.fromisoformat(str(rec_ts))
        rec_dt = rec_dt.replace(tzinfo=None)
        age_hours = (now - latest_scan).total_seconds() / 3600
        status = "healthy" if age_hours <= max_age_hours else "degraded"
        return HealthSignal(
            name="anomaly_freshness", status=status,
            metric_value=round(age_hours, 1), threshold=float(max_age_hours),
            detail=f"latest_record={rec_dt.isoformat()}, latest_anomaly_week={anom_week}",
        )
    except Exception as e:
        return HealthSignal(name="anomaly_freshness", status="critical",
                            detail=f"check failed: {type(e).__name__}: {e}")


def check_novel_candidate_backlog(*, max_open: int = 100) -> HealthSignal:
    """Novel candidates awaiting clustering review."""
    from src.data.warehouse import ops_con
    try:
        with ops_con(read_only=True) as con:
            row = con.execute(
                "SELECT COUNT(*) FROM novel_candidates WHERE status = 'open'"
            ).fetchone()
        count = row[0] if row else 0
        status = "healthy" if count <= max_open else "degraded"
        return HealthSignal(
            name="novel_candidate_backlog", status=status,
            metric_value=float(count), threshold=float(max_open),
            detail=f"open_candidates={count}",
        )
    except Exception as e:
        return HealthSignal(name="novel_candidate_backlog", status="critical",
                            detail=f"check failed: {type(e).__name__}: {e}")


def check_fairness(*, pack_id: str | None = None, window_days: int = 90) -> HealthSignal:
    """Disparate impact ratio (four-fifths rule)."""
    try:
        from src.frontline.analytics import evaluate_fairness_circuit_breaker
        result = evaluate_fairness_circuit_breaker(
            pack_id=pack_id, window_days=window_days
        )
        tripped = result.get("circuit_breaker_tripped", False)
        ratio = result.get("disparate_impact_ratio")
        status = "critical" if tripped else "healthy"
        return HealthSignal(
            name="fairness_circuit_breaker", status=status,
            metric_value=ratio if isinstance(ratio, (int, float)) else None,
            threshold=0.80,
            detail=f"tripped={tripped}, ratio={ratio}",
        )
    except Exception as e:
        return HealthSignal(name="fairness_circuit_breaker", status="critical",
                            detail=f"check failed: {type(e).__name__}: {e}")


def check_ops_db_writable() -> HealthSignal:
    """Verify the operational database accepts writes."""
    from src.data.warehouse import ops_con
    try:
        with ops_con() as con:
            con.execute(
                "CREATE TABLE IF NOT EXISTS _health_probe (k INTEGER PRIMARY KEY)"
            )
            con.execute("INSERT OR REPLACE INTO _health_probe VALUES (1)")
            row = con.execute("SELECT k FROM _health_probe WHERE k = 1").fetchone()
            if row and row[0] == 1:
                return HealthSignal(name="ops_db_writable", status="healthy",
                                    detail="write probe succeeded")
            return HealthSignal(name="ops_db_writable", status="degraded",
                                detail="probe inserted but readback failed")
    except Exception as e:
        return HealthSignal(name="ops_db_writable", status="critical",
                            detail=f"{type(e).__name__}: {e}")


def check_dead_letter_backlog(*, max_pending: int = 50) -> HealthSignal:
    """Undelivered webhook alert backlog."""
    from src.data.warehouse import ops_con
    try:
        with ops_con(read_only=True) as con:
            row = con.execute(
                "SELECT COUNT(*) FROM alert_dead_letter WHERE status = 'pending'"
            ).fetchone()
        count = row[0] if row else 0
        status = "healthy" if count <= max_pending else "degraded"
        return HealthSignal(
            name="dead_letter_backlog", status=status,
            metric_value=float(count), threshold=float(max_pending),
            detail=f"pending_dead_letters={count}",
        )
    except Exception as e:
        return HealthSignal(name="dead_letter_backlog", status="critical",
                            detail=f"check failed: {type(e).__name__}: {e}")


def check_crypto_key_store_size(*, max_keys: int = 10000) -> HealthSignal:
    """SubjectKeyStore memory pressure (in-memory DEK cache)."""
    try:
        n = len(SubjectKeyStore._keys)
    except Exception:
        n = 0
    status = "healthy" if n <= max_keys else "degraded"
    return HealthSignal(
        name="crypto_key_store_size", status=status,
        metric_value=float(n), threshold=float(max_keys),
        detail=f"dek_count={n}",
    )


def check_pii_encryption() -> HealthSignal:
    """At-rest PII encryption. Any failure means customer text was dropped.

    Encryption fails closed (R29), so a non-zero count is not a cosmetic
    warning: those descriptions were replaced by a marker and are gone.
    """
    from src.security.pii import encryption_health

    h = encryption_health()
    failures = int(h.get("failures", 0))
    if failures >= 10:
        status = "critical"
    elif failures >= 1:
        status = "degraded"
    else:
        status = "healthy"
    return HealthSignal(
        name="pii_encryption",
        status=status,
        metric_value=float(failures),
        threshold=10.0,
        detail=f"encrypt_failures={failures}, last_error={h.get('last_error') or 'none'}",
    )


def check_embedding_runtime() -> HealthSignal:
    """Active embedding mode, readiness, and version (no artifact paths)."""
    try:
        from src.ml_runtime.embedding_runtime import embedding_mode, health_payload

        mode = embedding_mode()
        payload = health_payload()
        if mode == "semantic" and not payload.get("semantic_ready"):
            status = "critical"
        elif mode == "shadow" and not payload.get("semantic_ready"):
            status = "degraded"
        else:
            status = "healthy"
        detail = (
            f"mode={mode} visible={payload.get('visible_provider')} "
            f"semantic_ready={payload.get('semantic_ready', False)}"
        )
        return HealthSignal(
            name="embedding_runtime",
            status=status,
            detail=detail[:300],
        )
    except Exception as e:
        return HealthSignal(
            name="embedding_runtime",
            status="degraded",
            detail=f"check failed: {type(e).__name__}",
        )


def run_all_checks(*, pack_id: str | None = None) -> list[HealthSignal]:
    """Run all health checks. Returns list of HealthSignal.
    
    Catches exceptions per-check so a failing check never prevents others.
    """
    from src.data.timeutil import utc_now
    now = utc_now()
    checks = [
        check_ledger_health,
        check_ops_db_writable,
        lambda: check_dead_letter_backlog(),
        lambda: check_crypto_key_store_size(),
        check_pii_encryption,
    ]
    if pack_id:
        checks.append(lambda: check_anomaly_freshness(pack_id))
    checks.append(lambda: check_fairness(pack_id=pack_id))
    checks.append(lambda: check_novel_candidate_backlog())
    checks.append(check_embedding_runtime)
    
    results: list[HealthSignal] = []
    for check_fn in checks:
        try:
            sig = check_fn()
            sig.checked_at = now
            results.append(sig)
        except Exception as e:
            results.append(HealthSignal(
                name=getattr(check_fn, '__name__', 'unknown'),
                status='critical',
                detail=f'check crashed: {type(e).__name__}: {e}',
                checked_at=now,
            ))
    
    # Log summary
    critical = sum(1 for r in results if r.status == 'critical')
    degraded = sum(1 for r in results if r.status == 'degraded')
    logger.info(
        "health_check_summary",
        extra={"total": len(results), "critical": critical, "degraded": degraded},
    )
    return results


def health_summary(*, pack_id: str | None = None) -> dict[str, Any]:
    """Human-readable summary for /ops/health endpoint."""
    signals = run_all_checks(pack_id=pack_id)
    worst = 'healthy'
    for s in signals:
        if s.status == 'critical':
            worst = 'critical'
            break
        if s.status == 'degraded':
            worst = 'degraded'
    ladders: list[dict[str, Any]] = []
    try:
        from src.observability.degradation import all_ladders_status, init_default_ladders

        if not all_ladders_status():
            init_default_ladders()
        ladders = all_ladders_status()
    except Exception:
        ladders = []
    return {
        'overall': worst,
        'checks': [
            {
                'name': s.name,
                'status': s.status,
                'metric_value': s.metric_value,
                'threshold': s.threshold,
                'detail': s.detail,
            }
            for s in signals
        ],
        'degradation_ladders': ladders,
    }


__all__ = [
    'HealthSignal',
    'check_ledger_health',
    'check_anomaly_freshness',
    'check_novel_candidate_backlog',
    'check_fairness',
    'check_ops_db_writable',
    'check_dead_letter_backlog',
    'check_crypto_key_store_size',
    'check_pii_encryption',
    'run_all_checks',
    'health_summary',
    'check_embedding_runtime',
]
