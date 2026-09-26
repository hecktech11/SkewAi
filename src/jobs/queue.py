"""Background job queue (feature #46) — Postgres-backed table or in-process.

Hermetic: default in-memory + DuckDB ``job_queue`` table. Workers call
``run_next``; no arq/redis required for pilot.

Reliability (item 18):
- Atomic claim (``UPDATE ... WHERE status='pending'`` + explicit re-SELECT
  verification — DuckDB reports ``rowcount=None`` for UPDATEs, which a
  previous version misread as a lost race and leaked ``running`` jobs).
- Lease / visibility timeout: claims carry ``lease_owner`` + ``lease_expires``.
  ``run_next`` first reaps expired leases back to ``pending`` (crashed-worker
  recovery) so a dead worker never wedges the queue.
- Retry cap (``MAX_ATTEMPTS``) with terminal ``dead``/``failed`` states.
- Idempotent completion: finishing checks ownership + status; duplicate
  completions return the stored result instead of double-applying.
- ``Idempotency-Key`` style dedupe on enqueue: same key returns the existing
  job instead of inserting a duplicate.
"""

from __future__ import annotations

import json
import re
import traceback
from datetime import timedelta
from typing import Any, Callable

from src.data.timeutil import utc_now
from src.data.warehouse import ops_con
from src.ids import new_ulid

_HANDLERS: dict[str, Callable[[dict[str, Any]], Any]] = {}

#: Permission required to enqueue *and* to execute each job type.
#:
#: Enqueueing is equivalent to invoking the operation — a worker runs the
#: payload with full process authority — so submission is gated like the
#: operation itself rather than by the router's API key alone (R01). Checking
#: only ``/jobs/run-next`` never protected execution by the automatic worker.
JOB_PERMISSIONS: dict[str, str] = {
    "audit_contact": "ops:write",
    "rebuild_clusters": "ops:write",
    "ingest_scale": "ops:write",
    "ingest_source": "ops:write",
    "recompute_anomalies": "ops:write",
    "scheduled_scan": "ops:write",
    "reenrich": "ops:write",
    "build_digest": "ops:write",
    "embedding_backfill": "ops:write",
    "rebuild_cluster_build": "ops:write",
    "audit_export": "dsr:export",
    "export_audit_regressions": "dsr:export",
    "erasure_drill": "dsr:delete",
}

#: Permission demanded of job types registered at runtime via
#: ``register_handler`` without an entry in JOB_PERMISSIONS. Admin-only: an
#: unmapped handler is an unreviewed privileged operation.
DEFAULT_JOB_PERMISSION = "jobs:run"

#: Role recorded for in-process callers (orchestrator backfill, erasure-drill
#: scheduler). These are not behind the HTTP trust boundary; every request-path
#: caller must pass the principal's resolved role explicitly.
SYSTEM_ROLE = "system"

# Known pilot job types (enqueue / run_next refuse free-form strings).
ALLOWED_JOB_TYPES = frozenset(JOB_PERMISSIONS)

MAX_ATTEMPTS = 3
DEFAULT_LEASE_S = 300

#: Payload keys carrying a filesystem reference, by job type.
_SOURCE_TAG_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")


def permission_for_job(job_type: str) -> str:
    """Permission enforced for *job_type*. Raises for unknown types."""
    jt = (job_type or "").strip()
    perm = JOB_PERMISSIONS.get(jt)
    if perm:
        return perm
    if jt in _HANDLERS:
        return DEFAULT_JOB_PERMISSION
    raise ValueError(
        f"job_type not allowlisted: {job_type!r}; allowed={sorted(ALLOWED_JOB_TYPES)}"
    )


def authorize_job(job_type: str, role: str | None) -> str:
    """Enforce the job's permission for *role*; return the permission checked.

    ``role=None`` / ``SYSTEM_ROLE`` denotes an in-process caller. Any other
    role is checked against the RBAC matrix and raises 403 when it lacks the
    permission — the same answer ``/jobs/run-next`` gives.
    """
    perm = permission_for_job(job_type)
    actor_role = (role or SYSTEM_ROLE).strip().lower()
    if actor_role == SYSTEM_ROLE:
        return perm
    from src.api.rbac import require_perm

    require_perm(actor_role, perm)
    return perm


def validate_job_payload(job_type: str, payload: Any) -> dict[str, Any]:
    """Return a validated payload for *job_type*.

    Identifiers that become path or database-name components are jailed, and
    caller-supplied file references get the same allow-root jail the upload
    routes use — a job payload reaches local file handling just as an upload
    does. Raises ValueError (InvalidIdentifier) on anything unusable.
    """
    from src.security.identifiers import (
        safe_csv_path,
        safe_mapping_path,
        safe_pack_id,
        safe_token_id,
    )

    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError("payload must be a JSON object")
    out = dict(payload)

    if out.get("pack_id") not in (None, ""):
        out["pack_id"] = safe_pack_id(str(out["pack_id"]))
    for key in ("interaction_id", "case_id"):
        if out.get(key) not in (None, ""):
            out[key] = safe_token_id(str(out[key]), kind=key)
    if out.get("limit") not in (None, ""):
        try:
            limit = int(out["limit"])
        except (TypeError, ValueError) as e:
            raise ValueError("limit must be an integer") from e
        if limit < 1 or limit > 5_000_000:
            raise ValueError("limit out of range")
        out["limit"] = limit

    if job_type == "ingest_source":
        if not str(out.get("csv_path") or "").strip():
            raise ValueError("ingest_source requires csv_path")
        out["csv_path"] = str(safe_csv_path(str(out["csv_path"])))
        if out.get("mapping_path") not in (None, ""):
            out["mapping_path"] = str(safe_mapping_path(str(out["mapping_path"])))
        tag = str(out.get("source") or "").strip().lower()
        if not _SOURCE_TAG_RE.match(tag):
            raise ValueError("ingest_source requires a short alphanumeric source tag")
        out["source"] = tag
    elif out.get("csv_path") not in (None, "") or out.get("mapping_path") not in (None, ""):
        raise ValueError(f"{job_type} does not accept file paths")

    return out


def _ensure(con) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS job_queue (
            job_id VARCHAR PRIMARY KEY,
            job_type VARCHAR NOT NULL,
            status VARCHAR NOT NULL,
            payload_json VARCHAR,
            result_json VARCHAR,
            error VARCHAR,
            attempts INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT current_timestamp,
            started_at TIMESTAMP,
            finished_at TIMESTAMP,
            lease_owner VARCHAR,
            lease_expires TIMESTAMP,
            idempotency_key VARCHAR,
            requested_by VARCHAR,
            requested_role VARCHAR
        )
        """
    )
    # Forward-compatible lease columns for pre-item-18 tables.
    for ddl in (
        "ALTER TABLE job_queue ADD COLUMN lease_owner VARCHAR",
        "ALTER TABLE job_queue ADD COLUMN lease_expires TIMESTAMP",
        "ALTER TABLE job_queue ADD COLUMN idempotency_key VARCHAR",
        "ALTER TABLE job_queue ADD COLUMN requested_by VARCHAR",
        "ALTER TABLE job_queue ADD COLUMN requested_role VARCHAR",
    ):
        try:
            con.execute(ddl)
        except Exception:
            pass


def register_handler(job_type: str, fn: Callable[[dict[str, Any]], Any]) -> None:
    _HANDLERS[job_type] = fn


def enqueue(
    job_type: str,
    payload: dict[str, Any] | None = None,
    *,
    idempotency_key: str | None = None,
    role: str | None = None,
    principal: str | None = None,
) -> dict[str, Any]:
    """Queue a job after authorizing *role* for it.

    Request-path callers MUST pass the principal's resolved ``role`` (and
    ``principal`` for the audit trail). Omitting ``role`` records the
    in-process ``system`` principal and is only correct for callers that are
    not behind the HTTP trust boundary.
    """
    jt = (job_type or "").strip()
    authorize_job(jt, role)  # raises ValueError (unknown) / HTTPException (403)
    clean_payload = validate_job_payload(jt, payload)
    actor_role = (role or SYSTEM_ROLE).strip().lower()
    actor = (principal or "").strip()[:80] or (
        SYSTEM_ROLE if actor_role == SYSTEM_ROLE else actor_role
    )
    with ops_con() as con:
        _ensure(con)
        # Idempotent enqueue: same key returns the existing live job.
        if idempotency_key:
            try:
                row = con.execute(
                    """
                    SELECT job_id, job_type, status FROM job_queue
                    WHERE idempotency_key = ? AND status IN ('pending', 'running')
                    """,
                    [idempotency_key],
                ).fetchone()
            except Exception:
                row = None
            if row:
                return {"job_id": row[0], "job_type": row[1], "status": row[2],
                        "duplicate": True}
        jid = f"job_{new_ulid()}"
        con.execute(
            """
            INSERT INTO job_queue
                (job_id, job_type, status, payload_json, idempotency_key,
                 requested_by, requested_role)
            VALUES (?, ?, 'pending', ?, ?, ?, ?)
            """,
            [jid, jt, json.dumps(clean_payload), idempotency_key, actor, actor_role],
        )
    return {"job_id": jid, "job_type": jt, "status": "pending"}


def _reap_expired_leases(con, *, now) -> int:
    """Return expired ``running`` leases to ``pending`` (crashed-worker recovery).

    Each reap counts as an attempt; jobs exhausting MAX_ATTEMPTS go ``dead``.
    Returns the number of jobs reaped.
    """
    try:
        stale = con.execute(
            """
            SELECT job_id, attempts FROM job_queue
            WHERE status = 'running' AND lease_expires IS NOT NULL AND lease_expires < ?
            """,
            [now],
        ).fetchall()
    except Exception:
        return 0
    n = 0
    for jid, attempts in stale:
        attempts = int(attempts or 0) + 1
        if attempts >= MAX_ATTEMPTS:
            con.execute(
                """
                UPDATE job_queue
                SET status = 'dead', finished_at = ?, lease_owner = NULL,
                    lease_expires = NULL,
                    error = ? WHERE job_id = ? AND status = 'running'
                """,
                [now, f"lease expired {attempts}x; worker presumed crashed", jid],
            )
        else:
            con.execute(
                """
                UPDATE job_queue
                SET status = 'pending', attempts = ?, lease_owner = NULL,
                    lease_expires = NULL, started_at = NULL,
                    error = ? WHERE job_id = ? AND status = 'running'
                """,
                [attempts, f"lease expired; requeued (attempt {attempts})", jid],
            )
        n += 1
    return n


def run_next(
    *,
    worker_id: str | None = None,
    lease_s: int = DEFAULT_LEASE_S,
) -> dict[str, Any] | None:
    """Claim and run one pending job (at-most-one-worker executes it).

    - Reaps expired leases first (visibility timeout).
    - Claims atomically; verifies the claim with an explicit re-SELECT
      (DuckDB UPDATE rowcount is unreliable — never infer races from it).
    - Completion is idempotent: only the lease owner transitions
      running -> done/failed/pending.
    """
    owner = worker_id or f"worker_{new_ulid()}"
    now = utc_now()
    lease_until = now + timedelta(seconds=max(1, int(lease_s)))
    with ops_con() as con:
        _ensure(con)
        _reap_expired_leases(con, now=now)
        row = con.execute(
            """
            SELECT job_id, job_type, payload_json, attempts, requested_role, requested_by
            FROM job_queue WHERE status = 'pending'
            ORDER BY created_at ASC LIMIT 1
            """
        ).fetchone()
        if not row:
            return None
        jid, jtype, payload_raw, attempts = row[0], row[1], row[2], int(row[3] or 0)
        req_role, req_by = row[4], row[5]
        if int(attempts or 0) >= MAX_ATTEMPTS:
            con.execute(
                "UPDATE job_queue SET status='dead', finished_at=?, error=? WHERE job_id=? AND status='pending'",
                [now, f"max attempts ({MAX_ATTEMPTS}) exceeded", jid],
            )
            return {"job_id": jid, "status": "dead", "error": "max attempts exceeded"}
        con.execute(
            """
            UPDATE job_queue
            SET status='running', started_at=?, attempts=?, lease_owner=?, lease_expires=?
            WHERE job_id=? AND status='pending'
            """,
            [now, attempts + 1, owner, lease_until, jid],
        )
        # Explicit verification — never trust rowcount (None/-1 on DuckDB).
        check = con.execute(
            "SELECT status, lease_owner FROM job_queue WHERE job_id = ?", [jid]
        ).fetchone()
        if not check or check[0] != "running" or check[1] != owner:
            return None  # lost the race; another worker claimed it
    # Re-enforce the enqueue-time policy: a worker must not lend its process
    # authority to a job whose initiating principal could not authorize it,
    # whatever path inserted the row (R01).
    denial = _authorization_denial(jtype, req_role, req_by)
    if denial is not None:
        return _finish(jid, owner, ok=False, error=denial, terminal=True)
    payload = json.loads(payload_raw or "{}")
    fn = _HANDLERS.get(jtype)

    import threading

    stop_heartbeat = threading.Event()

    def _heartbeat() -> None:
        interval = max(0.1, min(float(lease_s) / 3.0, 5.0))
        while not stop_heartbeat.wait(interval):
            try:
                new_expiry = utc_now() + timedelta(seconds=max(1, int(lease_s)))
                with ops_con() as c:
                    c.execute(
                        "UPDATE job_queue SET lease_expires = ? "
                        "WHERE job_id = ? AND status = 'running' AND lease_owner = ?",
                        [new_expiry, jid, owner],
                    )
            except Exception:
                pass

    hb_thread = threading.Thread(target=_heartbeat, daemon=True)
    hb_thread.start()

    try:
        from src.observability.otel import start_span as _span

        with _span(f"job.{jtype}", attributes={"job_id": jid, "owner": owner}):
            if fn is None:
                # Built-in no-op handlers for known pilot jobs
                result = _default_handler(jtype, payload)
            else:
                result = fn(payload)

        is_failure = False
        err_msg = None
        if isinstance(result, dict):
            if result.get("ok") is False:
                is_failure = True
                err_msg = result.get("error") or "handler returned ok=False"
            elif result.get("success") is False:
                is_failure = True
                err_msg = result.get("error") or "handler returned success=False"

        if is_failure:
            return _finish(jid, owner, ok=False, error=RuntimeError(err_msg), result=result)
        return _finish(jid, owner, ok=True, result=result)
    except Exception as e:
        return _finish(jid, owner, ok=False, error=e)
    finally:
        stop_heartbeat.set()
        hb_thread.join(timeout=1.0)


def _authorization_denial(
    job_type: str, requested_role: str | None, requested_by: str | None
) -> Exception | None:
    """Return the denial to fail the job with, or None when it may execute.

    A NULL ``requested_role`` is a row written by a build that predates
    principal recording. Those are treated as in-process ``system`` jobs so an
    upgrade does not wedge an existing queue, but the acceptance is reported to
    the security log rather than passing silently.
    """
    role = (requested_role or "").strip().lower()
    if not role:
        try:
            from src.security.audit_log import security_event

            security_event(
                "jobs.principal_missing",
                outcome="accepted",
                role=SYSTEM_ROLE,
                detail={"job_type": job_type, "requested_by": requested_by},
            )
        except Exception:
            pass
        return None
    try:
        authorize_job(job_type, role)
    except Exception as e:  # HTTPException (403) or ValueError (unknown type)
        try:
            from src.security.audit_log import security_event

            security_event(
                "jobs.execution_denied",
                outcome="denied",
                role=role,
                detail={"job_type": job_type, "requested_by": requested_by},
            )
        except Exception:
            pass
        return PermissionError(
            f"job_type {job_type!r} not permitted for initiating role {role!r}: "
            f"{getattr(e, 'detail', None) or e}"
        )
    return None


def _finish(
    jid: str,
    owner: str,
    *,
    ok: bool,
    result: Any = None,
    error: BaseException | None = None,
    terminal: bool = False,
) -> dict[str, Any]:
    """Idempotent completion: only the lease owner may transition the job.

    A duplicate completion (job already done/failed) returns the stored
    result instead of double-applying side effects. ``terminal=True`` skips
    retries for failures that cannot succeed on a second attempt (e.g. an
    authorization denial).
    """
    now = utc_now()
    with ops_con() as con:
        _ensure(con)
        cur = con.execute(
            "SELECT status, attempts, result_json, error FROM job_queue WHERE job_id = ?",
            [jid],
        ).fetchone()
        if not cur:
            return {"job_id": jid, "status": "unknown", "error": "job vanished"}
        status, attempts, result_json, err = cur[0], int(cur[1] or 0), cur[2], cur[3]
        if status in ("done", "failed", "dead"):
            try:
                stored = json.loads(result_json or "{}")
            except (json.JSONDecodeError, TypeError):
                stored = {}
            return {"job_id": jid, "status": status, "result": stored,
                    "duplicate_completion": True}
        if status != "running":
            return {"job_id": jid, "status": status, "error": err or "not running"}
        if ok:
            con.execute(
                """
                UPDATE job_queue
                SET status='done', finished_at=?, result_json=?,
                    lease_owner=NULL, lease_expires=NULL
                WHERE job_id=? AND status='running' AND lease_owner=?
                """,
                [now, json.dumps(result if isinstance(result, dict) else {"result": result}), jid, owner],
            )
            return {"job_id": jid, "status": "done", "result": result}
        n = attempts
        res_json = (
            json.dumps(result if isinstance(result, dict) else {"result": result})
            if result is not None
            else None
        )
        if terminal or n >= MAX_ATTEMPTS:
            con.execute(
                """
                UPDATE job_queue SET status='failed', finished_at=?, error=?, result_json=?,
                    lease_owner=NULL, lease_expires=NULL
                WHERE job_id=? AND status='running' AND lease_owner=?
                """,
                [now, f"{error}\n{traceback.format_exc()[-500:]}", res_json, jid, owner],
            )
        else:
            con.execute(
                """
                UPDATE job_queue SET status='pending', finished_at=NULL, error=?, result_json=?,
                    lease_owner=NULL, lease_expires=NULL, started_at=NULL
                WHERE job_id=? AND status='running' AND lease_owner=?
                """,
                [f"retry {n}/{MAX_ATTEMPTS}: {error}", res_json, jid, owner],
            )
        return {"job_id": jid, "status": "failed", "error": str(error), "result": result}


def _default_handler(jtype: str, payload: dict[str, Any]) -> dict[str, Any]:
    if jtype == "audit_contact":
        return {"ok": True, "interaction_id": payload.get("interaction_id"), "deferred": True}
    if jtype == "rebuild_clusters":
        pack = payload.get("pack_id") or "automotive_nhtsa"
        try:
            from src.ml_runtime.clustering import rebuild_clusters

            return rebuild_clusters(pack)
        except Exception as e:
            return {"ok": False, "error": str(e)}
    if jtype == "ingest_scale":
        return {"ok": True, "queued": True, "pack_id": payload.get("pack_id")}
    if jtype == "ingest_source":
        pack = payload.get("pack_id") or "automotive_nhtsa"
        try:
            from src.domains.source_ingest import ingest_source

            return ingest_source(
                pack,
                payload.get("source") or "service",
                payload.get("csv_path") or "",
                mapping_path=payload.get("mapping_path"),
            )
        except Exception as e:
            return {"ok": False, "error": str(e)}
    if jtype == "scheduled_scan":
        pack = payload.get("pack_id") or "automotive_nhtsa"
        try:
            from src.frontline.fleet_scan import run_fleet_scan

            return run_fleet_scan(
                pack,
                rebuild_clusters=bool(payload.get("rebuild_clusters", False)),
                alert=bool(payload.get("alert", True)),
            )
        except Exception as e:
            return {"ok": False, "error": str(e)}
    if jtype == "reenrich":
        try:
            from src.frontline.reenrich import backfill_brief

            return backfill_brief(
                payload.get("interaction_id") or "",
                case_id=payload.get("case_id"),
            )
        except Exception as e:
            return {"ok": False, "error": str(e)}
    if jtype == "embedding_backfill":
        from src.ml_runtime.embedding_backfill import job_handler as _emb_bf

        return _emb_bf(payload)
    if jtype == "rebuild_cluster_build":
        from src.ml_runtime.cluster_builds import rebuild_cluster_build

        return rebuild_cluster_build(
            payload.get("pack_id") or "automotive_nhtsa",
            payload.get("embedding_version") or "",
            k=int(payload.get("k") or 5),
            dry_run=bool(payload.get("dry_run")),
        )
    if jtype == "recompute_anomalies":
        pack = payload.get("pack_id") or "automotive_nhtsa"
        try:
            from src.ml_runtime.anomalies import recompute_weekly_anomalies

            rows = recompute_weekly_anomalies(
                pack,
                category=payload.get("category"),
                entity_2=payload.get("entity_2"),
            )
            return {"ok": True, "pack_id": pack, "rows": len(rows)}
        except Exception as e:
            return {"ok": False, "error": str(e)}
    if jtype == "audit_export":
        return {"ok": True, "deferred": True, "params": payload}
    if jtype == "build_digest":
        return {"ok": True, "deferred": True, "params": payload}
    if jtype == "erasure_drill":
        from src.compliance.erasure_drill import maybe_run_weekly_drill, run_erasure_drill

        if payload.get("weekly"):
            return maybe_run_weekly_drill() or {"ok": True, "skipped": True, "reason": "not_due"}
        return run_erasure_drill()
    if jtype == "export_audit_regressions":
        from src.frontline.validation_queue import export_audit_regressions

        return export_audit_regressions(since_days=int(payload.get("since_days") or 7))
    # Unknown types must not echo arbitrary payloads (allowlist gate on enqueue).
    raise ValueError(f"no handler for job_type: {jtype!r}")


def list_jobs(*, status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    with ops_con(read_only=True) as con:
        try:
            _ensure(con)
            sql = "SELECT job_id, job_type, status, attempts, created_at, finished_at FROM job_queue"
            params: list[Any] = []
            if status:
                sql += " WHERE status = ?"
                params.append(status)
            sql += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            rows = con.execute(sql, params).fetchall()
        except Exception:
            return []
    cols = ["job_id", "job_type", "status", "attempts", "created_at", "finished_at"]
    return [dict(zip(cols, r)) for r in rows]
