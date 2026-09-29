"""Data-trust layer: freshness, completeness, dedup, schema-contract, lineage.

Groundedness is a property of the ingested row, not only the last agent claim.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from typing import Any, Iterable, Iterator

from src.data.timeutil import utc_now
from src.data.warehouse import domain_con, ops_con
from src.domains.mapping_ingest import CANONICAL_RECORD_FIELDS
from src.ids import new_ulid

DEFAULT_FRESHNESS_SLA_DAYS = 90
REQUIRED_FIELDS = ("record_id", "text")


def _as_dt(val: Any) -> datetime | None:
    if val is None:
        return None
    if isinstance(val, datetime):
        return val.replace(tzinfo=None)
    s = str(val).strip()
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


_SCHEMA_OK_EXTRA = frozenset({"embedding", "pack_id"})


def schema_contract_ok(row: dict[str, Any]) -> tuple[bool, str]:
    for f in REQUIRED_FIELDS:
        if not str(row.get(f) or "").strip():
            return False, f"schema: missing {f}"
    for k in row:
        if k not in CANONICAL_RECORD_FIELDS and k not in _SCHEMA_OK_EXTRA:
            return False, f"schema: unknown field {k}"
    received = row.get("received_at")
    if received is not None and _as_dt(received) is None and not isinstance(received, datetime):
        return False, "schema: received_at not a timestamp"
    return True, "schema: ok"


def completeness_ok(row: dict[str, Any], *, required: Iterable[str] = ("record_id", "text", "source")) -> tuple[bool, str]:
    missing = [f for f in required if not str(row.get(f) or "").strip()]
    if missing:
        return False, "completeness: empty " + ",".join(missing)
    return True, "completeness: ok"


def freshness_ok(
    row: dict[str, Any],
    *,
    sla_days: int = DEFAULT_FRESHNESS_SLA_DAYS,
    now: datetime | None = None,
) -> tuple[bool, str]:
    ts = _as_dt(row.get("received_at") or row.get("occurred_at"))
    if ts is None:
        return False, "freshness: no timestamp"
    age = (now or utc_now().replace(tzinfo=None)) - ts
    if age > timedelta(days=sla_days):
        return False, f"freshness: stale ({age.days}d > {sla_days}d SLA)"
    return True, f"freshness: {age.days}d within {sla_days}d"


def content_hash(row: dict[str, Any]) -> str:
    body = {
        "record_id": str(row.get("record_id") or ""),
        "text": str(row.get("text") or ""),
        "source": str(row.get("source") or ""),
    }
    raw = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def dedup_ok(row: dict[str, Any], seen_ids: set[str], seen_hashes: set[str]) -> tuple[bool, str]:
    rid = str(row.get("record_id") or "")
    h = content_hash(row)
    if rid and rid in seen_ids:
        return False, f"dedup: record_id {rid} already ingested"
    if h in seen_hashes:
        return False, "dedup: identical content already ingested"
    return True, "dedup: unique"


def evaluate_record(
    row: dict[str, Any],
    *,
    seen_ids: set[str] | None = None,
    seen_hashes: set[str] | None = None,
    sla_days: int = DEFAULT_FRESHNESS_SLA_DAYS,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Pure: return lineage badge + per-check results. Does not persist."""
    seen_ids = seen_ids if seen_ids is not None else set()
    seen_hashes = seen_hashes if seen_hashes is not None else set()
    checks = {
        "schema": schema_contract_ok(row),
        "completeness": completeness_ok(row),
        "freshness": freshness_ok(row, sla_days=sla_days, now=now),
        "dedup": dedup_ok(row, seen_ids, seen_hashes),
    }
    failed = [name for name, (ok, _msg) in checks.items() if not ok]
    trusted = not failed
    source = str(row.get("source") or "unknown")
    freshness_verdict = "fresh" if checks["freshness"][0] else "stale"
    grounded = "trusted" if trusted else "untrusted"
    badge = {
        "source": source,
        "freshness": freshness_verdict,
        "grounded_verdict": grounded,
        "failed_checks": failed,
        "why_trusted": (
            f"{source} → {freshness_verdict} → {grounded}"
            if trusted
            else f"{source} → rejected ({', '.join(failed)})"
        ),
    }
    return {
        "record_id": str(row.get("record_id") or ""),
        "trusted": trusted,
        "checks": {k: {"ok": ok, "detail": msg} for k, (ok, msg) in checks.items()},
        "badge": badge,
        "content_hash": content_hash(row),
    }


def _ensure(con) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS record_trust (
            trust_id      VARCHAR PRIMARY KEY,
            pack_id       VARCHAR NOT NULL,
            record_id     VARCHAR NOT NULL,
            trusted       BOOLEAN NOT NULL,
            badge_json    TEXT NOT NULL,
            checks_json   TEXT NOT NULL,
            content_hash  VARCHAR NOT NULL,
            evaluated_at  TIMESTAMP NOT NULL
        )
        """
    )


#: Evaluations buffered before one write to ``record_trust``. A per-row
#: persist opens its own ops connection — measured at 20.2 ms/row, of which
#: 11.7 ms is connect-and-close — so a 500k load spent ~2.8 hours here alone,
#: dwarfing both the embedding cost and the domain-warehouse write.
_TRUST_BATCH_ROWS = 10_000

_TRUST_INSERT_COLUMNS = (
    "trust_id",
    "pack_id",
    "record_id",
    "trusted",
    "badge_json",
    "checks_json",
    "content_hash",
    "evaluated_at",
)


def _trust_arrow_schema():
    import pyarrow as pa

    return pa.schema(
        [
            ("trust_id", pa.string()),
            ("pack_id", pa.string()),
            ("record_id", pa.string()),
            ("trusted", pa.bool_()),
            ("badge_json", pa.string()),
            ("checks_json", pa.string()),
            ("content_hash", pa.string()),
            # utc_now() is naive UTC, which is what the column stores.
            ("evaluated_at", pa.timestamp("us")),
        ]
    )


def _trust_staged_row(pack_id: str, evaluation: dict[str, Any], tid: str, now) -> dict[str, Any]:
    return {
        "trust_id": tid,
        "pack_id": pack_id,
        "record_id": evaluation.get("record_id") or "",
        "trusted": bool(evaluation.get("trusted")),
        "badge_json": json.dumps(evaluation.get("badge") or {}, default=str),
        "checks_json": json.dumps(evaluation.get("checks") or {}, default=str),
        "content_hash": evaluation.get("content_hash") or "",
        "evaluated_at": now,
    }


def persist_trust_many(pack_id: str, evaluations: Iterable[dict[str, Any]]) -> list[str]:
    """Write a batch of evaluations over one ops connection.

    Returns the new trust_ids in input order. An empty batch is a no-op and
    must not open a connection — a load whose rows are all rejected still
    reaches here once per batch.
    """
    import pyarrow as pa

    now = utc_now()
    staged: list[dict[str, Any]] = []
    tids: list[str] = []
    for ev in evaluations:
        tid = "tr_" + new_ulid()
        tids.append(tid)
        staged.append(_trust_staged_row(pack_id, ev, tid, now))
    if not staged:
        return []
    cols = ", ".join(_TRUST_INSERT_COLUMNS)
    table = pa.Table.from_pylist(staged, schema=_trust_arrow_schema())
    with ops_con() as con:
        _ensure(con)
        con.register("_trust_batch", table)
        try:
            con.execute(f"INSERT INTO record_trust ({cols}) SELECT {cols} FROM _trust_batch")
        finally:
            con.unregister("_trust_batch")
    return tids


def persist_trust(pack_id: str, evaluation: dict[str, Any]) -> str:
    """Write one evaluation. Batch loads should use ``persist_trust_many``."""
    return persist_trust_many(pack_id, [evaluation])[0]


def load_seen(pack_id: str) -> tuple[set[str], set[str]]:
    """Seed dedup sets from prior trust rows and warehouse record_ids."""
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    with ops_con(read_only=True) as con:
        try:
            rows = con.execute(
                """
                SELECT record_id, content_hash FROM record_trust
                WHERE pack_id = ?
                """,
                [pack_id],
            ).fetchall()
        except Exception:
            rows = []
    for rid, h in rows:
        if rid:
            seen_ids.add(str(rid))
        if h:
            seen_hashes.add(str(h))
    try:
        with domain_con(pack_id) as dcon:
            live = dcon.execute(
                "SELECT record_id, text, source FROM records"
            ).fetchall()
    except (FileNotFoundError, Exception):
        live = []
    for rid, text, source in live:
        if rid:
            seen_ids.add(str(rid))
        seen_hashes.add(
            content_hash(
                {"record_id": rid or "", "text": text or "", "source": source or ""}
            )
        )
    return seen_ids, seen_hashes


def iter_trusted(
    pack_id: str,
    rows: Iterable[dict[str, Any]],
    *,
    sla_days: int = DEFAULT_FRESHNESS_SLA_DAYS,
    persist: bool = True,
    keep_untrusted: bool = False,
) -> tuple[dict[str, int], Iterator[dict[str, Any]]]:
    """Stream rows through the trust contract, yielding the ones to write.

    Returns ``(stats, stream)``. ``stats`` is filled in as the stream is
    consumed and is complete once it is exhausted; reading it early gives a
    partial count, not a wrong one.

    This exists so a bulk load never holds the whole CSV in memory:
    ``evaluate_ingest`` accumulates every mapped row plus every evaluation,
    which is ~1.5 GB over a 500k load and does not fit a full-corpus one.

    The dedup seed is loaded eagerly, before the generator is returned, and
    that ordering is load-bearing rather than incidental. ``load_seen``
    opens the pack's domain warehouse, and the caller feeds this stream to
    ``upsert_records``, which holds a write connection to that same
    warehouse for the length of the iteration. DuckDB is single-writer and
    ``domain_con`` takes a non-reentrant per-pack lock, so seeding from
    inside the generator body would run under the writer and deadlock.
    """
    seen_ids, seen_hashes = load_seen(pack_id)
    stats: dict[str, int] = {"evaluated": 0, "trusted": 0, "rejected": 0}

    def _gen() -> Iterator[dict[str, Any]]:
        pending: list[dict[str, Any]] = []
        for row in rows:
            ev = evaluate_record(
                row, seen_ids=seen_ids, seen_hashes=seen_hashes, sla_days=sla_days
            )
            stats["evaluated"] += 1
            rid = ev["record_id"]
            if rid:
                seen_ids.add(rid)
            seen_hashes.add(ev["content_hash"])
            if ev["trusted"]:
                stats["trusted"] += 1
            else:
                stats["rejected"] += 1
            if persist:
                pending.append(ev)
                if len(pending) >= _TRUST_BATCH_ROWS:
                    persist_trust_many(pack_id, pending)
                    pending = []
            if ev["trusted"] or keep_untrusted:
                yield row
        if persist and pending:
            persist_trust_many(pack_id, pending)

    return stats, _gen()


def evaluate_ingest(
    pack_id: str,
    rows: Iterable[dict[str, Any]],
    *,
    sla_days: int = DEFAULT_FRESHNESS_SLA_DAYS,
    persist: bool = True,
) -> dict[str, Any]:
    """Evaluate a batch and return every evaluation.

    Holds one evaluation per row, so it suits a preview or a sandbox sample.
    A bulk load should use ``iter_trusted``, which streams.
    """
    seen_ids, seen_hashes = load_seen(pack_id)
    results: list[dict[str, Any]] = []
    trusted_rows: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for row in rows:
        ev = evaluate_record(
            row, seen_ids=seen_ids, seen_hashes=seen_hashes, sla_days=sla_days
        )
        results.append(ev)
        rid = ev["record_id"]
        if rid:
            seen_ids.add(rid)
        seen_hashes.add(ev["content_hash"])
        if ev["trusted"]:
            trusted_rows.append(row)
        if persist:
            pending.append(ev)
            if len(pending) >= _TRUST_BATCH_ROWS:
                persist_trust_many(pack_id, pending)
                pending = []
    if persist and pending:
        persist_trust_many(pack_id, pending)
    return {
        "pack_id": pack_id,
        "evaluated": len(results),
        "trusted": len(trusted_rows),
        "rejected": len(results) - len(trusted_rows),
        "results": results,
        "trusted_rows": trusted_rows,
    }


def lineage_for_record(pack_id: str, record_id: str) -> dict[str, Any] | None:
    with ops_con(read_only=True) as con:
        try:
            row = con.execute(
                """
                SELECT trust_id, trusted, badge_json, checks_json, content_hash, evaluated_at
                FROM record_trust
                WHERE pack_id = ? AND record_id = ?
                ORDER BY evaluated_at DESC LIMIT 1
                """,
                [pack_id, record_id],
            ).fetchone()
        except Exception:
            return None
    if not row:
        return None
    badge = json.loads(row[2]) if row[2] else {}
    checks = json.loads(row[3]) if row[3] else {}
    return {
        "trust_id": row[0],
        "pack_id": pack_id,
        "record_id": record_id,
        "trusted": bool(row[1]),
        "badge": badge,
        "checks": checks,
        "content_hash": row[4],
        "evaluated_at": str(row[5]),
        "chain_of_custody": [
            {"step": "source", "value": badge.get("source")},
            {"step": "freshness", "value": badge.get("freshness")},
            {"step": "grounded_verdict", "value": badge.get("grounded_verdict")},
            {"step": "why", "value": badge.get("why_trusted")},
        ],
    }


__all__ = [
    "DEFAULT_FRESHNESS_SLA_DAYS",
    "evaluate_record",
    "evaluate_ingest",
    "iter_trusted",
    "load_seen",
    "lineage_for_record",
    "persist_trust",
    "persist_trust_many",
    "schema_contract_ok",
    "completeness_ok",
    "freshness_ok",
    "dedup_ok",
    "content_hash",
]
