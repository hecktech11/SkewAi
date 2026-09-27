from __future__ import annotations

from collections import defaultdict
import json
import os
import threading
import time
from typing import Any

from src.data.timeutil import utc_now
from src.data.warehouse import ops_con
from src.ids import new_ulid

_dsr_rate_limit_lock = threading.Lock()
_dsr_export_history: dict[str, list[float]] = defaultdict(list)


def check_dsr_export_rate_limit(
    principal: str,
    limit: int | None = None,
    window_s: float = 3600.0,
) -> tuple[bool, int]:
    """Sliding-window rate limiter for DSR exports (default 5/hour per principal)."""
    if limit is None:
        try:
            limit = int(os.getenv("FRONTLINE_DSR_EXPORT_RATE_LIMIT", "5"))
        except Exception:
            limit = 5
    now = time.time()
    cutoff = now - window_s
    with _dsr_rate_limit_lock:
        history = [ts for ts in _dsr_export_history[principal] if ts > cutoff]
        if len(history) >= limit:
            oldest = min(history) if history else now
            retry_after = max(1, int(window_s - (now - oldest)))
            _dsr_export_history[principal] = history
            return False, retry_after
        history.append(now)
        _dsr_export_history[principal] = history
        return True, 0


def reset_dsr_export_rate_limits() -> None:
    """Reset rate limit history for tests."""
    with _dsr_rate_limit_lock:
        _dsr_export_history.clear()


def _ensure_dsr_audit_table(con) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS dsr_export_audit (
            audit_id VARCHAR PRIMARY KEY,
            principal VARCHAR NOT NULL,
            scope VARCHAR NOT NULL,
            record_count INTEGER NOT NULL,
            ip VARCHAR,
            created_at TIMESTAMP NOT NULL
        )
        """
    )


def log_dsr_export_audit(
    *,
    principal: str,
    scope: str,
    record_count: int,
    ip: str | None = None,
) -> None:
    """Log an audit row for DSR export."""
    aid = "dsr_aud_" + new_ulid()
    now = utc_now()
    try:
        with ops_con() as con:
            _ensure_dsr_audit_table(con)
            con.execute(
                """
                INSERT INTO dsr_export_audit
                (audit_id, principal, scope, record_count, ip, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [aid, str(principal), str(scope), int(record_count), ip, now],
            )
    except Exception:
        pass


def export_interaction(interaction_id: str, *, redact_pii: bool = False) -> dict[str, Any]:
    """Export all ops rows linked to one interaction.

    redact_pii=False by default because DSR/legal export must be complete for
    the data subject. Pass redact_pii=True for support-screen views.
    """
    iid = interaction_id
    out: dict[str, Any] = {"interaction_id": iid, "exported_at": utc_now().isoformat() + "Z"}
    with ops_con(read_only=True) as con:
        for name, sql, params in (
            ("interaction", "SELECT * FROM interactions WHERE interaction_id = ?", [iid]),
            ("turns", "SELECT * FROM interaction_turns WHERE interaction_id = ? ORDER BY seq", [iid]),
            ("actions", "SELECT * FROM agent_actions WHERE interaction_id = ? ORDER BY ts", [iid]),
            ("cases", "SELECT * FROM cases WHERE interaction_id = ?", [iid]),
            (
                "version_stamps",
                "SELECT * FROM interaction_version_stamps WHERE interaction_id = ?",
                [iid],
            ),
        ):
            try:
                cur = con.execute(sql, params)
                cols = [d[0] for d in cur.description]
                rows = [dict(zip(cols, r)) for r in cur.fetchall()]
                for r in rows:
                    for k, v in list(r.items()):
                        if hasattr(v, "isoformat"):
                            r[k] = v.isoformat()
                if name == "turns":
                    from src.data.turns import decrypt_turn_rows

                    rows = decrypt_turn_rows(iid, rows)
                if name == "cases":
                    from src.security.pii import decrypt_case_rows

                    rows = decrypt_case_rows(rows)
                out[name] = rows if name != "interaction" else (rows[0] if rows else None)
            except Exception as e:
                out[name] = {"error": f"{type(e).__name__}:{e}"}

        # notes for cases
        case_ids = [c["case_id"] for c in (out.get("cases") or []) if c.get("case_id")]
        notes = []
        for cid in case_ids:
            try:
                cur = con.execute(
                    "SELECT * FROM case_notes WHERE case_id = ?", [cid]
                )
                cols = [d[0] for d in cur.description]
                for r in cur.fetchall():
                    d = dict(zip(cols, r))
                    for k, v in list(d.items()):
                        if hasattr(v, "isoformat"):
                            d[k] = v.isoformat()
                    notes.append(d)
            except Exception:
                pass
        out["case_notes"] = notes
    if redact_pii:
        try:
            from src.security.pii import redact_pii as _redact

            def _scrub(obj: Any) -> Any:
                if isinstance(obj, dict):
                    return {k: (_redact(v) if isinstance(v, str) else _scrub(v)) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [_scrub(x) for x in obj]
                if isinstance(obj, str):
                    return _redact(obj)
                return obj

            for key in ("interaction", "turns", "actions", "cases", "case_notes"):
                if key in out:
                    out[key] = _scrub(out[key])
        except Exception:
            pass
    return out


def erasure_succeeded(counts: dict[str, Any]) -> bool:
    """False when a step failed or a plaintext copy was still present afterward."""
    if int(counts.get("leftovers") or 0) > 0:
        return False
    for value in counts.values():
        if value == -1 or value == "failed":
            return False
    return True


def _relation_exists(con: Any, name: str) -> bool:
    row = con.execute(
        """
        SELECT 1 FROM information_schema.tables
        WHERE table_name = ?
        LIMIT 1
        """,
        [name],
    ).fetchone()
    return row is not None


def _plaintext_leftovers(con: Any, interaction_id: str, stamp: str) -> int:
    """Rows whose erasable free text is not the tombstone we just wrote."""
    leftover = 0
    for summary, draft in con.execute(
        "SELECT description_summary, followup_draft FROM cases WHERE interaction_id = ?",
        [interaction_id],
    ).fetchall():
        if summary != stamp or draft != stamp:
            leftover += 1
    for (body,) in con.execute(
        """
        SELECT body FROM case_notes
        WHERE case_id IN (SELECT case_id FROM cases WHERE interaction_id = ?)
        """,
        [interaction_id],
    ).fetchall():
        if body != stamp:
            leftover += 1
    if _relation_exists(con, "contact_issues"):
        for (description,) in con.execute(
            "SELECT description FROM contact_issues WHERE interaction_id = ?",
            [interaction_id],
        ).fetchall():
            if description != stamp:
                leftover += 1
    for (desc,) in con.execute(
        "SELECT description FROM interactions WHERE interaction_id = ?",
        [interaction_id],
    ).fetchall():
        if desc is not None and desc != "":
            leftover += 1
    return leftover


_erasure_barrier: set[str] = set()
_erasure_lock = threading.Lock()


def is_interaction_erased(interaction_id: str, con: Any = None) -> bool:
    """Return True if the interaction has been tombstoned, erased, or shredded."""
    with _erasure_lock:
        if interaction_id in _erasure_barrier:
            return True

    from src.security.pii import SubjectKeyStore

    if not SubjectKeyStore.has_dek(interaction_id, con=con):
        if interaction_id in getattr(SubjectKeyStore, "_shredded", set()):
            return True

    def _check(c):
        try:
            row = c.execute(
                "SELECT erased, description FROM interactions WHERE interaction_id = ?",
                [interaction_id],
            ).fetchone()
            if row is not None:
                erased, desc = row[0], row[1]
                if erased:
                    return True
                if desc and "[ERASED" in str(desc):
                    return True
            turn_row = c.execute(
                "SELECT 1 FROM interaction_turns WHERE interaction_id = ? AND (erased = TRUE OR text LIKE '[ERASED%') LIMIT 1",
                [interaction_id],
            ).fetchone()
            if turn_row:
                return True
        except Exception:
            pass
        return False

    if con is not None:
        if _check(con):
            with _erasure_lock:
                _erasure_barrier.add(interaction_id)
            return True
    else:
        try:
            with ops_con(read_only=True) as read_con:
                if _check(read_con):
                    with _erasure_lock:
                        _erasure_barrier.add(interaction_id)
                    return True
        except Exception:
            pass

    return False


def invalidate_active_orchestrator(interaction_id: str) -> None:
    """Invalidate in-memory state and remove active entry for an erased contact (FU02)."""
    with _erasure_lock:
        _erasure_barrier.add(interaction_id)
    try:
        from src.api.routes.interactions import _active

        entry = _active.pop(interaction_id, None)
        if entry is not None:
            entry.capability_token = None
            entry.ws_attached = False
            if hasattr(entry, "orch") and entry.orch is not None:
                entry.orch._erased = True
                if hasattr(entry.orch, "ctx") and entry.orch.ctx is not None:
                    ctx = entry.orch.ctx
                    ctx.slots.clear()
                    ctx.slots["description"] = ""
                    if hasattr(ctx, "turns") and ctx.turns is not None:
                        ctx.turns.clear()
                    ctx.pending_safety_script = None
                    if hasattr(ctx, "facts") and ctx.facts is not None:
                        ctx.facts.clear()
    except Exception:
        pass


def delete_interaction(interaction_id: str, *, mode: str = "erase") -> dict[str, Any]:
    """Delete linked ops rows for one interaction.

    Modes (audit 7.3 & 5.1 — hash chain vs erasure):
    - ``"erase"`` (default, backward compatible): hard delete. The contact
      is gone entirely; its chain segment cannot be re-verified afterward.
    - ``"tombstone"`` (recommended for regulated data): PII-bearing content
      (turn text, action summaries, evidence-pin bodies) is replaced with a
      dated tombstone and flagged ``erased``, while ``row_hash``/``prev_hash``
      linkage is preserved — ``verify_chain`` still passes and neighbors are
      unaffected. Merkle leaves (hashes of hashes) keep verifying.
    - ``"crypto_shred"``: destroys the per-subject Data Encryption Key (DEK)
      permanently rendering ciphertexts mathematically irrecoverable, and
      applies tombstoning without mutating historical chain hashes.
    """
    if mode not in ("erase", "tombstone", "crypto_shred"):
        raise ValueError("mode must be 'erase', 'tombstone', or 'crypto_shred'")
    if mode == "crypto_shred":
        from src.security.pii import SubjectKeyStore
        shredded = SubjectKeyStore.shred_dek(interaction_id)
        res = tombstone_interaction(interaction_id)
        res["crypto_shredded"] = shredded
        res["mode"] = "crypto_shred"
        return res
    if mode == "tombstone":
        return tombstone_interaction(interaction_id)
    iid = interaction_id
    invalidate_active_orchestrator(iid)
    from src.frontline.archive import purge_audit_archives

    archives_purged = purge_audit_archives(iid)
    deleted: dict[str, int] = {"archives_purged": archives_purged}
    with ops_con() as con:
        # case notes first
        case_ids = [
            r[0]
            for r in con.execute(
                "SELECT case_id FROM cases WHERE interaction_id = ?", [iid]
            ).fetchall()
        ]
        n = 0
        for cid in case_ids:
            con.execute("DELETE FROM case_notes WHERE case_id = ?", [cid])
            n += 1
        deleted["case_notes_batches"] = n
        from src.security.sql_ident import safe_column, safe_table

        tables = [
            ("cases", "interaction_id"),
            ("agent_actions", "interaction_id"),
            ("interaction_turns", "interaction_id"),
            ("interaction_version_stamps", "interaction_id"),
            ("risk_snapshots", "interaction_id"),
            ("interactions", "interaction_id"),
        ]
        if _relation_exists(con, "contact_issues"):
            tables.append(("contact_issues", "interaction_id"))
        for table, col in tables:
            try:
                t = safe_table(table)
                c = safe_column(col)
                before = con.execute(
                    f"SELECT COUNT(*) FROM {t} WHERE {c} = ?", [iid]
                ).fetchone()[0]
                con.execute(f"DELETE FROM {t} WHERE {c} = ?", [iid])
                deleted[table] = int(before)
            except Exception:
                deleted[table] = -1
    return {"interaction_id": iid, "deleted": deleted, "ok": erasure_succeeded(deleted)}


def tombstone_interaction(interaction_id: str) -> dict[str, Any]:
    """Chain-preserving erasure for one interaction (audit 7.3).

    Replaces PII content with tombstones, keeps every hash link intact:
    - interaction_turns.text → tombstone, erased=TRUE
    - agent_actions input/output_summary → tombstone, erased=TRUE
      (row_hash/prev_hash untouched → verify_chain passes, linkage checked)
    - cited_evidence_snapshots body_json → tombstone, erased=TRUE
      (body_hash of the ORIGINAL kept alongside, so drift-audit can tell
      "erased by request" apart from "tampered")
    - interactions description/category/entity slots → cleared (operational
      copies; the case row keeps non-PII analytics fields)
    Structural rows (cases, interactions headers) are kept so corpus counts
    and foreign-key-shaped joins don't silently shift.
    """
    from src.data.timeutil import utc_now
    from src.frontline.archive import purge_audit_archives
    from src.qubot.evidence_pin import _ensure_pin_table

    iid = interaction_id
    invalidate_active_orchestrator(iid)
    archives_purged = purge_audit_archives(iid)
    stamp = f"[ERASED {utc_now().date().isoformat()} per erasure request]"
    out: dict[str, int] = {"archives_purged": archives_purged}
    with ops_con() as con:
        _ensure_pin_table(con)
        try:
            from src.ledger.chain import compute_content_hash

            rows = con.execute(
                "SELECT action_id, interaction_id, case_id, agent, action_type, "
                "input_summary, output_summary, evidence_ids, ok, error, duration_ms, ts, "
                "hash_version, claims FROM agent_actions "
                "WHERE interaction_id = ? AND (content_hash IS NULL OR content_hash = '') AND hash_version >= 2",
                [iid],
            ).fetchall()
            cols = [d[0] for d in con.description]
            for r in rows:
                rd = dict(zip(cols, r))
                ch = compute_content_hash(rd, version=int(rd.get("hash_version") or 2))
                con.execute(
                    "UPDATE agent_actions SET content_hash = ? WHERE action_id = ?",
                    [ch, rd["action_id"]],
                )
        except Exception:
            pass
        try:
            from src.security.sql_ident import safe_column, safe_table

            t = safe_table("interaction_turns")
            c = safe_column("text")
            n = con.execute(
                f"SELECT COUNT(*) FROM {t} WHERE interaction_id = ?", [iid]
            ).fetchone()[0]
            con.execute(
                f"UPDATE {t} SET {c} = ?, erased = TRUE WHERE interaction_id = ?",
                [stamp, iid],
            )
            out["interaction_turns.tombstoned"] = int(n)
        except Exception:
            out["interaction_turns.tombstoned"] = -1
        # v2+ only: v1 row_hash covers current summaries, so tombstoning v1
        # actions would fail verify_chain. New writes are hash_version=2.
        try:
            n = con.execute(
                "SELECT COUNT(*) FROM agent_actions WHERE interaction_id = ? "
                "AND COALESCE(hash_version, 1) >= 2",
                [iid],
            ).fetchone()[0]
            con.execute(
                "UPDATE agent_actions SET input_summary = ?, erased = TRUE "
                "WHERE interaction_id = ? AND COALESCE(hash_version, 1) >= 2",
                [stamp, iid],
            )
            out["agent_actions.tombstoned"] = int(n)
        except Exception:
            out["agent_actions.tombstoned"] = -1
        try:
            con.execute(
                "UPDATE agent_actions SET output_summary = ?, erased = TRUE "
                "WHERE interaction_id = ? AND COALESCE(hash_version, 1) >= 2",
                [stamp, iid],
            )
            out["agent_actions.output_tombstoned"] = "ok"
        except Exception:
            out["agent_actions.output_tombstoned"] = "failed"
        try:
            n = con.execute(
                "SELECT COUNT(*) FROM cited_evidence_snapshots WHERE interaction_id = ?",
                [iid],
            ).fetchone()[0]
            con.execute(
                "UPDATE cited_evidence_snapshots SET body_json = ?, erased = TRUE"
                " WHERE interaction_id = ?",
                ['{"erased": true}', iid],
            )
            out["pins.tombstoned"] = int(n)
        except Exception:
            out["pins.tombstoned"] = -1
        try:
            try:
                con.execute(
                    "UPDATE interactions SET erased = TRUE, description = NULL, category = NULL,"
                    " entity_1 = NULL, entity_2 = NULL, entity_3 = NULL"
                    " WHERE interaction_id = ?",
                    [iid],
                )
            except Exception:
                con.execute(
                    "UPDATE interactions SET description = NULL, category = NULL,"
                    " entity_1 = NULL, entity_2 = NULL, entity_3 = NULL"
                    " WHERE interaction_id = ?",
                    [iid],
                )
            out["interaction.slots_cleared"] = 1
        except Exception:
            out["interaction.slots_cleared"] = -1
        try:
            n = con.execute(
                "SELECT COUNT(*) FROM cases WHERE interaction_id = ?", [iid]
            ).fetchone()[0]
            con.execute(
                "UPDATE cases SET description_summary = ?, followup_draft = ? "
                "WHERE interaction_id = ?",
                [stamp, stamp, iid],
            )
            out["cases.description_tombstoned"] = int(n)
            out["cases.followup_tombstoned"] = int(n)
        except Exception:
            out["cases.description_tombstoned"] = -1
            out["cases.followup_tombstoned"] = -1
        try:
            n = con.execute(
                """
                SELECT COUNT(*) FROM case_notes
                WHERE case_id IN (SELECT case_id FROM cases WHERE interaction_id = ?)
                """,
                [iid],
            ).fetchone()[0]
            con.execute(
                """
                UPDATE case_notes SET body = ?
                WHERE case_id IN (SELECT case_id FROM cases WHERE interaction_id = ?)
                """,
                [stamp, iid],
            )
            out["case_notes.body_tombstoned"] = int(n)
        except Exception:
            out["case_notes.body_tombstoned"] = -1
        if _relation_exists(con, "contact_issues"):
            try:
                n = con.execute(
                    "SELECT COUNT(*) FROM contact_issues WHERE interaction_id = ?",
                    [iid],
                ).fetchone()[0]
                con.execute(
                    "UPDATE contact_issues SET description = ? WHERE interaction_id = ?",
                    [stamp, iid],
                )
                out["contact_issues.description_tombstoned"] = int(n)
            except Exception:
                out["contact_issues.description_tombstoned"] = -1
        else:
            out["contact_issues.description_tombstoned"] = 0
        out["leftovers"] = _plaintext_leftovers(con, iid, stamp)
    ok = erasure_succeeded(out)
    try:
        from src.security.audit_log import security_event

        security_event(
            "dsr.tombstone",
            outcome="success" if ok else "failure",
            resource=iid,
            detail={"tombstoned": out},
        )
    except Exception:
        pass
    return {"interaction_id": iid, "mode": "tombstone", "tombstoned": out, "ok": ok}


__all__ = [
    "export_interaction",
    "delete_interaction",
    "tombstone_interaction",
    "erasure_succeeded",
    "is_interaction_erased",
    "invalidate_active_orchestrator",
]
