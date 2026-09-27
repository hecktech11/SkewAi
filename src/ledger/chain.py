"""Tamper-evident hash chain for agent_actions.

Each row stores:
  prev_hash — row_hash of the previous action for this interaction (or GENESIS)
  row_hash  — sha256 of canonical payload + prev_hash
  hash_version — 1 (legacy) or 2 (includes claims and content_hash verification)
  content_hash — sha256 of canonical action content for hash_version=2

Verification walks the chain and re-computes hashes. Any insert/delete/edit
breaks the chain.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

GENESIS = "0" * 64


def _canon(action: dict[str, Any]) -> str:
    """Stable string for hashing (v1 legacy format)."""
    payload = {
        "action_id": action.get("action_id") or "",
        "interaction_id": action.get("interaction_id") or "",
        "case_id": action.get("case_id") or "",
        "agent": action.get("agent") or "",
        "action_type": action.get("action_type") or "",
        "input_summary": action.get("input_summary") or "",
        "output_summary": action.get("output_summary") or "",
        "evidence_ids": action.get("evidence_ids")
        if isinstance(action.get("evidence_ids"), str)
        else json.dumps(action.get("evidence_ids") or []),
        "ok": bool(action.get("ok", True)),
        "error": action.get("error") or "",
        "duration_ms": action.get("duration_ms"),
        "ts": str(action.get("ts") or ""),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _canon_v2(action: dict[str, Any]) -> str:
    """Stable string for hashing (v2 format: includes claims)."""
    claims = action.get("claims")
    if isinstance(claims, str):
        try:
            claims = json.loads(claims)
        except Exception:
            pass
    claims_json = json.dumps(claims if claims is not None else [], sort_keys=True, separators=(",", ":"))
    payload = {
        "action_id": action.get("action_id") or "",
        "interaction_id": action.get("interaction_id") or "",
        "case_id": action.get("case_id") or "",
        "agent": action.get("agent") or "",
        "action_type": action.get("action_type") or "",
        "input_summary": action.get("input_summary") or "",
        "output_summary": action.get("output_summary") or "",
        "evidence_ids": action.get("evidence_ids")
        if isinstance(action.get("evidence_ids"), str)
        else json.dumps(action.get("evidence_ids") or []),
        "claims": claims_json,
        "ok": bool(action.get("ok", True)),
        "error": action.get("error") or "",
        "duration_ms": action.get("duration_ms"),
        "ts": str(action.get("ts") or ""),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def compute_metadata_commitment(action: dict[str, Any]) -> str:
    """Commitment to immutable metadata fields."""
    claims = action.get("claims")
    if isinstance(claims, str):
        try:
            claims = json.loads(claims)
        except Exception:
            pass
    claims_json = json.dumps(claims if claims is not None else [], sort_keys=True, separators=(",", ":"))
    payload = {
        "action_id": action.get("action_id") or "",
        "interaction_id": action.get("interaction_id") or "",
        "case_id": action.get("case_id") or "",
        "agent": action.get("agent") or "",
        "action_type": action.get("action_type") or "",
        "evidence_ids": action.get("evidence_ids")
        if isinstance(action.get("evidence_ids"), str)
        else json.dumps(action.get("evidence_ids") or []),
        "claims": claims_json,
        "ok": bool(action.get("ok", True)),
        "error": action.get("error") or "",
        "duration_ms": action.get("duration_ms"),
        "ts": str(action.get("ts") or ""),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def compute_summary_commitment(action: dict[str, Any]) -> str:
    """Commitment to erasable customer content summaries."""
    payload = {
        "input_summary": action.get("input_summary") or "",
        "output_summary": action.get("output_summary") or "",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _is_tombstone(val: Any) -> bool:
    if not isinstance(val, str):
        return False
    s = val.strip()
    return s.startswith("[ERASED") and s.endswith("]")


def compute_content_hash(action: dict[str, Any], version: int = 2) -> str:
    """Compute content hash for an action before erasure or for v2 verification."""
    if version == 1:
        raw = _canon(action)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
    meta_h = compute_metadata_commitment(action)
    summary_h = compute_summary_commitment(action)
    return f"{meta_h}:{summary_h}"


def compute_row_hash_from_content(content_hash: str, prev_hash: str) -> str:
    """Compute row hash by linking prev_hash and content_hash."""
    raw = f"{prev_hash}|{content_hash}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def compute_row_hash(
    action: dict[str, Any],
    prev_hash: str,
    *,
    version: int | None = None,
) -> str:
    """Compute row hash for an action row.

    Supports version=1 (legacy) and version=2 (content_hash + claims).
    """
    v = version if version is not None else action.get("hash_version", 2)
    if v == 1:
        raw = f"{prev_hash}|{_canon(action)}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    if action.get("erased") and action.get("content_hash"):
        content_h = action["content_hash"]
    else:
        content_h = compute_content_hash(action, version=2)
    return compute_row_hash_from_content(content_h, prev_hash)


def verify_chain(
    rows: list[dict[str, Any]],
    *,
    allow_partial: bool = False,
) -> dict[str, Any]:
    """Verify an ordered list of agent_actions rows for one interaction.

    Returns {ok, checked, first_bad_action_id, detail, error}.
    """
    if not rows:
        return {"ok": True, "checked": 0, "first_bad_action_id": None, "detail": "ok"}

    # 1. Prefix truncation rejection: must start at genesis
    first = rows[0]
    if first.get("prev_hash") != GENESIS:
        return {
            "ok": False,
            "error": "chain_starts_midway",
            "checked": 0,
            "first_bad_action_id": first.get("action_id"),
            "detail": "chain_starts_midway",
        }

    # 2. Completeness check: must end at latest row in DB unless allow_partial=True
    if not allow_partial:
        iid = first.get("interaction_id")
        if iid:
            try:
                from src.data.warehouse import ops_con

                with ops_con(read_only=True) as con:
                    latest = con.execute(
                        "SELECT action_id FROM agent_actions WHERE interaction_id = ? ORDER BY ts DESC, action_id DESC LIMIT 1",
                        [iid],
                    ).fetchone()
                    if latest and latest[0] != rows[-1].get("action_id"):
                        return {
                            "ok": False,
                            "error": "chain_incomplete",
                            "checked": len(rows),
                            "first_bad_action_id": rows[-1].get("action_id"),
                            "detail": "chain_incomplete",
                        }
            except Exception as e:
                return {
                    "ok": False,
                    "error": "head_lookup_failed",
                    "checked": 0,
                    "first_bad_action_id": rows[-1].get("action_id") if rows else None,
                    "detail": f"failed to check stored head for complete chain: {e}",
                }

    prev = GENESIS
    for i, row in enumerate(rows):
        expected_prev = row.get("prev_hash") or GENESIS
        if expected_prev != prev and i > 0:
            return {
                "ok": False,
                "error": "prev_hash_mismatch",
                "checked": i,
                "first_bad_action_id": row.get("action_id"),
                "detail": f"prev_hash mismatch at index {i}",
            }

        v = row.get("hash_version", 1)
        actual = row.get("row_hash") or ""

        if v == 1:
            expected = compute_row_hash(row, expected_prev, version=1)
            if actual != expected:
                return {
                    "ok": False,
                    "error": "row_hash_mismatch",
                    "checked": i + 1,
                    "first_bad_action_id": row.get("action_id"),
                    "detail": f"row_hash mismatch at index {i}",
                }
            prev = actual or expected
        else:
            if row.get("erased"):
                content_h = row.get("content_hash")
                if not content_h:
                    return {
                        "ok": False,
                        "error": "erased_row_missing_content_hash",
                        "checked": i + 1,
                        "first_bad_action_id": row.get("action_id"),
                        "detail": "erased_row_missing_content_hash",
                    }
                # Check tombstone shape: erased content must follow exact tombstone format
                in_sum = row.get("input_summary") or ""
                out_sum = row.get("output_summary") or ""
                if not _is_tombstone(out_sum) or (in_sum and not _is_tombstone(in_sum)):
                    return {
                        "ok": False,
                        "error": "invalid_tombstone_shape",
                        "checked": i + 1,
                        "first_bad_action_id": row.get("action_id"),
                        "detail": f"erased row {row.get('action_id')} has non-tombstone summaries",
                    }
                # Verify immutable metadata commitment if separated content_hash is present
                if ":" in str(content_h):
                    meta_expected, _ = str(content_h).split(":", 1)
                    meta_actual = compute_metadata_commitment(row)
                    if meta_actual != meta_expected:
                        return {
                            "ok": False,
                            "error": "metadata_commitment_mismatch",
                            "checked": i + 1,
                            "first_bad_action_id": row.get("action_id"),
                            "detail": f"metadata commitment mismatch for erased row at index {i}",
                        }
                expected = compute_row_hash_from_content(content_h, expected_prev)
                if actual != expected:
                    return {
                        "ok": False,
                        "error": "content_hash_mismatch",
                        "checked": i + 1,
                        "first_bad_action_id": row.get("action_id"),
                        "detail": f"row_hash mismatch at index {i}",
                    }
                prev = actual or expected
            else:
                content_h = compute_content_hash(row, version=2)
                # If row already had a legacy single-hash content_hash, check if legacy matches
                if row.get("content_hash") and ":" not in str(row.get("content_hash")):
                    legacy_content_h = hashlib.sha256(_canon_v2(row).encode("utf-8")).hexdigest()
                    if row.get("content_hash") == legacy_content_h:
                        content_h = legacy_content_h
                expected = compute_row_hash_from_content(content_h, expected_prev)
                if actual != expected:
                    return {
                        "ok": False,
                        "error": "row_hash_mismatch",
                        "checked": i + 1,
                        "first_bad_action_id": row.get("action_id"),
                        "detail": f"row_hash mismatch at index {i}",
                    }
                prev = actual or expected

    return {"ok": True, "checked": len(rows), "first_bad_action_id": None, "detail": "ok"}


__all__ = [
    "GENESIS",
    "compute_content_hash",
    "compute_metadata_commitment",
    "compute_summary_commitment",
    "compute_row_hash_from_content",
    "compute_row_hash",
    "verify_chain",
]
