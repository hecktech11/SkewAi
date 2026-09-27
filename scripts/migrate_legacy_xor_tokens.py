"""Migrate legacy ``enc:x1:`` XOR tokens or proven legacy ``enc:v1:`` rows to authenticated AES-GCM.

Usage:
    python scripts/migrate_legacy_xor_tokens.py [--dry-run] [--provenance-file PATH] [--proven-legacy-turn-ids ID1,ID2...]

Security invariant (FU01 / R30):
Never treat an arbitrary AES-GCM decryption failure as a legacy XOR token.
A failing AES-GCM token is corrupt or tampered; rewriting it with the XOR decoder
produces authenticated garbage and destroys forensic evidence.

Only rows already labelled ``enc:x1:`` or rows explicitly documented in a
provenance manifest as pre-dating the AES-GCM migration are eligible for migration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.warehouse import ops_con
from src.security.pii import (
    LEGACY_XOR_PREFIX,
    TOKEN_PREFIX,
    PiiIntegrityError,
    SubjectKeyStore,
    migrate_legacy_xor_token,
    relabel_legacy_xor_token,
)


def _check_aes_gcm(subject_id: str, token: str) -> bool:
    """Return True if *token* successfully authenticates and decrypts with AES-GCM."""
    if not token or not token.startswith(TOKEN_PREFIX):
        return False
    if not SubjectKeyStore.has_dek(subject_id):
        return False
    import base64

    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    try:
        dek = SubjectKeyStore.get_or_create_dek(subject_id)
        raw = base64.urlsafe_b64decode(token[len(TOKEN_PREFIX) :].encode("ascii"))
        nonce, cipher_bytes = raw[:12], raw[12:]
        AESGCM(dek).decrypt(nonce, cipher_bytes, None)
        return True
    except Exception:
        return False


def load_proven_ids(
    provenance_file: str | Path | None = None,
    proven_legacy_turn_ids: Sequence[str] | set[str] | None = None,
) -> set[str]:
    """Load verified turn IDs known to have been written before AES-GCM introduction."""
    proven: set[str] = set()
    if proven_legacy_turn_ids:
        proven.update(t.strip() for t in proven_legacy_turn_ids if t.strip())
    if provenance_file:
        p = Path(provenance_file)
        if p.exists():
            content = p.read_text(encoding="utf-8").strip()
            if content.startswith("[") or content.startswith("{"):
                try:
                    data = json.loads(content)
                    if isinstance(data, list):
                        proven.update(str(x).strip() for x in data if str(x).strip())
                    elif isinstance(data, dict):
                        ids = data.get("turn_ids") or data.get("proven_ids") or []
                        proven.update(str(x).strip() for x in ids if str(x).strip())
                except Exception:
                    pass
            else:
                for line in content.splitlines():
                    line = line.strip()
                    if line and not line.startswith("#"):
                        proven.add(line)
    return proven


def migrate(
    *,
    dry_run: bool = False,
    provenance_file: str | Path | None = None,
    proven_legacy_turn_ids: Sequence[str] | set[str] | None = None,
) -> dict[str, int]:
    """Scan turns, relabel + re-encrypt legacy XOR tokens with explicit provenance.

    Returns a dict with counts: scanned, legacy, migrated, corrupt, errors.
    """
    stats = {
        "scanned": 0,
        "legacy": 0,
        "migrated": 0,
        "corrupt": 0,
        "errors": 0,
    }

    proven_ids = load_proven_ids(provenance_file, proven_legacy_turn_ids)

    with ops_con(read_only=True) as con:
        rows = con.execute(
            "SELECT turn_id, interaction_id, text FROM interaction_turns "
            "WHERE text LIKE 'enc:v1:%' OR text LIKE 'enc:x1:%'"
        ).fetchall()

    stats["scanned"] = len(rows)
    print(f"→ Scanning {len(rows)} encrypted turn rows …")

    pending_updates: list[tuple[str, str]] = []

    for row in rows:
        turn_id = row[0] if isinstance(row, (list, tuple)) else row["turn_id"]
        iid = row[1] if isinstance(row, (list, tuple)) else row["interaction_id"]
        token = row[2] if isinstance(row, (list, tuple)) else row["text"]

        is_x1 = str(token).startswith(LEGACY_XOR_PREFIX)
        is_proven_v1 = (not is_x1) and (str(token).startswith(TOKEN_PREFIX)) and (turn_id in proven_ids)

        if not is_x1 and not is_proven_v1:
            # An enc:v1: row not in the proven legacy set.
            # Verify if it authenticates with AES-GCM.
            if not _check_aes_gcm(iid, token):
                # Never assume AES failure means legacy XOR (FU01). Record as corrupt/integrity issue.
                stats["corrupt"] += 1
                print(f"  ⚠ turn_id={turn_id} interaction_id={iid}: AES-GCM authentication failed; preserving as corrupted ciphertext (not legacy XOR)")
            continue

        # Row is either enc:x1: or explicitly proven legacy
        stats["legacy"] += 1

        if not SubjectKeyStore.has_dek(iid):
            stats["errors"] += 1
            print(f"  ✗ turn_id={turn_id}: Subject DEK for {iid!r} is missing or shredded; cannot migrate")
            continue

        if dry_run:
            print(f"  [DRY-RUN] turn_id={turn_id} interaction_id={iid} → would migrate")
            continue

        try:
            # Step 1: ensure token carries enc:x1:
            relabelled = relabel_legacy_xor_token(token) if is_proven_v1 else token
            # Step 2: decrypt via XOR and re-encrypt via AES-GCM → authenticated enc:v1:
            new_token = migrate_legacy_xor_token(iid, relabelled)
            pending_updates.append((new_token, turn_id))
        except (PiiIntegrityError, KeyError, Exception) as exc:
            stats["errors"] += 1
            print(f"  ✗ turn_id={turn_id}: {type(exc).__name__}: {exc}")

    if not dry_run and pending_updates:
        try:
            with ops_con() as con:
                con.execute("BEGIN TRANSACTION")
                try:
                    for new_tok, tid in pending_updates:
                        con.execute(
                            "UPDATE interaction_turns SET text = ? WHERE turn_id = ?",
                            [new_tok, tid],
                        )
                    con.execute("COMMIT")
                    stats["migrated"] += len(pending_updates)
                    for _, tid in pending_updates:
                        print(f"  ✓ migrated turn_id={tid}")
                except Exception as exc:
                    con.execute("ROLLBACK")
                    stats["errors"] += len(pending_updates)
                    print(f"  ✗ Transaction failed and rolled back: {exc}")
                    raise
        except Exception:
            pass

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print which rows would be migrated without modifying the database.",
    )
    parser.add_argument(
        "--provenance-file",
        type=str,
        default=None,
        help="Path to file listing turn IDs proven to be pre-AES XOR tokens.",
    )
    parser.add_argument(
        "--proven-legacy-turn-ids",
        type=str,
        default=None,
        help="Comma-separated turn IDs proven to be pre-AES XOR tokens.",
    )
    args = parser.parse_args()
    proven_ids = (
        [t.strip() for t in args.proven_legacy_turn_ids.split(",") if t.strip()]
        if args.proven_legacy_turn_ids
        else None
    )
    stats = migrate(
        dry_run=args.dry_run,
        provenance_file=args.provenance_file,
        proven_legacy_turn_ids=proven_ids,
    )
    print(
        f"\nDone — scanned={stats['scanned']}, legacy={stats['legacy']}, "
        f"migrated={stats['migrated']}, corrupt={stats['corrupt']}, errors={stats['errors']}"
    )
    return 1 if stats["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
