"""Migrate legacy ``enc:v1:`` XOR tokens to ``enc:x1:`` label then re-encrypt.

Usage:
    python scripts/migrate_legacy_xor_tokens.py [--dry-run]

For each row in ``interaction_turns`` whose ``text`` starts with ``enc:v1:``,
attempts AES-GCM decryption. If it fails (meaning the row was written by the
pre-AES XOR encoder), the token is relabelled ``enc:x1:`` and then fully
re-encrypted as authenticated ``enc:v1:`` via ``migrate_legacy_xor_token``.

After this migration, no rows should reach the ``_decode_legacy_xor`` path
during normal runtime.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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


def _is_legacy_xor(subject_id: str, token: str) -> bool:
    """Return True if *token* looks like ``enc:v1:`` but fails AES-GCM."""
    if not token or not token.startswith(TOKEN_PREFIX):
        return False
    # If AES-GCM decryption fails, it was encoded with the old XOR format.
    import base64

    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    try:
        dek = SubjectKeyStore.get_or_create_dek(subject_id)
        raw = base64.urlsafe_b64decode(token[len(TOKEN_PREFIX) :].encode("ascii"))
        nonce, cipher_bytes = raw[:12], raw[12:]
        AESGCM(dek).decrypt(nonce, cipher_bytes, None)
        return False  # Valid AES-GCM — not legacy
    except Exception:
        return True  # AES-GCM failed — it's a legacy XOR token


def migrate(*, dry_run: bool = False) -> dict[str, int]:
    """Scan turns, relabel + re-encrypt legacy XOR tokens.

    Returns a dict with counts: scanned, legacy, migrated, errors.
    """
    stats = {"scanned": 0, "legacy": 0, "migrated": 0, "errors": 0}

    with ops_con() as con:
        rows = con.execute(
            "SELECT turn_id, interaction_id, text FROM interaction_turns "
            "WHERE text LIKE 'enc:v1:%'"
        ).fetchall()

    stats["scanned"] = len(rows)
    print(f"→ Scanning {len(rows)} enc:v1: rows …")

    for row in rows:
        turn_id = row[0] if isinstance(row, (list, tuple)) else row["turn_id"]
        iid = row[1] if isinstance(row, (list, tuple)) else row["interaction_id"]
        token = row[2] if isinstance(row, (list, tuple)) else row["text"]

        if not _is_legacy_xor(iid, token):
            continue

        stats["legacy"] += 1
        if dry_run:
            print(f"  [DRY-RUN] turn_id={turn_id}  interaction_id={iid}  → would migrate")
            continue

        try:
            # Step 1: relabel enc:v1: → enc:x1:
            relabelled = relabel_legacy_xor_token(token)
            # Step 2: decrypt via XOR and re-encrypt via AES-GCM → new enc:v1: token
            new_token = migrate_legacy_xor_token(iid, relabelled)

            with ops_con() as con:
                con.execute(
                    "UPDATE interaction_turns SET text = ? WHERE turn_id = ?",
                    [new_token, turn_id],
                )
            stats["migrated"] += 1
            print(f"  ✓ migrated turn_id={turn_id}")
        except (PiiIntegrityError, KeyError, Exception) as exc:
            stats["errors"] += 1
            print(f"  ✗ turn_id={turn_id}: {type(exc).__name__}: {exc}")

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print which rows would be migrated without modifying the database.",
    )
    args = parser.parse_args()
    stats = migrate(dry_run=args.dry_run)
    print(
        f"\nDone — scanned={stats['scanned']}, legacy={stats['legacy']}, "
        f"migrated={stats['migrated']}, errors={stats['errors']}"
    )
    return 1 if stats["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
