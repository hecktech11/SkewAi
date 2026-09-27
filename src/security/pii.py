"""Basic PII detection/redaction for pilot transcripts (regex, not NER)."""

from __future__ import annotations

import re
from typing import Any

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE = re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CC = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
# Conservative street-address pattern: house number + capitalized words +
# street suffix. Word-boundary anchored to avoid flagging prose.
_ADDRESS = re.compile(
    r"\b\d{1,5}\s+(?:[A-Z][a-z]+\s+){1,3}"
    r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|"
    r"Court|Ct|Circle|Cir|Parkway|Pkwy|Terrace|Plaza|Way)\b"
)


def redact_pii(text: str) -> str:
    """Replace common PII patterns with placeholders."""
    if not text:
        return text
    out = _EMAIL.sub("[EMAIL]", text)
    out = _PHONE.sub("[PHONE]", out)
    out = _SSN.sub("[SSN]", out)
    out = _CC.sub("[CARD]", out)
    out = _ADDRESS.sub("[ADDRESS]", out)
    return out


def find_pii(text: str) -> list[str]:
    kinds: list[str] = []
    if _EMAIL.search(text or ""):
        kinds.append("email")
    if _PHONE.search(text or ""):
        kinds.append("phone")
    if _SSN.search(text or ""):
        kinds.append("ssn")
    if _ADDRESS.search(text or ""):
        kinds.append("address")
    return kinds


#: Free-text fields that may carry customer PII across read/export surfaces.
PII_TEXT_FIELDS = frozenset({
    "description",
    "description_summary",
    "followup_draft",
    "text",
    "note",
    "notes",
    "summary",
    "input_summary",
    "output_summary",
    "body",
})

#: Structured slot keys that carry sensitive customer identifiers / PII.
SENSITIVE_STRUCTURED_KEYS = frozenset({
    "vin",
    "ssn",
    "account_number",
    "card_number",
    "license_number",
    "ssn_last4",
    "pan",
    "cvv",
    "driver_license",
    "phone",
    "email",
})


def redact_dict(row: dict[str, Any], fields: frozenset[str] = PII_TEXT_FIELDS) -> dict[str, Any]:
    """Return a copy of *row* with PII redacted in known free-text fields.

    Walks nested dicts and lists so a header, a case, a note, or an explain
    payload cannot hide a free-text field one level down. Only the named
    fields are rewritten; identifiers and status columns pass through.
    """
    out: dict[str, Any] = {}
    for key, val in row.items():
        if isinstance(val, dict):
            out[key] = redact_dict(val, fields)
        elif isinstance(val, list):
            out[key] = [
                redact_dict(item, fields) if isinstance(item, dict) else item
                for item in val
            ]
        elif isinstance(val, str) and val:
            k_lower = key.lower()
            if k_lower in SENSITIVE_STRUCTURED_KEYS:
                tag = k_lower.upper()
                out[key] = f"[{tag}]"
            elif key in fields:
                out[key] = redact_pii(val)
            else:
                out[key] = val
        else:
            out[key] = val
    return out


def redact_turns(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Redact PII in conversation turn text (returns new list)."""
    redacted: list[dict[str, Any]] = []
    for turn in turns:
        t = dict(turn)
        if isinstance(t.get("text"), str):
            t["text"] = redact_pii(t["text"])
        redacted.append(t)
    return redacted


import base64
import hashlib
import logging
import secrets
import threading

_log = logging.getLogger("skewai.pii")

#: Prefix of an authenticated (AES-GCM) at-rest token.
TOKEN_PREFIX = "enc:v1:"

#: Prefix of the pre-AES-GCM blake2b/XOR format. Unauthenticated, so it is
#: reachable only when a row says so — see ``relabel_legacy_xor_token``.
LEGACY_XOR_PREFIX = "enc:x1:"


class PiiIntegrityError(ValueError):
    """Stored ciphertext did not authenticate.

    Distinct from a shredded key (KeyError): this is corruption or tampering,
    and it must surface as an error rather than being silently re-read with a
    decoder that has no integrity check.
    """


class PiiEncryptionError(RuntimeError):
    """PII could not be encrypted.

    Raised instead of handing the plaintext back: a caller that receives its own
    input has no way to tell success from failure, and every caller here writes
    the result straight into a warehouse column.
    """


#: Written in place of text that could not be encrypted. Never the plaintext —
#: an unencrypted description is indistinguishable on read from a decrypted one.
UNENCRYPTED_PLACEHOLDER = "[ENCRYPTION_UNAVAILABLE]"

_health_lock = threading.Lock()
_encrypt_failures = 0
_encrypt_last_error = ""


def _record_encryption_failure(kind: str) -> None:
    global _encrypt_failures, _encrypt_last_error
    with _health_lock:
        _encrypt_failures += 1
        _encrypt_last_error = kind


def encryption_health() -> dict[str, Any]:
    """Whether at-rest PII encryption is working.

    Any non-zero ``failures`` means customer text was dropped rather than
    stored, so operators need this on the health surface — silently writing
    plaintext instead used to make the same outage invisible.
    """
    with _health_lock:
        failures = _encrypt_failures
        last = _encrypt_last_error
    return {
        "healthy": failures == 0,
        "failures": failures,
        "last_error": last,
        "placeholder": UNENCRYPTED_PLACEHOLDER,
    }


def reset_encryption_health() -> None:
    """Clear the failure counters (tests / after an operator acknowledges)."""
    global _encrypt_failures, _encrypt_last_error
    with _health_lock:
        _encrypt_failures = 0
        _encrypt_last_error = ""


def _ensure_dek_table(con) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS subject_deks (
            subject_id VARCHAR PRIMARY KEY,
            dek_hex VARCHAR NOT NULL,
            created_at TIMESTAMP NOT NULL,
            shredded BOOLEAN DEFAULT FALSE,
            shredded_at TIMESTAMP
        )
        """
    )


class SubjectKeyStore:
    """Vault for per-subject data encryption keys enabling crypto-shredding (audit 5.1).

    Destroying the subject's DEK renders stored encrypted ciphertext permanently
    unrecoverable without breaking historical hash chains or Merkle tree leaves.
    Keys are persisted across restarts and replicas in DuckDB ops warehouse.
    """
    _keys: dict[str, bytes] = {}
    _shredded: set[str] = set()
    _lock = threading.RLock()

    @classmethod
    def get_or_create_dek(cls, subject_id: str) -> bytes:
        with cls._lock:
            if subject_id in cls._shredded:
                raise KeyError(f"Subject DEK for '{subject_id}' has been shredded (GDPR/CCPA Art. 17)")
            if subject_id in cls._keys:
                return cls._keys[subject_id]
            from src.data.timeutil import utc_now
            from src.data.warehouse import ops_con

            with ops_con() as con:
                _ensure_dek_table(con)
                row = con.execute(
                    "SELECT dek_hex, shredded FROM subject_deks WHERE subject_id = ?",
                    [subject_id],
                ).fetchone()
                if row:
                    dek_hex, shredded = row
                    if shredded:
                        cls._shredded.add(subject_id)
                        cls._keys.pop(subject_id, None)
                        raise KeyError(f"Subject DEK for '{subject_id}' has been shredded (GDPR/CCPA Art. 17)")
                    dek = bytes.fromhex(dek_hex)
                    cls._keys[subject_id] = dek
                    return dek
                new_key = secrets.token_bytes(32)
                con.execute(
                    """
                    INSERT INTO subject_deks (subject_id, dek_hex, created_at, shredded)
                    VALUES (?, ?, ?, FALSE)
                    """,
                    [subject_id, new_key.hex(), utc_now().replace(tzinfo=None)],
                )
                cls._keys[subject_id] = new_key
                return new_key

    @classmethod
    def shred_dek(cls, subject_id: str) -> bool:
        """Permanently erase the subject's encryption key (GDPR Art. 17)."""
        with cls._lock:
            cls._shredded.add(subject_id)
            cls._keys.pop(subject_id, None)
            from src.data.timeutil import utc_now
            from src.data.warehouse import ops_con

            with ops_con() as con:
                _ensure_dek_table(con)
                now = utc_now().replace(tzinfo=None)
                row = con.execute(
                    "SELECT shredded FROM subject_deks WHERE subject_id = ?",
                    [subject_id],
                ).fetchone()
                if row:
                    if not row[0]:
                        con.execute(
                            "UPDATE subject_deks SET dek_hex = '', shredded = TRUE, shredded_at = ? WHERE subject_id = ?",
                            [now, subject_id],
                        )
                        return True
                    return False
                # If subject_id exists in interactions table, record shredding tombstones
                try:
                    int_row = con.execute(
                        "SELECT 1 FROM interactions WHERE interaction_id = ?",
                        [subject_id],
                    ).fetchone()
                    if int_row:
                        con.execute(
                            """
                            INSERT INTO subject_deks (subject_id, dek_hex, created_at, shredded, shredded_at)
                            VALUES (?, '', ?, TRUE, ?)
                            """,
                            [subject_id, now, now],
                        )
                        return True
                except Exception:
                    pass
                return False

    @classmethod
    def has_dek(cls, subject_id: str) -> bool:
        with cls._lock:
            if subject_id in cls._shredded:
                return False
            if subject_id in cls._keys:
                return True
            try:
                from src.data.warehouse import ops_con

                with ops_con(read_only=True) as con:
                    _ensure_dek_table(con)
                    row = con.execute(
                        "SELECT shredded FROM subject_deks WHERE subject_id = ?",
                        [subject_id],
                    ).fetchone()
                    if row:
                        if row[0]:
                            cls._shredded.add(subject_id)
                            cls._keys.pop(subject_id, None)
                            return False
                        return True
            except Exception:
                pass
            return False

    @classmethod
    def clear(cls) -> None:
        with cls._lock:
            cls._keys.clear()
            cls._shredded.clear()
            try:
                from src.data.warehouse import ops_con

                with ops_con() as con:
                    _ensure_dek_table(con)
                    con.execute("DELETE FROM subject_deks")
            except Exception:
                pass


def encrypt_subject_pii(subject_id: str, plaintext: str) -> str:
    """Encrypt PII under the subject's dedicated DEK."""
    if not plaintext:
        return ""
    dek = SubjectKeyStore.get_or_create_dek(subject_id)
    nonce = secrets.token_bytes(12)
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    aesgcm = AESGCM(dek)
    ciphertext = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)
    payload = nonce + ciphertext
    return TOKEN_PREFIX + base64.urlsafe_b64encode(payload).decode("ascii")


def decrypt_subject_pii(subject_id: str, token: str, *, allow_legacy: bool = False) -> str:
    """Decrypt PII.

    Raises KeyError if the key has been shredded, PiiIntegrityError if the
    ciphertext does not authenticate. An AES-GCM authentication failure is never
    retried with the legacy decoder (R30).

    Live reads accept authenticated ``enc:v1:`` ciphertext only (N07).
    Unauthenticated legacy ``enc:x1:`` tokens are rejected during normal operation
    and are only permitted during explicit offline migration (``allow_legacy=True``).
    """
    if not token:
        return token
    legacy = token.startswith(LEGACY_XOR_PREFIX)
    if legacy:
        if not allow_legacy:
            raise PiiIntegrityError(
                f"unauthenticated legacy XOR token '{LEGACY_XOR_PREFIX}...' is not permitted in live reads; "
                "run offline migration scripts/migrate_legacy_xor_tokens.py"
            )
    elif not token.startswith(TOKEN_PREFIX):
        return token
    if not SubjectKeyStore.has_dek(subject_id):
        raise KeyError(f"Subject DEK for '{subject_id}' has been shredded (GDPR/CCPA Art. 17)")
    dek = SubjectKeyStore.get_or_create_dek(subject_id)
    prefix = LEGACY_XOR_PREFIX if legacy else TOKEN_PREFIX
    try:
        raw = base64.urlsafe_b64decode(token[len(prefix):].encode("ascii"))
    except Exception as exc:
        raise PiiIntegrityError(
            f"malformed at-rest PII token for subject {subject_id!r}"
        ) from exc
    nonce, cipher_bytes = raw[:12], raw[12:]
    if legacy:
        return _decode_legacy_xor(dek, nonce, cipher_bytes, subject_id)
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        aesgcm = AESGCM(dek)
        plain = aesgcm.decrypt(nonce, cipher_bytes, None)
    except Exception as exc:
        _log.error(
            "pii_token_authentication_failed subject=%s bytes=%d: %s",
            subject_id,
            len(cipher_bytes),
            type(exc).__name__,
        )
        raise PiiIntegrityError(
            f"at-rest PII for subject {subject_id!r} failed authentication "
            f"({type(exc).__name__}); stored ciphertext is corrupt or was modified"
        ) from exc
    return plain.decode("utf-8", errors="replace")


def _decode_legacy_xor(dek: bytes, nonce: bytes, cipher_bytes: bytes, subject_id: str) -> str:
    """Decode an ``enc:x1:`` row written before AES-GCM.

    Unauthenticated by construction — there is no tag to check, which is exactly
    why it is reachable only through its own prefix. Migrate these rows with
    ``migrate_legacy_xor_token`` and this path goes away.

    Guard: the legacy XOR format stored only short identifiers (≤ 64 bytes) with
    no authentication tag. AES-GCM ciphertext always contains a 16-byte GCM tag
    appended to the plaintext bytes. If the cipher payload after the nonce is
    longer than the legacy maximum, it is almost certainly an AES-GCM payload
    whose prefix was mutated from ``enc:v1:`` to ``enc:x1:`` — reject it.
    """
    # Legacy XOR never produced ciphertext longer than the plaintext itself
    # (no authentication tag), and plaintext was limited to short identifiers.
    if len(cipher_bytes) > 64:
        raise PiiIntegrityError(
            f"legacy PII token for subject {subject_id!r} exceeds the "
            "64-byte keystream the legacy format could produce"
        )
    # AES-GCM tag is 16 bytes — any ciphertext that is long enough to
    # contain meaningful plaintext + a 16-byte tag is suspicious.  The
    # absolute minimum AES-GCM output for a 1-byte plaintext is 17 bytes.
    # Legacy XOR tokens for real-world identifiers (phone, email) are
    # typically ≤ 40 bytes.  We accept up to 48 bytes to give headroom
    # but reject anything above that as a potential prefix-swap attack.
    _MAX_LEGACY_PAYLOAD = 48
    if len(cipher_bytes) > _MAX_LEGACY_PAYLOAD:
        raise PiiIntegrityError(
            f"legacy PII token for subject {subject_id!r} has {len(cipher_bytes)}-byte "
            f"payload (max {_MAX_LEGACY_PAYLOAD}); possible enc:v1: → enc:x1: prefix swap"
        )
    _log.warning("pii_legacy_xor_token_read subject=%s (unauthenticated format)", subject_id)
    stream_key = hashlib.blake2b(
        dek, key=nonce, digest_size=max(1, len(cipher_bytes))
    ).digest()
    return bytes(b ^ k for b, k in zip(cipher_bytes, stream_key)).decode(
        "utf-8", errors="replace"
    )


def relabel_legacy_xor_token(token: str) -> str:
    """Tag a pre-AES-GCM token with the format version it actually uses.

    Rows written before AES-GCM carry the same ``enc:v1:`` prefix as
    authenticated ciphertext, which is why decryption used to fall back to the
    unauthenticated decoder whenever AES failed. Operators who know a row
    predates the change relabel it; only ``enc:x1:`` reaches the legacy path.
    """
    if token.startswith(LEGACY_XOR_PREFIX):
        return token
    if not token.startswith(TOKEN_PREFIX):
        raise ValueError("not an at-rest PII token")
    return LEGACY_XOR_PREFIX + token[len(TOKEN_PREFIX):]


def migrate_legacy_xor_token(subject_id: str, token: str) -> str:
    """Re-encrypt an ``enc:x1:`` row as authenticated ``enc:v1:`` ciphertext."""
    if not token.startswith(LEGACY_XOR_PREFIX):
        raise ValueError(f"expected a {LEGACY_XOR_PREFIX} token")
    return encrypt_subject_pii(
        subject_id, decrypt_subject_pii(subject_id, token, allow_legacy=True)
    )


ERASED_TEXT = "[ERASED]"
#: Read-side marker for ciphertext that exists but does not authenticate. Kept
#: distinct from ERASED_TEXT so a corruption/tampering incident is never filed
#: as a completed Art. 17 erasure.
CORRUPT_TEXT = "[UNREADABLE]"


def encrypt_subject_text(subject_id: str, plaintext: str) -> str:
    """Encrypt free text under the subject's DEK, or raise.

    Fails closed. This used to return *plaintext* on any exception "so persist
    cannot stall", which meant a locked ``subject_deks`` table, a shredded DEK,
    or a missing ``cryptography`` wheel wrote the customer's own words into
    ``cases.description_summary`` in the clear. Nothing downstream can tell that
    row from a decrypted one, so the leak never surfaces (R29).
    """
    if not plaintext:
        return plaintext
    try:
        token = encrypt_subject_pii(subject_id, plaintext)
    except Exception as exc:
        _record_encryption_failure(type(exc).__name__)
        raise PiiEncryptionError(
            f"could not encrypt PII for subject {subject_id!r}: {type(exc).__name__}"
        ) from exc
    if not isinstance(token, str) or not token.startswith(TOKEN_PREFIX):
        # A non-token result is the same leak by another route.
        _record_encryption_failure("unencrypted_result")
        raise PiiEncryptionError(
            f"encryption for subject {subject_id!r} did not produce a {TOKEN_PREFIX} token"
        )
    return token


def store_subject_text(subject_id: str, plaintext: str, *, field: str) -> str:
    """Value to persist in a PII free-text column — ciphertext or a marker.

    When encryption is unavailable the row is still written (dropping a safety
    complaint is worse than dropping its description) but the customer's words
    are not persisted in the clear. ``encryption_health()`` and the
    ``pii.encrypt_failed`` security event tell operators what was lost.
    """
    if not plaintext:
        return plaintext
    try:
        return encrypt_subject_text(subject_id, plaintext)
    except PiiEncryptionError as exc:
        _log.error("pii_encrypt_failed field=%s subject=%s: %s", field, subject_id, exc)
        try:
            from src.security.audit_log import security_event

            security_event(
                "pii.encrypt_failed",
                outcome="failure",
                resource=f"{field}:{subject_id}",
                detail={"error": str(exc), "chars_dropped": len(plaintext)},
            )
        except Exception:
            pass
        return UNENCRYPTED_PLACEHOLDER


def reveal_subject_text(subject_id: str, text: str | None) -> str:
    """Decrypt an at-rest token for display. Plaintext passes through.

    Shredded key → ``[ERASED]``. Ciphertext that fails authentication →
    ``[UNREADABLE]`` plus a security event: a read surface should not raise, but
    it must not report corruption as a completed erasure either (R30).
    """
    raw = text or ""
    if not (raw.startswith(TOKEN_PREFIX) or raw.startswith(LEGACY_XOR_PREFIX)):
        return raw
    try:
        return decrypt_subject_pii(subject_id, raw)
    except PiiIntegrityError as exc:
        _log.error("pii_reveal_integrity_failure subject=%s: %s", subject_id, exc)
        try:
            from src.security.audit_log import security_event

            security_event(
                "pii.token_integrity_failure",
                outcome="failure",
                resource=subject_id,
                detail={"error": str(exc)},
            )
        except Exception:
            pass
        return CORRUPT_TEXT
    except Exception:
        return ERASED_TEXT


def decrypt_case_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return row
    item = dict(row)
    iid = str(item.get("interaction_id") or "")
    if iid and "description_summary" in item:
        item["description_summary"] = reveal_subject_text(iid, item.get("description_summary"))
    return item


def decrypt_case_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [decrypt_case_row(r) or r for r in rows]


__all__ = [
    "redact_pii",
    "find_pii",
    "PII_TEXT_FIELDS",
    "redact_dict",
    "redact_turns",
    "SubjectKeyStore",
    "PiiEncryptionError",
    "PiiIntegrityError",
    "TOKEN_PREFIX",
    "LEGACY_XOR_PREFIX",
    "UNENCRYPTED_PLACEHOLDER",
    "encryption_health",
    "reset_encryption_health",
    "encrypt_subject_pii",
    "decrypt_subject_pii",
    "encrypt_subject_text",
    "store_subject_text",
    "reveal_subject_text",
    "relabel_legacy_xor_token",
    "migrate_legacy_xor_token",
    "decrypt_case_row",
    "decrypt_case_rows",
    "ERASED_TEXT",
    "CORRUPT_TEXT",
]
