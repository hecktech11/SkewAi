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
})


def redact_dict(row: dict[str, Any], fields: frozenset[str] = PII_TEXT_FIELDS) -> dict[str, Any]:
    """Return a copy of *row* with PII redacted in known free-text fields."""
    out = dict(row)
    for key in fields:
        val = out.get(key)
        if isinstance(val, str) and val:
            out[key] = redact_pii(val)
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

    @classmethod
    def get_or_create_dek(cls, subject_id: str) -> bytes:
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
                if row and not row[0]:
                    return True
        except Exception:
            pass
        return False

    @classmethod
    def clear(cls) -> None:
        cls._keys.clear()
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
    return "enc:v1:" + base64.urlsafe_b64encode(payload).decode("ascii")


def decrypt_subject_pii(subject_id: str, token: str) -> str:
    """Decrypt PII. Raises KeyError if the key has been shredded."""
    if not token or not token.startswith("enc:v1:"):
        return token
    if not SubjectKeyStore.has_dek(subject_id):
        raise KeyError(f"Subject DEK for '{subject_id}' has been shredded (GDPR/CCPA Art. 17)")
    dek = SubjectKeyStore.get_or_create_dek(subject_id)
    raw = base64.urlsafe_b64decode(token[len("enc:v1:"):].encode("ascii"))
    nonce, cipher_bytes = raw[:12], raw[12:]
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        aesgcm = AESGCM(dek)
        return aesgcm.decrypt(nonce, cipher_bytes, None).decode("utf-8", errors="replace")
    except Exception:
        if len(cipher_bytes) <= 64:
            stream_key = hashlib.blake2b(dek, key=nonce, digest_size=max(1, len(cipher_bytes))).digest()
            return bytes(b ^ k for b, k in zip(cipher_bytes, stream_key)).decode("utf-8", errors="replace")
        raise


ERASED_TEXT = "[ERASED]"


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
    """Decrypt an at-rest enc:v1: payload. Plaintext passes through. Shredded → [ERASED]."""
    raw = text or ""
    if not raw.startswith("enc:v1:"):
        return raw
    try:
        return decrypt_subject_pii(subject_id, raw)
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
    "TOKEN_PREFIX",
    "UNENCRYPTED_PLACEHOLDER",
    "encryption_health",
    "reset_encryption_health",
    "encrypt_subject_pii",
    "decrypt_subject_pii",
    "encrypt_subject_text",
    "store_subject_text",
    "reveal_subject_text",
    "decrypt_case_row",
    "decrypt_case_rows",
    "ERASED_TEXT",
]
