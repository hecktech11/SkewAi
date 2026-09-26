"""An AES-GCM authentication failure must never fall through to a legacy decoder (R30).

``decrypt_subject_pii`` caught every AES-GCM error and, for payloads of 64 bytes
or fewer, re-decoded the same bytes with an unauthenticated blake2b/XOR stream
under the same ``enc:v1:`` prefix. Tampered ciphertext therefore came back as
text instead of an error, and anyone able to write to ``cases`` or
``interaction_turns`` could pick which decoder ran by flipping one byte —
a downgrade from authenticated encryption to a keystream with no integrity at
all. The two formats need distinct versions so only an explicitly-tagged legacy
row reaches the legacy path.
"""

from __future__ import annotations

import base64
import hashlib

import pytest

from src.ids import new_ulid
from src.security import pii as pii_mod
from src.security.pii import (
    LEGACY_XOR_PREFIX,
    TOKEN_PREFIX,
    PiiIntegrityError,
    SubjectKeyStore,
    decrypt_subject_pii,
    encrypt_subject_pii,
)

SHORT_SECRET = "SSN 123-45-6789"
LONG_SECRET = "Customer statement: " + ("brake failure at speed; " * 8)


@pytest.fixture(autouse=True)
def _clean_keys():
    SubjectKeyStore.clear()
    yield
    SubjectKeyStore.clear()


def _payload(token: str) -> bytearray:
    prefix = LEGACY_XOR_PREFIX if token.startswith(LEGACY_XOR_PREFIX) else TOKEN_PREFIX
    return bytearray(base64.urlsafe_b64decode(token[len(prefix) :].encode("ascii")))


def _retoken(payload: bytes, prefix: str = TOKEN_PREFIX) -> str:
    return prefix + base64.urlsafe_b64encode(bytes(payload)).decode("ascii")


def _legacy_xor_token(dek: bytes, nonce: bytes, plaintext: str, *, prefix: str) -> str:
    """Rebuild a token in the pre-AES-GCM format this module used to write."""
    raw = plaintext.encode("utf-8")
    stream = hashlib.blake2b(dek, key=nonce, digest_size=max(1, len(raw))).digest()
    return _retoken(nonce + bytes(b ^ k for b, k in zip(raw, stream)), prefix)


# ── tampering must be an error, not a different decoder ──────────────────────


def test_tampered_short_ciphertext_is_rejected(reset_ops_db):
    iid = "int_r30_" + new_ulid()[:8]
    token = encrypt_subject_pii(iid, SHORT_SECRET)

    payload = _payload(token)
    payload[-1] ^= 0x01

    with pytest.raises(PiiIntegrityError):
        decrypt_subject_pii(iid, _retoken(payload))


def test_tampered_short_ciphertext_never_returns_text(reset_ops_db):
    """The precise regression: a 31-byte payload took the <= 64 legacy branch."""
    iid = "int_r30_" + new_ulid()[:8]
    token = encrypt_subject_pii(iid, SHORT_SECRET)
    payload = _payload(token)
    assert len(payload) - 12 <= 64, "fixture must sit inside the old fallback window"
    payload[-1] ^= 0x01

    try:
        result = decrypt_subject_pii(iid, _retoken(payload))
    except PiiIntegrityError:
        return
    pytest.fail(f"tampered token decoded to {result!r} instead of raising")


def test_tampered_nonce_is_rejected(reset_ops_db):
    iid = "int_r30_" + new_ulid()[:8]
    payload = _payload(encrypt_subject_pii(iid, SHORT_SECRET))
    payload[0] ^= 0xFF

    with pytest.raises(PiiIntegrityError):
        decrypt_subject_pii(iid, _retoken(payload))


def test_truncated_token_is_rejected(reset_ops_db):
    iid = "int_r30_" + new_ulid()[:8]
    payload = _payload(encrypt_subject_pii(iid, SHORT_SECRET))

    with pytest.raises(PiiIntegrityError):
        decrypt_subject_pii(iid, _retoken(payload[:20]))


def test_malformed_base64_is_rejected(reset_ops_db):
    iid = "int_r30_" + new_ulid()[:8]
    SubjectKeyStore.get_or_create_dek(iid)

    with pytest.raises(PiiIntegrityError):
        decrypt_subject_pii(iid, TOKEN_PREFIX + "not base64 !!!")


def test_tampered_long_ciphertext_is_still_rejected(reset_ops_db):
    """Payloads over 64 bytes already raised; keep that true."""
    iid = "int_r30_" + new_ulid()[:8]
    payload = _payload(encrypt_subject_pii(iid, LONG_SECRET))
    assert len(payload) - 12 > 64
    payload[-1] ^= 0x01

    with pytest.raises(PiiIntegrityError):
        decrypt_subject_pii(iid, _retoken(payload))


def test_another_subjects_key_cannot_read_the_token(reset_ops_db):
    victim = "int_r30_v_" + new_ulid()[:8]
    other = "int_r30_o_" + new_ulid()[:8]
    token = encrypt_subject_pii(victim, SHORT_SECRET)
    SubjectKeyStore.get_or_create_dek(other)

    with pytest.raises(PiiIntegrityError):
        decrypt_subject_pii(other, token)


# ── the legacy format needs its own version ──────────────────────────────────


def test_legacy_xor_bytes_are_not_readable_under_the_v1_prefix(reset_ops_db):
    iid = "int_r30_" + new_ulid()[:8]
    dek = SubjectKeyStore.get_or_create_dek(iid)
    nonce = b"\x01" * 12
    mislabelled = _legacy_xor_token(dek, nonce, SHORT_SECRET, prefix=TOKEN_PREFIX)

    with pytest.raises(PiiIntegrityError):
        decrypt_subject_pii(iid, mislabelled)


def test_legacy_xor_token_decodes_under_its_own_prefix(reset_ops_db):
    iid = "int_r30_" + new_ulid()[:8]
    dek = SubjectKeyStore.get_or_create_dek(iid)
    nonce = b"\x02" * 12
    legacy = _legacy_xor_token(dek, nonce, SHORT_SECRET, prefix=LEGACY_XOR_PREFIX)

    # Live reads reject legacy XOR tokens to prevent prefix downgrade attacks (N07)
    with pytest.raises(PiiIntegrityError, match="not permitted in live reads"):
        decrypt_subject_pii(iid, legacy)

    # Offline migration explicitly permits decoding
    assert decrypt_subject_pii(iid, legacy, allow_legacy=True) == SHORT_SECRET


def test_relabel_then_migrate_yields_authenticated_ciphertext(reset_ops_db):
    iid = "int_r30_" + new_ulid()[:8]
    dek = SubjectKeyStore.get_or_create_dek(iid)
    nonce = b"\x03" * 12
    historical = _legacy_xor_token(dek, nonce, SHORT_SECRET, prefix=TOKEN_PREFIX)

    relabelled = pii_mod.relabel_legacy_xor_token(historical)
    assert relabelled.startswith(LEGACY_XOR_PREFIX)

    migrated = pii_mod.migrate_legacy_xor_token(iid, relabelled)
    assert migrated.startswith(TOKEN_PREFIX)
    assert decrypt_subject_pii(iid, migrated) == SHORT_SECRET

    # And the migrated token is now tamper-evident.
    payload = _payload(migrated)
    payload[-1] ^= 0x01
    with pytest.raises(PiiIntegrityError):
        decrypt_subject_pii(iid, _retoken(payload))


# ── reads keep working, and say which failure they hit ───────────────────────


def test_roundtrip_is_unchanged(reset_ops_db):
    iid = "int_r30_" + new_ulid()[:8]

    assert decrypt_subject_pii(iid, encrypt_subject_pii(iid, SHORT_SECRET)) == SHORT_SECRET
    assert decrypt_subject_pii(iid, encrypt_subject_pii(iid, LONG_SECRET)) == LONG_SECRET


def test_shredded_key_still_raises_key_error_not_integrity_error(reset_ops_db):
    iid = "int_r30_" + new_ulid()[:8]
    token = encrypt_subject_pii(iid, SHORT_SECRET)
    assert SubjectKeyStore.shred_dek(iid) is True

    with pytest.raises(KeyError, match="shredded"):
        decrypt_subject_pii(iid, token)


def test_reveal_distinguishes_corruption_from_erasure(reset_ops_db):
    """[ERASED] means the subject exercised Art. 17. Corruption is a different
    incident and must not be reported as a completed erasure."""
    erased_iid = "int_r30_e_" + new_ulid()[:8]
    erased_token = encrypt_subject_pii(erased_iid, SHORT_SECRET)
    assert SubjectKeyStore.shred_dek(erased_iid) is True
    assert pii_mod.reveal_subject_text(erased_iid, erased_token) == pii_mod.ERASED_TEXT

    live_iid = "int_r30_c_" + new_ulid()[:8]
    payload = _payload(encrypt_subject_pii(live_iid, SHORT_SECRET))
    payload[-1] ^= 0x01

    revealed = pii_mod.reveal_subject_text(live_iid, _retoken(payload))
    assert revealed == pii_mod.CORRUPT_TEXT
    assert revealed != pii_mod.ERASED_TEXT
    assert SHORT_SECRET not in revealed
