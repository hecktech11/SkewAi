"""PII encryption must fail closed, never persist the text it could not protect (R29).

``encrypt_subject_text`` returned its own input whenever encryption raised, and
both case-creation callers wrapped it in a second plaintext fallback. A DuckDB
lock on ``subject_deks``, a shredded DEK, or a missing ``cryptography`` wheel
therefore wrote the customer's own words into ``cases.description_summary`` in
the clear — and nothing on the read path can tell an unencrypted row apart from
a successfully decrypted one, so the leak is silent and permanent.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.agents.base import InteractionContext
from src.agents.case_agent import CaseAgent
from src.data.warehouse import ops_con
from src.ids import new_ulid
from src.security import pii as pii_mod
from src.security.pii import (
    SubjectKeyStore,
    encrypt_subject_text,
    encryption_health,
    reset_encryption_health,
)

SECRET = "Caller Jane Doe 555-0199, SSN 123-45-6789, brakes grind on the CR-V"


@pytest.fixture(autouse=True)
def _clean_encryption_state():
    SubjectKeyStore.clear()
    reset_encryption_health()
    yield
    reset_encryption_health()


def _break_encryption(monkeypatch, exc: Exception) -> None:
    """Make the AES-GCM writer fail the way a key-store outage would."""

    def _boom(subject_id: str, plaintext: str) -> str:
        raise exc

    monkeypatch.setattr(pii_mod, "encrypt_subject_pii", _boom)


def _seed_interaction(iid: str, pack_id: str = "automotive_nhtsa") -> None:
    with ops_con() as con:
        con.execute(
            """
            INSERT INTO interactions
            (interaction_id, pack_id, pack_version, started_at, channel, status, supervised, llm_calls)
            VALUES (?, ?, 't', ?, 'web_text', 'active', FALSE, 0)
            """,
            [iid, pack_id, datetime.now(timezone.utc)],
        )


def _stored_description(case_id: str) -> str:
    with ops_con(read_only=True) as con:
        row = con.execute(
            "SELECT description_summary FROM cases WHERE case_id = ?", [case_id]
        ).fetchone()
    return str(row[0] if row else "")


# ── the helper itself ────────────────────────────────────────────────────────


def test_encrypt_subject_text_raises_instead_of_returning_plaintext(monkeypatch):
    _break_encryption(monkeypatch, OSError("subject_deks is locked by another process"))

    with pytest.raises(pii_mod.PiiEncryptionError):
        encrypt_subject_text("int_fc_1", SECRET)


@pytest.mark.parametrize(
    "exc",
    [
        OSError("db locked"),
        RuntimeError("no entropy source"),
        ValueError("bad key length"),
        ImportError("no module named cryptography"),
        KeyError("subject DEK has been shredded"),
    ],
    ids=["oserror", "runtime", "value", "import", "shredded"],
)
def test_no_failure_mode_leaks_the_plaintext_back_to_the_caller(monkeypatch, exc):
    _break_encryption(monkeypatch, exc)

    with pytest.raises(pii_mod.PiiEncryptionError) as raised:
        encrypt_subject_text("int_fc_2", SECRET)

    # The error itself must not carry what it refused to encrypt.
    assert SECRET not in str(raised.value)
    assert "123-45-6789" not in str(raised.value)


def test_shredded_subject_is_never_re_encrypted_as_plaintext(reset_ops_db):
    iid = "int_fc_shred_" + new_ulid()[:8]
    SubjectKeyStore.get_or_create_dek(iid)
    assert SubjectKeyStore.shred_dek(iid) is True

    with pytest.raises(pii_mod.PiiEncryptionError):
        encrypt_subject_text(iid, SECRET)


def test_store_subject_text_substitutes_a_marker_not_the_plaintext(monkeypatch):
    _break_encryption(monkeypatch, OSError("db locked"))

    stored = pii_mod.store_subject_text("int_fc_3", SECRET, field="cases.description_summary")

    assert stored == pii_mod.UNENCRYPTED_PLACEHOLDER
    assert SECRET not in stored


def test_encryption_failures_are_visible_to_operators(monkeypatch):
    assert encryption_health()["failures"] == 0
    _break_encryption(monkeypatch, OSError("db locked"))

    pii_mod.store_subject_text("int_fc_4", SECRET, field="cases.description_summary")

    health = encryption_health()
    assert health["failures"] == 1
    assert health["healthy"] is False
    assert SECRET not in str(health)


def test_encryption_health_check_reports_degraded(monkeypatch):
    from src.observability.health_checks import check_pii_encryption

    assert check_pii_encryption().status == "healthy"
    _break_encryption(monkeypatch, OSError("db locked"))
    pii_mod.store_subject_text("int_fc_5", SECRET, field="cases.description_summary")

    assert check_pii_encryption().status in {"degraded", "critical"}


def test_marker_is_not_mistaken_for_ciphertext_on_read():
    revealed = pii_mod.reveal_subject_text("int_fc_6", pii_mod.UNENCRYPTED_PLACEHOLDER)

    assert revealed == pii_mod.UNENCRYPTED_PLACEHOLDER
    assert SECRET not in revealed


# ── the callers that used to repeat the fallback ─────────────────────────────


def _ctx(pack, iid: str, description: str) -> InteractionContext:
    ctx = InteractionContext(interaction_id=iid, pack=pack)
    ctx.slots = {
        "entity_1": "2019",
        "entity_2": "HONDA",
        "entity_3": "CR-V",
        "category": "SERVICE BRAKES",
        "description": description,
    }
    ctx.severity = "Medium"
    ctx.priority = 2
    return ctx


@pytest.mark.asyncio
async def test_case_agent_never_writes_a_plaintext_description(
    reset_ops_db, pack, monkeypatch
):
    iid = "int_fc_case_" + new_ulid()[:8]
    _seed_interaction(iid, pack.id)
    _break_encryption(monkeypatch, OSError("subject_deks is locked by another process"))

    result = await CaseAgent(_ctx(pack, iid, SECRET)).run()

    stored = _stored_description(result["case_id"])
    assert SECRET not in stored
    assert "123-45-6789" not in stored
    assert stored == pii_mod.UNENCRYPTED_PLACEHOLDER


@pytest.mark.asyncio
async def test_case_agent_still_encrypts_on_the_happy_path(reset_ops_db, pack):
    iid = "int_fc_ok_" + new_ulid()[:8]
    _seed_interaction(iid, pack.id)

    result = await CaseAgent(_ctx(pack, iid, SECRET)).run()

    stored = _stored_description(result["case_id"])
    assert stored.startswith("enc:v1:")
    assert "123-45-6789" not in stored
    assert pii_mod.reveal_subject_text(iid, stored) == SECRET[:500]
    assert encryption_health()["failures"] == 0


def test_multi_issue_never_writes_a_plaintext_description(reset_ops_db, monkeypatch):
    from src.frontline.multi_issue import record_multi_issues

    iid = "int_fc_multi_" + new_ulid()[:8]
    _seed_interaction(iid)
    _break_encryption(monkeypatch, OSError("subject_deks is locked by another process"))

    record_multi_issues(
        iid,
        pack_id="automotive_nhtsa",
        issues=[
            {"category": "SERVICE BRAKES", "description": "primary issue text"},
            {"category": "ELECTRICAL", "description": SECRET},
        ],
    )

    with ops_con(read_only=True) as con:
        rows = con.execute(
            "SELECT description_summary FROM cases WHERE interaction_id = ?", [iid]
        ).fetchall()
    assert rows, "the extra case must still be created — losing a complaint is worse"
    for (value,) in rows:
        assert SECRET not in str(value)
        assert "123-45-6789" not in str(value)
