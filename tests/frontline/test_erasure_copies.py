"""Crypto-shred must remove every plaintext copy it claims to erase (R03).

tombstone_interaction cleared cases.description_summary and then returned
ok=true, while cases.followup_draft, case_notes.body and
contact_issues.description still held the customer's email and phone.
A per-table error was stored as -1 and ignored by that same ok flag.
"""

from __future__ import annotations

from datetime import datetime, timezone

from src.data.warehouse import ops_con
from src.frontline.dsr import delete_interaction
from src.frontline.multi_issue import ensure_contact_issues_table

EMAIL = "ada.r03@example.com"
PHONE = "555-014-2298"
IID = "int_r03_shred"
CID = "case_r03_shred"


def _seed() -> None:
    now = datetime.now(timezone.utc)
    text = f"email {EMAIL} phone {PHONE}"
    ensure_contact_issues_table()
    with ops_con() as con:
        con.execute(
            """
            INSERT INTO interactions
            (interaction_id, pack_id, pack_version, started_at, channel, status, description)
            VALUES (?, 'automotive_nhtsa', 't', ?, 'web_text', 'completed', ?)
            """,
            [IID, now, text],
        )
        con.execute(
            """
            INSERT INTO cases
            (case_id, interaction_id, pack_id, created_at, category, description_summary,
             onset, severity, severity_source, priority, safety_flags, status, followup_draft)
            VALUES (?, ?, 'automotive_nhtsa', ?, 'brakes', ?, ?, 'Low', 'rules', 3, '{}', 'open', ?)
            """,
            [CID, IID, now, text, now, text],
        )
        con.execute(
            """
            INSERT INTO case_notes (note_id, case_id, author, body, created_at)
            VALUES ('note_r03', ?, 'operator', ?, ?)
            """,
            [CID, text, now],
        )
        con.execute(
            """
            INSERT INTO contact_issues
            (issue_id, interaction_id, seq, category, description, case_id, created_at)
            VALUES ('iss_r03', ?, 1, 'brakes', ?, ?, ?)
            """,
            [IID, text, CID, now],
        )


def _copies() -> tuple[str, str, str]:
    with ops_con(read_only=True) as con:
        draft = con.execute(
            "SELECT followup_draft FROM cases WHERE case_id = ?", [CID]
        ).fetchone()[0]
        body = con.execute(
            "SELECT body FROM case_notes WHERE note_id = 'note_r03'"
        ).fetchone()[0]
        issue = con.execute(
            "SELECT description FROM contact_issues WHERE issue_id = 'iss_r03'"
        ).fetchone()[0]
    return draft or "", body or "", issue or ""


def test_crypto_shred_clears_followup_notes_and_issues(reset_ops_db):
    _seed()
    res = delete_interaction(IID, mode="crypto_shred")
    assert res["ok"] is True
    for value in _copies():
        assert EMAIL not in value
        assert PHONE not in value
        assert "ERASED" in value


def test_erase_deletes_contact_issues(reset_ops_db):
    _seed()
    res = delete_interaction(IID, mode="erase")
    assert res["ok"] is True
    with ops_con(read_only=True) as con:
        left = con.execute(
            "SELECT COUNT(*) FROM contact_issues WHERE interaction_id = ?", [IID]
        ).fetchone()[0]
    assert left == 0


def test_incomplete_erasure_is_not_ok():
    from src.frontline.dsr import erasure_succeeded

    assert erasure_succeeded({"cases.description_tombstoned": 1, "leftovers": 0}) is True
    assert erasure_succeeded({"interaction_turns.tombstoned": -1}) is False
    assert erasure_succeeded({"agent_actions.output_tombstoned": "failed"}) is False
    assert erasure_succeeded({"leftovers": 2}) is False
