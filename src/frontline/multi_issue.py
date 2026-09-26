"""Multi-issue linkage for one contact (feature #18, minimal).

Stores secondary issues on ``contact_issues`` and optionally creates extra
case rows sharing the same interaction_id.
"""

from __future__ import annotations

import re
from typing import Any

from src.data.timeutil import utc_now
from src.data.warehouse import ops_con
from src.ids import new_ulid
from src.ledger import AgentAction, record_action

_SPLIT = re.compile(
    r"\b(?:also|and also|plus|second(?:ly)?|another issue|additionally)\b",
    re.I,
)


def ensure_contact_issues_table() -> None:
    with ops_con() as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS contact_issues (
                issue_id           VARCHAR PRIMARY KEY,
                interaction_id     VARCHAR NOT NULL,
                seq                INTEGER NOT NULL,
                category           VARCHAR,
                description        TEXT NOT NULL,
                case_id            VARCHAR,
                created_at         TIMESTAMP NOT NULL
            )
            """
        )
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_contact_issues_ix "
            "ON contact_issues(interaction_id, seq)"
        )


def split_issues_from_text(text: str) -> list[str]:
    """Split multi-problem utterance into distinct issue snippets."""
    raw = (text or "").strip()
    if not raw:
        return []
    parts = [p.strip(" .,;") for p in _SPLIT.split(raw) if p and p.strip(" .,;")]
    # Deduplicate trivial splits
    out = []
    for p in parts:
        if len(p) < 8:
            continue
        if p not in out:
            out.append(p)
    if len(out) < 2:
        return [raw] if raw else []
    return out


def record_multi_issues(
    interaction_id: str,
    *,
    pack_id: str,
    issues: list[dict[str, Any]],
    create_extra_cases: bool = True,
) -> list[dict[str, Any]]:
    """Persist issue rows (+ optional extra cases). First issue may already be primary case."""
    ensure_contact_issues_table()
    created: list[dict[str, Any]] = []
    now = utc_now()
    with ops_con() as con:
        for seq, issue in enumerate(issues):
            desc = (issue.get("description") or "").strip()
            if not desc:
                continue
            # Full ULID + seq: truncated ULID is timestamp-heavy and collides
            # when two issues are minted in the same millisecond.
            issue_id = f"iss_{new_ulid()}_{seq}"
            case_id = issue.get("case_id")
            if create_extra_cases and seq > 0 and not case_id:
                case_id = f"case_{new_ulid()}_{seq}"
                from src.security.pii import store_subject_text

                # No plaintext fallback — see store_subject_text (R29).
                desc_stored = store_subject_text(
                    interaction_id, desc[:500], field="cases.description_summary"
                )
                con.execute(
                    """
                    INSERT INTO cases (
                        case_id, interaction_id, pack_id, created_at,
                        category, description_summary, onset, severity,
                        severity_source, priority, safety_flags,
                        advisory_match_id, cluster_match_id, similar_record_count,
                        investigation_id, status, followup_draft
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'rules', ?, '{}',
                              NULL, NULL, 0, NULL, 'open', ?)
                    """,
                    [
                        case_id,
                        interaction_id,
                        pack_id,
                        now,
                        issue.get("category"),
                        desc_stored,
                        now,
                        issue.get("severity") or "Medium",
                        int(issue.get("priority") or 2),
                        f"Linked multi-issue #{seq + 1} from contact {interaction_id}",
                    ],
                )
            con.execute(
                """
                INSERT INTO contact_issues
                (issue_id, interaction_id, seq, category, description, case_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    issue_id,
                    interaction_id,
                    seq,
                    issue.get("category"),
                    desc[:1000],
                    case_id,
                    now,
                ],
            )
            created.append(
                {
                    "issue_id": issue_id,
                    "seq": seq,
                    "description": desc[:200],
                    "case_id": case_id,
                    "category": issue.get("category"),
                }
            )
    if created:
        try:
            record_action(
                AgentAction(
                    interaction_id=interaction_id,
                    agent="case",
                    action_type="case_created",
                    input_summary="multi_issue_link",
                    output_summary=f"issues={len(created)}",
                    evidence_ids=[c["issue_id"] for c in created],
                )
            )
        except Exception:
            pass
    return created


def attach_multi_issues_from_description(
    interaction_id: str,
    *,
    pack_id: str,
    description: str,
    primary_case_id: str | None = None,
    category: str | None = None,
) -> list[dict[str, Any]]:
    snippets = split_issues_from_text(description)
    if len(snippets) < 2:
        return []
    issues = []
    for i, snip in enumerate(snippets):
        issues.append(
            {
                "description": snip,
                "category": category,
                "case_id": primary_case_id if i == 0 else None,
            }
        )
    return record_multi_issues(
        interaction_id, pack_id=pack_id, issues=issues, create_extra_cases=True
    )


__all__ = [
    "ensure_contact_issues_table",
    "split_issues_from_text",
    "record_multi_issues",
    "attach_multi_issues_from_description",
]
