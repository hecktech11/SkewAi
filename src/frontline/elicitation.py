"""Engineer-registered diagnostic questions asked mid-contact + spoken confirm."""

from __future__ import annotations

import re
from typing import Any

from src.data.timeutil import utc_now
from src.data.warehouse import ops_con
from src.ids import new_ulid
from src.ledger import AgentAction, record_action


def _ensure(con) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS diagnostic_questions (
            question_id    VARCHAR PRIMARY KEY,
            pack_id        VARCHAR NOT NULL,
            prompt         TEXT NOT NULL,
            category       VARCHAR,
            entity_2       VARCHAR,
            created_at     TIMESTAMP NOT NULL
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS diagnostic_answers (
            answer_id      VARCHAR PRIMARY KEY,
            interaction_id VARCHAR NOT NULL,
            question_id    VARCHAR NOT NULL,
            prompt         TEXT NOT NULL,
            answer         TEXT NOT NULL,
            confirmed      BOOLEAN NOT NULL DEFAULT TRUE,
            created_at     TIMESTAMP NOT NULL
        )
        """
    )
    try:
        con.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS ux_diag_answers_qa "
            "ON diagnostic_answers (interaction_id, question_id)"
        )
    except Exception:
        pass
    for ddl in (
        "ALTER TABLE diagnostic_answers ADD COLUMN provenance VARCHAR",
        "ALTER TABLE diagnostic_answers ADD COLUMN polarity VARCHAR",
        "ALTER TABLE diagnostic_answers ADD COLUMN uncertainty VARCHAR",
        "ALTER TABLE diagnostic_answers ADD COLUMN symptom VARCHAR",
        "ALTER TABLE diagnostic_answers ADD COLUMN source_turn VARCHAR",
    ):
        try:
            con.execute(ddl)
        except Exception:
            pass


def register_question(
    pack_id: str,
    prompt: str,
    *,
    category: str | None = None,
    entity_2: str | None = None,
) -> dict[str, Any]:
    qid = "dq_" + new_ulid()
    with ops_con() as con:
        _ensure(con)
        con.execute(
            """
            INSERT INTO diagnostic_questions
            (question_id, pack_id, prompt, category, entity_2, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [qid, pack_id, prompt.strip(), category, entity_2, utc_now()],
        )
    return {
        "question_id": qid,
        "pack_id": pack_id,
        "prompt": prompt.strip(),
        "category": category,
        "entity_2": entity_2,
    }


def _open_hypotheses(hypotheses: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    out = []
    for row in hypotheses or []:
        if not isinstance(row, dict):
            continue
        if row.get("open", True) is False:
            continue
        if row.get("category"):
            out.append(row)
    return out


def _splits_open_hypotheses(category: str | None, hypotheses: list[dict[str, Any]] | None) -> bool:
    """True when this question's category belongs to exactly one still-open hypothesis."""
    open_rows = _open_hypotheses(hypotheses)
    if len(open_rows) < 2 or not category:
        return False
    owners = [row for row in open_rows if str(row.get("category")) == str(category)]
    return 0 < len(owners) < len(open_rows)


def next_question(
    pack_id: str,
    slots: dict[str, Any] | None = None,
    *,
    interaction_id: str | None = None,
    hypotheses: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """First matching unanswered question (item 35: no duplicate questioning).

    When ``interaction_id`` is given, questions already answered in this
    contact (see ``diagnostic_answers``) are skipped — the contact is never
    asked the same diagnostic question twice.
    """
    slots = slots or {}
    cat = (slots.get("category") or "").strip()
    ent = (slots.get("entity_2") or "").strip()

    def _load(con):
        answered_ids: set[str] = set()
        if interaction_id:
            answered_ids = {
                str(r[0])
                for r in con.execute(
                    "SELECT question_id FROM diagnostic_answers WHERE interaction_id = ?",
                    [interaction_id],
                ).fetchall()
            }
        found = con.execute(
            """
            SELECT question_id, prompt, category, entity_2
            FROM diagnostic_questions WHERE pack_id = ?
            ORDER BY created_at
            """,
            [pack_id],
        ).fetchall()
        return answered_ids, found

    # A read-only connection cannot CREATE or ALTER, even when the table
    # already exists. Read first. Create the tables only if that read fails.
    try:
        with ops_con(read_only=True) as con:
            answered, rows = _load(con)
    except Exception:
        try:
            with ops_con() as con:
                _ensure(con)
                answered, rows = _load(con)
        except Exception:
            return None
    pending = []
    for qid, prompt, qcat, qent in rows:
        if str(qid) in answered:
            continue
        pending.append((qid, prompt, qcat, qent))
    # A question that separates two open hypotheses wins over first-match order.
    for qid, prompt, qcat, qent in pending:
        if _splits_open_hypotheses(qcat, hypotheses):
            return {
                "question_id": qid,
                "prompt": prompt,
                "category": qcat,
                "entity_2": qent,
                "splits_hypotheses": True,
            }
    for qid, prompt, qcat, qent in pending:
        if qcat and cat and str(qcat) != cat:
            continue
        if qent and ent and str(qent) != ent:
            continue
        return {
            "question_id": qid,
            "prompt": prompt,
            "category": qcat,
            "entity_2": qent,
            "splits_hypotheses": False,
        }
    return None


_HAZARD_WORDS = (
    "smoke", "fire", "grind", "grinding", "brake", "brakes", "leak",
    "stall", "stalled", "cut", "noise", "squeal", "injury", "hurt",
)
_UNCERTAIN_RE = re.compile(r"\b(maybe|think|not sure|unsure|might|possibly)\b", re.I)
_LEAD_NO = re.compile(r"^(no|nope|nah)\b", re.I)
_LEAD_YES = re.compile(r"^(yes|yeah|yep|yup|yea)\b", re.I)


def classify_diagnostic_answer(prompt: str, answer: str) -> dict[str, str]:
    """Polarity, uncertainty, and the symptom constraint an answer adds."""
    text = (answer or "").strip()
    low = text.lower()
    uncertain = "uncertain" if _UNCERTAIN_RE.search(low) else "stated"
    if _LEAD_NO.search(low):
        polarity = "no"
    elif _LEAD_YES.search(low):
        polarity = "yes"
    elif text:
        polarity = "descriptive"
    else:
        polarity = "unknown"
    blob = f"{prompt or ''} {text}".lower()
    source = blob if polarity == "yes" else low
    if polarity == "no":
        symptom = ""
    else:
        symptom = " ".join(word for word in _HAZARD_WORDS if re.search(rf"\b{re.escape(word)}\b", source))
    return {
        "polarity": polarity,
        "uncertainty": uncertain,
        "symptom": symptom,
        "provenance": "customer_turn",
    }


def effective_symptom_text(description: str, answers: list[dict[str, Any]] | None = None) -> str:
    """Description plus affirmed diagnostic symptoms. A 'no' does not add the hazard."""
    parts = [(description or "").strip()]
    for row in answers or []:
        if not isinstance(row, dict):
            continue
        if str(row.get("polarity") or "").lower() == "no":
            continue
        symptom = str(row.get("symptom") or "").strip()
        if symptom:
            parts.append(symptom)
        elif str(row.get("polarity") or "").lower() == "yes":
            extra = str(row.get("answer") or "").strip()
            if extra:
                parts.append(extra)
    return " ".join(part for part in parts if part).strip()


def symptom_text_for_ctx(ctx: Any) -> str:
    """Symptom text the live severity and retrieval paths should score."""
    slots = getattr(ctx, "slots", None) or {}
    base = str(slots.get("description") or "")
    iid = getattr(ctx, "interaction_id", None)
    answers: list[dict[str, Any]] = []
    if iid:
        try:
            answers = answers_for_interaction(str(iid))
        except Exception:
            answers = []
    return effective_symptom_text(base, answers)


def record_elicitation_answer(
    interaction_id: str,
    question_id: str,
    answer: str,
    *,
    prompt: str = "",
    confirmed: bool = True,
    provenance: str = "",
    polarity: str = "",
    uncertainty: str = "",
    symptom: str = "",
    source_turn: str = "",
) -> dict[str, Any]:
    """Persist a diagnostic answer with provenance (item 35).

    Idempotent per (interaction, question): a repeated answer UPDATES the
    stored text rather than duplicating the question. Returns the record —
    investigators attach it to the brief, case agents to hypotheses.
    """
    judged = classify_diagnostic_answer(prompt, answer)
    provenance = provenance or judged["provenance"]
    polarity = polarity or judged["polarity"]
    uncertainty = uncertainty or judged["uncertainty"]
    if symptom == "" and "symptom" not in (answer or ""):
        symptom = judged["symptom"]
    elif not symptom:
        symptom = judged["symptom"]
    aid = "da_" + new_ulid()
    stored_answer = answer.strip()[:1000]
    with ops_con() as con:
        _ensure(con)
        existing = con.execute(
            "SELECT answer_id FROM diagnostic_answers WHERE interaction_id = ? AND question_id = ?",
            [interaction_id, question_id],
        ).fetchone()
        if existing:
            con.execute(
                """
                UPDATE diagnostic_answers
                SET answer = ?, prompt = ?, confirmed = ?, created_at = ?,
                    provenance = ?, polarity = ?, uncertainty = ?, symptom = ?,
                    source_turn = ?
                WHERE interaction_id = ? AND question_id = ?
                """,
                [stored_answer, prompt[:500], confirmed, utc_now(),
                 provenance[:80], polarity[:20], uncertainty[:20], symptom[:200],
                 (source_turn or stored_answer)[:300],
                 interaction_id, question_id],
            )
            return {
                "answer_id": existing[0], "interaction_id": interaction_id,
                "question_id": question_id, "prompt": prompt,
                "answer": stored_answer, "updated": True,
                "provenance": provenance, "polarity": polarity,
                "uncertainty": uncertainty, "symptom": symptom,
            }
        con.execute(
            """
            INSERT INTO diagnostic_answers
            (answer_id, interaction_id, question_id, prompt, answer, confirmed, created_at,
             provenance, polarity, uncertainty, symptom, source_turn)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [aid, interaction_id, question_id, prompt[:500],
             stored_answer, confirmed, utc_now(),
             provenance[:80], polarity[:20], uncertainty[:20], symptom[:200],
             (source_turn or stored_answer)[:300]],
        )
    return {
        "answer_id": aid, "interaction_id": interaction_id,
        "question_id": question_id, "prompt": prompt,
        "answer": stored_answer, "updated": False,
        "provenance": provenance, "polarity": polarity,
        "uncertainty": uncertainty, "symptom": symptom,
    }


def answers_for_interaction(interaction_id: str) -> list[dict[str, Any]]:
    """All diagnostic answers for a contact, oldest first (brief attachment)."""

    def _load(con) -> list[dict[str, Any]]:
        cur = con.execute(
            """
            SELECT answer_id, question_id, prompt, answer, confirmed, created_at,
                   provenance, polarity, uncertainty, symptom, source_turn
            FROM diagnostic_answers WHERE interaction_id = ?
            ORDER BY created_at
            """,
            [interaction_id],
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    try:
        with ops_con(read_only=True) as con:
            return _load(con)
    except Exception:
        try:
            with ops_con() as con:
                _ensure(con)
                return _load(con)
        except Exception:
            return []


def record_spoken_confirmation(
    interaction_id: str,
    utterance: str,
    *,
    confirmed: bool,
    question_id: str | None = None,
) -> str:
    """Ledger a hash-chained spoken confirmation (read-back)."""
    return record_action(
        AgentAction(
            interaction_id=interaction_id,
            agent="orchestrator",
            action_type="spoken_confirmation",
            input_summary=(question_id or "")[:200],
            output_summary=f"confirmed={confirmed}; utterance={utterance[:400]}",
            evidence_ids=[question_id] if question_id else [],
            ok=True,
        )
    )


def looks_like_confirmation(text: str) -> bool:
    t = (text or "").strip().lower().rstrip(".!?").strip()
    if t in {"yes", "y", "correct", "yeah", "yep", "that's right", "that is right"}:
        return True
    if "correct" in t and "not" not in t:
        return True
    # "Yes, that's right." — yes-led with no correction cue.
    if re.match(r"^(yes|yeah|yep|y|correct)[,!\s]", t):
        if not re.search(r"\b(but|actually|however|though|no\b|not|wrong|instead|rather)\b", t):
            return True
    return False


__all__ = [
    "register_question",
    "next_question",
    "record_spoken_confirmation",
    "record_elicitation_answer",
    "answers_for_interaction",
    "looks_like_confirmation",
    "classify_diagnostic_answer",
    "effective_symptom_text",
    "symptom_text_for_ctx",
]
