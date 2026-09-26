"""Regression tests for R17: PII redaction before external LLM narration."""

from __future__ import annotations

import pytest

from src.ai.narration import phrase_followup_draft, phrase_intake_question, phrase_investigation_brief
from src.ai.provider import narrate


def test_phrase_followup_draft_redacts_pii(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    def _fake_narrate(*, site: str, system: str, user: str, fallback: str, model: str | None = None):
        captured["system"] = system
        captured["user"] = user
        return fallback

    monkeypatch.setattr("src.ai.narration.narrate", _fake_narrate)

    phrase_followup_draft(
        case_id="case_123",
        category="BRAKES",
        severity="high",
        description="My email is alice@example.com and phone is 555-123-4567 please call me.",
    )

    assert "alice@example.com" not in captured["user"]
    assert "555-123-4567" not in captured["user"]
    assert "[EMAIL]" in captured["user"]
    assert "[PHONE]" in captured["user"]


def test_narrate_boundary_redacts_pii(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    def _fake_chat(system: str, user: str, model: str) -> str:
        captured["system"] = system
        captured["user"] = user
        return "Clean response"

    monkeypatch.setattr("src.ai.provider.llm_enabled", lambda: True)
    monkeypatch.setattr("src.ai.provider.can_spend", lambda: (True, "ok"))
    monkeypatch.setattr("src.ai.provider._http_chat", _fake_chat)

    res = narrate(
        site="intake_phrasing",
        system="System prompt with bob@test.com",
        user="User prompt with 800-555-0199 and SSN 123-45-6789",
        fallback="fallback",
    )

    assert res.ok is True
    assert "bob@test.com" not in captured["system"]
    assert "[EMAIL]" in captured["system"]
    assert "800-555-0199" not in captured["user"]
    assert "[PHONE]" in captured["user"]
    assert "123-45-6789" not in captured["user"]
    assert "[SSN]" in captured["user"]
