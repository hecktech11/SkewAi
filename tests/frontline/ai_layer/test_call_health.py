"""Meta call health: latency, success/failure, fallback per call type.  [AI-LAYER]

Every Meta call is faked (FakeMeta transport). No key, no network.
"""

from __future__ import annotations

import pytest

from src.ai_layer import call_health
from src.ai_layer.severity import SEVERITY_SCHEMA

OK_SEV = {"severity": "Critical", "confidence": 0.9, "reason": "r", "evidence": "brakes failed"}


def _chat(name="severity"):
    from src.ai_layer import meta_client

    return meta_client.chat_json(system="s", user="u", schema=SEVERITY_SCHEMA, schema_name=name)


def test_no_key_reports_zero_and_not_run(no_meta, reset_ops_db):
    from src.ai_layer.meta_client import MetaError

    s = call_health.summary()
    assert s["status"] == "not run" and s["totals"]["calls"] == 0
    assert all(v["calls"] == 0 and v["status"] == "not run" for v in s["by_type"].values())
    for t in ("intake", "severity", "vision", "brief", "video"):
        assert t in s["by_type"]
    with pytest.raises(MetaError):
        _chat()  # meta_disabled: never left the process -> not recorded
    assert call_health.summary()["totals"]["calls"] == 0
    assert no_meta == []


def test_adapter_entry_points_are_wrapped_transparently():
    from src.ai_layer import evidence, meta_client, severity

    assert getattr(meta_client.chat_json, "_ai_call_health", False)
    assert getattr(meta_client.sam_segment, "_ai_call_health", False)
    # modules that imported the names see the same wrapped objects
    assert severity.chat_json is meta_client.chat_json
    assert evidence.sam_segment is meta_client.sam_segment
    assert meta_client.chat_json.__name__ == "chat_json"


def test_success_is_recorded_with_latency(fake_meta, reset_ops_db):
    fake_meta.chat["severity"] = lambda p: OK_SEV
    assert _chat()["severity"] == "Critical"
    s = call_health.summary()
    sev = s["by_type"]["severity"]
    assert (sev["calls"], sev["ok"], sev["failed"], sev["fallback_to_rules"]) == (1, 1, 0, 0)
    assert sev["latency_ms"]["n"] == 1 and sev["latency_ms"]["p50"] >= 0
    assert s["status"] == "recorded" and s["by_type"]["intake"]["calls"] == 0


def test_http_failure_counts_as_fallback(fake_meta, reset_ops_db):
    from src.ai_layer.meta_client import MetaError

    fake_meta.status_queue = [500, 500]
    with pytest.raises(MetaError):
        _chat()
    sev = call_health.summary()["by_type"]["severity"]
    assert (sev["calls"], sev["ok"], sev["failed"], sev["fallback_to_rules"]) == (1, 0, 1, 1)
    assert sev["last_error"] == "meta_http_500"


def test_timeout_is_counted(fake_meta, reset_ops_db):
    from src.ai_layer.meta_client import MetaError

    fake_meta.raise_exc = TimeoutError("slow")
    with pytest.raises(MetaError):
        _chat()
    sev = call_health.summary()["by_type"]["severity"]
    assert sev["failed"] == 1 and sev["timeouts"] == 1 and "TimeoutError" in sev["last_error"]


def test_schema_invalid_output_is_a_failure(fake_meta, reset_ops_db):
    from src.ai_layer.meta_client import MetaError

    fake_meta.chat["severity"] = lambda p: {"severity": "Apocalyptic"}
    with pytest.raises(MetaError):
        _chat()
    sev = call_health.summary()["by_type"]["severity"]
    assert sev["failed"] == 1 and sev["last_error"].startswith("meta_schema_invalid")


@pytest.mark.asyncio
async def test_real_caller_falls_back_and_both_counters_move(fake_meta, orchestrator_factory):
    from src.agents.intake import IntakeAgent
    from src.agents.triage import TriageAgent
    from src.ai_layer.severity import maybe_raise_severity

    text = "my 2019 Honda Civic headlight goes dark at night"
    orch, _ = orchestrator_factory()
    await orch.start()
    orch.ctx.record_turn("customer", text)
    await IntakeAgent(orch.ctx).run(customer_turn=text)
    orch.ctx.slots.setdefault("category", "EXTERIOR LIGHTING")
    orch.ctx.slots.setdefault("description", text)
    tres = await TriageAgent(orch.ctx).run()
    rule = orch.ctx.severity
    fake_meta.status_queue = [503, 503]
    out = await maybe_raise_severity(orch.ctx, tres)
    assert out is tres and orch.ctx.severity == rule  # fell back to rules, unchanged
    s = call_health.summary()
    assert s["by_type"]["severity"]["fallback_to_rules"] == 1
    assert s["caller_fallback_events_by_kind"].get("severity") == 1


def test_classify():
    assert call_health.classify("chat", {"schema_name": "intake_slots"}) == "intake"
    assert call_health.classify("chat", {"schema_name": "normalize"}) == "normalize"
    assert call_health.classify("chat", {"schema_name": "grounded_brief"}) == "brief"
    assert call_health.classify("chat", {"schema_name": "evidence_vision", "images": ["x"]}) == "vision"
    assert call_health.classify("chat", {"schema_name": "video_evidence", "images": ["x"]}) == "video"
    assert call_health.classify("chat", {"schema_name": "narration_brief"}) == "narration"
    assert call_health.classify("chat", {"schema_name": "zzz", "images": ["x"]}) == "vision"
    assert call_health.classify("sam", {}) == "segmentation"


def test_record_failure_never_raises(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("db down")

    monkeypatch.setattr(call_health, "ops_con", boom)
    assert call_health.record_call(call_type="severity", ok=True, latency_ms=1.0) is None
    assert call_health.summary()["status"] == "unavailable"


@pytest.fixture
def client(reset_ops_db, monkeypatch):
    from fastapi.testclient import TestClient

    from src.api.main import app

    monkeypatch.delenv("FRONTLINE_API_KEY", raising=False)
    monkeypatch.setenv("FRONTLINE_ENABLED", "1")
    with TestClient(app) as c:
        yield c


def test_routes_no_key(no_meta, client):
    r = client.get("/api/ai/call-health")
    assert r.status_code == 200 and r.json()["status"] == "not run" and r.json()["totals"]["calls"] == 0
    o = client.get("/api/ai/status/overview").json()
    assert o["status"]["mode"] == "deterministic"
    assert o["call_health"]["totals"]["calls"] == 0
    assert o["raise_reviews"]["reviewed"] == 0 and o["raise_reviews"]["agreement_rate"] is None
