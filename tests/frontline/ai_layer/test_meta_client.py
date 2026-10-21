"""Meta adapter: config, allowlist, retries, schema validation, SAM parsing.  [AI-LAYER]"""

from __future__ import annotations

import pytest

from src.ai_layer import config, meta_client
from src.ai_layer.meta_client import MetaError, chat_json, parse_sam_output, sam_segment
from src.ai_layer.schema import validate

SCHEMA = {
    "type": "object",
    "properties": {"x": {"type": "integer", "minimum": 0}, "tag": {"type": "string", "enum": ["a", "b"]}},
    "required": ["x"],
}


def test_mode_deterministic_without_key(no_meta, monkeypatch):
    assert config.ai_mode() == "deterministic"
    st = config.ai_status()
    assert st["mode"] == "deterministic"
    assert st["key_configured"] is False
    assert st["features"]["photo_evidence_upload"] is True
    assert st["features"]["photo_evidence_analysis"] is False
    assert st["verified_against_live_api"] is False
    with pytest.raises(MetaError) as e:
        chat_json(system="s", user="u", schema=SCHEMA, schema_name="t")
    assert e.value.reason == "meta_disabled"
    assert no_meta == []


def test_mode_enhanced_with_key_and_switch_off(fake_meta, monkeypatch):
    assert config.ai_mode() == "enhanced"
    monkeypatch.setenv("FRONTLINE_AI_LAYER", "off")
    assert config.ai_mode() == "deterministic"


def test_model_api_key_alias(monkeypatch):
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.setenv("MODEL_API_KEY", "placeholder")
    assert config.meta_enabled()


def test_status_never_contains_key(fake_meta):
    import json

    assert "test-placeholder" not in json.dumps(config.ai_status())


def test_chat_json_happy_path_and_payload_shape(fake_meta):
    fake_meta.chat["t"] = lambda p: {"x": 3, "tag": "a"}
    out = chat_json(system="sys", user="call me at 555-123-4567", schema=SCHEMA, schema_name="t")
    assert out == {"x": 3, "tag": "a"}
    call = fake_meta.calls[0]
    assert call["url"] == "https://api.meta.ai/v1/chat/completions"
    assert call["headers"]["Authorization"].startswith("Bearer ")
    assert call["payload"]["model"] == "muse-spark-1.3"
    assert call["payload"]["response_format"]["type"] == "json_schema"
    # PII is redacted before leaving the process
    assert "555-123-4567" not in call["payload"]["messages"][1]["content"]


def test_chat_json_strips_code_fences(fake_meta):
    fake_meta.chat["t"] = lambda p: "```json\n{\"x\": 1}\n```"
    assert chat_json(system="s", user="u", schema=SCHEMA, schema_name="t") == {"x": 1}


@pytest.mark.parametrize("bad", [{"x": -1}, {"tag": "a"}, {"x": "1"}, "not json", {"x": 1, "tag": "z"}])
def test_chat_json_rejects_schema_invalid(fake_meta, bad):
    fake_meta.chat["t"] = lambda p: bad
    with pytest.raises(MetaError):
        chat_json(system="s", user="u", schema=SCHEMA, schema_name="t")


def test_retry_on_5xx_then_success(fake_meta):
    fake_meta.chat["t"] = lambda p: {"x": 2}
    fake_meta.status_queue = [503]
    assert chat_json(system="s", user="u", schema=SCHEMA, schema_name="t") == {"x": 2}
    assert len(fake_meta.calls) == 2


def test_no_retry_on_4xx(fake_meta):
    fake_meta.chat["t"] = lambda p: {"x": 2}
    fake_meta.status_queue = [400]
    with pytest.raises(MetaError) as e:
        chat_json(system="s", user="u", schema=SCHEMA, schema_name="t")
    assert e.value.reason == "meta_http_400"
    assert len(fake_meta.calls) == 1


def test_network_error_exhausts_retries(fake_meta):
    fake_meta.raise_exc = TimeoutError("slow")
    with pytest.raises(MetaError) as e:
        chat_json(system="s", user="u", schema=SCHEMA, schema_name="t")
    assert e.value.reason.startswith("meta_network")
    assert len(fake_meta.calls) == 2  # 1 + META_RETRIES=1


def test_host_allowlist_blocks_key_exfiltration(fake_meta, monkeypatch):
    monkeypatch.setenv("META_API_BASE", "https://evil.example.com/v1")
    with pytest.raises(MetaError) as e:
        chat_json(system="s", user="u", schema=SCHEMA, schema_name="t")
    assert "not_allowlisted" in e.value.reason
    assert fake_meta.calls == []
    monkeypatch.setenv("META_ALLOWED_HOSTS", "evil.example.com")
    fake_meta.chat["t"] = lambda p: {"x": 1}
    assert chat_json(system="s", user="u", schema=SCHEMA, schema_name="t") == {"x": 1}


def test_http_base_refused(fake_meta, monkeypatch):
    monkeypatch.setenv("META_API_BASE", "http://api.meta.ai/v1")
    with pytest.raises(MetaError) as e:
        chat_json(system="s", user="u", schema=SCHEMA, schema_name="t")
    assert e.value.reason == "meta_url_must_be_https"


def test_private_ip_blocked_even_if_allowlisted(fake_meta, monkeypatch):
    monkeypatch.setenv("META_API_BASE", "https://169.254.169.254/v1")
    monkeypatch.setenv("META_ALLOWED_HOSTS", "169.254.169.254")
    with pytest.raises(MetaError) as e:
        chat_json(system="s", user="u", schema=SCHEMA, schema_name="t")
    assert e.value.reason.startswith("meta_url_blocked")


def test_daily_cap(fake_meta, monkeypatch):
    monkeypatch.setenv("META_DAILY_CALL_CAP", "1")
    fake_meta.chat["t"] = lambda p: {"x": 1}
    chat_json(system="s", user="u", schema=SCHEMA, schema_name="t")
    with pytest.raises(MetaError) as e:
        chat_json(system="s", user="u", schema=SCHEMA, schema_name="t")
    assert e.value.reason == "daily_call_cap"


def test_parse_sam_output_boxes_and_pct():
    text = (
        "<0f>0<|box;x1=181;y1=365;x2=556;y2=486;w=1516;h=1600|><|mask;x=0;y=0;data=122,376,~|>,"
        "1<|box;x1=10;y1=20;x2=30;y2=40;w=100;h=200|><|mask;x=0;y=0;data=1~|>"
    )
    boxes = parse_sam_output(text)
    assert len(boxes) == 2
    assert boxes[0]["x1"] == 181 and boxes[0]["has_mask"]
    assert boxes[1]["pct"] == {"left": 10.0, "top": 10.0, "width": 20.0, "height": 10.0}
    assert parse_sam_output("") == []


def test_sam_segment_request_shape(fake_meta):
    fake_meta.sam = lambda p: "<0f>0<|box;x1=1;y1=2;x2=3;y2=4;w=10;h=10|>"
    boxes = sam_segment(concept="brake disc; rm -rf", image_data_url="data:image/png;base64,AAAA")
    assert len(boxes) == 1
    call = fake_meta.calls[0]
    assert call["url"].endswith("/v1/responses")
    assert call["payload"]["model"] == "sam-3.1"
    content = call["payload"]["input"][0]["content"]
    assert content[0] == {"type": "input_text", "text": "brake disc rm -rf"}
    assert content[1]["type"] == "input_image"
    assert call["headers"]["Accept"] == "text/event-stream"


def test_meta_narrate_falls_back(no_meta):
    res = meta_client.meta_narrate(site="x", system="s", user="u", fallback="template")
    assert res.text == "template" and res.ok is False and res.used_llm is False


def test_meta_narrate_rejects_invented_citation(fake_meta):
    fake_meta.chat["narration_x"] = lambda p: {"text": "See rec_FAKE123 for details."}
    res = meta_client.meta_narrate(site="x", system="s", user="u", fallback="template")
    assert res.text == "template" and res.reason == "narration_violation"


def test_schema_validator_basics():
    assert validate({"x": 1}, SCHEMA) == []
    assert validate({"x": True}, SCHEMA)  # bool is not integer
    assert validate([1, 2], {"type": "array", "maxItems": 1})
    assert validate({"a": 1}, {"type": "object", "additionalProperties": False, "properties": {}})
