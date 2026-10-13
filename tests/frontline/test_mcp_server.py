"""Read-only MCP tools. No network and no warehouse writes."""

from __future__ import annotations

import io
import json
from datetime import datetime

def _open_local(monkeypatch) -> None:
    monkeypatch.delenv("FRONTLINE_API_KEY", raising=False)
    monkeypatch.delenv("FRONTLINE_AUTH_REQUIRED", raising=False)
    monkeypatch.setenv("FRONTLINE_OPEN_MODE", "1")
    monkeypatch.setenv("FRONTLINE_OPEN_BIND_ACK", "1")
    from src.api.auth import is_open_mode

    if not is_open_mode():
        monkeypatch.setenv("FRONTLINE_API_KEY", "local-demo-key")


def test_list_packs_includes_the_automotive_pack():
    from src.mcp_server.tools import call_tool

    packs = call_tool("list_packs")["packs"]
    assert "automotive_nhtsa" in packs


def test_list_spikes_uses_current_spikes(monkeypatch):
    seen = {}

    def fake(pack_id):
        seen["pack_id"] = pack_id
        return {"spike_week": None, "spikes": [], "censored_weeks": ["2026-W01"]}

    monkeypatch.setattr("src.ml_runtime.anomalies.current_spikes", fake)
    from src.mcp_server.tools import call_tool

    out = call_tool("list_spikes", {})
    assert seen["pack_id"] == "automotive_nhtsa"
    assert out["spikes"] == []
    assert out["censored_weeks"] == ["2026-W01"]


def test_list_investigations_filters_pack_and_status(monkeypatch):
    def fake(window_days=30):
        assert window_days == 90
        return [
            {
                "investigation_id": "inv_a",
                "pack_id": "automotive_nhtsa",
                "status": "open",
                "opened_at": datetime(2026, 1, 2, 3, 4, 5),
            },
            {
                "investigation_id": "inv_b",
                "pack_id": "automotive_nhtsa",
                "status": "closed",
                "opened_at": datetime(2026, 1, 3),
            },
            {
                "investigation_id": "inv_c",
                "pack_id": "finance_cfpb",
                "status": "open",
                "opened_at": datetime(2026, 1, 4),
            },
        ]

    monkeypatch.setattr("src.qubot.retrievers.investigation_status", fake)
    from src.mcp_server.tools import call_tool

    out = call_tool(
        "list_investigations",
        {"pack_id": "automotive_nhtsa", "status": "open"},
    )
    assert [row["investigation_id"] for row in out["investigations"]] == ["inv_a"]
    assert out["investigations"][0]["opened_at"] == "2026-01-02T03:04:05"


def test_brief_uses_the_http_builder(monkeypatch):
    def fake(investigation_id):
        return {"investigation_id": investigation_id, "pack_id": "automotive_nhtsa"}

    monkeypatch.setattr(
        "src.frontline.investigation_brief.build_investigation_brief",
        fake,
    )
    from src.mcp_server.tools import call_tool

    out = call_tool("get_investigation_brief", {"investigation_id": "inv_demo"})
    assert out == {"investigation_id": "inv_demo", "pack_id": "automotive_nhtsa"}


def test_missing_brief_is_a_tool_error(monkeypatch):
    def missing(_investigation_id):
        raise LookupError("investigation not found: inv_missing")

    monkeypatch.setattr(
        "src.frontline.investigation_brief.build_investigation_brief",
        missing,
    )
    _open_local(monkeypatch)
    from src.mcp_server.protocol import handle_message

    reply = handle_message(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "get_investigation_brief",
                "arguments": {"investigation_id": "inv_missing"},
            },
        }
    )
    assert reply["result"]["isError"] is True
    assert "inv_missing" in reply["result"]["content"][0]["text"]


def test_recall_summary_reads_markdown_and_skips_other_packs(tmp_path):
    from src.mcp_server.tools import call_tool, recall_lead_summary

    sample = recall_lead_summary("recall_lead_sample")
    assert sample["hits"] == 2
    assert sample["misses"] == 1
    assert sample["unconfirmed"] == 3
    assert sample["report_path"].endswith("recall_lead_time.md")

    (tmp_path / "other.md").write_text(
        "Pack `other_pack`. Hits 1. Misses 4. Unconfirmed flags 0.\n",
        encoding="utf-8",
    )
    other = recall_lead_summary("other_pack", tmp_path)
    assert other["hits"] == 1
    missing = recall_lead_summary("missing_pack", tmp_path)
    assert missing["hits"] is None
    assert "no lead-time report" in missing["note"]
    # The public tool does not accept a caller-chosen directory.
    public = call_tool("get_recall_lead_summary", {"pack_id": "recall_lead_sample"})
    assert public["hits"] == 2


def test_stdio_lists_spikes_and_stays_closed_without_a_key(monkeypatch):
    from src.mcp_server.protocol import handle_message, read_message, serve, write_message

    monkeypatch.setattr(
        "src.ml_runtime.anomalies.current_spikes",
        lambda pack_id: {"spike_week": None, "spikes": [], "censored_weeks": []},
    )
    _open_local(monkeypatch)
    inbound = io.BytesIO()
    write_message(
        inbound,
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "list_spikes", "arguments": {}},
        },
    )
    inbound.seek(0)
    outbound = io.BytesIO()
    serve(inbound, outbound)
    outbound.seek(0)
    reply = read_message(outbound)
    body = json.loads(reply["result"]["content"][0]["text"])
    assert reply["result"]["isError"] is False
    assert body["spikes"] == []

    monkeypatch.setenv("FRONTLINE_AUTH_REQUIRED", "1")
    monkeypatch.delenv("FRONTLINE_OPEN_MODE", raising=False)
    monkeypatch.delenv("FRONTLINE_API_KEY", raising=False)
    closed = handle_message({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert closed["error"]["code"] == -32001


def test_unknown_tool_is_an_error_result(monkeypatch):
    _open_local(monkeypatch)
    from src.mcp_server.protocol import handle_message

    reply = handle_message(
        {
            "jsonrpc": "2.0",
            "id": 9,
            "method": "tools/call",
            "params": {"name": "delete_pack", "arguments": {}},
        }
    )
    assert reply["result"]["isError"] is True
