"""Live smoke script: no key -> no network; fake Meta -> each check reported.  [AI-LAYER]"""

from __future__ import annotations


def test_smoke_skips_without_key(no_meta, capsys):
    from scripts.ai_meta_smoke import main

    assert main([]) == 0
    out = capsys.readouterr().out
    assert "skipped" in out and "checks" in out


def test_smoke_reports_each_check(fake_meta):
    from scripts.ai_meta_smoke import run_checks

    fake_meta.chat["smoke_normalize"] = lambda p: {"normalized_en": "My car's brakes failed.", "confidence": 0.9}
    fake_meta.chat["smoke_vision"] = lambda p: {"description": "A brake disc.", "confidence": 0.8}
    fake_meta.sam = lambda p: "0<|box;x1=10;y1=10;x2=50;y2=40;w=240;h=180|>"
    out = run_checks()
    names = [c["check"].split(":")[0] for c in out["checks"]]
    assert names == ["text", "vision", "segmentation"]
    assert out["all_ok"] is True
    assert all("key" not in str(c).lower() or "keys" in str(c) for c in out["checks"])


def test_smoke_reports_failure_reason(fake_meta):
    from scripts.ai_meta_smoke import run_checks

    fake_meta.status_queue = [401, 401, 401, 401, 401, 401]
    out = run_checks()
    assert out["all_ok"] is False
    assert all(not c["ok"] and c.get("reason") for c in out["checks"])
