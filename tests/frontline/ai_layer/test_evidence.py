"""Photo evidence: upload validation, local storage, case/cluster attach,
SAM 3.1 + Muse Spark analysis (faked), and no-key behaviour.  [AI-LAYER]"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from tests.frontline.ai_layer.conftest import tiny_jpeg, tiny_png


def _insert_case(case_id="case_ai_test01", interaction_id="int_ai_test01", cluster=7, inv="inv_0042",
                 category="SERVICE BRAKES"):
    from src.data.warehouse import ops_con

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    with ops_con() as con:
        con.execute(
            "INSERT INTO interactions (interaction_id, pack_id, pack_version, started_at, channel, status) "
            "VALUES (?, 'automotive_nhtsa', 'v', ?, 'web_text', 'closed')",
            [interaction_id, now],
        )
        con.execute(
            "INSERT INTO cases (case_id, interaction_id, pack_id, created_at, category, description_summary, "
            "severity, severity_source, priority, cluster_match_id, investigation_id) "
            "VALUES (?, ?, 'automotive_nhtsa', ?, ?, 'brake grinding noise', 'Medium', 'rules', 2, ?, ?)",
            [case_id, interaction_id, now, category, cluster, inv],
        )
    return case_id, interaction_id


@pytest.fixture
def client(reset_ops_db, monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from src.api.main import app

    monkeypatch.delenv("FRONTLINE_API_KEY", raising=False)
    monkeypatch.setenv("FRONTLINE_ENABLED", "1")
    monkeypatch.setenv("AI_EVIDENCE_DIR", str(tmp_path / "evidence"))
    with TestClient(app) as c:
        yield c


def test_validate_image_rules(monkeypatch):
    from src.ai_layer.evidence import EvidenceError, validate_image

    assert validate_image(tiny_png(4, 3))["width"] == 4
    meta = validate_image(tiny_jpeg(16, 8))
    assert (meta["mime"], meta["width"], meta["height"]) == ("image/jpeg", 16, 8)
    with pytest.raises(EvidenceError) as e:
        validate_image(b"GIF89a....")
    assert e.value.status == 415
    with pytest.raises(EvidenceError) as e:
        validate_image(b"<svg xmlns='http://www.w3.org/2000/svg'/>", "image/svg+xml")
    assert e.value.status == 415
    with pytest.raises(EvidenceError) as e:
        validate_image(tiny_png(), "application/pdf")
    assert e.value.status == 415
    with pytest.raises(EvidenceError) as e:
        validate_image(b"")
    assert e.value.status == 400
    monkeypatch.setenv("AI_EVIDENCE_MAX_BYTES", "1024")
    with pytest.raises(EvidenceError) as e:
        validate_image(tiny_png() + b"\x00" * 2000)
    assert e.value.status == 413
    with pytest.raises(EvidenceError):
        validate_image(b"\x89PNG\r\n\x1a\n" + b"\x00" * 10)  # truncated header


def test_upload_without_key_stores_and_attaches(no_meta, client, tmp_path):
    case_id, iid = _insert_case()
    r = client.post(f"/api/cases/{case_id}/evidence", files={"file": ("../../etc/brake.png", tiny_png(), "image/png")})
    assert r.status_code == 201, r.text
    ev = r.json()
    assert ev["analysis_status"] == "not_run"
    assert ev["case_id"] == case_id and ev["cluster_id"] == 7 and ev["investigation_id"] == "inv_0042"
    assert ev["filename"] == "brake.png"  # path components stripped
    assert "storage_path" not in ev
    files = list((tmp_path / "evidence").rglob("*.png"))
    assert len(files) == 1 and files[0].parent.name == case_id
    lst = client.get(f"/api/cases/{case_id}/evidence").json()["evidence"]
    assert [e["evidence_id"] for e in lst] == [ev["evidence_id"]]
    f = client.get(ev["file_url"])
    assert f.status_code == 200 and f.content == tiny_png() and f.headers["content-type"] == "image/png"
    cl = client.get("/api/ai/clusters/7/evidence", params={"pack_id": "automotive_nhtsa"}).json()
    assert len(cl["evidence"]) == 1
    inv = client.get("/api/ai/investigations/inv_0042/evidence").json()
    assert len(inv["evidence"]) == 1
    an = client.post(f"/api/evidence/{ev['evidence_id']}/analyze").json()
    assert an["analysis_status"] == "not_run" and an["analysis"] is None
    assert no_meta == []


def test_upload_rejections(no_meta, client):
    case_id, _ = _insert_case()
    assert client.post("/api/cases/case_missing01/evidence",
                       files={"file": ("a.png", tiny_png(), "image/png")}).status_code == 404
    assert client.post(f"/api/cases/{case_id}/evidence",
                       files={"file": ("a.gif", b"GIF89a" + b"0" * 20, "image/gif")}).status_code == 415
    assert client.post(f"/api/cases/{case_id}/evidence",
                       files={"file": ("a.html", b"<html></html>", "text/html")}).status_code == 415
    assert client.post("/api/cases/bad..id/evidence",
                       files={"file": ("a.png", tiny_png(), "image/png")}).status_code in (400, 404, 422)
    assert client.get("/api/evidence/evd_nope/file").status_code == 404


def test_mid_call_upload_attaches_to_case_later(no_meta, client):
    from src.data.warehouse import ops_con

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    with ops_con() as con:
        con.execute(
            "INSERT INTO interactions (interaction_id, pack_id, pack_version, started_at, channel, status) "
            "VALUES ('int_live_0001', 'automotive_nhtsa', 'v', ?, 'web_text', 'closed')", [now])
    r = client.post("/api/interactions/int_live_0001/evidence",
                    files={"file": ("p.jpg", tiny_jpeg(), "image/jpeg")})
    assert r.status_code == 201, r.text
    ev = r.json()
    assert ev["case_id"] is None
    with ops_con() as con:
        con.execute(
            "INSERT INTO cases (case_id, interaction_id, pack_id, created_at, category, severity, "
            "severity_source, priority, cluster_match_id) VALUES ('case_live_0001', 'int_live_0001', "
            "'automotive_nhtsa', ?, 'TIRES', 'Low', 'rules', 3, 3)", [now])
    got = client.get(f"/api/evidence/{ev['evidence_id']}").json()
    assert got["case_id"] == "case_live_0001" and got["cluster_id"] == 3
    lst = client.get("/api/cases/case_live_0001/evidence").json()["evidence"]
    assert len(lst) == 1
    assert len(client.get("/api/ai/clusters/3/evidence").json()["evidence"]) == 1
    assert client.post("/api/interactions/int_unknown1/evidence",
                       files={"file": ("p.jpg", tiny_jpeg(), "image/jpeg")}).status_code == 404


def test_analysis_with_fake_meta_maps_category_and_boxes(fake_meta, reset_ops_db, monkeypatch, tmp_path):
    from src.ai_layer.evidence import analyze_evidence, store_evidence

    monkeypatch.setenv("AI_EVIDENCE_DIR", str(tmp_path / "ev"))
    case_id, _ = _insert_case()
    ev = store_evidence(data=tiny_png(100, 50), filename="disc.png", declared_mime="image/png", case_id=case_id)
    assert ev["analysis_status"] == "pending"
    fake_meta.chat["evidence_vision"] = lambda p: {
        "component_category": "SERVICE BRAKES", "confidence": 0.91,
        "defect_description": "Deep scoring on the brake disc surface", "damaged_part": "brake disc",
        "visible_hazard": True, "hazard_reason": "worn disc reduces braking"}
    fake_meta.sam = lambda p: "<0f>0<|box;x1=10;y1=5;x2=60;y2=45;w=100;h=50|><|mask;x=0;y=0;data=1~|>"
    out = analyze_evidence(ev["evidence_id"])
    assert out["analysis_status"] == "ok"
    assert out["mapped_category"] == "SERVICE BRAKES"
    a = out["analysis"]
    assert a["agrees_with_case_category"] is True
    assert a["review_flag"] is True
    assert a["segmentation"]["concept"] == "brake disc"
    assert a["segmentation"]["boxes"][0]["pct"] == {"left": 10.0, "top": 10.0, "width": 50.0, "height": 80.0}
    vision_call = fake_meta.chat_calls("evidence_vision")[0]
    parts = vision_call["payload"]["messages"][1]["content"]
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
    # The photo never changes the case
    from src.data.warehouse import ops_con

    with ops_con() as con:
        sev, cat = con.execute("SELECT severity, category FROM cases WHERE case_id = ?", [case_id]).fetchone()
    assert (sev, cat) == ("Medium", "SERVICE BRAKES")


def test_analysis_rejects_low_confidence_and_handles_sam_failure(fake_meta, reset_ops_db, monkeypatch, tmp_path):
    from src.ai_layer.evidence import analyze_evidence, store_evidence

    monkeypatch.setenv("AI_EVIDENCE_DIR", str(tmp_path / "ev"))
    case_id, _ = _insert_case()
    ev = store_evidence(data=tiny_png(), filename="x.png", declared_mime="image/png", case_id=case_id)
    fake_meta.chat["evidence_vision"] = lambda p: {
        "component_category": "TIRES", "confidence": 0.3, "defect_description": "maybe a tire",
        "damaged_part": "tire", "visible_hazard": False}
    fake_meta.sam = None

    def _sam_fail(p):
        raise OSError("sam down")

    out_ok = analyze_evidence(ev["evidence_id"])
    assert out_ok["mapped_category"] is None  # below confidence floor
    fake_meta.chat["evidence_vision"] = lambda p: "garbage"
    fake_meta.status_queue = []
    out = analyze_evidence(ev["evidence_id"])
    assert out["analysis_status"] in ("partial", "failed")
    assert any(e.startswith("vision:") for e in out["analysis"]["errors"])


def test_dsr_erase_removes_ai_rows_and_files(no_meta, reset_ops_db, monkeypatch, tmp_path):
    from pathlib import Path

    from src.ai_layer.evidence import get_evidence, store_evidence
    from src.data.warehouse import ops_con
    from src.frontline.dsr import delete_interaction

    monkeypatch.setenv("AI_EVIDENCE_DIR", str(tmp_path / "ev"))
    case_id, iid = _insert_case()
    ev = store_evidence(data=tiny_png(), filename="x.png", declared_mime="image/png", case_id=case_id)
    path = Path(get_evidence(ev["evidence_id"], include_path=True)["storage_path"])
    assert path.exists()
    res = delete_interaction(iid)
    assert res["deleted"].get("case_evidence") == 1
    assert not path.exists()
    with ops_con() as con:
        assert con.execute("SELECT COUNT(*) FROM case_evidence").fetchone()[0] == 0
