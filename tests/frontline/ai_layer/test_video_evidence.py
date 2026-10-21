"""Short video evidence: container sniffing, duration/size limits, storage,
flag-gated (unverified) analysis, DSR erase.  [AI-LAYER]"""

from __future__ import annotations

import struct

import pytest

from tests.frontline.ai_layer.test_evidence import _insert_case


def _box(kind: bytes, body: bytes) -> bytes:
    return struct.pack(">I", 8 + len(body)) + kind + body


def mp4(seconds: float, brand: bytes = b"isom", v1: bool = False) -> bytes:
    ts = 1000
    if v1:
        mvhd = bytes([1, 0, 0, 0]) + b"\0" * 16 + struct.pack(">IQ", ts, int(seconds * ts)) + b"\0" * 80
    else:
        mvhd = bytes(4) + b"\0" * 8 + struct.pack(">II", ts, int(seconds * ts)) + b"\0" * 80
    return _box(b"ftyp", brand + b"\0\0\0\0" + brand) + _box(b"moov", _box(b"mvhd", mvhd)) + _box(b"mdat", b"\0" * 64)


def _el(eid: int, body: bytes) -> bytes:
    idb = eid.to_bytes((eid.bit_length() + 7) // 8, "big")
    return idb + bytes([0x80 | 0]) [:0] + (0x01 << 56 | len(body)).to_bytes(8, "big") + body


def webm(seconds: float | None) -> bytes:
    hdr = _el(0x1A45DFA3, _el(0x4282, b"webm"))
    info = _el(0x2AD7B1, (1_000_000).to_bytes(3, "big"))
    if seconds is not None:
        info += _el(0x4489, struct.pack(">d", seconds * 1000))
    return hdr + _el(0x18538067, _el(0x1549A966, info) + _el(0x1F43B675, b"\0" * 16))


def test_validate_containers_and_limits(monkeypatch):
    from src.ai_layer.video_evidence import VideoError, validate_video

    assert validate_video(mp4(7.5))["duration_s"] == 7.5
    assert validate_video(mp4(6, v1=True))["duration_s"] == 6
    assert validate_video(mp4(5, brand=b"qt  "))["mime"] == "video/quicktime"
    assert validate_video(webm(8.0)) == {"mime": "video/webm", "ext": "webm", "duration_s": 8.0, "declared_mime_matches": True}
    for data, code in ((mp4(12), 422), (mp4(0.2), 422), (webm(None), 422), (b"\x89PNG\r\n\x1a\n" + b"0" * 40, 415), (b"", 400)):
        with pytest.raises(VideoError) as e:
            validate_video(data)
        assert e.value.status == code
    monkeypatch.setenv("AI_VIDEO_MAX_BYTES", str(64 * 1024))
    with pytest.raises(VideoError) as e:
        validate_video(mp4(5) + b"\0" * (70 * 1024))
    assert e.value.status == 413


def test_store_without_analysis_and_erase(no_meta, reset_ops_db, monkeypatch, tmp_path):
    from pathlib import Path

    from src.ai_layer.store import erase_interaction
    from src.ai_layer.video_evidence import analyze_video, list_for_case, store_video
    from src.data.warehouse import ops_con

    monkeypatch.setenv("AI_EVIDENCE_DIR", str(tmp_path / "ev"))
    monkeypatch.setenv("AI_VIDEO_ANALYSIS", "1")  # flag on but no key -> still no analysis
    cid, iid = _insert_case()
    v = store_video(data=mp4(6), filename="../clip.mp4", declared_mime="video/mp4", case_id=cid)
    assert v["analysis_status"] == "stored_no_analysis" and v["duration_s"] == 6 and "unverified" in v["analysis_note"]
    assert "storage_path" not in v and list_for_case(cid)[0]["video_id"] == v["video_id"]
    assert analyze_video(v["video_id"])["analysis_status"] == "stored_no_analysis"
    assert no_meta == []
    files = list((tmp_path / "ev" / "video").rglob("*.mp4"))
    assert len(files) == 1
    with ops_con() as con:
        out = erase_interaction(con, iid)
    assert out["case_video_evidence"] == 1 and not Path(files[0]).exists()


def test_flagged_analysis_with_fake_meta(fake_meta, reset_ops_db, monkeypatch, tmp_path):
    from src.ai_layer.video_evidence import store_video, analyze_video

    monkeypatch.setenv("AI_EVIDENCE_DIR", str(tmp_path / "ev"))
    cid, _ = _insert_case()
    seen = {}

    def _reply(payload):
        seen["p"] = payload
        return {"observed": "smoke from wheel", "component": "SERVICE BRAKES", "visible_hazards": ["smoke"], "confidence": 0.7}

    fake_meta.chat["video_evidence"] = _reply
    v = store_video(data=mp4(6), filename="c.mp4", declared_mime="video/mp4", case_id=cid)
    assert v["analysis_status"] == "stored_no_analysis"  # flag off
    monkeypatch.setenv("AI_VIDEO_ANALYSIS", "1")
    out = analyze_video(v["video_id"])
    assert out["analysis_status"] == "done" and out["analysis"]["note"].startswith("unverified")
    assert "data:video/mp4;base64," in str(seen["p"])


def test_video_routes(reset_ops_db, monkeypatch, tmp_path, no_meta):
    from fastapi.testclient import TestClient

    from src.api.main import app

    monkeypatch.delenv("FRONTLINE_API_KEY", raising=False)
    monkeypatch.setenv("AI_EVIDENCE_DIR", str(tmp_path / "ev"))
    cid, _ = _insert_case()
    c = TestClient(app)
    r = c.post(f"/api/ai/cases/{cid}/video", files={"file": ("c.mp4", mp4(5), "video/mp4")})
    assert r.status_code == 201, r.text
    vid = r.json()["video_id"]
    assert c.get(f"/api/ai/video/{vid}/file").headers["content-type"] == "video/mp4"
    assert c.get(f"/api/ai/cases/{cid}/video").json()["videos"][0]["video_id"] == vid
    assert c.post(f"/api/ai/cases/{cid}/video", files={"file": ("c.mp4", mp4(30), "video/mp4")}).status_code == 422
    assert c.post("/api/ai/cases/case_nope01/video", files={"file": ("c.mp4", mp4(5), "video/mp4")}).status_code == 404
