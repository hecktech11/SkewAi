"""Short video evidence (5–10 s clips) attached to a case.  [AI-LAYER]

Validation is strict and local: container sniffed from magic bytes (MP4 /
QuickTime MOV / WebM), size capped (``AI_VIDEO_MAX_BYTES``, default 20 MB) and
duration read from the container itself (MP4/MOV ``mvhd`` box, WebM
``Segment/Info/Duration``) and capped at ``AI_VIDEO_MAX_SECONDS`` (default 10).
A clip whose duration cannot be read is rejected rather than guessed.

Analysis is OFF by default. ``AI_VIDEO_ANALYSIS=1`` plus a Meta key sends the
clip to Muse Spark — **unverified: video input support not confirmed** in the
Meta Model API docs. Otherwise the clip is stored with
``analysis_status = "stored_no_analysis"``. Analysis output is informational
only and never changes severity or any existing decision.
"""

from __future__ import annotations

import base64
import hashlib
import os
import struct
from pathlib import Path
from typing import Any

from src.ai_layer import config
from src.data.timeutil import utc_now
from src.data.warehouse import ops_con
from src.ids import new_ulid

UNVERIFIED_NOTE = "unverified: video input support not confirmed"
ALLOWED = {"video/mp4": "mp4", "video/quicktime": "mov", "video/webm": "webm"}
MAX_PER_CASE = 3
MIN_SECONDS = 1.0

_DDL = """
CREATE TABLE IF NOT EXISTS case_video_evidence (
    video_id         VARCHAR PRIMARY KEY,
    case_id          VARCHAR NOT NULL,
    interaction_id   VARCHAR,
    pack_id          VARCHAR,
    filename         VARCHAR,
    mime             VARCHAR NOT NULL,
    size_bytes       INTEGER NOT NULL,
    duration_s       DOUBLE NOT NULL,
    sha256           VARCHAR NOT NULL,
    storage_path     VARCHAR NOT NULL,
    analysis_status  VARCHAR NOT NULL,
    analysis         JSON,
    created_at       TIMESTAMP NOT NULL
)
"""


class VideoError(Exception):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status, self.detail = status, detail


def max_bytes() -> int:
    return max(64 * 1024, config._env_int("AI_VIDEO_MAX_BYTES", 20 * 1024 * 1024))


def max_seconds() -> float:
    return float(max(1, config._env_int("AI_VIDEO_MAX_SECONDS", 10)))


def analysis_enabled() -> bool:
    return (os.getenv("AI_VIDEO_ANALYSIS") or "").strip().lower() in {"1", "true", "yes", "on"} and config.meta_enabled()


# ── Container sniffing / duration ──────────────────────────────────────────


def sniff(data: bytes) -> str | None:
    if len(data) >= 12 and data[4:8] == b"ftyp":
        return "video/quicktime" if data[8:12] == b"qt  " else "video/mp4"
    if data[:4] == b"\x1a\x45\xdf\xa3" and b"webm" in data[:64]:
        return "video/webm"
    return None


def _boxes(data: bytes, start: int, end: int):
    i = start
    while i + 8 <= end:
        size, kind = struct.unpack(">I4s", data[i:i + 8])
        hdr = 8
        if size == 1:
            if i + 16 > end:
                return
            size = struct.unpack(">Q", data[i + 8:i + 16])[0]
            hdr = 16
        elif size == 0:
            size = end - i
        if size < hdr or i + size > end:
            return
        yield kind, i + hdr, i + size
        i += size


def mp4_duration(data: bytes) -> float | None:
    for kind, s, e in _boxes(data, 0, len(data)):
        if kind != b"moov":
            continue
        for k2, s2, e2 in _boxes(data, s, e):
            if k2 != b"mvhd" or e2 - s2 < 20:
                continue
            ver = data[s2]
            if ver == 1 and e2 - s2 >= 32:
                timescale, dur = struct.unpack(">IQ", data[s2 + 20:s2 + 32])
            else:
                timescale, dur = struct.unpack(">II", data[s2 + 12:s2 + 20])
            return (dur / timescale) if timescale else None
    return None


def _vint(data: bytes, i: int, keep_marker: bool) -> tuple[int, int] | None:
    if i >= len(data) or data[i] == 0:
        return None
    first = data[i]
    n = 1
    while not (first & (0x80 >> (n - 1))):
        n += 1
    if i + n > len(data):
        return None
    val = first if keep_marker else first & (0xFF >> n)
    for b in data[i + 1:i + n]:
        val = (val << 8) | b
    return val, n


def _ebml(data: bytes, start: int, end: int):
    i = start
    while i < end:
        idv = _vint(data, i, True)
        if not idv:
            return
        sz = _vint(data, i + idv[1], False)
        if not sz:
            return
        body = i + idv[1] + sz[1]
        size = sz[0]
        unknown = size == (1 << (7 * sz[1])) - 1
        stop = end if unknown else min(end, body + size)
        yield idv[0], body, stop
        if unknown:
            return
        i = stop


def webm_duration(data: bytes) -> float | None:
    for eid, s, e in _ebml(data, 0, len(data)):
        if eid != 0x18538067:  # Segment
            continue
        for eid2, s2, e2 in _ebml(data, s, e):
            if eid2 != 0x1549A966:  # Info
                continue
            scale, dur = 1_000_000, None
            for eid3, s3, e3 in _ebml(data, s2, e2):
                if eid3 == 0x2AD7B1:
                    scale = int.from_bytes(data[s3:e3], "big") or scale
                elif eid3 == 0x4489:
                    raw = data[s3:e3]
                    dur = struct.unpack(">f", raw)[0] if len(raw) == 4 else struct.unpack(">d", raw)[0] if len(raw) == 8 else None
            return (dur * scale / 1e9) if dur else None
    return None


def validate_video(data: bytes, declared_mime: str | None = None) -> dict[str, Any]:
    if not data:
        raise VideoError(400, "empty file")
    if len(data) > max_bytes():
        raise VideoError(413, f"video too large (max {max_bytes()} bytes)")
    mime = sniff(data)
    if mime is None:
        raise VideoError(415, "unsupported video type (allowed: mp4, mov, webm)")
    dur = webm_duration(data) if mime == "video/webm" else mp4_duration(data)
    if dur is None:
        raise VideoError(422, "could not read the clip duration from the file; please upload an mp4/mov/webm with duration metadata")
    if dur > max_seconds():
        raise VideoError(422, f"clip is {dur:.1f}s; maximum is {max_seconds():.0f}s (aim for 5–10s)")
    if dur < MIN_SECONDS:
        raise VideoError(422, f"clip is {dur:.1f}s; minimum is {MIN_SECONDS:.0f}s")
    return {"mime": mime, "ext": ALLOWED[mime], "duration_s": round(dur, 2),
            "declared_mime_matches": (declared_mime or "").split(";")[0].strip().lower() in ("", mime)}


# ── Storage ─────────────────────────────────────────────────────────────────


def _dir() -> Path:
    from src.ai_layer.evidence import evidence_dir

    return evidence_dir() / "video"


def _public(row: dict[str, Any]) -> dict[str, Any]:
    import json

    out = {k: v for k, v in row.items() if k != "storage_path"}
    if isinstance(out.get("analysis"), str):
        try:
            out["analysis"] = json.loads(out["analysis"])
        except ValueError:
            pass
    if hasattr(out.get("created_at"), "isoformat"):
        out["created_at"] = out["created_at"].isoformat()
    out["file_url"] = f"/api/ai/video/{row['video_id']}/file"
    out["analysis_note"] = UNVERIFIED_NOTE
    return out


def _rows(con, where: str, args: list[Any]) -> list[dict[str, Any]]:
    cur = con.execute(f"SELECT * FROM case_video_evidence WHERE {where} ORDER BY created_at", args)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _exists(con) -> bool:
    return con.execute("SELECT 1 FROM information_schema.tables WHERE table_name = 'case_video_evidence'").fetchone() is not None


def store_video(*, data: bytes, filename: str | None, declared_mime: str | None, case_id: str) -> dict[str, Any]:
    from src.ai_layer.evidence import _safe_filename
    from src.security.identifiers import safe_token_id

    safe_token_id(case_id, kind="case_id")
    meta = validate_video(data, declared_mime)
    with ops_con() as con:
        case = con.execute("SELECT interaction_id, pack_id FROM cases WHERE case_id = ?", [case_id]).fetchone()
        if case is None:
            raise VideoError(404, "case not found")
        con.execute(_DDL)
        n = con.execute("SELECT COUNT(*) FROM case_video_evidence WHERE case_id = ?", [case_id]).fetchone()[0]
        if int(n) >= MAX_PER_CASE:
            raise VideoError(409, f"video limit reached ({MAX_PER_CASE} per case)")
        vid = "vid_" + new_ulid()
        folder = _dir() / case_id
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{vid}.{meta['ext']}"
        tmp = path.with_suffix(path.suffix + ".tmp")
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
        status = "pending" if analysis_enabled() else "stored_no_analysis"
        con.execute("INSERT INTO case_video_evidence VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [vid, case_id, case[0], case[1], _safe_filename(filename), meta["mime"], len(data), meta["duration_s"],
                     hashlib.sha256(data).hexdigest(), str(path), status, None, utc_now()])
    return get_video(vid)


def get_video(video_id: str, *, include_path: bool = False) -> dict[str, Any]:
    with ops_con() as con:
        rows = _rows(con, "video_id = ?", [video_id]) if _exists(con) else []
    if not rows:
        raise VideoError(404, "video not found")
    return rows[0] if include_path else _public(rows[0])


def list_for_case(case_id: str) -> list[dict[str, Any]]:
    with ops_con() as con:
        return [_public(r) for r in _rows(con, "case_id = ?", [case_id])] if _exists(con) else []


def _schema(categories: list[str]) -> dict[str, Any]:
    return {"type": "object", "additionalProperties": False,
            "required": ["observed", "component", "visible_hazards", "confidence"],
            "properties": {"observed": {"type": "string", "maxLength": 400},
                           "component": {"type": "string", "enum": categories + ["UNKNOWN"]},
                           "visible_hazards": {"type": "array", "items": {"type": "string", "maxLength": 80}, "maxItems": 6},
                           "confidence": {"type": "number", "minimum": 0, "maximum": 1}}}


def analyze_video(video_id: str) -> dict[str, Any]:
    """Muse Spark on the raw clip. Unverified API capability; informational only."""
    import json

    from src.ai_layer import meta_client
    from src.ai_layer.store import record_event

    row = get_video(video_id, include_path=True)
    if not analysis_enabled():
        return {**_public(row), "analysis_status": row["analysis_status"],
                "reason": "AI_VIDEO_ANALYSIS is off or no Meta key: stored without analysis"}
    from src.ai_layer.evidence import _pack_categories

    cats, _ = _pack_categories(row.get("pack_id"))
    url = f"data:{row['mime']};base64," + base64.b64encode(Path(row["storage_path"]).read_bytes()).decode()
    try:
        out = meta_client.chat_json(
            system="You inspect a short vehicle video clip submitted with a fault report. Describe only what is visible. "
            "Pick the closest component from the enum or UNKNOWN. Do not guess beyond the footage.",
            user="Describe the visible fault in this clip.", schema=_schema(cats), schema_name="video_evidence",
            images=[url], model=config.model_vision())
        status, analysis = "done", {**out, "note": UNVERIFIED_NOTE, "model": config.model_vision()}
    except meta_client.MetaError as e:
        status, analysis = "failed", {"error": getattr(e, "reason", "meta_error"), "note": UNVERIFIED_NOTE}
    with ops_con() as con:
        con.execute("UPDATE case_video_evidence SET analysis_status = ?, analysis = ? WHERE video_id = ?",
                    [status, json.dumps(analysis), video_id])
    record_event(kind="video_analysis", interaction_id=row.get("interaction_id"), case_id=row["case_id"],
                 ai_proposal=analysis, accepted={"severity_changed": False}, reason=UNVERIFIED_NOTE,
                 model_id=config.model_vision())
    return get_video(video_id)


def erase_for_interaction(con, interaction_id: str) -> int:
    if not _exists(con):
        return 0
    rows = con.execute("SELECT storage_path FROM case_video_evidence WHERE interaction_id = ?", [interaction_id]).fetchall()
    for (p,) in rows:
        try:
            Path(p).unlink(missing_ok=True)
        except OSError:
            pass
    con.execute("DELETE FROM case_video_evidence WHERE interaction_id = ?", [interaction_id])
    return len(rows)


__all__ = ["UNVERIFIED_NOTE", "VideoError", "analyze_video", "erase_for_interaction", "get_video", "list_for_case",
           "store_video", "validate_video"]
