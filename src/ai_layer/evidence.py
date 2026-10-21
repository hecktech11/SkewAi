"""Photo evidence: upload, validate, store locally, optional SAM 3.1 + Muse Spark.  [AI-LAYER]

Upload works with or without a key. Without a key the photo is validated,
stored and attached; ``analysis_status`` is ``not_run``. With a key:

1. Muse Spark (vision) describes the visible defect and proposes a component
   category. The category is accepted only if it is a canonical value of the
   pack's own ``categories`` gazetteer and at or above META_MIN_CONFIDENCE.
2. SAM 3.1 segments the damaged part named by a short noun phrase and returns
   boxes (rendered as an overlay in the dashboard EvidencePanel).

The photo never changes the case severity or category: AI output is stored
beside the case for a human to review (and a visible hazard is surfaced as a
review flag only).

Evidence is attached to a case (directly, or through its interaction when it
is uploaded mid-call before the case exists) and, through that case, to its
cluster (``cases.cluster_match_id``) and investigation.
"""

from __future__ import annotations

import base64
import hashlib
import os
import struct
from pathlib import Path
from typing import Any

from src.ai_layer import config
from src.ai_layer.meta_client import MetaError, chat_json, sam_segment
from src.ai_layer.store import ensure_tables, record_event
from src.data.timeutil import utc_now
from src.data.warehouse import ops_con
from src.ids import new_ulid

ALLOWED = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp"}
MAX_DIM = 12000
MAX_PER_CASE = 12


class EvidenceError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


# ── Validation ─────────────────────────────────────────────────────────────


def sniff_mime(data: bytes) -> str | None:
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _png_dims(data: bytes) -> tuple[int, int] | None:
    if len(data) < 24 or data[12:16] != b"IHDR":
        return None
    w, h = struct.unpack(">II", data[16:24])
    return int(w), int(h)


def _jpeg_dims(data: bytes) -> tuple[int, int] | None:
    i = 2
    n = len(data)
    while i + 9 < n:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        if i + 4 > n:
            return None
        seg_len = struct.unpack(">H", data[i + 2:i + 4])[0]
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            if i + 9 > n:
                return None
            h, w = struct.unpack(">HH", data[i + 5:i + 9])
            return int(w), int(h)
        if seg_len < 2:
            return None
        i += 2 + seg_len
    return None


def _webp_dims(data: bytes) -> tuple[int, int] | None:
    chunk = data[12:16]
    if chunk == b"VP8 " and len(data) >= 30:
        w, h = struct.unpack("<HH", data[26:30])
        return int(w & 0x3FFF), int(h & 0x3FFF)
    if chunk == b"VP8L" and len(data) >= 25:
        b = data[21:25]
        w = 1 + (((b[1] & 0x3F) << 8) | b[0])
        h = 1 + (((b[3] & 0x0F) << 10) | (b[2] << 2) | ((b[1] & 0xC0) >> 6))
        return int(w), int(h)
    if chunk == b"VP8X" and len(data) >= 30:
        w = 1 + int.from_bytes(data[24:27], "little")
        h = 1 + int.from_bytes(data[27:30], "little")
        return int(w), int(h)
    return None


def validate_image(data: bytes, declared_mime: str | None = None) -> dict[str, Any]:
    if not data:
        raise EvidenceError(400, "empty upload")
    limit = config.evidence_max_bytes()
    if len(data) > limit:
        raise EvidenceError(413, f"file too large (max {limit} bytes)")
    declared = (declared_mime or "").split(";")[0].strip().lower()
    if declared and not declared.startswith("image/") and declared != "application/octet-stream":
        raise EvidenceError(415, "only JPEG, PNG or WebP images are accepted")
    mime = sniff_mime(data)
    if mime is None:
        raise EvidenceError(415, "only JPEG, PNG or WebP images are accepted")
    dims = {"image/png": _png_dims, "image/jpeg": _jpeg_dims, "image/webp": _webp_dims}[mime](data)
    if not dims or dims[0] <= 0 or dims[1] <= 0:
        raise EvidenceError(415, "image could not be decoded")
    if dims[0] > MAX_DIM or dims[1] > MAX_DIM:
        raise EvidenceError(413, f"image dimensions exceed {MAX_DIM}px")
    return {"mime": mime, "ext": ALLOWED[mime], "width": dims[0], "height": dims[1]}


def _safe_filename(name: str | None) -> str:
    base = os.path.basename(name or "photo")
    clean = "".join(ch for ch in base if ch.isalnum() or ch in "._- ")[:80].strip()
    return clean or "photo"


def evidence_dir() -> Path:
    raw = (os.getenv("AI_EVIDENCE_DIR") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    from src.config import settings

    return (settings.frontline_db_path.parent / "ai_evidence").resolve()


# ── Case / interaction resolution ──────────────────────────────────────────


def _case_row(con, case_id: str) -> dict[str, Any] | None:
    cur = con.execute(
        "SELECT case_id, interaction_id, pack_id, cluster_match_id, investigation_id, category, "
        "description_summary, severity FROM cases WHERE case_id = ?",
        [case_id],
    )
    row = cur.fetchone()
    if row is None:
        return None
    return dict(zip([d[0] for d in cur.description], row))


def _case_for_interaction(con, interaction_id: str) -> dict[str, Any] | None:
    cur = con.execute(
        "SELECT case_id, interaction_id, pack_id, cluster_match_id, investigation_id, category, "
        "description_summary, severity FROM cases WHERE interaction_id = ? ORDER BY created_at DESC LIMIT 1",
        [interaction_id],
    )
    row = cur.fetchone()
    if row is None:
        return None
    return dict(zip([d[0] for d in cur.description], row))


def _interaction_pack(con, interaction_id: str) -> str | None:
    try:
        row = con.execute(
            "SELECT pack_id FROM interactions WHERE interaction_id = ?", [interaction_id]
        ).fetchone()
    except Exception:
        return None
    return None if row is None else str(row[0] or "")


# ── Store ──────────────────────────────────────────────────────────────────


def store_evidence(*, data: bytes, filename: str | None, declared_mime: str | None,
                   case_id: str | None = None, interaction_id: str | None = None) -> dict[str, Any]:
    from src.security.identifiers import safe_token_id

    if not case_id and not interaction_id:
        raise EvidenceError(400, "case_id or interaction_id required")
    meta = validate_image(data, declared_mime)
    with ops_con() as con:
        ensure_tables(con)
        case = None
        pack_id = None
        if case_id:
            safe_token_id(case_id, kind="case_id")
            case = _case_row(con, case_id)
            if case is None:
                raise EvidenceError(404, "case not found")
            interaction_id = case.get("interaction_id")
            pack_id = case.get("pack_id")
        else:
            safe_token_id(interaction_id, kind="interaction_id")
            pack_id = _interaction_pack(con, str(interaction_id))
            if pack_id is None:
                raise EvidenceError(404, "interaction not found")
            case = _case_for_interaction(con, str(interaction_id))
            if case:
                case_id = case["case_id"]
        owner_col, owner_val = ("case_id", case_id) if case_id else ("interaction_id", interaction_id)
        count = con.execute(
            f"SELECT COUNT(*) FROM case_evidence WHERE {owner_col} = ?", [owner_val]
        ).fetchone()[0]
        if int(count) >= MAX_PER_CASE:
            raise EvidenceError(409, f"evidence limit reached ({MAX_PER_CASE} per case)")
        evidence_id = "evd_" + new_ulid()
        folder = evidence_dir() / (case_id or f"pending_{interaction_id}")
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{evidence_id}.{meta['ext']}"
        tmp = path.with_suffix(path.suffix + ".tmp")
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
        status = "pending" if config.meta_enabled() else "not_run"
        con.execute(
            """
            INSERT INTO case_evidence (evidence_id, case_id, interaction_id, pack_id, cluster_id,
                investigation_id, filename, mime, size_bytes, width, height, sha256, storage_path,
                analysis_status, analysis, mapped_category, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                evidence_id, case_id, interaction_id, pack_id,
                (case or {}).get("cluster_match_id"), (case or {}).get("investigation_id"),
                _safe_filename(filename), meta["mime"], len(data), meta["width"], meta["height"],
                hashlib.sha256(data).hexdigest(), str(path), status, None, None, utc_now(),
            ],
        )
    return get_evidence(evidence_id)


def _row_public(row: dict[str, Any]) -> dict[str, Any]:
    import json

    out = {k: v for k, v in row.items() if k != "storage_path"}
    if isinstance(out.get("analysis"), str):
        try:
            out["analysis"] = json.loads(out["analysis"])
        except json.JSONDecodeError:
            out["analysis"] = None
    if out.get("created_at") is not None:
        out["created_at"] = str(out["created_at"])
    out["file_url"] = f"/api/evidence/{row['evidence_id']}/file"
    return out


def _select(con, where: str, args: list[Any]) -> list[dict[str, Any]]:
    cur = con.execute(f"SELECT * FROM case_evidence WHERE {where} ORDER BY created_at ASC", args)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _tables_exist(con) -> bool:
    return con.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_name = 'case_evidence'"
    ).fetchone() is not None


def get_evidence(evidence_id: str, *, include_path: bool = False) -> dict[str, Any]:
    with ops_con() as con:
        if not _tables_exist(con):
            raise EvidenceError(404, "evidence not found")
        rows = _select(con, "evidence_id = ?", [evidence_id])
        if not rows:
            raise EvidenceError(404, "evidence not found")
        row = rows[0]
        # Late attach: uploaded mid-call, the case now exists.
        if not row.get("case_id") and row.get("interaction_id"):
            case = _case_for_interaction(con, row["interaction_id"])
            if case:
                row["case_id"] = case["case_id"]
                row["cluster_id"] = case.get("cluster_match_id")
                row["investigation_id"] = case.get("investigation_id")
    out = _row_public(row)
    if include_path:
        out["storage_path"] = row["storage_path"]
    return out


def list_for_case(case_id: str) -> list[dict[str, Any]]:
    with ops_con() as con:
        case = _case_row(con, case_id)
        if case is None:
            raise EvidenceError(404, "case not found")
        if not _tables_exist(con):
            return []
        rows = _select(con, "case_id = ? OR (case_id IS NULL AND interaction_id = ?)",
                       [case_id, case.get("interaction_id")])
    for r in rows:
        r["case_id"] = case_id
        r["cluster_id"] = r.get("cluster_id") if r.get("cluster_id") is not None else case.get("cluster_match_id")
        r["investigation_id"] = r.get("investigation_id") or case.get("investigation_id")
    return [_row_public(r) for r in rows]


def list_for_interaction(interaction_id: str) -> list[dict[str, Any]]:
    with ops_con() as con:
        if not _tables_exist(con):
            return []
        rows = _select(con, "interaction_id = ?", [interaction_id])
    return [_row_public(r) for r in rows]


def list_for_cluster(pack_id: str, cluster_id: int) -> list[dict[str, Any]]:
    with ops_con() as con:
        if not _tables_exist(con):
            return []
        cur = con.execute(
            """
            SELECT e.* FROM case_evidence e
            LEFT JOIN cases c ON c.case_id = e.case_id
                OR (e.case_id IS NULL AND c.interaction_id = e.interaction_id)
            WHERE e.pack_id = ? AND (e.cluster_id = ? OR c.cluster_match_id = ?)
            ORDER BY e.created_at ASC
            """,
            [pack_id, int(cluster_id), int(cluster_id)],
        )
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    seen: set[str] = set()
    out = []
    for r in rows:
        if r["evidence_id"] in seen:
            continue
        seen.add(r["evidence_id"])
        out.append(_row_public(r))
    return out


def list_for_investigation(investigation_id: str) -> list[dict[str, Any]]:
    with ops_con() as con:
        if not _tables_exist(con):
            return []
        cur = con.execute(
            """
            SELECT DISTINCT e.* FROM case_evidence e
            LEFT JOIN cases c ON c.case_id = e.case_id
                OR (e.case_id IS NULL AND c.interaction_id = e.interaction_id)
            WHERE e.investigation_id = ? OR c.investigation_id = ?
            ORDER BY e.created_at ASC
            """,
            [investigation_id, investigation_id],
        )
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    return [_row_public(r) for r in rows]


# ── Analysis ───────────────────────────────────────────────────────────────

# Fallback SAM concept per pack category, used when the vision step gives no
# usable noun phrase. Plain noun phrases (SAM 3.1 prompts are concepts).
_CONCEPT_BY_CATEGORY = {
    "SERVICE BRAKES": "brake disc", "PARKING BRAKE": "brake", "TIRES": "tire", "WHEELS": "wheel",
    "ENGINE": "engine", "ELECTRICAL SYSTEM": "wiring", "EXTERIOR LIGHTING": "headlight",
    "AIR BAGS": "airbag", "SEAT BELTS": "seat belt", "SEATS": "seat", "STEERING": "steering wheel",
    "SUSPENSION": "suspension", "STRUCTURE": "dent", "VISIBILITY": "windshield",
    "VISIBILITY/WIPER": "wiper", "FUEL SYSTEM": "fuel leak", "POWER TRAIN": "gearbox",
    "LATCHES/LOCKS/LINKAGES": "door latch",
}


def vision_schema(categories: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "component_category": {"type": "string", "enum": categories + ["UNKNOWN"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "defect_description": {"type": "string", "maxLength": 300},
            "damaged_part": {"type": "string", "maxLength": 40},
            "visible_hazard": {"type": "boolean"},
            "hazard_reason": {"type": "string", "maxLength": 200},
        },
        "required": ["component_category", "confidence", "defect_description", "damaged_part", "visible_hazard"],
    }


def _pack_categories(pack_id: str | None) -> tuple[list[str], Any]:
    try:
        from src.domains.loader import load_pack

        pack = load_pack(pack_id or "automotive_nhtsa")
        gaz = pack.gazetteer_for_slot("category")
        if gaz is not None:
            return list(gaz.values), gaz
    except Exception:
        pass
    return [], None


def _save_analysis(evidence_id: str, status: str, analysis: dict[str, Any] | None, mapped: str | None) -> None:
    import json

    with ops_con() as con:
        ensure_tables(con)
        con.execute(
            "UPDATE case_evidence SET analysis_status = ?, analysis = ?, mapped_category = ? WHERE evidence_id = ?",
            [status, json.dumps(analysis, default=str) if analysis is not None else None, mapped, evidence_id],
        )


def analyze_evidence(evidence_id: str) -> dict[str, Any]:
    """Run Muse Spark vision + SAM 3.1. No key -> ``not_run`` (no I/O)."""
    row = get_evidence(evidence_id, include_path=True)
    if not config.meta_enabled():
        _save_analysis(evidence_id, "not_run", None, None)
        return get_evidence(evidence_id)
    path = Path(row["storage_path"])
    data = path.read_bytes()
    data_url = f"data:{row['mime']};base64,{base64.b64encode(data).decode('ascii')}"
    categories, gaz = _pack_categories(row.get("pack_id"))
    case_ctx = ""
    if row.get("case_id"):
        with ops_con() as con:
            case = _case_row(con, row["case_id"])
        if case:
            case_ctx = f"case category={case.get('category')} description={(case.get('description_summary') or '')[:300]}"
    analysis: dict[str, Any] = {"vision": None, "segmentation": None, "errors": []}
    mapped = None
    try:
        vis = chat_json(
            system=(
                "You inspect a customer's photo of a vehicle/product defect for a quality team. Return JSON only. "
                "Describe only what is visible. Choose component_category strictly from the enum (UNKNOWN if unsure). "
                "damaged_part is a short noun phrase naming the damaged part (e.g. 'brake disc', 'tire'). "
                "Never invent part numbers, recalls or causes."
            ),
            user=f"{case_ctx}\nDescribe the defect in the attached photo.",
            schema=vision_schema(categories or ["UNKNOWN OR OTHER"]),
            schema_name="evidence_vision",
            images=[data_url],
            timeout=max(config.timeout_s(), 20.0),
        )
        cat = str(vis.get("component_category") or "")
        conf = float(vis.get("confidence") or 0)
        if gaz is not None and cat != "UNKNOWN" and conf >= config.min_confidence():
            mapped = gaz.lookup(cat)
        analysis["vision"] = {
            "model_id": config.model_vision(),
            "proposed_category": cat,
            "confidence": round(conf, 3),
            "mapped_category": mapped,
            "defect_description": str(vis.get("defect_description") or "")[:300],
            "damaged_part": str(vis.get("damaged_part") or "")[:40],
            "visible_hazard": bool(vis.get("visible_hazard")),
            "hazard_reason": str(vis.get("hazard_reason") or "")[:200],
        }
    except MetaError as e:
        analysis["errors"].append(f"vision:{e.reason}")
    concept = ((analysis.get("vision") or {}).get("damaged_part") or "").strip()
    if not concept:
        concept = _CONCEPT_BY_CATEGORY.get(mapped or "", "") or _CONCEPT_BY_CATEGORY.get(
            str((analysis.get("vision") or {}).get("proposed_category") or ""), "")
    if concept:
        try:
            boxes = sam_segment(concept=concept, image_data_url=data_url, timeout=max(config.timeout_s(), 30.0))
            analysis["segmentation"] = {"model_id": config.model_sam(), "concept": concept,
                                        "count": len(boxes), "boxes": boxes,
                                        "image": {"width": row.get("width"), "height": row.get("height")}}
        except MetaError as e:
            analysis["errors"].append(f"segmentation:{e.reason}")
    if analysis["vision"] and analysis["segmentation"] is not None:
        status = "ok"
    elif analysis["vision"] or analysis["segmentation"] is not None:
        status = "partial"
    else:
        status = "failed"
    if row.get("case_id") and mapped:
        with ops_con() as con:
            case = _case_row(con, row["case_id"])
        analysis["agrees_with_case_category"] = bool(case and case.get("category") == mapped)
    analysis["review_flag"] = bool((analysis.get("vision") or {}).get("visible_hazard"))
    _save_analysis(evidence_id, status, analysis, mapped)
    record_event(kind="evidence_analysis", interaction_id=row.get("interaction_id"), case_id=row.get("case_id"),
                 model_id=f"{config.model_vision()}+{config.model_sam()}",
                 deterministic={"case_category": None}, ai_proposal=analysis,
                 accepted={"mapped_category": mapped, "status": status}, reason=status)
    return get_evidence(evidence_id)


__all__ = [
    "EvidenceError",
    "analyze_evidence",
    "evidence_dir",
    "get_evidence",
    "list_for_case",
    "list_for_cluster",
    "list_for_interaction",
    "list_for_investigation",
    "sniff_mime",
    "store_evidence",
    "validate_image",
]
