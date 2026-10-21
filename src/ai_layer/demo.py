"""Seeded India demo scenario (hackathon demo path).  [AI-LAYER]

Runs ONE clearly-labelled synthetic contact through the EXISTING orchestrator
(same intake, kill-switch, severity, case creation and Qubot audit as a real
call) on the ``simulated`` channel, so it is excluded from live metrics by
default, then attaches a synthetic demo photo to the resulting case.

Scenario: a driver in Pune calls in Hinglish/Hindi about a 2019 Honda CR-V
whose brake pedal sinks and grinds (the same slice as ``make demo-issue``, so
the case joins that early-warning investigation when it has been seeded).

The photo is a synthetic drawing of a scored brake rotor generated in code
(no external asset). With a Meta key, ``--muse-image`` asks Muse Image for a
photorealistic synthetic photo instead; either way it is labelled synthetic.
Nothing here invents warehouse records or metrics.
"""

from __future__ import annotations

import base64
import math
import struct
import zlib
from typing import Any

INDIA_SCENARIO_TURNS = [
    "Namaste. Meri Honda CR-V 2019 ke brakes fail ho gaye, pedal neeche tak chala jata hai aur grinding ki awaaz aati hai.",
    "कल हाईवे पर ब्रेक लगाने में बहुत देर लगी, लगभग accident हो गया था।",
]
# Replies to the agent's follow-up questions, chosen by keyword in its last line.
_FOLLOW_UPS = [
    (("safe location", "are you safe"), "Yes, I am safe. Gaadi side mein park ki hai."),
    (("hurt", "injur"), "No, nobody is hurt. Hum sab theek hain."),
    (("year", "model", "make", "vehicle"), "It is a 2019 Honda CR-V."),
    (("right", "correct", "confirm", "did i get"), "Haan, sahi hai."),
]
_DEFAULT_REPLY = "Brake pedal neeche tak chala jata hai aur brakes grind karte hain."
SCENARIO_LABEL = "synthetic demo contact (India, Hinglish + Hindi)"
MUSE_IMAGE_PROMPT = (
    "Photorealistic close-up of a car front wheel with the tyre removed, showing a deeply scored, "
    "rusty brake disc and a caliper with worn pads, Indian roadside garage, no text, no logos"
)


def synthetic_rotor_png(width: int = 480, height: int = 360) -> bytes:
    """Deterministic synthetic 'worn brake rotor' picture as PNG bytes."""
    cx, cy = width * 0.45, height * 0.5
    r_out, r_hub = min(width, height) * 0.42, min(width, height) * 0.16
    rows = []
    for y in range(height):
        row = bytearray([0])
        for x in range(width):
            dx, dy = x - cx, y - cy
            d = math.hypot(dx, dy)
            # caliper block on the right edge of the disc
            if cx + r_out * 0.55 < x < cx + r_out * 1.05 and abs(dy) < r_out * 0.45:
                g = 70 + int(10 * math.sin(x * 0.3))
                row += bytes((g, g - 5, g - 10))
            elif d <= r_hub:
                rust = 120 + int(25 * math.sin(dx * 0.7) * math.cos(dy * 0.5))
                row += bytes((rust, int(rust * 0.55), int(rust * 0.3)))
            elif d <= r_out:
                groove = 35 if int(d) % 9 in (0, 1) else 0
                score = 25 if int(d * 1.7 + math.atan2(dy, dx) * 3) % 23 == 0 else 0
                base = 150 - groove - score + int(8 * math.sin(d * 0.9))
                row += bytes((base, base, base + 4))
            else:
                bg = 40 + int(12 * (y / height))
                row += bytes((bg, bg + 3, bg + 6))
        rows.append(bytes(row))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    text = chunk(b"tEXt", b"Comment\x00SkewAi SYNTHETIC demo image - not a real vehicle photo")
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + text
            + chunk(b"IDAT", zlib.compress(b"".join(rows), 6)) + chunk(b"IEND", b""))


def muse_image_photo() -> tuple[bytes, str] | None:
    """Optional: a synthetic photo from Muse Image. None on any failure."""
    from src.ai_layer import config
    from src.ai_layer.meta_client import MetaError, image_generate

    if not config.meta_enabled():
        return None
    try:
        out = image_generate(prompt=MUSE_IMAGE_PROMPT, size="1024x768")
        if out.get("b64_json"):
            return base64.b64decode(out["b64_json"]), "synthetic_muse_image_brake_rotor.png"
    except (MetaError, ValueError):
        return None
    return None


def safety_contrast(text: str, pack_id: str | None = None) -> dict[str, Any]:
    """Hero-moment check: run the pack's EXISTING kill-switch lexicon on the
    caller's own words, then on the English normalisation (Muse Spark when a key
    is set, otherwise the deterministic glossary, which leaves Hinglish as is).

    This is read-only. It does not touch any interaction, and on a real call
    the enhanced intake does the same thing (raise-only).
    """
    from types import SimpleNamespace

    from src.agents.intake import _check_kill_switch
    from src.ai_layer.normalize import normalize_text
    from src.domains.active_pack import resolve_active_pack_id
    from src.domains.loader import load_pack

    pack = load_pack(pack_id or resolve_active_pack_id())
    ctx = SimpleNamespace(pack=pack)
    text = str(text or "")[:2000]
    original_hit = _check_kill_switch(text, ctx)
    norm = normalize_text(text)
    translated = norm.get("method") == "meta_muse_spark"
    normalized_hit = _check_kill_switch(norm["normalized_en"], ctx) if norm["normalized_en"] != text else None
    return {
        "original": text,
        "lang": norm.get("lang"),
        "original_kill_switch": original_hit,
        "normalized_en": norm["normalized_en"],
        "normalization_method": norm.get("method"),
        "normalized_kill_switch": normalized_hit,
        "translated_by_meta": translated,
        "fallback_reason": norm.get("fallback_reason"),
        "decided_by": "existing pack kill-switch lexicon (deterministic)",
        "final_kill_switch": original_hit or normalized_hit,
    }


async def seed_india_scenario(*, pack_id: str | None = None, use_muse_image: bool = False,
                              analyze: bool = True) -> dict[str, Any]:
    from src.agents.orchestrator import OrchestratorHooks, create_interaction
    from src.ai_layer import config
    from src.ai_layer.evidence import analyze_evidence, store_evidence

    last_agent: list[str] = [""]
    transcript: list[dict[str, str]] = []

    async def _agent_said(text: str, meta: dict[str, Any]) -> None:
        last_agent[0] = str(text or "")
        transcript.append({"speaker": "agent", "text": last_agent[0]})

    orch, _greeting = await create_interaction(channel="simulated", pack_id=pack_id,
                                               hooks=OrchestratorHooks(emit_customer_turn=_agent_said),
                                               locale="hi-IN")

    async def _say(text: str) -> None:
        transcript.append({"speaker": "customer", "text": text})
        await orch.handle_customer_turn(text)

    for turn in INDIA_SCENARIO_TURNS:
        if orch.ctx.state in ("DONE", "ABANDONED", "SUPERVISED"):
            break
        await _say(turn)
    for _ in range(8):  # answer follow-ups until the existing flow closes the contact
        if orch.ctx.state in ("DONE", "ABANDONED", "SUPERVISED", "CLOSING"):
            break
        said = last_agent[0].lower()
        reply = next((r for keys, r in _FOLLOW_UPS if any(k in said for k in keys)), _DEFAULT_REPLY)
        await _say(reply)
    if orch.ctx.state != "DONE":
        await orch.hangup()
    ctx = orch.ctx
    result: dict[str, Any] = {
        "label": SCENARIO_LABEL,
        "interaction_id": ctx.interaction_id,
        "state": ctx.state,
        "case_id": ctx.case_id,
        "investigation_id": ctx.investigation_id,
        "cluster_id": (ctx.investigation_brief or {}).get("cluster_id"),
        "severity": ctx.severity,
        "safety_flags": sorted(k for k, v in (ctx.safety_flags or {}).items() if v),
        "slots": {k: v for k, v in (ctx.slots or {}).items() if not str(k).startswith("__")},
        "transcript": transcript,
        "ai_mode": config.ai_mode(),
        "evidence": None,
        "links": {},
    }
    photo = muse_image_photo() if use_muse_image else None
    data, name = photo if photo else (synthetic_rotor_png(), "synthetic_demo_brake_rotor.png")
    owner = {"case_id": ctx.case_id} if ctx.case_id else {"interaction_id": ctx.interaction_id}
    ev = store_evidence(data=data, filename=name, declared_mime="image/png", **owner)
    if analyze and config.meta_enabled():
        try:
            ev = analyze_evidence(ev["evidence_id"])
        except Exception:
            pass
    result["evidence"] = {"evidence_id": ev.get("evidence_id"), "filename": name,
                          "analysis_status": ev.get("analysis_status"),
                          "source": "muse_image" if photo else "synthetic_drawing"}
    result["links"] = {
        "call": "#call",
        "case": f"#demo?case={ctx.case_id}&inv={ctx.investigation_id or ''}" if ctx.case_id else None,
        "issue": f"#issue/{ctx.investigation_id}" if ctx.investigation_id else None,
        "early_warning": "#warning",
        "demo": "#demo",
    }
    return result


__all__ = ["INDIA_SCENARIO_TURNS", "SCENARIO_LABEL", "seed_india_scenario", "synthetic_rotor_png"]
