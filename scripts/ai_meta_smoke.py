"""Live smoke check of the Meta Model API settings.  [AI-LAYER]

The layer's Meta endpoints are configured from public docs but have never been
called live from this repo (no key exists here). Anyone with a key can verify
them in about 30 seconds:

    META_API_KEY=... python -m scripts.ai_meta_smoke            # text, vision, SAM 3.1
    META_API_KEY=... python -m scripts.ai_meta_smoke --image    # + one Muse Image call

Each check makes one small call through the same client the product uses
(host allow-list, timeout, call cap, schema validation) and prints ok/failed
with the reason. Without a key it makes no network call and exits 0. The key
is never printed.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from typing import Any, Callable

_SCHEMA = {
    "type": "object",
    "properties": {"normalized_en": {"type": "string"}, "confidence": {"type": "number"}},
    "required": ["normalized_en", "confidence"],
}
_VISION_SCHEMA = {
    "type": "object",
    "properties": {"description": {"type": "string"}, "confidence": {"type": "number"}},
    "required": ["description", "confidence"],
}


def _run(name: str, fn: Callable[[], Any]) -> dict[str, Any]:
    from src.ai_layer.meta_client import MetaError

    t0 = time.time()
    try:
        detail = fn()
        return {"check": name, "ok": True, "ms": int((time.time() - t0) * 1000), "detail": detail}
    except MetaError as e:
        return {"check": name, "ok": False, "ms": int((time.time() - t0) * 1000), "reason": e.reason}
    except Exception as e:  # report, never raise
        return {"check": name, "ok": False, "ms": int((time.time() - t0) * 1000), "reason": type(e).__name__}


def run_checks(*, image: bool = False) -> dict[str, Any]:
    from src.ai_layer import config
    from src.ai_layer.demo import synthetic_rotor_png
    from src.ai_layer.meta_client import chat_json, image_generate, sam_segment

    status = config.ai_status()
    if not config.meta_enabled():
        return {"status": status, "skipped": "no META_API_KEY / MODEL_API_KEY (or FRONTLINE_AI_LAYER=off)", "checks": []}
    data_url = "data:image/png;base64," + base64.b64encode(synthetic_rotor_png(240, 180)).decode()
    checks = [
        _run("text:" + config.model_text(), lambda: chat_json(
            system="Translate to English. JSON only.", user="Meri gaadi ke brakes fail ho gaye.",
            schema=_SCHEMA, schema_name="smoke_normalize")),
        _run("vision:" + config.model_vision(), lambda: chat_json(
            system="Describe the image in one sentence. JSON only.", user="What vehicle part is shown?",
            schema=_VISION_SCHEMA, schema_name="smoke_vision", images=[data_url])),
        _run("segmentation:" + config.model_sam(), lambda: {"boxes": len(sam_segment(concept="brake disc", image_data_url=data_url))}),
    ]
    if image:
        checks.append(_run("image:" + config.model_image(), lambda: {"keys": sorted(image_generate(
            prompt="Synthetic photo of a scored brake disc, no text", size="512x512").keys())}))
    return {"status": status, "checks": checks, "all_ok": all(c["ok"] for c in checks)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--image", action="store_true", help="also call Muse Image once")
    args = p.parse_args(argv)
    out = run_checks(image=args.image)
    print(json.dumps(out, indent=2, default=str))
    return 0 if out.get("skipped") or out.get("all_ok") else 1


if __name__ == "__main__":
    sys.exit(main())
