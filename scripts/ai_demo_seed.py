"""Seed the hackathon India demo scenario.  [AI-LAYER]

    python -m scripts.ai_demo_seed [--muse-image] [--no-analyze]

Runs one synthetic Hinglish/Hindi contact through the existing orchestrator
(``simulated`` channel), attaches a synthetic brake-rotor photo to the case
and prints the ids + dashboard links. Safe to re-run (each run is one more
synthetic contact). See src/ai_layer/demo.py.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Seed the India demo scenario")
    p.add_argument("--pack", default=None)
    p.add_argument("--muse-image", action="store_true", help="generate the synthetic photo with Muse Image (needs META_API_KEY)")
    p.add_argument("--no-analyze", action="store_true")
    p.add_argument("--out", default=None, help="also write the result JSON to this file")
    args = p.parse_args(argv)
    from src.ai_layer.demo import seed_india_scenario

    out = asyncio.run(seed_india_scenario(pack_id=args.pack, use_muse_image=args.muse_image,
                                          analyze=not args.no_analyze))
    text = json.dumps(out, indent=2, default=str)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0 if out.get("interaction_id") else 1


if __name__ == "__main__":
    raise SystemExit(main())
