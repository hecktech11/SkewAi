"""Deterministic vs enhanced intake on the synthetic call set.  [AI-LAYER]

    python -m eval.enhanced.run_eval [--calls eval/enhanced/calls.jsonl]

For every call: a fresh InteractionContext, the EXISTING ``IntakeAgent`` on
the text (deterministic), then, only when a Meta key is configured, the
optional ``maybe_enhance_intake`` on top (enhanced). Scores slot accuracy
(make, model, year, component) and kill-switch recall / false alarms per
language. With no key the enhanced column is reported as "not run": no
numbers are invented. Runs on throwaway temp warehouses, never the pilot DB.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

SLOTS = ("entity_2", "entity_3", "entity_1", "category")
SLOT_LABEL = {"entity_2": "make", "entity_3": "model", "entity_1": "year", "category": "component"}


def _isolate() -> None:
    root = tempfile.mkdtemp(prefix="skewai-enh-eval-")
    os.environ.setdefault("FRONTLINE_DB_PATH", str(Path(root) / "frontline.duckdb"))
    os.environ.setdefault("DOMAIN_DB_PATH", str(Path(root) / "domains"))
    os.environ.setdefault("AI_EVIDENCE_DIR", str(Path(root) / "evidence"))


def load_calls(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


async def run_call(pack, call: dict[str, Any], *, enhanced: bool) -> dict[str, Any]:
    from src.agents.base import InteractionContext
    from src.agents.intake import IntakeAgent

    ctx = InteractionContext(interaction_id=f"int_eval_{call['id']}", pack=pack, channel="web_text")
    ctx.record_turn("customer", call["text"])
    t0 = time.perf_counter()
    ires = await IntakeAgent(ctx).run(customer_turn=call["text"])
    if enhanced:
        from src.ai_layer.intake_enhancer import maybe_enhance_intake

        ires = await maybe_enhance_intake(ctx, call["text"], ires)
    ms = (time.perf_counter() - t0) * 1000
    slots = {k: (ctx.slots or {}).get(k) for k in SLOTS}
    return {"slots": slots, "kill_switch": bool(ires.get("kill_switch") or (ctx.safety_flags or {}).get("escalation")),
            "ms": round(ms, 1)}


def score(calls: list[dict[str, Any]], results: list[dict[str, Any]]) -> dict[str, Any]:
    by_lang: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    ks = defaultdict(lambda: {"tp": 0, "fn": 0, "fp": 0, "tn": 0})
    for c, r in zip(calls, results):
        for lang in (c["lang"], "all"):
            # Slots are scored on non-safety calls: when the kill-switch fires the existing
            # pipeline escalates first and defers slot capture, by design.
            if not c["kill_switch"]:
                for s in SLOTS:
                    got = str(r["slots"].get(s) or "").upper()
                    by_lang[lang][s].append(int(got == str(c["expected"][s]).upper()))
            exp, got = bool(c["kill_switch"]), bool(r["kill_switch"])
            key = "tp" if exp and got else "fn" if exp else "fp" if got else "tn"
            ks[lang][key] += 1
    out: dict[str, Any] = {}
    for lang, slots in by_lang.items():
        acc = {SLOT_LABEL[s]: round(sum(v) / len(v), 3) for s, v in slots.items()}
        all_four = [all(slots[s][i] for s in SLOTS) for i in range(len(slots["entity_2"]))]
        k = ks[lang]
        out[lang] = {
            "n_slot_scored": len(all_four), "n_safety_calls": ks[lang]["tp"] + ks[lang]["fn"], "slot_accuracy": acc, "all_slots_correct": round(sum(all_four) / len(all_four), 3),
            "kill_switch_recall": round(k["tp"] / (k["tp"] + k["fn"]), 3) if (k["tp"] + k["fn"]) else None,
            "kill_switch_false_alarms": k["fp"],
        }
    return out


def render(det: dict[str, Any], enh: dict[str, Any] | None, meta: dict[str, Any]) -> str:
    L = ["# Enhanced vs deterministic intake (synthetic call set)", "",
         f"{meta['n']} SYNTHETIC calls (`eval/enhanced/calls.jsonl`, built by `build_calls.py`): "
         "English, Hindi (Devanagari) and Hinglish, labelled with make / model / year / component and whether the "
         "pack kill-switch should fire.", "",
         f"Enhanced mode: **{'run with ' + meta['model'] if enh else 'not run (no META_API_KEY configured)'}**.", ""]
    langs = ["en", "hi", "hinglish", "all"]
    L += ["| Language | Mode | Make | Model | Year | Component | All 4 slots | Kill-switch recall | False alarms |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for lang in langs:
        for mode, res in (("deterministic", det), ("enhanced", enh)):
            if res is None:
                L.append(f"| {lang} | enhanced | not run | not run | not run | not run | not run | not run | not run |")
                continue
            r = res.get(lang)
            if not r:
                continue
            a = r["slot_accuracy"]
            L.append(f"| {lang} | {mode} | {a['make']} | {a['model']} | {a['year']} | {a['component']} | "
                     f"{r['all_slots_correct']} | {r['kill_switch_recall']} | {r['kill_switch_false_alarms']} |")
    L += ["", "Slot accuracy is the share of NON-safety calls whose slot exactly matches the label (single customer turn,",
          "no re-asks); on safety calls the existing pipeline escalates first and defers slot capture, so those calls are",
          "scored on kill-switch recall only.",
          "The deterministic path is the unchanged product. Enhanced only fills EMPTY slots with validated values and",
          "can only add a kill-switch (never remove one), so it cannot score lower on a slot the deterministic path got right.",
          "", f"Generated {meta['generated_at']}.", ""]
    return "\n".join(L)


async def main_async(args) -> int:
    _isolate()
    from src.ai_layer import config
    from src.data.warehouse import init_ops_db
    from src.domains.loader import load_pack

    init_ops_db()
    try:
        from scripts.seed_domains import build as build_domain

        build_domain("automotive_nhtsa")
    except Exception:
        pass
    pack = load_pack("automotive_nhtsa")
    calls = load_calls(Path(args.calls))
    det_res = [await run_call(pack, c, enhanced=False) for c in calls]
    det = score(calls, det_res)
    enh = None
    enh_res = None
    if config.meta_enabled():
        enh_res = [await run_call(pack, c, enhanced=True) for c in calls]
        enh = score(calls, enh_res)
    meta = {"n": len(calls), "model": config.model_text(), "mode": config.ai_mode(),
            "generated_at": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())}
    Path(args.out_md).write_text(render(det, enh, meta), encoding="utf-8")
    Path(args.out_json).write_text(json.dumps({
        "meta": meta, "deterministic": det, "enhanced": enh if enh else "not run",
        "per_call": [{"id": c["id"], "lang": c["lang"], "expected": c["expected"], "kill_switch_expected": c["kill_switch"],
                      "deterministic": d, "enhanced": (e if e else "not run")}
                     for c, d, e in zip(calls, det_res, enh_res or [None] * len(calls))],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"deterministic": det.get("all"), "enhanced": (enh or {}).get("all", "not run")}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--calls", default=str(HERE / "calls.jsonl"))
    p.add_argument("--out-md", default=str(HERE / "REPORT.md"))
    p.add_argument("--out-json", default=str(HERE / "results.json"))
    return asyncio.run(main_async(p.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
