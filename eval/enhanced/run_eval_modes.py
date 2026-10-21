"""Severity on the 60-call set: rules-only vs rules+Meta, per language.  [AI-LAYER]

    python -m eval.enhanced.run_eval_modes [--out-dir reports/eval_modes]

For every synthetic call (``eval/enhanced/calls.jsonl``, en / hi / hinglish):

* rules-only: the EXISTING ``IntakeAgent`` then the EXISTING ``TriageAgent``
  (``score_severity``), exactly as the orchestrator runs them, nothing else;
* rules+Meta (only when ``META_API_KEY`` is set): same, plus the optional
  ``maybe_enhance_intake`` after intake and ``maybe_raise_severity`` after
  triage (the raise-only hook).

Reported per language: severity distribution, triage-skipped count (triage
needs category + description), pipeline errors, kill-switch hits; and, when
enhanced ran: severity agreement (same final severity), raises (enhanced >
rules), lowers (must be 0: raise-only), Meta call failures / fallbacks
(from ``ai_call_health`` rows written during the run).

The call set has NO human severity labels, so "agreement" means agreement
between the two modes, not accuracy. With no key the enhanced column is
written as "not run" and no enhanced number is produced. Runs on throwaway
temp warehouses, never the pilot DB.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

ORDER = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
LANGS = ("en", "hi", "hinglish")


def _isolate() -> None:
    root = tempfile.mkdtemp(prefix="skewai-eval-modes-")
    os.environ.setdefault("FRONTLINE_DB_PATH", str(Path(root) / "frontline.duckdb"))
    os.environ.setdefault("DOMAIN_DB_PATH", str(Path(root) / "domains"))
    os.environ.setdefault("AI_EVIDENCE_DIR", str(Path(root) / "evidence"))


def load_calls(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _health_counts() -> tuple[int, int]:
    """(calls, failed) recorded in ai_call_health so far (0, 0 when the table does not exist)."""
    from src.data.warehouse import ops_con

    try:
        with ops_con() as con:
            if not con.execute("SELECT 1 FROM information_schema.tables WHERE table_name='ai_call_health'").fetchone():
                return 0, 0
            row = con.execute("SELECT COUNT(*), COALESCE(SUM(CASE WHEN ok THEN 0 ELSE 1 END), 0) FROM ai_call_health").fetchone()
            return int(row[0]), int(row[1])
    except Exception:
        return 0, 0


async def run_call(pack, call: dict[str, Any], *, enhanced: bool) -> dict[str, Any]:
    from src.agents.base import InteractionContext
    from src.agents.intake import IntakeAgent
    from src.agents.triage import TriageAgent

    mode = "enh" if enhanced else "rules"
    ctx = InteractionContext(interaction_id=f"int_evm_{mode}_{call['id']}", pack=pack, channel="web_text")
    ctx.record_turn("customer", call["text"])
    c0, f0 = _health_counts()
    t0 = time.perf_counter()
    out: dict[str, Any] = {"error": None}
    try:
        ires = await IntakeAgent(ctx).run(customer_turn=call["text"])
        if enhanced:
            from src.ai_layer.intake_enhancer import maybe_enhance_intake

            ires = await maybe_enhance_intake(ctx, call["text"], ires)
        tres = await TriageAgent(ctx).run()
        rule_sev = None if tres.get("skipped") else tres.get("severity")
        if enhanced:
            from src.ai_layer.severity import maybe_raise_severity

            tres = await maybe_raise_severity(ctx, tres)
        out.update({
            "severity": None if tres.get("skipped") else tres.get("severity"),
            "rule_severity_before_ai": rule_sev,
            "triage_skipped": bool(tres.get("skipped")),
            "kill_switch": bool(ires.get("kill_switch") or (ctx.safety_flags or {}).get("escalation")),
        })
    except Exception as e:  # counted, never hidden
        out.update({"severity": None, "rule_severity_before_ai": None, "triage_skipped": False,
                    "kill_switch": None, "error": f"{type(e).__name__}: {str(e)[:160]}"})
    c1, f1 = _health_counts()
    out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
    out["meta_calls"], out["meta_failures"] = c1 - c0, f1 - f0
    return out


def summarise_rules(calls: list[dict[str, Any]], res: list[dict[str, Any]]) -> dict[str, Any]:
    by: dict[str, dict[str, Any]] = {}
    for lang in (*LANGS, "all"):
        idx = [i for i, c in enumerate(calls) if lang == "all" or c["lang"] == lang]
        sev = Counter(str(res[i]["severity"]) for i in idx if res[i]["severity"])
        by[lang] = {
            "n": len(idx),
            "severity_counts": dict(sorted(sev.items(), key=lambda kv: ORDER.get(kv[0], 9))),
            "triage_skipped": sum(1 for i in idx if res[i]["triage_skipped"]),
            "pipeline_errors": sum(1 for i in idx if res[i]["error"]),
            "kill_switch_hits": sum(1 for i in idx if res[i]["kill_switch"]),
            "kill_switch_expected": sum(1 for i in idx if calls[i]["kill_switch"]),
            "meta_calls": sum(res[i]["meta_calls"] for i in idx),
        }
    return by


def summarise_compare(calls, rules, enh) -> dict[str, Any]:
    by: dict[str, dict[str, Any]] = {}
    for lang in (*LANGS, "all"):
        idx = [i for i, c in enumerate(calls) if lang == "all" or c["lang"] == lang]
        both = [i for i in idx if rules[i]["severity"] and enh[i]["severity"]]
        agree = sum(1 for i in both if rules[i]["severity"] == enh[i]["severity"])
        raises = sum(1 for i in both if ORDER.get(enh[i]["severity"], -1) > ORDER.get(rules[i]["severity"], -1))
        lowers = sum(1 for i in both if ORDER.get(enh[i]["severity"], 9) < ORDER.get(rules[i]["severity"], 9))
        by[lang] = {
            "n": len(idx), "n_both_scored": len(both),
            "severity_agreement": round(agree / len(both), 3) if both else None, "agree": agree,
            "raises": raises, "lowers": lowers,
            "enhanced_triage_only_scored": sum(1 for i in idx if enh[i]["severity"] and not rules[i]["severity"]),
            "meta_calls": sum(enh[i]["meta_calls"] for i in idx),
            "meta_failures_fallback_to_rules": sum(enh[i]["meta_failures"] for i in idx),
            "pipeline_errors": sum(1 for i in idx if enh[i]["error"]),
        }
    return by


def render(meta: dict[str, Any], rules: dict[str, Any], cmp_: dict[str, Any] | None) -> str:
    L = ["# Severity: rules-only vs rules+Meta on the 60-call set", "",
         f"{meta['n']} SYNTHETIC calls (`eval/enhanced/calls.jsonl`), en / hi (Devanagari) / hinglish. "
         "Generated by `eval/enhanced/run_eval_modes.py`; raw per-call output in `eval_modes.json` next to this file.", "",
         f"Rules+Meta mode: **{('RUN with ' + meta['model']) if cmp_ else 'NOT RUN: no META_API_KEY configured on this box. No enhanced number below is real; every enhanced cell says not run.'}**",
         "", "The call set carries no human severity labels, so agreement is between the two modes, not accuracy.", "",
         "## Rules-only (run)", "",
         "| Language | Calls | Severity counts | Triage skipped | Pipeline errors | Kill-switch hits / expected | Meta calls |",
         "| --- | --- | --- | --- | --- | --- | --- |"]
    for lang in (*LANGS, "all"):
        r = rules[lang]
        sev = ", ".join(f"{k} {v}" for k, v in r["severity_counts"].items()) or "none"
        L.append(f"| {lang} | {r['n']} | {sev} | {r['triage_skipped']} | {r['pipeline_errors']} | "
                 f"{r['kill_switch_hits']} / {r['kill_switch_expected']} | {r['meta_calls']} |")
    L += ["", "## Rules+Meta vs rules-only", "",
          "| Language | Severity agreement | Raises | Lowers (must be 0) | Meta calls | Meta failures (fell back to rules) | Pipeline errors |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for lang in (*LANGS, "all"):
        if not cmp_:
            L.append(f"| {lang} | not run | not run | not run | not run | not run | not run |")
            continue
        c = cmp_[lang]
        L.append(f"| {lang} | {c['severity_agreement']} ({c['agree']}/{c['n_both_scored']}) | {c['raises']} | "
                 f"{c['lowers']} | {c['meta_calls']} | {c['meta_failures_fallback_to_rules']} | {c['pipeline_errors']} |")
    L += ["", "Notes:",
          "- Triage needs category + description; a single customer turn that leaves either empty is counted as",
          "  'triage skipped' (the live orchestrator would ask a follow-up question first).",
          "- On safety calls the kill-switch fires during intake; severity there comes from the existing rules.",
          f"- Generated {meta['generated_at']}.", ""]
    return "\n".join(L)


async def run_all(calls: list[dict[str, Any]], *, force_enhanced: bool = False) -> dict[str, Any]:
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
    rules_res = [await run_call(pack, c, enhanced=False) for c in calls]
    enh_res = None
    if config.meta_enabled() or force_enhanced:
        enh_res = [await run_call(pack, c, enhanced=True) for c in calls]
    rules = summarise_rules(calls, rules_res)
    cmp_ = summarise_compare(calls, rules_res, enh_res) if enh_res else None
    meta = {"n": len(calls), "mode": config.ai_mode(), "model": config.model_text(),
            "enhanced_run": bool(enh_res), "generated_at": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())}
    return {"meta": meta, "rules_only": rules, "rules_plus_meta": cmp_ if cmp_ else "not run",
            "per_call": [{"id": c["id"], "lang": c["lang"], "kill_switch_expected": c["kill_switch"],
                          "rules_only": r, "rules_plus_meta": (e if e else "not run")}
                         for c, r, e in zip(calls, rules_res, enh_res or [None] * len(calls))]}


async def main_async(args) -> int:
    _isolate()
    calls = load_calls(Path(args.calls))
    out = await run_all(calls)
    od = Path(args.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    (od / "eval_modes.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    cmp_ = out["rules_plus_meta"] if isinstance(out["rules_plus_meta"], dict) else None
    (od / "EVAL_MODES.md").write_text(render(out["meta"], out["rules_only"], cmp_), encoding="utf-8")
    print(json.dumps({"rules_only": out["rules_only"], "rules_plus_meta": out["rules_plus_meta"] if cmp_ else "not run"},
                     indent=2, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--calls", default=str(HERE / "calls.jsonl"))
    p.add_argument("--out-dir", default=str(ROOT / "reports" / "eval_modes"))
    return asyncio.run(main_async(p.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
