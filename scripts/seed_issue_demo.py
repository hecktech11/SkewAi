"""Dense one-issue demo pack: NHTSA-scale corpus + warranty/service + lots.

Builds the 8-minute Issue-360 walk end to end (offline, deterministic):

  1. ops reset + NHTSA-scale corpus (10k rows by default)
  2. dense second/third sources: WARRANTY + SERVICE CSVs for one failure
     (2019 HONDA CR-V · SERVICE BRAKES · grinding pads)
  3. cluster rebuild + backtest (lead time vs advisory)
  4. supplier lots (AcmeBrakes L-9 = bad lot, CleanCo L-1 = control)
  5. simulated traffic → live cases + auto-opened investigation
  6. ownership (assignee, SLA, hypotheses) on that investigation
  7. prints the 8-minute walk with concrete IDs

Usage:
  FRONTLINE_ALLOW_DEFAULT_DB_RESET=1 python -m scripts.seed_issue_demo \\
      [--n-scale 10000] [--n-warranty 1200] [--n-service 700] \\
      [--simulate 30] [--seed 7] [--force]

Order matters: scale wipes the domain DB, so sources/lots/clusters come after.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import random
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PACK_ID = "automotive_nhtsa"
MAKE, MODEL, CATEGORY = "HONDA", "CR-V", "SERVICE BRAKES"

WARRANTY_NOTES = [
    "front pads worn to backing plate grinding",
    "grinding noise front brakes pad replacement",
    "customer hears grinding when braking at low speed front axle",
    "front rotors scored pads worn uneven grinding on stop",
    "brake pedal pulsation plus grinding noise front",
    "pads replaced 8k miles ago already grinding again",
    "metal on metal grinding front brakes tow in",
    "squeal then grind front pads below minimum thickness",
]
FAIL_CODES = ["PAD-WEAR", "PAD-WEAR", "PAD-WEAR", "ROTOR-SCORING", "CALIPER-DRAG", "SHIM-NOISE"]
SERVICE_NOTES = [
    "customer reports grinding when braking at low speed",
    "front brake pads and rotors replaced grinding resolved",
    "inspected front brakes pads at 2mm rotors scored",
    "road test confirms grinding front axle under light braking",
    "replaced front pads and hardware grinding gone on retest",
    "customer returned grinding persists after pad slap",
]
OP_CODES = ["INSP", "INSP", "BRK-RPL", "BRK-RPL", "BRK-PAD", "BRK-ROT"]
STATES = ["CA", "TX", "FL", "OH", "NY", "AZ", "WA", "MI"]
YEARS = ["2018", "2019", "2019", "2019", "2020"]


def _recent_date(rng: random.Random, now: datetime, span_days: int = 60) -> datetime:
    # Rising toward now so the spike reads on the wall.
    u = rng.random()
    back = int(span_days * (u ** 2))
    return now - timedelta(days=back, hours=rng.randrange(24))


def _write_warranty_csv(path: Path, n: int, seed: int) -> None:
    rng = random.Random(seed)
    now = datetime.now(timezone.utc)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["CLAIM_ID", "CLAIM_DATE", "REPAIR_DATE", "MODEL_YR", "MAKETXT",
                    "MODELTXT", "COMPNAME", "FAIL_CODE", "TECH_NOTES",
                    "CLAIM_SEVERITY", "STATE"])
        for i in range(n):
            claim = _recent_date(rng, now)
            repair = claim - timedelta(days=rng.randrange(1, 14))
            sev = "Critical" if rng.random() < 0.08 else "Medium"
            w.writerow([
                f"W{100001 + i}",
                claim.strftime("%Y-%m-%d"),
                repair.strftime("%Y-%m-%d"),
                rng.choice(YEARS), MAKE, MODEL, CATEGORY,
                rng.choice(FAIL_CODES), rng.choice(WARRANTY_NOTES),
                sev, rng.choice(STATES),
            ])


def _write_service_csv(path: Path, n: int, seed: int) -> None:
    rng = random.Random(seed + 1)
    now = datetime.now(timezone.utc)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["RO_NUMBER", "VISIT_DATE", "MODEL_YR", "MAKE", "MODEL",
                    "SYSTEM", "OP_CODE", "TECH_NOTES", "STATE"])
        for i in range(n):
            visit = _recent_date(rng, now)
            w.writerow([
                f"RO-{90001 + i}",
                visit.strftime("%Y-%m-%d"),
                rng.choice(YEARS), MAKE, MODEL, CATEGORY,
                rng.choice(OP_CODES), rng.choice(SERVICE_NOTES),
                rng.choice(STATES),
            ])


def main() -> int:
    ap = argparse.ArgumentParser(description="Seed the dense one-issue demo pack")
    ap.add_argument("--n-scale", type=int, default=10000)
    ap.add_argument("--n-warranty", type=int, default=1200)
    ap.add_argument("--n-service", type=int, default=700)
    ap.add_argument("--simulate", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--force", action="store_true",
                    help="Allow wiping pilot DBs (or set FRONTLINE_ALLOW_DEFAULT_DB_RESET=1)")
    args = ap.parse_args()

    if not args.force and os.getenv("FRONTLINE_ALLOW_DEFAULT_DB_RESET", "").strip().lower() not in {
        "1", "true", "yes", "on",
    }:
        print("Refusing to wipe pilot DBs. Re-run with --force or "
              "FRONTLINE_ALLOW_DEFAULT_DB_RESET=1", file=sys.stderr)
        return 2
    if args.force:
        # Same allow path `make frontline-db` uses for intentional rebuilds.
        os.environ["FRONTLINE_ALLOW_DEFAULT_DB_RESET"] = "1"

    from src.data.warehouse import domain_con, init_ops_db, ops_con, reset_ops_db
    from scripts.ingest_scale import build_scaled
    from src.domains.source_ingest import ingest_source
    from src.backtest.engine import run_backtest
    from src.frontline.supplier import attach_supply
    from src.frontline.simulator import simulate
    from src.enterprise.investigation_workspace import (
        add_hypothesis,
        update_investigation,
    )

    print("→ [1/7] ops reset + NHTSA-scale corpus "
          f"(n={args.n_scale})…")
    reset_ops_db()
    init_ops_db()
    build_scaled(PACK_ID, max(10, min(int(args.n_scale), 100_000)), force=True)
    print("  ✓ scale corpus + advisories + backtest")

    print(f"→ [2/7] warranty ({args.n_warranty}) + service ({args.n_service}) sources…")
    tmp = Path(tempfile.mkdtemp(prefix="issue_demo_"))
    wcsv, scsv = tmp / "warranty.csv", tmp / "service.csv"
    _write_warranty_csv(wcsv, max(0, args.n_warranty), args.seed)
    _write_service_csv(scsv, max(0, args.n_service), args.seed)
    rep_w = ingest_source(PACK_ID, "warranty", wcsv) if args.n_warranty > 0 else {"upserted": 0}
    rep_s = ingest_source(PACK_ID, "service", scsv) if args.n_service > 0 else {"upserted": 0}
    print(f"  ✓ warranty upserted={rep_w['upserted']} service upserted={rep_s['upserted']}")

    print("→ [3/7] cluster assignment + backtest…")
    # NOTE: no k-means here on purpose. scripts.ingest_scale ships the pack's
    # fixture clusters; pure-Python k-means over 5k rows takes 30+ min and
    # would rediscover the same category slices. Instead we deterministically
    # attach the new same-category rows to their cluster (transparent, fast),
    # recompute honest weekly anomaly stats from the real corpus below, and
    # re-run the backtest so lead times reflect the dense data.
    from src.data.warehouse import domain_con as _dcon

    with _dcon(PACK_ID, read_only=False) as dcon:
        cmap = {r[0]: r[1] for r in dcon.execute(
            "SELECT category, MIN(cluster_id) FROM clusters WHERE pack_id = ? "
            "GROUP BY category", [PACK_ID]).fetchall()}
        n_assign = 0
        for cat, cid in cmap.items():
            rows = dcon.execute(
                "SELECT record_id FROM records WHERE category = ? "
                "AND record_id NOT IN (SELECT record_id FROM cluster_assignments)",
                [cat]).fetchall()
            dcon.executemany(
                "INSERT OR REPLACE INTO cluster_assignments "
                "(record_id, cluster_id, distance) VALUES (?, ?, 0.0)",
                [(r[0], cid) for r in rows])
            n_assign += len(rows)
    print(f"  ✓ attached {n_assign} rows to {len(cmap)} clusters")
    with _dcon(PACK_ID, read_only=False) as dcon:
        # Keep the cluster header counts honest (they seed from fixtures).
        dcon.execute(
            """
            UPDATE clusters SET record_count = (
                SELECT COUNT(*) FROM cluster_assignments a
                WHERE a.cluster_id = clusters.cluster_id)
            WHERE pack_id = ?
            """,
            [PACK_ID])

    # Honest weekly anomaly stats computed from the real corpus (trailing
    # 12-week mean/std per (category, entity_2); latest-week z-score).
    import math as _math

    with _dcon(PACK_ID, read_only=False) as dcon:
        try:
            dcon.execute("DELETE FROM weekly_anomalies WHERE pack_id = ?", [PACK_ID])
        except Exception:
            pass
        slices = dcon.execute(
            "SELECT DISTINCT category, entity_2 FROM records "
            "WHERE category IS NOT NULL AND entity_2 IS NOT NULL").fetchall()
        n_anom = 0
        for cat, ent in slices:
            try:
                weeks = dcon.execute(
                    """
                    SELECT strftime(received_at, '%Y-W%V') AS w, COUNT(*) AS n
                    FROM records
                    WHERE category = ? AND entity_2 = ?
                      AND received_at >= now() - INTERVAL 84 DAY
                    GROUP BY 1 ORDER BY 1
                    """, [cat, ent]).fetchall()
            except Exception:
                continue
            if len(weeks) < 4:
                continue
            hist = [w[1] for w in weeks[:-1]]
            mean = sum(hist) / len(hist)
            var = sum((x - mean) ** 2 for x in hist) / len(hist)
            std = _math.sqrt(var) if var > 0 else 0.0
            last_w, last_n = weeks[-1]
            z = (last_n - mean) / std if std > 0 else 0.0
            dcon.execute(
                """
                INSERT INTO weekly_anomalies (
                    pack_id, iso_week, category, entity_2, record_count,
                    baseline_mean, baseline_std, z_score, is_anomaly)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [PACK_ID, last_w, cat, ent, last_n,
                      round(mean, 3), round(std, 3), round(z, 3), bool(z > 3.0)])
            n_anom += 1
    print(f"  ✓ weekly anomalies recomputed ({n_anom} slices)")

    bres = run_backtest(PACK_ID)
    print(f"  ✓ backtest_rows={len(bres) if isinstance(bres, list) else bres}")

    print("→ [4/7] supplier lots (AcmeBrakes L-9 bad lot, CleanCo L-1 control)…")
    with domain_con(PACK_ID, read_only=True) as dcon:
        wids = [r[0] for r in dcon.execute(
            "SELECT record_id FROM records WHERE source = 'WARRANTY' "
            "AND category = 'SERVICE BRAKES' ORDER BY received_at DESC LIMIT 40",
        ).fetchall()]
        sids = [r[0] for r in dcon.execute(
            "SELECT record_id FROM records WHERE source = 'SERVICE' "
            "AND category = 'SERVICE BRAKES' ORDER BY received_at DESC LIMIT 8",
        ).fetchall()]
    for rid in wids[:30]:
        attach_supply(record_id=rid, pack_id=PACK_ID, supplier="AcmeBrakes",
                      lot_id="L-9", lot_start="2025-11-01", lot_end="2025-11-30")
    for rid in wids[30:]:
        attach_supply(record_id=rid, pack_id=PACK_ID, supplier="AcmeBrakes",
                      lot_id="L-11", lot_start="2025-12-01", lot_end="2025-12-31")
    for rid in sids:
        attach_supply(record_id=rid, pack_id=PACK_ID, supplier="CleanCo",
                      lot_id="L-1", lot_start="2025-11-01", lot_end="2025-12-31")
    print(f"  ✓ lots: L-9={min(30, len(wids))} L-11={max(0, len(wids) - 30)} control={len(sids)}")

    print(f"→ [5/7] simulated traffic (n={args.simulate})…")
    sim = asyncio.run(simulate(count=max(0, args.simulate), speed="instant",
                               pack_id=PACK_ID, seed=args.seed))
    print(f"  ✓ completed={getattr(sim, 'completed', '?')} "
          f"cases={getattr(sim, 'cases_created', '?')} "
          f"investigations_opened={getattr(sim, 'investigations_opened', '?')}")

    print("→ [6/7] ownership on the hottest investigation…")
    with ops_con() as con:
        inv = con.execute(
            "SELECT investigation_id, cluster_id, title, case_count FROM investigations "
            "ORDER BY case_count DESC LIMIT 1",
        ).fetchone()
    issue_cid = None
    if inv:
        iid, issue_cid = inv[0], inv[1]
        sla = datetime.now(timezone.utc) + timedelta(days=7)
        update_investigation(iid, assignee="Priya Nair", sla_due_at=sla)
        add_hypothesis(iid, "Front pad compound wear on 2019 CR-V, lot L-9 (AcmeBrakes)")
        add_hypothesis(iid, "Caliper slide-pin corrosion in salt-belt states")
        print(f"  ✓ {iid} cluster={issue_cid} assignee=Priya Nair sla={sla.date()} + 2 hypotheses")
    else:
        print("  ! no investigation opened (raise --simulate)")

    print("→ [7/7] verify…")
    with domain_con(PACK_ID, read_only=True) as dcon:
        nrec = dcon.execute("SELECT COUNT(*) FROM records").fetchone()[0]
        nwar = dcon.execute("SELECT COUNT(*) FROM records WHERE source = 'WARRANTY'").fetchone()[0]
        nsvc = dcon.execute("SELECT COUNT(*) FROM records WHERE source = 'SERVICE'").fetchone()[0]
    print(f"  ✓ records={nrec} warranty={nwar} service={nsvc}")

    issue_url = f"http://127.0.0.1:8000/ui/#issue/{issue_cid}" if issue_cid is not None else "(re-run with --simulate 30)"
    dashes = "─" * 64
    print(f"\n{dashes}\n8-MINUTE WALK — SERVICE BRAKES / HONDA CR-V\n{dashes}")
    print("Beat 1 — spike on the wall (0:00–2:00)")
    print("  Open Command Center. Hero reads the brake cluster and its lead")
    print("  time vs the advisory ($ joins the hero once live calls land).")
    print("  Early warning shows the rising SERVICE BRAKES / HONDA slice.")
    print("Beat 2 — Open Issue 360 (2:00–4:00)")
    print(f"  Click Open issue (or visit {issue_url}).")
    print("  Four signal types on one page: voice turn, warranty row, service")
    print("  row, supplier lot L-9. Verify the voice evidence — Qubot green.")
    print("Beat 3 — live call moves the score (4:00–6:30)")
    print("  Voice agent → describe grinding brakes on a 2019 CR-V.")
    print("  Watch the cluster score and open-case count move on the wall.")
    print("Beat 4 — record the fix (6:30–8:00)")
    print("  On the issue page: Record fix & close. Before/after drops, reopen")
    print("  rate stays on the page. Export the audit CSV for the record.")
    print(dashes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
