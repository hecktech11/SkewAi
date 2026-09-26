"""Non-destructive container startup.

Creates a missing ops or domain schema. With FRONTLINE_SEED_DEMO=1, fills a
domain warehouse only when its file is absent. Never calls reset_ops_db and
never inspects record ids.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.data.bootstrap import bootstrap_actions
from src.data.warehouse import init_domain_db, init_ops_db


def _db_has_rows(path: Path, table: str) -> bool:
    """Return True if *path* exists, is a valid DuckDB file, and *table* has ≥ 1 row."""
    if not path.exists():
        return False
    try:
        import duckdb

        con = duckdb.connect(str(path), read_only=True)
        try:
            result = con.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone()
            return result is not None
        except Exception:
            return False
        finally:
            con.close()
    except Exception:
        return False


def main() -> int:
    ops = settings.frontline_db_path
    automotive = settings.domain_db_path("automotive_nhtsa")
    finance = settings.domain_db_path("finance_cfpb")
    seed_demo = os.getenv("FRONTLINE_SEED_DEMO", "").strip() == "1"
    actions = bootstrap_actions(
        ops_exists=ops.exists(),
        automotive_exists=automotive.exists(),
        finance_exists=finance.exists(),
        automotive_populated=_db_has_rows(automotive, "complaints"),
        finance_populated=_db_has_rows(finance, "complaints"),
        seed_demo=seed_demo,
    )
    print("→ Container bootstrap:", ", ".join(actions))
    if "init-ops" in actions:
        init_ops_db()
    if "init-automotive" in actions:
        init_domain_db("automotive_nhtsa")
    if "init-finance" in actions:
        init_domain_db("finance_cfpb")
    if "seed-automotive" in actions:
        from scripts.seed_domains import build as build_automotive

        build_automotive("automotive_nhtsa", force=False)
    if "seed-finance" in actions:
        from scripts.seed_finance_cfpb import build as build_finance

        build_finance("finance_cfpb", force=False)

    # N08: Migrate legacy XOR tokens on startup so existing enc:v1: rows are upgraded to authenticated AES-GCM
    try:
        from scripts.migrate_legacy_xor_tokens import migrate as migrate_legacy_tokens

        stats = migrate_legacy_tokens(dry_run=False)
        if stats.get("migrated", 0) > 0:
            print(f"→ Migrated {stats['migrated']} legacy XOR PII tokens to authenticated AES-GCM")
    except Exception as e:
        print(f"→ Legacy XOR token migration note: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
