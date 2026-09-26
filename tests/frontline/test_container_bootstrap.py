"""Container startup must not reseed an existing database (R06).

The entrypoint treated a missing warehouse, or a finance record id that did
not start with CFPB-, as a reason to run seed_frontline_fixtures, which
resets the ops database.
"""

from __future__ import annotations

from pathlib import Path

from src.data.bootstrap import bootstrap_actions

ROOT = Path(__file__).resolve().parents[2]


def test_missing_warehouse_initializes_schema_and_does_not_reseed():
    actions = bootstrap_actions(
        ops_exists=False,
        automotive_exists=False,
        finance_exists=False,
        finance_first_record_id=None,
        seed_demo=False,
    )
    assert "reset-ops" not in actions
    assert "seed-automotive" not in actions
    assert "seed-finance" not in actions
    assert "init-ops" in actions
    assert "init-automotive" in actions
    assert "init-finance" in actions


def test_non_cfpb_record_id_does_not_reseed_existing_data():
    quiet = bootstrap_actions(
        ops_exists=True,
        automotive_exists=True,
        finance_exists=True,
        finance_first_record_id="BANK-100",
        seed_demo=False,
    )
    assert "seed-finance" not in quiet
    assert "reset-ops" not in quiet

    opted_in = bootstrap_actions(
        ops_exists=True,
        automotive_exists=True,
        finance_exists=True,
        finance_first_record_id="BANK-100",
        seed_demo=True,
    )
    assert "seed-finance" not in opted_in
    assert "reset-ops" not in opted_in


def test_demo_seed_fills_only_a_missing_domain_database():
    actions = bootstrap_actions(
        ops_exists=True,
        automotive_exists=True,
        finance_exists=False,
        finance_first_record_id=None,
        seed_demo=True,
    )
    assert actions.count("seed-finance") == 1
    assert "seed-automotive" not in actions
    assert "reset-ops" not in actions


def test_running_bootstrap_keeps_an_existing_ops_row(reset_ops_db, monkeypatch):
    from scripts.container_bootstrap import main
    from src.data.warehouse import ops_con

    monkeypatch.delenv("FRONTLINE_SEED_DEMO", raising=False)
    with ops_con() as con:
        con.execute(
            """
            INSERT INTO interactions
            (interaction_id, pack_id, pack_version, started_at, channel, status, description)
            VALUES ('int_r06', 'automotive_nhtsa', 't', CURRENT_TIMESTAMP, 'web_text', 'completed', 'keep me')
            """
        )
    assert main() == 0
    with ops_con(read_only=True) as con:
        row = con.execute(
            "SELECT description FROM interactions WHERE interaction_id = 'int_r06'"
        ).fetchone()
    assert row is not None
    assert row[0] == "keep me"


def test_entrypoint_does_not_call_the_destructive_seeder():
    text = (ROOT / "docker" / "entrypoint.sh").read_text(encoding="utf-8")
    assert "seed_frontline_fixtures" not in text
    assert "CFPB-" not in text
    assert "container_bootstrap" in text
