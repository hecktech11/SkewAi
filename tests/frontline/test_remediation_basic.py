"""Remediation regression tests — BASIC items 41-50."""

from __future__ import annotations

import pytest


# ── ITEM 41: eval independence ───────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_item41_persona_expectations_and_negative_controls(
    reset_ops_db, seed_automotive_pack
):
    from eval.frontline.personas import personas_for_pack
    from eval.frontline.run_eval import (
        _independently_verify_advisory,
        eval_pack,
        run_persona,
    )

    personas = {p.name: p for p in personas_for_pack("automotive_nhtsa")}
    # Expectations are independently defined on personas (not hardcoded gates).
    assert personas["safety_critical"].expected.should_escalate_safety is True
    assert personas["abandoner"].expected.should_complete is False
    assert personas["angry"].expected.should_flag_frustration is True

    results = await eval_pack("automotive_nhtsa")
    by_name = {r.name: r for r in results}
    # Negative controls exist and are capable of failing.
    assert "negative_off_topic" in by_name and "negative_abandoner" in by_name
    assert by_name["negative_abandoner"].passed is True
    assert by_name["negative_abandoner"].actual.startswith("case_id=None")
    # Escalation timing is asserted (≤1 turn), not just "any flag".
    assert "turns_to_escalation" in by_name["safety_escalation"].actual
    # Angry gate requires observed handoff behavior (not lexicon-vs-lexicon).
    assert "offers=" in by_name["frustration_flag"].actual
    # Advisory gate independently re-verifies scope.
    assert "independent_scope_verify=True" in by_name["advisory_notification"].detail
    # Cluster pin is enforced for the trigger gate.
    auto_open = by_name["investigation_auto_open"]
    assert "cluster_id=14" in auto_open.detail and "(expected 14)" in auto_open.detail


def test_item41_independent_verify_rejects_mismatch(seed_automotive_pack):
    from eval.frontline.run_eval import _independently_verify_advisory

    good_slots = {"entity_1": "2019", "entity_2": "HONDA", "entity_3": "CR-V",
                  "category": "SERVICE BRAKES"}
    assert _independently_verify_advisory(
        "automotive_nhtsa", "19V-12345", good_slots
    ) is True
    assert _independently_verify_advisory(
        "automotive_nhtsa", "19V-12345",
        {**good_slots, "entity_2": "FORD"},
    ) is False
    assert _independently_verify_advisory(
        "automotive_nhtsa", "NOPE-00000", good_slots
    ) is False


# ── ITEM 42: anomaly docs (covered by docstrings + these semantics) ──────────


def test_item42_baseline_semantics_documented():
    import inspect

    from src.ml_runtime import anomalies

    src = inspect.getsource(anomalies)
    assert "MIN_BASELINE_WEEKS" in src
    assert "4" in src  # 4 baseline weeks documented
    assert anomalies.MIN_BASELINE_WEEKS == 4
    assert "occurred_at" in anomalies.__doc__


# ── ITEM 43: backtest pack isolation ─────────────────────────────────────────


def test_item43_backtest_pack_isolation(tmp_path, monkeypatch):
    from datetime import datetime, timedelta

    from src.backtest.engine import run_backtest
    from src.data.warehouse import apply_domain_schema, domain_con

    monkeypatch.setenv("DOMAIN_DB_PATH", str(tmp_path))
    now = datetime(2025, 3, 1)
    for pack in ("pack_iso_a", "pack_iso_b"):
        with domain_con(pack, read_only=False) as con:
            apply_domain_schema(con)
            con.execute(
                """INSERT INTO records (record_id, occurred_at, received_at,
                   entity_2, category, text, source)
                   VALUES (?, ?, ?, 'HONDA', 'SERVICE BRAKES', 'grinding', 'NHTSA')""",
                [f"{pack}-1", now - timedelta(days=100), now - timedelta(days=100)],
            )
            con.execute(
                """INSERT INTO clusters (cluster_id, pack_id, top_terms, category,
                   record_count, first_seen, last_seen)
                   VALUES (1000, ?, '["g"]', 'SERVICE BRAKES', 1, ?, ?)""",
                [pack, now - timedelta(days=100), now - timedelta(days=90)],
            )
            con.execute(
                "INSERT INTO cluster_assignments (record_id, cluster_id, distance)"
                " VALUES (?, 1000, 0.2)",
                [f"{pack}-1"],
            )
            con.execute(
                """INSERT INTO advisories (advisory_id, issued_at, scope_category, summary)
                   VALUES (?, ?, 'SERVICE BRAKES', 'x')""",
                [f"ADV-{pack}", now - timedelta(days=10)],
            )
    run_backtest("pack_iso_a")
    run_backtest("pack_iso_b")
    with domain_con("pack_iso_a") as con:
        rows = con.execute(
            "SELECT pack_id, provenance FROM backtest_results"
        ).fetchall()
    assert rows and all(r[0] == "pack_iso_a" for r in rows)
    assert all(r[1] == "computed" for r in rows)
    # Re-running pack A must not delete pack B's rows (pack-scoped cleanup).
    with domain_con("pack_iso_b") as con:
        before = con.execute("SELECT COUNT(*) FROM backtest_results").fetchone()[0]
    run_backtest("pack_iso_a")
    with domain_con("pack_iso_b") as con:
        after = con.execute("SELECT COUNT(*) FROM backtest_results").fetchone()[0]
    assert before == after >= 1


# ── ITEM 44: pagination ──────────────────────────────────────────────────────


def test_item44_signed_cursors_and_totals():
    from src.api.pagination import (
        InvalidCursor,
        decode_cursor,
        decode_cursor_strict,
        encode_cursor,
        resolve_offset_strict,
    )

    c = encode_cursor(14)
    assert decode_cursor(c) == 14
    assert decode_cursor_strict(c) == 14
    assert resolve_offset_strict(cursor=c) == 14
    # Tampered cursor rejected (not silently reset to 0).
    import base64
    import json

    raw = json.dumps({"v": 1, "o": 9999, "sig": "0" * 32}).encode()
    forged = base64.urlsafe_b64encode(raw).decode().rstrip("=")
    with pytest.raises(InvalidCursor):
        decode_cursor_strict(forged)
    # Legacy unsigned cursor rejected.
    legacy = base64.urlsafe_b64encode(json.dumps({"o": 3}).encode()).decode().rstrip("=")
    with pytest.raises(InvalidCursor):
        decode_cursor_strict(legacy)
    assert decode_cursor(forged) is None  # lenient path stays None-safe


def test_item44_pages_first_next_and_tampered(reset_ops_db, seed_automotive_pack, monkeypatch):
    from fastapi.testclient import TestClient

    from src.api.main import app
    from src.data.timeutil import utc_now
    from src.data.warehouse import ops_con

    monkeypatch.delenv("FRONTLINE_API_KEY", raising=False)
    with ops_con() as con:
        for i in range(5):
            con.execute(
                """INSERT INTO cases (case_id, interaction_id, pack_id, created_at,
                   category, description_summary, onset, severity, severity_source,
                   priority, safety_flags, status)
                   VALUES (?, 'int_pg', 'automotive_nhtsa', ?, 'X', 'd', ?, 'Low',
                   'rules', 3, '{}', 'open')""",
                [f"case_pg_{i}", utc_now(), utc_now()],
            )
    with TestClient(app) as client:
        first = client.get("/api/frontline/cases?limit=2&offset=0").json()
        assert first["pagination"]["total"] == 5, "small tables expose COUNT(*)"
        assert first["pagination"]["has_more"] is True
        nxt = client.get(
            f"/api/frontline/cases?limit=2&cursor={first['pagination']['next_cursor']}"
        )
        assert nxt.status_code == 200
        assert nxt.json()["pagination"]["offset"] == 2
        bad = client.get("/api/frontline/cases?limit=2&cursor=forged.cursor==")
        assert bad.status_code == 400


# ── ITEM 45: idempotency ─────────────────────────────────────────────────────


def test_item45_duplicate_note_single_side_effect(reset_ops_db, seed_automotive_pack, monkeypatch):
    from fastapi.testclient import TestClient

    from src.api.main import app
    from src.data.timeutil import utc_now
    from src.data.warehouse import ops_con

    monkeypatch.delenv("FRONTLINE_API_KEY", raising=False)
    with ops_con() as con:
        con.execute(
            """INSERT INTO cases (case_id, interaction_id, pack_id, created_at,
               category, description_summary, onset, severity, severity_source,
               priority, safety_flags, status)
               VALUES ('case_idem', 'int_idem', 'automotive_nhtsa', ?, 'X', 'd',
               ?, 'Low', 'rules', 3, '{}', 'open')""",
            [utc_now(), utc_now()],
        )
    with TestClient(app) as client:
        h = {"Idempotency-Key": "note-key-1"}
        r1 = client.post("/api/frontline/cases/case_idem/notes",
                         json={"body": "hello"}, headers=h)
        assert r1.status_code == 200
        r2 = client.post("/api/frontline/cases/case_idem/notes",
                         json={"body": "hello"}, headers=h)
        assert r2.status_code == 200
        assert r2.json() == r1.json(), "duplicate returns the stored result"
        # Same key, different body → loud rejection, not silent aliasing.
        r3 = client.post("/api/frontline/cases/case_idem/notes",
                         json={"body": "different"}, headers=h)
        assert r3.status_code == 422
    with ops_con(read_only=True) as con:
        n = con.execute(
            "SELECT COUNT(*) FROM case_notes WHERE case_id = 'case_idem'"
        ).fetchone()[0]
    assert n == 1, "exactly one side effect for the duplicated request"


# ── ITEM 46: dashboard ───────────────────────────────────────────────────────


def test_item46_dashboard_centralization():
    from src.config import REPO_ROOT

    routes = REPO_ROOT / "dashboard" / "routes"
    assert routes.is_dir()
    offenders = []
    for f in routes.glob("*.jsx"):
        src = f.read_text(encoding="utf-8", errors="replace")
        if 'localStorage.getItem("frontline_api_key")' in src:
            offenders.append(f.name)
        if "function apiHeaders()" in src:
            offenders.append(f"{f.name}: local apiHeaders")
    assert offenders == [], f"direct credential handling outside apiAuth: {offenders}"
    # Accessibility landmarks survive: skip link, main, tab roles.
    app = (REPO_ROOT / "dashboard" / "src" / "App.jsx").read_text()
    assert "skip-link" in app and 'id="main"' in app
    board = (routes / "EarlyWarningBoard.jsx").read_text()
    assert 'role="tablist"' in board and "role=\"alert\"" in board


# ── ITEM 47: observability ───────────────────────────────────────────────────


def test_item47_instrumentation_hooks():
    from src.observability.otel import (
        backend,
        bind_request,
        clear_spans,
        emit_span,
        new_request_id,
        recent_spans,
        start_span,
    )
    from src.observability.slo import evaluate_slos

    clear_spans()
    rid = new_request_id()
    assert rid.startswith("req_")
    bind_request(rid, pack_id="automotive_nhtsa")
    with start_span("test.operation", attributes={"k": "v"}):
        pass
    spans = recent_spans()
    assert any(s["name"] == "test.operation" for s in spans)
    assert backend() in ("opentelemetry-sdk", "in-house")
    # Spans never crash business logic (SDK failures contained), but the
    # block's own exceptions still propagate — and the span still closes.
    with pytest.raises(ValueError, match="swallowed"):
        with start_span("test.broken", attributes=None):
            raise ValueError("swallowed?")
    assert any(s["name"] == "test.broken" for s in recent_spans())
    # SLO evaluation runs offline-safe with named objectives.
    ev = evaluate_slos(fire_alerts=False)
    assert "samples" in ev and "queue" in ev and "breached" in ev
    clear_spans()


def test_item47_request_id_header(reset_ops_db, monkeypatch):
    from fastapi.testclient import TestClient

    from src.api.main import app

    monkeypatch.delenv("FRONTLINE_API_KEY", raising=False)
    with TestClient(app) as c:
        r = c.get("/health")
        assert r.status_code == 200
        assert r.headers.get("x-request-id"), "every response carries a request ID"
        r2 = c.get("/health", headers={"X-Request-ID": "req_probe123"})
        assert r2.headers.get("x-request-id") == "req_probe123"


# ── ITEM 48: migrations + restore drill ──────────────────────────────────────


def test_item48_migrate_repeatable_and_ordered(tmp_path):
    from scripts.migrate import available_migrations, migrate_duckdb

    migs = available_migrations(target="ops")
    assert [v for v, _ in migs] == sorted(v for v, _ in migs)
    assert migs and migs[0][0] == "001"
    db = tmp_path / "mig_ops.duckdb"
    first = migrate_duckdb(db, target="ops")
    assert "001" in first
    second = migrate_duckdb(db, target="ops")
    assert second == [], "re-runs are no-ops"
    import duckdb

    con = duckdb.connect(str(db), read_only=True)
    try:
        versions = {r[0] for r in con.execute("SELECT version FROM schema_migrations").fetchall()}
        assert "001" in versions
    finally:
        con.close()


def test_item48_backup_restore_drill(tmp_path, reset_ops_db, seed_automotive_pack):
    from scripts.backup import backup, verify_backup

    out = tmp_path / "backup_drill"
    manifest = backup(out=out)
    assert manifest["files"], "backup must capture warehouse files"
    report = verify_backup(out)
    assert report["ok"] is True
    assert report["domain_files"] >= 1
    assert report["ops_counts"]["cases"] >= 0


# ── ITEM 50: secrets ─────────────────────────────────────────────────────────


def test_item50_key_permissions_and_rotation(tmp_path):
    import os
    import stat

    from src.qubot.locker import generate_keypair
    from src.security.harden import locker_key_permission_problems
    from src.security.secrets import redact_secrets, rotate_file_secret

    d = tmp_path / "keys"
    priv, pub = generate_keypair(d)
    mode = stat.S_IMODE(os.stat(priv).st_mode)
    assert mode == 0o600, f"private key must be created 0600, got {oct(mode)}"
    assert locker_key_permission_problems() == []

    # Rotation keeps 0600 + grace copy, refuses weak values.
    first = rotate_file_secret(d, "FRONTLINE_API_KEY", "x" * 40)
    assert first["rotated"] is True and first["previous_kept"] is False
    assert stat.S_IMODE(os.stat(d / "FRONTLINE_API_KEY").st_mode) == 0o600
    second = rotate_file_secret(d, "FRONTLINE_API_KEY", "y" * 40)
    assert second["previous_kept"] is True
    assert (d / "FRONTLINE_API_KEY.prev").read_text().strip() == "x" * 40
    with pytest.raises(ValueError, match="weak"):
        rotate_file_secret(d, "FRONTLINE_API_KEY", "short")


def test_item50_redaction_and_startup_guards(monkeypatch):
    import logging

    from src.security.secrets import (
        SecretRedactionFilter,
        install_secret_redaction,
        redact_secrets,
    )

    monkeypatch.setenv("FRONTLINE_API_KEY", "super-secret-value-32bytes!!!!!!")
    assert redact_secrets("key=super-secret-value-32bytes!!!!!! end") == "key=[REDACTED] end"
    assert redact_secrets("no secrets here") == "no secrets here"
    assert "[REDACTED-PRIVATE-KEY]" in redact_secrets(
        "k -----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY----- k"
    )
    rec = logging.LogRecord("t", logging.INFO, __file__, 1,
                            "leak %s", ("super-secret-value-32bytes!!!!!!",), None)
    assert SecretRedactionFilter().filter(rec) is True
    assert "super-secret-value-32bytes!!!!!!" not in str(rec.args)
    install_secret_redaction()

    # Startup refuses short/missing session secrets when hardened.
    from src.security.harden import validate_startup_security

    for var in ("ENV", "PILOT_HARDENED", "SOC2_MODE", "FRONTLINE_OPEN_MODE",
                "FRONTLINE_AUTH_REQUIRED", "FRONTLINE_API_KEY", "SESSION_SECRET",
                "API_HOST", "FRONTLINE_OPEN_BIND_ACK"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("PILOT_HARDENED", "1")
    monkeypatch.setenv("FRONTLINE_AUTH_REQUIRED", "1")
    monkeypatch.setenv("FRONTLINE_API_KEY", "a" * 40)
    monkeypatch.setenv("SESSION_SECRET", "short")
    with pytest.raises(RuntimeError, match="SESSION_SECRET"):
        validate_startup_security()
