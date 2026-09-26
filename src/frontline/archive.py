"""Immutable audit archive bundle for a contact (feature #35)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.config import REPO_ROOT
from src.data.timeutil import utc_now
from src.frontline.dsr import export_interaction
from src.ledger.chain import verify_chain
from src.ledger.writer import list_actions
from src.qubot.auditor import REPORTS_DIR


_archive_registry: set[Path] = set()


def build_audit_archive(interaction_id: str, *, out_dir: Path | None = None) -> dict[str, Any]:
    """Write a timestamped JSON bundle with SHA256 manifest."""
    from src.security.identifiers import safe_out_dir, safe_token_id

    iid = safe_token_id(interaction_id, kind="interaction_id")
    data = export_interaction(iid)
    actions = list_actions(iid)
    chain = verify_chain(actions)
    data["ledger_chain"] = chain
    data["actions_count"] = len(actions)

    report_path = REPORTS_DIR / f"{iid}.md"
    from src.security.identifiers import assert_under_roots as _assert_roots

    report_path = _assert_roots(report_path, [REPORTS_DIR])
    report_text = ""
    if report_path.is_file():
        report_text = report_path.read_text(encoding="utf-8")
    data["audit_report_path"] = str(report_path) if report_path.is_file() else None
    data["audit_report_sha256"] = (
        hashlib.sha256(report_text.encode("utf-8")).hexdigest() if report_text else None
    )

    base = safe_out_dir(out_dir, default=REPO_ROOT / "reports" / "qubot" / "archives")
    base.mkdir(parents=True, exist_ok=True)
    ts = utc_now().strftime("%Y%m%dT%H%M%SZ")
    bundle_path = base / f"{iid}_{ts}.json"
    payload = json.dumps(data, sort_keys=True, default=str, indent=2)
    bundle_path.write_text(payload + "\n", encoding="utf-8")
    bundle_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    manifest = {
        "interaction_id": iid,
        "bundle_path": str(bundle_path),
        "bundle_sha256": bundle_hash,
        "created_at": utc_now().isoformat() + "Z",
        "ledger_chain_ok": chain.get("ok"),
        "actions_count": len(actions),
    }
    man_path = base / f"{iid}_{ts}.manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    _archive_registry.add(bundle_path.resolve())
    _archive_registry.add(man_path.resolve())

    return {**manifest, "manifest_path": str(man_path)}


def purge_audit_archives(interaction_id: str, *, out_dir: Path | None = None) -> int:
    """Purge all archive bundles, manifests, and audit reports for an interaction (erasure)."""
    from src.security.identifiers import assert_under_roots as _assert_roots, safe_out_dir, safe_token_id

    iid = safe_token_id(interaction_id, kind="interaction_id")
    count = 0
    base = safe_out_dir(out_dir, default=REPO_ROOT / "reports" / "qubot" / "archives")
    if base.is_dir():
        for p in list(base.glob(f"{iid}_*.json")) + list(base.glob(f"{iid}.json")):
            try:
                p.unlink(missing_ok=True)
                count += 1
            except Exception:
                pass
    for p in list(_archive_registry):
        if p.name.startswith(f"{iid}_") or p.name.startswith(f"{iid}."):
            try:
                p.unlink(missing_ok=True)
                count += 1
            except Exception:
                pass
            _archive_registry.discard(p)

    report_path = REPORTS_DIR / f"{iid}.md"
    try:
        report_path = _assert_roots(report_path, [REPORTS_DIR])
        if report_path.is_file():
            report_path.unlink(missing_ok=True)
            count += 1
    except Exception:
        pass
    return count


__all__ = ["build_audit_archive", "purge_audit_archives"]
