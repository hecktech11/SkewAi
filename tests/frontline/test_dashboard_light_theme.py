"""Dashboard light-theme structural gate.

Drives the real shipped assert-light-theme.mjs against source + dist CSS
(after build). Proves light tokens + tokenized shell/cards exist and dark
defaults remain dark.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DASH = REPO / "dashboard"
ASSERT = DASH / "scripts" / "assert-light-theme.mjs"


def _toolchain_missing() -> str:
    """Reason the dashboard build cannot run here, or "" if it can.

    CI installs node deps before pytest (see .github/workflows), so this never
    excuses the gate there — it only keeps a Python-only checkout runnable.
    """
    if os.getenv("CI", "").strip().lower() in {"1", "true", "yes", "on"}:
        return ""
    if shutil.which("npm") is None or shutil.which("node") is None:
        return "node/npm not installed"
    if not (DASH / "node_modules").is_dir():
        return "dashboard/node_modules missing — run `npm ci` in dashboard/"
    return ""


@pytest.fixture(scope="module")
def built_dashboard() -> None:
    assert ASSERT.is_file(), f"missing theme assert script: {ASSERT}"
    reason = _toolchain_missing()
    if reason:
        pytest.skip(reason)
    # Ensure dist exists so the script checks production CSS
    r = subprocess.run(
        ["npm", "run", "build"],
        cwd=DASH,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 0, f"npm build failed:\n{r.stdout}\n{r.stderr}"


def test_assert_light_theme_script_passes(built_dashboard: None) -> None:
    r = subprocess.run(
        ["node", str(ASSERT)],
        cwd=DASH,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert r.returncode == 0, f"theme assert failed:\n{r.stdout}\n{r.stderr}"
    assert "assert-light-theme: PASSED" in r.stdout
    assert "light --bg" in r.stdout or "OK: light --bg" in r.stdout
    assert "dark :root --bg" in r.stdout or "OK: dark :root --bg" in r.stdout


def test_source_css_has_no_dark_only_shell_hexes() -> None:
    """Direct read of shipped sources — no reimplementation of the UI."""
    bad = (
        "#10151e",
        "#0a0d14",
        "#121824",
        "#111722",
        "#151b27",
        "#222a3c",
        "#0b0e14",
    )
    blobs = []
    for name in ("styles.css", "console.css", "ux-v21.css"):
        p = DASH / "src" / name
        assert p.is_file(), p
        blobs.append(p.read_text(encoding="utf-8").lower())
    joined = "\n".join(blobs)
    for hex_ in bad:
        assert hex_.lower() not in joined, f"dark-only fill still in CSS: {hex_}"


def test_light_token_block_and_shell_overrides_exist() -> None:
    console = (DASH / "src" / "console.css").read_text(encoding="utf-8")
    ux = (DASH / "src" / "ux-v21.css").read_text(encoding="utf-8")
    assert ':root[data-theme="light"]' in console
    assert "--bg:" in console
    assert "--ink:" in console
    for sel in (
        ':root[data-theme="light"] .sidebar',
        ':root[data-theme="light"] .main',
        ':root[data-theme="light"] .jump-tile',
        ':root[data-theme="light"] .stat-card',
    ):
        assert sel in ux or sel in console, f"missing {sel}"


def test_house_tokens_are_charcoal_not_teal_sora() -> None:
    """Drive shipped CSS: Image #1 charcoal system, not the old teal ops pair."""
    styles = (DASH / "src" / "styles.css").read_text(encoding="utf-8")
    ux = (DASH / "src" / "ux-v21.css").read_text(encoding="utf-8")
    prompt = (REPO / "docs" / "design" / "console-prompt.md").read_text(encoding="utf-8")
    assert "--bg: #141413" in styles
    assert "#090b10" not in styles
    assert "#2ec4a7" not in styles
    assert "Sora" not in styles.split("--sans:")[1][:80]
    assert "IBM Plex Mono" not in styles.split("--mono:")[1][:80]
    assert "brand-scan" not in ux
    assert "near-black warm charcoal" in prompt.lower() or "Near-black **warm charcoal**" in prompt
    assert "Sora + IBM Plex Mono" in prompt
    assert "image-718f63a4-a33d-4d6f-aa21-9a49784ef657.png" in prompt
    cc = (DASH / "routes" / "CommandCenter.jsx").read_text(encoding="utf-8")
    greet = (DASH / "src" / "ui" / "greeting.js").read_text(encoding="utf-8")
    assert "dayGreeting" in cc
    assert "Good afternoon" in greet
    assert "cc-greeting" in cc
