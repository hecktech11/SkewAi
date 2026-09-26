"""The container image must be able to import the API (R05).

requirements-docker.txt is what the image installs. It omitted
python-multipart, which FastAPI requires for the upload route, and
cryptography, which turn encryption imports at runtime.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _requirement_names(text: str) -> dict[str, str]:
    found: dict[str, str] = {}
    for line in text.splitlines():
        body = line.split("#", 1)[0].strip()
        if not body:
            continue
        name = body.split(">=")[0].split("==")[0].split("<")[0].split("[")[0].strip()
        found[name.lower()] = body
    return found


def test_docker_requirements_include_multipart_and_cryptography():
    text = (ROOT / "requirements-docker.txt").read_text(encoding="utf-8")
    names = _requirement_names(text)
    multipart = names["python-multipart"]
    crypto = names["cryptography"]
    assert "<" in multipart
    assert "<" in crypto
