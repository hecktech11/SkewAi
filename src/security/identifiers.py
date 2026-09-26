"""Jail for identifiers that become filesystem paths or DuckDB filenames.

pack_id is concatenated into ``{pack_id}.duckdb`` and ``domains/{pack_id}/``.
Anything that is not a short ASCII token is rejected before Path joins.
"""

from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path
from typing import Iterable

PACK_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,62}$")
TOKEN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,95}$")
MAX_PACK_ID_LEN = 64
MAX_TOKEN_ID_LEN = 96
MAX_CLUSTER_ID = 10_000_000


class InvalidIdentifier(ValueError):
    """Raised when a caller-supplied id cannot be used as a path or DB name."""


def _reject_binary(raw: str, kind: str) -> str:
    if not isinstance(raw, str):
        raise InvalidIdentifier(f"invalid {kind}")
    if "\x00" in raw or "\n" in raw or "\r" in raw:
        raise InvalidIdentifier(f"invalid {kind}")
    if any(ord(c) < 32 for c in raw):
        raise InvalidIdentifier(f"invalid {kind}")
    return raw


def safe_pack_id(pack_id: str | None, *, optional: bool = False) -> str | None:
    """Return a pack_id safe to use in a filename / directory name.

    Rejects empty (unless optional), oversize, path traversal, null bytes,
    and non-ASCII / NFKC-shifted strings.
    """
    if pack_id is None or pack_id == "":
        if optional:
            return None
        raise InvalidIdentifier("pack_id required")
    raw = _reject_binary(pack_id, "pack_id")
    if len(raw) > MAX_PACK_ID_LEN:
        raise InvalidIdentifier("pack_id too long")
    if unicodedata.normalize("NFKC", raw) != raw:
        raise InvalidIdentifier("invalid pack_id")
    if "/" in raw or "\\" in raw or ".." in raw:
        raise InvalidIdentifier("invalid pack_id")
    if not PACK_ID_RE.match(raw):
        raise InvalidIdentifier("invalid pack_id")
    return raw


def safe_token_id(value: str | None, *, kind: str = "id") -> str:
    """interaction_id / investigation_id / case_id / review_id."""
    if value is None or value == "":
        raise InvalidIdentifier(f"{kind} required")
    raw = _reject_binary(value, kind)
    if len(raw) > MAX_TOKEN_ID_LEN:
        raise InvalidIdentifier(f"{kind} too long")
    if unicodedata.normalize("NFKC", raw) != raw:
        raise InvalidIdentifier(f"invalid {kind}")
    if "/" in raw or "\\" in raw or ".." in raw:
        raise InvalidIdentifier(f"invalid {kind}")
    if not TOKEN_ID_RE.match(raw):
        raise InvalidIdentifier(f"invalid {kind}")
    return raw


def safe_cluster_id(value: object) -> int:
    try:
        n = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as e:
        raise InvalidIdentifier("invalid cluster_id") from e
    if n < 0 or n > MAX_CLUSTER_ID:
        raise InvalidIdentifier("invalid cluster_id")
    return n


def assert_under_roots(path: Path, roots: Iterable[Path]) -> Path:
    """Resolve *path* and require it to sit under one of *roots*."""
    try:
        resolved = path.expanduser().resolve()
    except OSError as e:
        raise InvalidIdentifier(f"invalid path: {e}") from e
    allowed = []
    for root in roots:
        try:
            allowed.append(root.expanduser().resolve())
        except OSError:
            allowed.append(root)
    for root in allowed:
        try:
            resolved.relative_to(root)
            return resolved
        except ValueError:
            continue
    raise InvalidIdentifier("path outside allow-root jail")


def safe_out_dir(out_dir: str | Path | None, *, default: Path) -> Path:
    """Jail export directories under reports/, eval/, or FRONTLINE_EXPORT_DIR."""
    from src.config import REPO_ROOT

    if out_dir is None or out_dir == "":
        return default
    raw = Path(str(out_dir))
    roots = [REPO_ROOT / "reports", REPO_ROOT / "eval"]
    extra = (os.getenv("FRONTLINE_EXPORT_DIR") or "").strip()
    if extra:
        roots.append(Path(extra))
    if os.getenv("PYTEST_CURRENT_TEST") or os.getenv("FRONTLINE_TEST_ISOLATION"):
        import tempfile

        roots.append(Path(tempfile.gettempdir()))
    return assert_under_roots(raw if raw.is_absolute() else (REPO_ROOT / raw), roots)


def _ingest_roots() -> list[Path]:
    """Allow-roots for caller-supplied ingest files."""
    from src.config import REPO_ROOT

    roots = [REPO_ROOT]
    for env_name in ("PACK_INSTALL_ROOT", "BUILDER_UPLOAD_DIR", "FRONTLINE_INGEST_ROOT"):
        extra = (os.getenv(env_name) or "").strip()
        if extra:
            roots.append(Path(extra))
    if os.getenv("PYTEST_CURRENT_TEST") or os.getenv("FRONTLINE_TEST_ISOLATION"):
        import tempfile

        roots.append(Path(tempfile.gettempdir()))
    return roots


def safe_csv_path(csv_path: str | Path) -> Path:
    """Jail CSV reads under the repo (or PACK_INSTALL_ROOT / BUILDER_UPLOAD_DIR)."""
    from src.config import REPO_ROOT

    raw = Path(str(csv_path))
    roots = _ingest_roots()
    p = raw if raw.is_absolute() else (REPO_ROOT / raw)
    resolved = assert_under_roots(p, roots)
    if resolved.suffix.lower() != ".csv":
        raise InvalidIdentifier("csv_path must be a .csv file")
    return resolved


def safe_mapping_path(mapping_path: str | Path) -> Path:
    """Jail mapping YAML reads under the same roots as ``safe_csv_path``."""
    from src.config import REPO_ROOT

    raw = Path(str(mapping_path))
    p = raw if raw.is_absolute() else (REPO_ROOT / raw)
    resolved = assert_under_roots(p, _ingest_roots())
    if resolved.suffix.lower() not in {".yaml", ".yml"}:
        raise InvalidIdentifier("mapping_path must be a .yaml file")
    return resolved


__all__ = [
    "InvalidIdentifier",
    "safe_pack_id",
    "safe_token_id",
    "safe_cluster_id",
    "assert_under_roots",
    "safe_out_dir",
    "safe_csv_path",
    "safe_mapping_path",
]
