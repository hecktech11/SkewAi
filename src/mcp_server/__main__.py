"""stdio entry: ``python -m src.mcp_server`` or this file by absolute path."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.mcp_server.protocol import serve


def main() -> None:
    serve(sys.stdin.buffer, sys.stdout.buffer)


if __name__ == "__main__":
    main()
