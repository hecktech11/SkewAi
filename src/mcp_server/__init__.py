"""Read-only MCP surface for local Claude and Cursor sessions."""

from src.mcp_server.protocol import handle_message, mcp_open
from src.mcp_server.tools import call_tool

__all__ = ["call_tool", "handle_message", "mcp_open"]
