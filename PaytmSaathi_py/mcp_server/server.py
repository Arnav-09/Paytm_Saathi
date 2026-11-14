"""Paytm Saathi MCP server (Model Context Protocol, stdio).

The chat assistant starts this as a subprocess and calls its tools through an MCP client, so the
same tools can also be used from any other MCP client (for example Claude Desktop):

    python -m mcp_server.server          # run from the project root

Tools are defined in ai_engine/assistant_tools.py. They are read-only and validate their inputs.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from ai_engine.assistant_tools import TOOLS

mcp = FastMCP("paytm-saathi")
for _tool in TOOLS:
    mcp.tool()(_tool)

if __name__ == "__main__":
    # stdout is the MCP protocol channel. Our logger writes to stderr, so nothing else touches it.
    mcp.run(transport="stdio")
