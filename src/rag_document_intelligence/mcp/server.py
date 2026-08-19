"""MCP server transport for RAG Document Intelligence.

This module provides the entry point for running the RAG pipeline as an
MCP (Model Context Protocol) server. It is kept separate from the core
RAG logic so that the RAG remains usable independently via its Python API
and CLI.

Usage:

    # Run as an MCP server over stdio:
    python -m rag_document_intelligence.mcp.server

    # Or call run() programmatically:
    from rag_document_intelligence.mcp.server import run
    run()
"""
import asyncio
import logging
import sys
from typing import Any, Optional

from mcp.server import InitializationOptions
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequestParams,
    CallToolResult,
    ListToolsResult,
    PaginatedRequestParams,
)

from .tools import get_tool_definitions, call_tool

logger = logging.getLogger(__name__)


def create_mcp_server(server_name: str = "rag-document-intelligence") -> Server:
    """Create an MCP Server instance with all RAG tools registered.

    The server uses the low-level MCP Server with ``on_list_tools`` and
    ``on_call_tool`` callbacks. The actual tool handlers in ``tools.py``
    delegate to the existing RAG pipeline — no logic is duplicated.

    Args:
        server_name: The MCP server name (default: "rag-document-intelligence").

    Returns:
        A configured MCP Server instance ready to run.
    """
    server = Server(server_name)

    async def _list_tools(
        ctx,
        params: Optional[PaginatedRequestParams] = None,
    ) -> ListToolsResult:
        return ListToolsResult(tools=get_tool_definitions())

    async def _call_tool(
        ctx,
        params: CallToolRequestParams,
    ) -> CallToolResult:
        return call_tool(params.name, params.arguments or {})

    server.on_list_tools = _list_tools
    server.on_call_tool = _call_tool

    return server


def run() -> None:
    """Run the MCP server over stdio.

    This is the main entry point for running the RAG as an MCP server.
    The server communicates via stdio, which is the standard transport
    for MCP servers used by Claude Desktop and compatible agents.
    """
    logging.basicConfig(
        level=logging.WARNING,
        stream=sys.stderr,
    )

    server = create_mcp_server()
    init_options = InitializationOptions(
        server=server,
    )

    try:
        asyncio.run(stdio_server().run(init_options))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    run()
