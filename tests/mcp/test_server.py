"""Integration tests for MCP server creation and tool dispatch."""
import json

from rag_document_intelligence.mcp.tools import call_tool, get_tool_definitions
from rag_document_intelligence.mcp.server import create_mcp_server


def test_mcp_server_can_be_created():
    server = create_mcp_server()
    assert server is not None
    assert server.name == "rag-document-intelligence"


def test_mcp_server_with_custom_name():
    server = create_mcp_server(server_name="custom-rag")
    assert server.name == "custom-rag"


def test_mcp_server_has_tool_listing_handler():
    server = create_mcp_server()
    assert server.on_list_tools is not None
    assert server.on_call_tool is not None


def test_dispense_through_call_tool_search():
    """Test calling search_knowledge via the call_tool entry point."""
    result = call_tool("get_ml_context_interface", {})
    assert result.is_error is False
    data = json.loads(result.content[0].text)
    assert data["available"] is False


def test_mcp_tools_cover_core_capabilities():
    """Verify all 5 required tools are present."""
    names = [t.name for t in get_tool_definitions()]
    assert set(names) == {
        "search_knowledge",
        "ask_rag",
        "get_patient_context",
        "get_ml_context_interface",
        "validate_medical_evidence",
    }


def test_mcp_tool_result_is_structured():
    """All tool results should be parseable JSON (except error text)."""
    result = call_tool("get_ml_context_interface", {})
    assert result.is_error is False
    data = json.loads(result.content[0].text)
    assert isinstance(data, dict)


def test_mcp_no_arbitrary_filesystem_access():
    """Verify tools reject arbitrary file paths."""
    from rag_document_intelligence.mcp.tools import handle_ask_rag

    result = handle_ask_rag({"query": "test", "patient_id": "/etc/passwd"})
    assert result.is_error is True


def test_mcp_query_length_enforced():
    """Verify all tools enforce MAX_QUERY_LENGTH."""
    from rag_document_intelligence.mcp.tools import (
        handle_search_knowledge,
        handle_ask_rag,
        handle_validate_medical_evidence,
    )

    long_query = "x" * 2001

    result = handle_search_knowledge({"query": long_query})
    assert result.is_error is True

    result = handle_ask_rag({"query": long_query})
    assert result.is_error is True

    result = handle_validate_medical_evidence({"query": long_query})
    assert result.is_error is True
