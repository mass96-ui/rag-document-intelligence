"""Tests for MCP tool listings and definitions."""

from rag_document_intelligence.mcp.tools import get_tool_definitions, call_tool


def test_tool_listing_returns_all_tools():
    tools = get_tool_definitions()
    names = [t.name for t in tools]
    assert "search_knowledge" in names
    assert "ask_rag" in names
    assert "get_patient_context" in names
    assert "get_ml_context_interface" in names
    assert "validate_medical_evidence" in names


def test_tool_definitions_return_copy():
    """get_tool_definitions should return a copy, not the internal list."""
    tools1 = get_tool_definitions()
    tools2 = get_tool_definitions()
    assert tools1 is not tools2
    assert tools1 == tools2


def test_search_knowledge_schema():
    tools = get_tool_definitions()
    tool = next(t for t in tools if t.name == "search_knowledge")
    assert tool.input_schema["required"] == ["query"]
    assert "query" in tool.input_schema["properties"]
    assert "top_k" in tool.input_schema["properties"]


def test_ask_rag_schema():
    tools = get_tool_definitions()
    tool = next(t for t in tools if t.name == "ask_rag")
    assert tool.input_schema["required"] == ["query"]
    assert "patient_id" in tool.input_schema["properties"]
    assert "top_k" in tool.input_schema["properties"]


def test_get_patient_context_schema():
    tools = get_tool_definitions()
    tool = next(t for t in tools if t.name == "get_patient_context")
    assert tool.input_schema["required"] == ["patient_id"]
    assert "patient_id" in tool.input_schema["properties"]


def test_get_ml_context_interface_schema():
    tools = get_tool_definitions()
    tool = next(t for t in tools if t.name == "get_ml_context_interface")
    assert tool.input_schema["properties"] == {}


def test_validate_medical_evidence_schema():
    tools = get_tool_definitions()
    tool = next(t for t in tools if t.name == "validate_medical_evidence")
    assert tool.input_schema["required"] == ["query"]
    assert "top_k" in tool.input_schema["properties"]


def test_unknown_tool_call_returns_error():
    result = call_tool("nonexistent_tool", {"query": "test"})
    assert result.is_error is True
    assert "Unknown tool" in result.content[0].text


def test_call_tool_with_empty_arguments():
    result = call_tool("get_ml_context_interface", {})
    assert result.is_error is False
    import json
    data = json.loads(result.content[0].text)
    assert data["available"] is False


def test_call_tool_with_none_arguments():
    result = call_tool("get_ml_context_interface", None)
    assert result.is_error is False
    import json
    data = json.loads(result.content[0].text)
    assert data["available"] is False
