"""Tests for search_knowledge MCP tool."""
import json
from unittest.mock import MagicMock


from rag_document_intelligence.mcp.tools import handle_search_knowledge


def test_search_knowledge_valid_query(sample_retrieved_docs, mock_pipeline_factory):
    pipeline = MagicMock()
    pipeline.retriever.retrieve.return_value = sample_retrieved_docs

    factory = mock_pipeline_factory(pipeline)
    result = handle_search_knowledge(
        {"query": "machine learning", "top_k": 3},
        pipeline_factory=factory,
    )

    assert result.is_error is False
    data = json.loads(result.content[0].text)
    assert data["query"] == "machine learning"
    assert data["result_count"] == 2
    assert data["sources"][0]["rank"] == 1
    assert data["sources"][0]["source"] == "ml.txt"
    assert data["sources"][1]["rank"] == 2
    assert "snippet" in data["sources"][0]


def test_search_knowledge_empty_query():
    result = handle_search_knowledge({"query": "", "top_k": 5})
    assert result.is_error is True
    assert "empty" in result.content[0].text.lower()


def test_search_knowledge_none_query():
    result = handle_search_knowledge({"query": None, "top_k": 5})
    assert result.is_error is True


def test_search_knowledge_whitespace_query():
    result = handle_search_knowledge({"query": "   ", "top_k": 5})
    assert result.is_error is True


def test_search_knowledge_default_top_k(sample_retrieved_docs, mock_pipeline_factory):
    pipeline = MagicMock()
    pipeline.retriever.retrieve.return_value = sample_retrieved_docs

    factory = mock_pipeline_factory(pipeline)
    result = handle_search_knowledge(
        {"query": "test"},
        pipeline_factory=factory,
    )

    assert result.is_error is False
    pipeline.retriever.retrieve.assert_called_once()
    call_kwargs = pipeline.retriever.retrieve.call_args
    assert call_kwargs.kwargs["top_k"] == 5


def test_search_knowledge_invalid_top_k():
    result = handle_search_knowledge({"query": "test", "top_k": 0})
    assert result.is_error is True
    assert "top_k" in result.content[0].text


def test_search_knowledge_negative_top_k():
    result = handle_search_knowledge({"query": "test", "top_k": -1})
    assert result.is_error is True


def test_search_knowledge_top_k_too_large():
    result = handle_search_knowledge({"query": "test", "top_k": 100})
    assert result.is_error is True


def test_search_knowledge_non_integer_top_k():
    result = handle_search_knowledge({"query": "test", "top_k": "abc"})
    assert result.is_error is True


def test_search_knowledge_pipeline_init_failure():
    def failing_factory():
        raise RuntimeError("ChromaDB is down")

    result = handle_search_knowledge(
        {"query": "test"},
        pipeline_factory=failing_factory,
    )
    assert result.is_error is True
    assert "Failed to initialize" in result.content[0].text


def test_search_knowledge_retrieval_failure(sample_retrieved_docs, mock_pipeline_factory):
    pipeline = MagicMock()
    pipeline.retriever.retrieve.side_effect = RuntimeError("Vector store down")

    factory = mock_pipeline_factory(pipeline)
    result = handle_search_knowledge(
        {"query": "test", "top_k": 3},
        pipeline_factory=factory,
    )
    assert result.is_error is True
    assert "Retrieval failed" in result.content[0].text


def test_search_knowledge_no_results(mock_pipeline_factory):
    pipeline = MagicMock()
    pipeline.retriever.retrieve.return_value = []

    factory = mock_pipeline_factory(pipeline)
    result = handle_search_knowledge(
        {"query": "test", "top_k": 3},
        pipeline_factory=factory,
    )

    assert result.is_error is False
    data = json.loads(result.content[0].text)
    assert data["result_count"] == 0
    assert data["sources"] == []


def test_search_knowledge_no_filesystem_path_exposure(sample_retrieved_docs, mock_pipeline_factory):
    """Ensure no internal filesystem paths are exposed in results."""
    pipeline = MagicMock()
    pipeline.retriever.retrieve.return_value = sample_retrieved_docs

    factory = mock_pipeline_factory(pipeline)
    result = handle_search_knowledge(
        {"query": "test", "top_k": 3},
        pipeline_factory=factory,
    )

    data = json.loads(result.content[0].text)
    for source in data["sources"]:
        assert "filepath" not in source
        assert "path" not in source
        assert "data/" not in str(source.get("source", ""))
        assert "/tmp/" not in str(source.get("source", ""))


def test_search_knowledge_query_length_validation():
    long_query = "x" * 2001
    result = handle_search_knowledge({"query": long_query, "top_k": 5})
    assert result.is_error is True
    assert "maximum length" in result.content[0].text.lower()
