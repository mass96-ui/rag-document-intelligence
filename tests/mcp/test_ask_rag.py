"""Tests for ask_rag MCP tool."""
import json
from unittest.mock import MagicMock


from rag_document_intelligence.mcp.tools import handle_ask_rag


def _make_mock_pipeline(
    answer="The answer is [1] based on evidence.",
    citations=None,
    refused=False,
    sources=None,
):
    """Create a mock pipeline with configurable answer result."""
    pipeline = MagicMock()
    result = {
        "query": "test",
        "answer": answer,
        "citations": citations if citations is not None else [1],
        "confidence": "low",
        "refused": refused,
        "reason": None,
        "sources": sources or [],
        "patient_context_used": False,
        "ml_context_used": False,
        "source_documents": [],
        "context_length": 100,
    }
    pipeline.answer.return_value = result
    return pipeline


def test_ask_rag_valid_query(mock_pipeline_factory):
    pipeline = _make_mock_pipeline()
    factory = mock_pipeline_factory(pipeline)

    result = handle_ask_rag(
        {"query": "What is machine learning?"},
        pipeline_factory=factory,
    )

    assert result.is_error is False
    data = json.loads(result.content[0].text)
    assert "answer" in data
    assert "[1]" in data["answer"]
    assert data["citations"] == [1]


def test_ask_rag_preserves_citations(mock_pipeline_factory):
    pipeline = _make_mock_pipeline(
        answer="Machine learning uses algorithms [1] and data [2].",
        citations=[1, 2],
    )
    factory = mock_pipeline_factory(pipeline)

    result = handle_ask_rag(
        {"query": "What is ML?", "top_k": 3},
        pipeline_factory=factory,
    )

    data = json.loads(result.content[0].text)
    assert "[1]" in data["answer"]
    assert "[2]" in data["answer"]
    assert data["citations"] == [1, 2]


def test_ask_rag_preserves_refused_flag(mock_pipeline_factory):
    pipeline = _make_mock_pipeline(
        answer="I could not find this information in the provided documents.",
        citations=[],
        refused=True,
    )
    factory = mock_pipeline_factory(pipeline)

    result = handle_ask_rag(
        {"query": "Unknown topic"},
        pipeline_factory=factory,
    )

    data = json.loads(result.content[0].text)
    assert data["refused"] is True
    assert "could not find" in data["answer"].lower()


def test_ask_rag_empty_query():
    result = handle_ask_rag({"query": ""})
    assert result.is_error is True
    assert "empty" in result.content[0].text.lower()


def test_ask_rag_none_query():
    result = handle_ask_rag({"query": None})
    assert result.is_error is True


def test_ask_rag_whitespace_query():
    result = handle_ask_rag({"query": "   "})
    assert result.is_error is True


def test_ask_rag_invalid_top_k():
    result = handle_ask_rag({"query": "test", "top_k": -1})
    assert result.is_error is True


def test_ask_rag_invalid_patient_id():
    result = handle_ask_rag({"query": "test", "patient_id": "../../../etc/passwd"})
    assert result.is_error is True
    assert "invalid characters" in result.content[0].text.lower()


def test_ask_rag_empty_patient_id():
    result = handle_ask_rag({"query": "test", "patient_id": ""})
    assert result.is_error is True


def test_ask_rag_oversized_patient_id():
    result = handle_ask_rag(
        {"query": "test", "patient_id": "x" * 200}
    )
    assert result.is_error is True
    assert "too long" in result.content[0].text.lower()


def test_ask_rag_query_length_validation():
    long_query = "x" * 2001
    result = handle_ask_rag({"query": long_query})
    assert result.is_error is True


def test_ask_rag_pipeline_init_failure():
    def failing_factory():
        raise RuntimeError("Ollama offline")

    result = handle_ask_rag(
        {"query": "test"},
        pipeline_factory=failing_factory,
    )
    assert result.is_error is True
    assert "Failed to initialize" in result.content[0].text


def test_ask_rag_generation_error(mock_pipeline_factory):
    pipeline = MagicMock()
    pipeline.answer.side_effect = RuntimeError("Generation failed")
    factory = mock_pipeline_factory(pipeline)

    result = handle_ask_rag(
        {"query": "test"},
        pipeline_factory=factory,
    )
    assert result.is_error is True
    assert "generation" in result.content[0].text.lower()


def test_ask_rag_with_patient_context(mock_pipeline_factory, mock_patient_context, mock_patient_provider):
    pipeline = _make_mock_pipeline()
    factory = mock_pipeline_factory(pipeline)
    patient_factory = lambda: mock_patient_provider

    result = handle_ask_rag(
        {"query": "test", "patient_id": "P001"},
        pipeline_factory=factory,
        patient_provider_factory=patient_factory,
    )

    assert result.is_error is False
    pipeline.answer.assert_called_once()
    call_kwargs = pipeline.answer.call_args
    assert call_kwargs.kwargs["patient_id"] == "P001"
    assert call_kwargs.kwargs["patient_context_provider"] is mock_patient_provider


def test_ask_rag_medical_safety_enforced(mock_pipeline_factory, mock_patient_context, mock_patient_provider):
    """Ask_rag must use the full pipeline, which enforces medical safety."""
    pipeline = MagicMock()
    pipeline.answer.return_value = {
        "query": "prescription",
        "answer": "I cannot provide a medical recommendation. Please consult a qualified healthcare provider.",
        "citations": [],
        "confidence": "unknown",
        "refused": True,
        "reason": "safe_refusal",
        "sources": [],
        "patient_context_used": True,
        "ml_context_used": False,
        "source_documents": [],
        "context_length": 100,
    }
    factory = mock_pipeline_factory(pipeline)

    result = handle_ask_rag(
        {"query": "What prescription should I take?", "patient_id": "P001"},
        pipeline_factory=factory,
        patient_provider_factory=lambda: mock_patient_provider,
    )

    data = json.loads(result.content[0].text)
    assert data["refused"] is True
    pipeline.answer.assert_called_once()
    assert "prescription" in pipeline.answer.call_args.kwargs["query"].lower()
