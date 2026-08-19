"""Tests for validate_medical_evidence and get_ml_context_interface MCP tools."""
import json
from unittest.mock import MagicMock


from rag_document_intelligence.mcp.tools import (
    call_tool,
    handle_validate_medical_evidence,
    handle_get_ml_context_interface,
)


def test_get_ml_context_interface_returns_contract():
    result = handle_get_ml_context_interface({})
    assert result.is_error is False
    data = json.loads(result.content[0].text)
    assert data["available"] is False
    assert "interface" in data
    assert data["interface"]["type"] == "MLSessionResult"
    assert "fields" in data["interface"]


def test_validate_medical_evidence_non_clinical_query(mock_pipeline_factory, sample_retrieved_docs):
    pipeline = MagicMock()
    pipeline.retriever.retrieve.return_value = sample_retrieved_docs
    factory = mock_pipeline_factory(pipeline)

    result = handle_validate_medical_evidence(
        {"query": "What is machine learning?", "top_k": 3},
        pipeline_factory=factory,
    )

    assert result.is_error is False
    data = json.loads(result.content[0].text)
    assert data["is_clinical_query"] is False
    assert data["safe_to_proceed"] is True
    assert "Non-clinical" in data["recommendation"]


def test_validate_medical_evidence_clinical_query_without_approved(
    mock_pipeline_factory, sample_retrieved_docs
):
    pipeline = MagicMock()
    pipeline.retriever.retrieve.return_value = sample_retrieved_docs
    factory = mock_pipeline_factory(pipeline)

    result = handle_validate_medical_evidence(
        {"query": "What prescription should I take?", "top_k": 3},
        pipeline_factory=factory,
    )

    assert result.is_error is False
    data = json.loads(result.content[0].text)
    assert data["is_clinical_query"] is True
    assert data["has_doctor_approved_evidence"] is False
    assert data["safe_to_proceed"] is False
    assert "physician" in data["recommendation"].lower()


def test_validate_medical_evidence_clinical_query_with_approved():
    docs = [
        {
            "rank": 1,
            "content": "Clinical guideline: resistance level 3.",
            "metadata": {
                "source": "clinical_guideline.pdf",
                "source_name": "clinical_guideline.pdf",
                "page": 1,
                "source_type": "clinical_guideline",
                "trust_level": "approved",
            },
            "distance": 0.1,
            "score": 0.9,
            "id": "clinical_guideline.pdf|0|abc",
        }
    ]
    pipeline = MagicMock()
    pipeline.retriever.retrieve.return_value = docs

    def factory():
        return pipeline

    result = handle_validate_medical_evidence(
        {"query": "What dosage should the patient take?", "top_k": 3},
        pipeline_factory=factory,
    )

    assert result.is_error is False
    data = json.loads(result.content[0].text)
    assert data["is_clinical_query"] is True
    assert data["has_doctor_approved_evidence"] is True
    assert data["safe_to_proceed"] is True
    assert "meets clinical" in data["recommendation"].lower()


def test_validate_medical_evidence_empty_query():
    result = handle_validate_medical_evidence({"query": ""})
    assert result.is_error is True


def test_validate_medical_evidence_invalid_top_k():
    result = handle_validate_medical_evidence({"query": "test", "top_k": 0})
    assert result.is_error is True


def test_validate_medical_evidence_pipeline_init_failure():
    def failing_factory():
        raise RuntimeError("ChromaDB down")

    result = handle_validate_medical_evidence(
        {"query": "test"},
        pipeline_factory=failing_factory,
    )
    assert result.is_error is True
    assert "Failed to initialize" in result.content[0].text


def test_validate_medical_evidence_retrieval_failure():
    pipeline = MagicMock()
    pipeline.retriever.retrieve.side_effect = RuntimeError("DB error")

    result = handle_validate_medical_evidence(
        {"query": "test", "top_k": 3},
        pipeline_factory=lambda: pipeline,
    )
    assert result.is_error is True
    assert "Retrieval failed" in result.content[0].text


def test_validate_medical_evidence_dosage_query():
    docs = [
        {
            "rank": 1,
            "content": "General exercise info.",
            "metadata": {
                "source": "general.pdf",
                "source_name": "general.pdf",
                "page": 1,
                "source_type": "general_reference",
                "trust_level": "unspecified",
            },
            "distance": 0.3,
            "score": 0.7,
            "id": "general.pdf|0|abc",
        }
    ]
    pipeline = MagicMock()
    pipeline.retriever.retrieve.return_value = docs

    result = handle_validate_medical_evidence(
        {"query": "What dosage should the patient take?", "top_k": 3},
        pipeline_factory=lambda: pipeline,
    )

    assert result.is_error is False
    data = json.loads(result.content[0].text)
    assert data["is_clinical_query"] is True
    assert data["has_doctor_approved_evidence"] is False
