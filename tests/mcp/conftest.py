"""Shared fixtures for MCP tests."""
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_pipeline_factory():
    """Return a factory that produces a configurable mock pipeline."""

    def _make(pipeline):
        def factory():
            return pipeline
        return factory

    return _make


@pytest.fixture
def mock_patient_provider_factory():
    """Return a factory that produces a configurable mock patient provider."""

    def _make(provider):
        def factory():
            return provider
        return factory

    return _make


@pytest.fixture
def mock_retriever_with_docs():
    """Return a mock retriever that returns preset documents."""

    def _make(docs):
        retriever = MagicMock()
        retriever.retrieve.return_value = docs
        return retriever

    return _make


@pytest.fixture
def sample_retrieved_docs():
    """Return sample retrieved documents with metadata."""
    return [
        {
            "rank": 1,
            "content": "Machine learning enables computers to learn from data.",
            "metadata": {
                "source": "ml.txt",
                "source_name": "ml.txt",
                "page": 1,
                "input_type": "txt",
                "source_type": "general_reference",
                "trust_level": "unspecified",
            },
            "distance": 0.1,
            "score": 0.9,
            "id": "ml.txt|0|abc123",
        },
        {
            "rank": 2,
            "content": "Python is a versatile programming language.",
            "metadata": {
                "source": "python.txt",
                "source_name": "python.txt",
                "page": 1,
                "input_type": "txt",
                "source_type": "general_reference",
                "trust_level": "unspecified",
            },
            "distance": 0.3,
            "score": 0.77,
            "id": "python.txt|0|def456",
        },
    ]


@pytest.fixture
def sample_cited_answer_docs():
    """Return documents suitable for citation-enforcement tests."""
    return [
        {
            "rank": 1,
            "content": "Python is a high-level programming language.",
            "metadata": {
                "source": "python.txt",
                "source_name": "python.txt",
                "page": 1,
                "input_type": "txt",
            },
            "distance": 0.1,
            "score": 0.9,
            "id": "python.txt|0|ghi789",
        }
    ]


@pytest.fixture
def mock_patient_context():
    """Return a mock PatientContext."""
    from rag_document_intelligence.patient_context import (
        PatientContext, Demographics,
    )
    return PatientContext(
        patient_id="P001",
        demographics=Demographics(age=45, sex="male", bmi=26.7),
        medical_conditions=["hypertension"],
        contraindications=["avoid heavy resistance"],
    )


@pytest.fixture
def mock_patient_provider(mock_patient_context):
    """Return a mock PatientContextProvider."""

    class _MockProvider:
        def __init__(self, context):
            self._context = context
            self.requested_id = None

        def get_patient_context(self, patient_id):
            self.requested_id = patient_id
            return self._context

        def patient_exists(self, patient_id):
            return patient_id == self._context.patient_id

    return _MockProvider(mock_patient_context)
