"""Shared test fixtures for the RAG Document Intelligence test suite."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from langchain_core.documents import Document

from rag_document_intelligence.chunking import DocumentChunker


@pytest.fixture
def tmp_docs_dir(tmp_path: Path) -> Path:
    """Create a temporary documents directory."""
    docs = tmp_path / "docs"
    docs.mkdir()
    return docs


@pytest.fixture
def sample_documents() -> list[Document]:
    """Return a small set of sample documents for testing."""
    return [
        Document(
            page_content="Python is a high-level programming language.",
            metadata={"source": "python.txt", "source_name": "python.txt", "input_type": "txt"},
        ),
        Document(
            page_content="Machine learning enables computers to learn from data.",
            metadata={"source": "ml.txt", "source_name": "ml.txt", "input_type": "txt"},
        ),
    ]


@pytest.fixture
def mock_embedding_manager():
    """Return a mock embedding manager that produces deterministic embeddings."""
    mgr = MagicMock()
    vector = np.random.default_rng(42).random(384)
    mgr.generate_embeddings.return_value = np.array([vector])
    return mgr


@pytest.fixture
def make_mock_collection():
    """Factory to build a mock ChromaDB collection with configurable results."""

    def _make(
        ids=None,
        documents=None,
        metadatas=None,
        distances=None,
    ):
        collection = MagicMock()
        collection.query.return_value = {
            "ids": [ids or []],
            "documents": [documents or []],
            "metadatas": [metadatas or []],
            "distances": [distances or []],
        }
        collection.count.return_value = len(ids or [])
        return collection

    return _make


@pytest.fixture
def mock_vector_store(make_mock_collection):
    """Return a mock vector store with a mock collection."""
    store = MagicMock()
    store.collection = make_mock_collection(
        ids=["doc1", "doc2"],
        documents=["Content of doc 1", "Content of doc 2"],
        metadatas=[
            {"source_name": "doc1.pdf", "page": 1, "input_type": "pdf"},
            {"source_name": "doc2.pdf", "page": 2, "input_type": "pdf"},
        ],
        distances=[0.3, 0.6],
    )
    return store


@pytest.fixture
def empty_mock_vector_store(make_mock_collection):
    """Return a mock vector store with an empty collection."""
    store = MagicMock()
    store.collection = make_mock_collection()
    return store


@pytest.fixture
def chunker() -> DocumentChunker:
    """Return a standard DocumentChunker for tests."""
    return DocumentChunker(chunk_size=100, chunk_overlap=20)
