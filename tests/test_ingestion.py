"""Tests for document ingestion and duplicate prevention."""

from pathlib import Path

import numpy as np
import pytest
from langchain_core.documents import Document

from rag_document_intelligence.vector_store import VectorStore
from rag_document_intelligence.chunking import DocumentChunker


def test_vector_store_document_id_stable_across_metadata_variants():
    """ID must be deterministic for the same source/page/content."""
    doc = Document(
        page_content="Same content text",
        metadata={"source_name": "file.pdf", "page": 3},
    )
    id1 = VectorStore._create_document_id(doc)
    id2 = VectorStore._create_document_id(doc)
    assert id1 == id2


def test_vector_store_document_id_different_content():
    """Different content must produce different IDs."""
    doc_a = Document(
        page_content="content A",
        metadata={"source_name": "file.pdf", "page": 1},
    )
    doc_b = Document(
        page_content="content B",
        metadata={"source_name": "file.pdf", "page": 1},
    )
    assert VectorStore._create_document_id(doc_a) != \
        VectorStore._create_document_id(doc_b)


def test_vector_store_document_id_uses_source_name():
    """ID should prefer source_name over source."""
    doc = Document(
        page_content="content",
        metadata={"source": "/full/path/file.pdf", "source_name": "file.pdf"},
    )
    doc_id = VectorStore._create_document_id(doc)
    assert "file.pdf" in doc_id
    assert "/full/path" not in doc_id


def test_vector_store_add_documents_prevents_duplicates(tmp_path):
    """Adding the same documents twice should skip them the second time."""
    persist_dir = tmp_path / "vector_store"
    persist_dir.mkdir()

    store = VectorStore(
        collection_name="test_dedup",
        persist_directory=persist_dir,
    )

    doc = Document(
        page_content="Unique chunk content",
        metadata={"source_name": "test.pdf", "page": 1},
    )

    embeddings = np.array([[0.1] * 384])

    first_count = store.add_documents([doc], embeddings)
    assert first_count == 1

    second_count = store.add_documents([doc], embeddings)
    assert second_count == 0
    assert store.count() == 1


def test_vector_store_add_documents_skips_only_duplicates(tmp_path):
    """Only duplicate IDs should be skipped, not all documents."""
    persist_dir = tmp_path / "vector_store"
    persist_dir.mkdir()

    store = VectorStore(
        collection_name="test_partial_dup",
        persist_directory=persist_dir,
    )

    doc_a = Document(
        page_content="Chunk A content",
        metadata={"source_name": "a.pdf", "page": 1},
    )
    doc_b = Document(
        page_content="Chunk B content",
        metadata={"source_name": "b.pdf", "page": 1},
    )

    embeddings = np.array([[0.1] * 384, [0.2] * 384])

    first_count = store.add_documents([doc_a, doc_b], embeddings)
    assert first_count == 2

    second_count = store.add_documents([doc_a, doc_b], embeddings)
    assert second_count == 0
    assert store.count() == 2


def test_vector_store_rejects_mismatched_counts(tmp_path):
    persist_dir = tmp_path / "vector_store"
    persist_dir.mkdir()

    store = VectorStore(
        collection_name="test_mismatch",
        persist_directory=persist_dir,
    )

    doc = Document(
        page_content="content",
        metadata={"source_name": "test.pdf"},
    )

    embeddings = np.array([[0.1] * 384, [0.2] * 384])

    with pytest.raises(ValueError, match="match"):
        store.add_documents([doc], embeddings)


def test_vector_store_empty_add_returns_zero(tmp_path):
    persist_dir = tmp_path / "vector_store"
    persist_dir.mkdir()

    store = VectorStore(
        collection_name="test_empty",
        persist_directory=persist_dir,
    )

    result = store.add_documents([], np.array([]))
    assert result == 0


def test_chunker_filters_empty_content():
    """Chunks with empty content after stripping should be removed."""
    doc = Document(page_content="A" * 200)
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.split_documents([doc])
    assert all(chunk.page_content.strip() for chunk in chunks)
