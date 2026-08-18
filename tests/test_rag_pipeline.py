from pathlib import Path

import numpy as np
import pytest
from langchain_core.documents import Document

from rag_document_intelligence.chunking import DocumentChunker
from rag_document_intelligence.loaders import DocumentLoader
from rag_document_intelligence.vector_store import VectorStore
from rag_document_intelligence.retriever import RAGRetriever


def test_document_loader_loads_txt(tmp_path: Path):
    document = tmp_path / "example.txt"
    document.write_text("Hello RAG", encoding="utf-8")

    loader = DocumentLoader(tmp_path)

    documents = loader.load_text_documents()

    assert len(documents) == 1
    assert documents[0].page_content == "Hello RAG"


def test_document_chunker_returns_chunks():
    documents = [
        Document(page_content="A " * 1000)
    ]

    chunker = DocumentChunker(
        chunk_size=100,
        chunk_overlap=20,
    )

    chunks = chunker.split_documents(documents)

    assert len(chunks) > 1
    assert all(chunk.page_content for chunk in chunks)


def test_document_chunker_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        DocumentChunker(chunk_size=100, chunk_overlap=100)

    with pytest.raises(ValueError):
        DocumentChunker(chunk_size=0)

    with pytest.raises(ValueError):
        DocumentChunker(chunk_size=100, chunk_overlap=-1)


def test_vector_store_document_id_is_deterministic():
    document = Document(
        page_content="Same content",
        metadata={
            "source": "example.pdf",
            "page": 1,
        },
    )

    first_id = VectorStore._create_document_id(document)
    second_id = VectorStore._create_document_id(document)

    assert first_id == second_id
    assert "example.pdf" in first_id


def test_retriever_rejects_empty_query():
    retriever = RAGRetriever(
        vector_store=None,
        embedding_manager=None,
    )

    with pytest.raises(ValueError):
        retriever.retrieve("")

    with pytest.raises(ValueError):
        retriever.retrieve("   ")


def test_retriever_rejects_invalid_top_k():
    retriever = RAGRetriever(
        vector_store=None,
        embedding_manager=None,
    )

    with pytest.raises(ValueError):
        retriever.retrieve("test", top_k=0)

    with pytest.raises(ValueError):
        retriever.retrieve("test", top_k=-1)


def test_retriever_rejects_oversized_query(mock_vector_store, mock_embedding_manager):
    retriever = RAGRetriever(
        vector_store=mock_vector_store,
        embedding_manager=mock_embedding_manager,
    )
    long_query = "x" * 5000
    with pytest.raises(ValueError, match="maximum length"):
        retriever.retrieve(long_query)


def test_retriever_returns_empty_when_no_results(empty_mock_vector_store, mock_embedding_manager):
    retriever = RAGRetriever(
        vector_store=empty_mock_vector_store,
        embedding_manager=mock_embedding_manager,
    )
    results = retriever.retrieve("test query")
    assert results == []


def test_retriever_sorts_deterministically(mock_vector_store, mock_embedding_manager):
    mock_vector_store.collection = mock_vector_store.collection
    mock_vector_store.collection.query.return_value = {
        "ids": [["c", "a", "b"]],
        "documents": [["Content C", "Content A", "Content B"]],
        "metadatas": [
            [{"source_name": "c.pdf"}, {"source_name": "a.pdf"}, {"source_name": "b.pdf"}]
        ],
        "distances": [[0.5, 0.1, 0.3]],
    }
    retriever = RAGRetriever(
        vector_store=mock_vector_store,
        embedding_manager=mock_embedding_manager,
    )
    results = retriever.retrieve("test")
    assert [r["id"] for r in results] == ["a", "b", "c"]
    assert [r["rank"] for r in results] == [1, 2, 3]


def test_retriever_deduplicates_ids(mock_vector_store, mock_embedding_manager):
    mock_vector_store.collection.query.return_value = {
        "ids": [["dup", "dup", "unique"]],
        "documents": [["Content A", "Content A", "Content B"]],
        "metadatas": [
            [{"source_name": "a.pdf"}, {"source_name": "a.pdf"}, {"source_name": "b.pdf"}]
        ],
        "distances": [[0.1, 0.1, 0.3]],
    }
    retriever = RAGRetriever(
        vector_store=mock_vector_store,
        embedding_manager=mock_embedding_manager,
    )
    results = retriever.retrieve("test")
    ids = [r["id"] for r in results]
    assert len(ids) == 2
    assert "unique" in ids
    assert ids.count("dup") == 1


def test_retriever_normalizes_score(mock_vector_store, mock_embedding_manager):
    mock_vector_store.collection.query.return_value = {
        "ids": [["doc1"]],
        "documents": [["content"]],
        "metadatas": [[{"source_name": "doc1.pdf"}]],
        "distances": [[0.0]],
    }
    retriever = RAGRetriever(
        vector_store=mock_vector_store,
        embedding_manager=mock_embedding_manager,
        score_threshold=None,
    )
    results = retriever.retrieve("test", score_threshold=None)
    assert results[0]["score"] == 1.0
    assert results[0]["distance"] == 0.0


def test_retriever_score_threshold_filters(mock_vector_store, mock_embedding_manager):
    mock_vector_store.collection.query.return_value = {
        "ids": [["a", "b", "c"]],
        "documents": [["A", "B", "C"]],
        "metadatas": [
            [{"source_name": "a.pdf"}, {"source_name": "b.pdf"}, {"source_name": "c.pdf"}]
        ],
        "distances": [[0.1, 0.5, 0.9]],
    }
    retriever = RAGRetriever(
        vector_store=mock_vector_store,
        embedding_manager=mock_embedding_manager,
    )
    results = retriever.retrieve("test", score_threshold=0.6)
    assert len(results) == 2
    assert all(r["distance"] <= 0.6 for r in results)


def test_retriever_handles_mismatched_list_lengths(mock_vector_store, mock_embedding_manager):
    mock_vector_store.collection.query.return_value = {
        "ids": [["a", "b", "c"]],
        "documents": [["A", "B"]],
        "metadatas": [[{"source_name": "a.pdf"}, {"source_name": "b.pdf"}]],
        "distances": [[0.1, 0.5, 0.9]],
    }
    retriever = RAGRetriever(
        vector_store=mock_vector_store,
        embedding_manager=mock_embedding_manager,
    )
    results = retriever.retrieve("test", score_threshold=2.0)
    assert len(results) == 2
    assert {r["id"] for r in results} == {"a", "b"}
