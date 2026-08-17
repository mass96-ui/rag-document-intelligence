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
