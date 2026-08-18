from unittest.mock import MagicMock

import pytest

from rag_document_intelligence.context_builder import ContextBuilder
from rag_document_intelligence.llm import MockLLMProvider
from rag_document_intelligence.pipeline import RAGPipeline


def test_context_builder_formats_correctly():
    builder = ContextBuilder()
    docs = [
        {
            "rank": 1,
            "content": "First content",
            "metadata": {"source": "doc1.pdf", "page": 10},
        },
        {
            "rank": 2,
            "content": "Second content",
            "metadata": {"source": "doc2.txt"},
        },
    ]

    context = builder.build_context(docs)

    assert "[1] Source: doc1.pdf, Page: 10" in context
    assert "First content" in context
    assert "[2] Source: doc2.txt, Page: unknown" in context
    assert "Second content" in context


def test_context_builder_handles_empty_input():
    builder = ContextBuilder()
    assert builder.build_context([]) == ""


def test_mock_llm_provider_generates_response():
    provider = MockLLMProvider()
    query = "What is RAG?"
    context = "RAG stands for Retrieval-Augmented Generation."

    response = provider.generate(query, context)

    assert "[MOCK RESPONSE]" in response
    assert query in response
    assert "RAG stands for" in response


def test_rag_pipeline_orchestrates_flow():
    # Mock the retriever
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        {"rank": 1, "content": "Retrieved content", "metadata": {"source": "test.pdf"}}
    ]

    builder = ContextBuilder()
    provider = MockLLMProvider()

    pipeline = RAGPipeline(
        retriever=mock_retriever,
        context_builder=builder,
        llm_provider=provider,
    )

    query = "Search query"
    result = pipeline.answer(query)

    assert result["query"] == query
    assert "[MOCK RESPONSE]" in result["answer"]
    assert len(result["source_documents"]) == 1
    assert result["source_documents"][0]["content"] == "Retrieved content"

    # Verify retriever was called
    mock_retriever.retrieve.assert_called_once_with(query=query, top_k=5)
