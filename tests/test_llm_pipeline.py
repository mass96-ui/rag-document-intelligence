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


def test_context_builder_includes_score_and_doc_id():
    builder = ContextBuilder()
    docs = [
        {
            "rank": 1,
            "id": "doc1.pdf|0|abc123",
            "score": 0.95,
            "content": "Important finding",
            "metadata": {"source": "doc1.pdf", "page": 3},
        },
    ]
    context = builder.build_context(docs)
    assert "[1] Source: doc1.pdf, Page: 3" in context
    assert "Score" in context
    assert "doc1.pdf|0|abc123" in context
    assert "Important finding" in context


def test_context_builder_citation_numbering_is_stable():
    builder = ContextBuilder()
    docs = [
        {"rank": 1, "content": "Alpha", "metadata": {"source": "a.pdf"}},
        {"rank": 2, "content": "Beta", "metadata": {"source": "b.pdf"}},
        {"rank": 3, "content": "Gamma", "metadata": {"source": "c.pdf"}},
    ]
    context1 = builder.build_context(docs)
    context2 = builder.build_context(docs)
    assert context1 == context2
    assert "[1]" in context1
    assert "[2]" in context1
    assert "[3]" in context1


def test_context_builder_handles_missing_metadata():
    builder = ContextBuilder()
    docs = [
        {"rank": 1, "content": "Some content", "metadata": {}},
    ]
    context = builder.build_context(docs)
    assert "[1] Source: unknown, Page: unknown" in context
    assert "Some content" in context


def test_context_builder_handles_missing_metadata_key():
    builder = ContextBuilder()
    docs = [{"rank": 1, "content": "Some content"}]
    context = builder.build_context(docs)
    assert "[1] Source: unknown, Page: unknown" in context


def test_context_builder_skips_empty_content():
    builder = ContextBuilder()
    docs = [
        {"rank": 1, "content": "", "metadata": {"source": "a.pdf"}},
        {"rank": 2, "content": "Real content", "metadata": {"source": "b.pdf"}},
    ]
    context = builder.build_context(docs)
    assert "[1]" not in context
    assert "[2] Source: b.pdf" in context


def test_context_builder_source_name_preferred_over_source():
    builder = ContextBuilder()
    docs = [
        {
            "rank": 1,
            "content": "content",
            "metadata": {"source": "/full/path/doc.pdf", "source_name": "doc.pdf"},
        },
    ]
    context = builder.build_context(docs)
    assert "Source: doc.pdf" in context
    assert "/full/path" not in context


def test_pipeline_rejects_empty_query():
    pipeline = RAGPipeline(
        retriever=MagicMock(),
        context_builder=ContextBuilder(),
        llm_provider=MockLLMProvider(),
    )
    result = pipeline.answer("")
    assert "empty" in result["answer"].lower()
    assert result["source_documents"] == []


def test_pipeline_returns_refusal_when_no_documents():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []
    pipeline = RAGPipeline(
        retriever=mock_retriever,
        context_builder=ContextBuilder(),
        llm_provider=MockLLMProvider(),
    )
    result = pipeline.answer("unknown question")
    assert "enough information" in result["answer"].lower()
    assert result["source_documents"] == []


def test_pipeline_returns_error_on_retrieval_failure():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.side_effect = RuntimeError("Vector store down")
    pipeline = RAGPipeline(
        retriever=mock_retriever,
        context_builder=ContextBuilder(),
        llm_provider=MockLLMProvider(),
    )
    result = pipeline.answer("test question")
    assert "retrieval" in result["answer"].lower()
    assert result["source_documents"] == []


def test_pipeline_returns_error_on_generation_failure():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        {"rank": 1, "content": "data", "metadata": {"source": "d.pdf"}}
    ]
    mock_llm = MagicMock()
    mock_llm.generate.side_effect = RuntimeError("Ollama exploded")
    pipeline = RAGPipeline(
        retriever=mock_retriever,
        context_builder=ContextBuilder(),
        llm_provider=mock_llm,
    )
    result = pipeline.answer("test question")
    assert "generation" in result["answer"].lower()
    assert len(result["source_documents"]) == 1
