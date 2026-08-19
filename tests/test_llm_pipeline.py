from unittest.mock import MagicMock

import pytest

from rag_document_intelligence.context_builder import ContextBuilder
from rag_document_intelligence.evaluation import RAGEvaluator
from rag_document_intelligence.llm import LLMProvider, MockLLMProvider
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

    assert "[1]" in context
    assert "doc1.pdf" in context
    assert "Page: 10" in context
    assert "First content" in context
    assert "[2]" in context
    assert "doc2.txt" in context
    assert "Page: unknown" in context
    assert "Second content" in context
    assert ContextBuilder.PROMPT_INJECTION_WARNING in context
    assert ContextBuilder.KNOWLEDGE_HEADER in context


def test_context_builder_handles_empty_input():
    builder = ContextBuilder()
    context = builder.build_context([])
    assert ContextBuilder.PROMPT_INJECTION_WARNING in context
    assert ContextBuilder.KNOWLEDGE_HEADER not in context


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

    class _TestableMockLLM(MockLLMProvider):
        def generate(self, query, context):
            return "[MOCK RESPONSE]\nQuestion: what is this?\nBased on [1], this is a simulated answer."

    pipeline = RAGPipeline(
        retriever=mock_retriever,
        context_builder=builder,
        llm_provider=_TestableMockLLM(),
    )

    query = "Search query"
    result = pipeline.answer(query)

    assert result["query"] == query
    assert "[MOCK RESPONSE]" in result["answer"]
    assert "[1]" in result["answer"]
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
    assert "[1]" in context
    assert "doc1.pdf" in context
    assert "Page: 3" in context
    assert "Score: 0.950" in context
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
    assert "[1]" in context
    assert "Source: unknown" in context
    assert "Page: unknown" in context
    assert "Some content" in context


def test_context_builder_handles_missing_metadata_key():
    builder = ContextBuilder()
    docs = [{"rank": 1, "content": "Some content"}]
    context = builder.build_context(docs)
    assert "[1]" in context
    assert "Source: unknown" in context
    assert "Page: unknown" in context


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


# ---------------------------------------------------------------------------
# Citation enforcement pipeline tests
# ---------------------------------------------------------------------------

class _ConfigurableMockLLM(LLMProvider):
    """Mock LLM that returns pre-set responses in order."""

    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.call_count = 0
        self.calls = []

    def generate(self, query, context):
        self.call_count += 1
        self.calls.append({"query": query, "context": context})
        if self.call_count <= len(self.responses):
            return self.responses[self.call_count - 1]
        return self.responses[-1] if self.responses else ""


def _mock_docs(count=3):
    docs = []
    for i in range(1, count + 1):
        docs.append(
            {
                "rank": i,
                "content": f"Content {i}",
                "metadata": {"source": f"doc{i}.pdf", "page": i},
                "distance": 0.5,
                "score": 0.67,
            }
        )
    return docs


def test_citation_enforcement_valid_citations_pass():
    docs = _mock_docs(3)
    llm = _ConfigurableMockLLM(["The answer is [1] and [2]."])
    pipeline = RAGPipeline(
        retriever=_MockRetriever(docs),
        context_builder=ContextBuilder(),
        llm_provider=llm,
    )
    result = pipeline.answer("test")
    assert "The answer is [1] and [2]." in result["answer"]
    assert llm.call_count == 1


def test_citation_enforcement_fabricated_triggers_regeneration():
    docs = _mock_docs(3)
    llm = _ConfigurableMockLLM([
        "Based on [99] the answer is X.",
        "Based on [1] the answer is X.",
    ])
    pipeline = RAGPipeline(
        retriever=_MockRetriever(docs),
        context_builder=ContextBuilder(),
        llm_provider=llm,
    )
    result = pipeline.answer("test")
    assert "[1]" in result["answer"]
    assert "[99]" not in result["answer"]
    assert llm.call_count == 2


def test_citation_enforcement_missing_triggers_regeneration():
    docs = _mock_docs(3)
    llm = _ConfigurableMockLLM([
        "The answer is X without citations.",
        "The answer is [1] based on retrieved context.",
    ])
    pipeline = RAGPipeline(
        retriever=_MockRetriever(docs),
        context_builder=ContextBuilder(),
        llm_provider=llm,
    )
    result = pipeline.answer("test")
    assert "[1]" in result["answer"]
    assert llm.call_count == 2


def test_citation_enforcement_empty_then_regenerate():
    docs = _mock_docs(3)
    llm = _ConfigurableMockLLM([
        "",
        "The answer is [1] and [2].",
    ])
    pipeline = RAGPipeline(
        retriever=_MockRetriever(docs),
        context_builder=ContextBuilder(),
        llm_provider=llm,
    )
    result = pipeline.answer("test")
    assert "[1]" in result["answer"]
    assert llm.call_count == 2


def test_citation_enforcement_failed_regeneration_refuses():
    docs = _mock_docs(3)
    llm = _ConfigurableMockLLM([
        "Based on [99] the answer.",
        "Based on [98] the answer.",
    ])
    pipeline = RAGPipeline(
        retriever=_MockRetriever(docs),
        context_builder=ContextBuilder(),
        llm_provider=llm,
    )
    result = pipeline.answer("test")
    assert "could not find this information" in result["answer"].lower()
    assert llm.call_count == 4


def test_citation_enforcement_regeneration_at_most_once():
    docs = _mock_docs(3)
    llm = _ConfigurableMockLLM([
        "Based on [99] the answer.",
        "Based on [98] the answer.",
        "Based on [97] the answer.",
    ])
    pipeline = RAGPipeline(
        retriever=_MockRetriever(docs),
        context_builder=ContextBuilder(),
        llm_provider=llm,
    )
    result = pipeline.answer("test")
    assert "could not find this information" in result["answer"].lower()
    assert llm.call_count == 4


def test_citation_enforcement_unsupported_question_refuses():
    llm = _ConfigurableMockLLM(["anything"])
    pipeline = RAGPipeline(
        retriever=_MockRetriever([]),
        context_builder=ContextBuilder(),
        llm_provider=llm,
    )
    result = pipeline.answer("What is the capital of France?")
    assert "enough information" in result["answer"].lower()
    assert llm.call_count == 0


def test_citation_enforcement_mock_provider_compatible():
    """Existing MockLLMProvider behavior must remain compatible."""
    docs = _mock_docs(1)
    pipeline = RAGPipeline(
        retriever=_MockRetriever(docs),
        context_builder=ContextBuilder(),
        llm_provider=MockLLMProvider(),
    )
    result = pipeline.answer("test question")
    assert "[MOCK RESPONSE]" in result["answer"]


class _MockRetriever:
    """Simple retriever mock returning preset documents."""

    def __init__(self, documents):
        self._documents = documents

    def retrieve(self, query, top_k=5, score_threshold=None):
        return self._documents


# ---------------------------------------------------------------------------
# Structured generation pipeline tests
# ---------------------------------------------------------------------------


class _StructuredMockLLM(LLMProvider):
    """Mock LLM that returns pre-set structured responses in order."""

    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.call_count = 0
        self.calls = []

    def generate(self, query, context):
        return ""

    def generate_structured(self, query, context):
        self.call_count += 1
        self.calls.append({"query": query, "context": context})
        if self.call_count <= len(self.responses):
            return self.responses[self.call_count - 1]
        return (
            self.responses[-1]
            if self.responses
            else {"answer": "", "citations": []}
        )


def test_structured_pipeline_success():
    docs = _mock_docs(3)
    llm = _StructuredMockLLM([
        {"answer": "The answer is based on evidence.", "citations": [1, 2]},
    ])
    pipeline = RAGPipeline(
        retriever=_MockRetriever(docs),
        context_builder=ContextBuilder(),
        llm_provider=llm,
    )
    result = pipeline.answer("test")
    assert "The answer is based on evidence." in result["answer"]
    assert "[1]" in result["answer"]
    assert "[2]" in result["answer"]
    assert "[3]" not in result["answer"]
    assert llm.call_count == 1


def test_structured_pipeline_multiple_citations_sorted():
    docs = _mock_docs(4)
    llm = _StructuredMockLLM([
        {"answer": "Multi-source answer.", "citations": [3, 1, 2]},
    ])
    pipeline = RAGPipeline(
        retriever=_MockRetriever(docs),
        context_builder=ContextBuilder(),
        llm_provider=llm,
    )
    result = pipeline.answer("test")
    assert "Multi-source answer." in result["answer"]
    assert "[1]" in result["answer"]
    assert "[2]" in result["answer"]
    assert "[3]" in result["answer"]
    assert "[4]" not in result["answer"]
    idx1 = result["answer"].index("[1]")
    idx2 = result["answer"].index("[2]")
    idx3 = result["answer"].index("[3]")
    assert idx1 < idx2 < idx3


def test_structured_pipeline_refusal_with_empty_citations():
    docs = _mock_docs(3)
    llm = _StructuredMockLLM([
        {
            "answer": (
                "I could not find enough information in the "
                "provided documents to answer this confidently."
            ),
            "citations": [],
        },
    ])
    pipeline = RAGPipeline(
        retriever=_MockRetriever(docs),
        context_builder=ContextBuilder(),
        llm_provider=llm,
    )
    result = pipeline.answer("test")
    assert "could not find" in result["answer"].lower()
    assert llm.call_count == 1


def test_structured_pipeline_fabricated_citation_triggers_fallback():
    docs = _mock_docs(3)
    llm = MagicMock()
    llm.generate_structured.return_value = {
        "answer": "Answer based on [99].", "citations": [99],
    }
    llm.generate.return_value = "Answer based on [1]."
    pipeline = RAGPipeline(
        retriever=_MockRetriever(docs),
        context_builder=ContextBuilder(),
        llm_provider=llm,
    )
    result = pipeline.answer("test")
    assert "[1]" in result["answer"]
    assert "[99]" not in result["answer"]


def test_structured_pipeline_missing_citations_triggers_fallback():
    docs = _mock_docs(3)
    llm = MagicMock()
    llm.generate_structured.return_value = {
        "answer": "Answer without citations.", "citations": [],
    }
    llm.generate.return_value = "Answer without citations."
    pipeline = RAGPipeline(
        retriever=_MockRetriever(docs),
        context_builder=ContextBuilder(),
        llm_provider=llm,
    )
    result = pipeline.answer("test")
    assert "could not find" in result["answer"].lower()


def test_structured_pipeline_malformed_json_falls_back_to_text():
    docs = _mock_docs(3)
    llm = MagicMock()
    llm.generate_structured.side_effect = RuntimeError(
        "not valid JSON"
    )
    llm.generate.return_value = "The answer is [1] based on context."
    pipeline = RAGPipeline(
        retriever=_MockRetriever(docs),
        context_builder=ContextBuilder(),
        llm_provider=llm,
    )
    result = pipeline.answer("test")
    assert "The answer is [1]" in result["answer"]
    llm.generate_structured.assert_called_once()
    llm.generate.assert_called_once()


def test_structured_pipeline_no_duplicate_citation_markers():
    docs = _mock_docs(3)
    llm = MagicMock()
    llm.generate_structured.return_value = {
        "answer": "Answer [1] already cited.", "citations": [1, 2],
    }
    pipeline = RAGPipeline(
        retriever=_MockRetriever(docs),
        context_builder=ContextBuilder(),
        llm_provider=llm,
    )
    result = pipeline.answer("test")
    assert result["answer"].count("[1]") == 1
    assert "[2]" in result["answer"]


def test_structured_pipeline_unsupported_refuses_without_llm():
    llm = MagicMock()
    pipeline = RAGPipeline(
        retriever=_MockRetriever([]),
        context_builder=ContextBuilder(),
        llm_provider=llm,
    )
    result = pipeline.answer("What is the capital of France?")
    assert "enough information" in result["answer"].lower()
    llm.generate_structured.assert_not_called()
    llm.generate.assert_not_called()


def test_structured_pipeline_max_one_regeneration():
    docs = _mock_docs(3)
    llm = _StructuredMockLLM([
        {"answer": "Bad answer.", "citations": [99]},
        {"answer": "Bad answer.", "citations": [98]},
        {"answer": "Should not be called.", "citations": [97]},
    ])
    pipeline = RAGPipeline(
        retriever=_MockRetriever(docs),
        context_builder=ContextBuilder(),
        llm_provider=llm,
    )
    result = pipeline.answer("test")
    assert "could not find" in result["answer"].lower()
    assert llm.call_count == 2


def test_structured_pipeline_mock_compatible():
    docs = _mock_docs(1)
    pipeline = RAGPipeline(
        retriever=_MockRetriever(docs),
        context_builder=ContextBuilder(),
        llm_provider=MockLLMProvider(),
    )
    result = pipeline.answer("test question")
    assert "[MOCK RESPONSE]" in result["answer"]


def test_structured_pipeline_generate_only_provider():
    """A provider that only implements generate() must still work."""

    class _GenerateOnly:
        def generate(self, query, context):
            return "The answer is [1] based on context."

    docs = _mock_docs(3)
    pipeline = RAGPipeline(
        retriever=_MockRetriever(docs),
        context_builder=ContextBuilder(),
        llm_provider=_GenerateOnly(),
    )
    result = pipeline.answer("test")
    assert "[1]" in result["answer"]


def test_structured_pipeline_regeneration_success():
    docs = _mock_docs(3)
    llm = _StructuredMockLLM([
        {"answer": "No citations here.", "citations": []},
        {"answer": "Good answer.", "citations": [1]},
    ])
    pipeline = RAGPipeline(
        retriever=_MockRetriever(docs),
        context_builder=ContextBuilder(),
        llm_provider=llm,
    )
    result = pipeline.answer("test")
    assert "Good answer." in result["answer"]
    assert "[1]" in result["answer"]
    assert llm.call_count == 2


def test_structured_pipeline_normalizes_citations():
    docs = _mock_docs(5)
    llm = _StructuredMockLLM([
        {
            "answer": "Machine learning enables systems to learn.",
            "citations": [3, 1, 2, 1, 3],
        },
    ])
    pipeline = RAGPipeline(
        retriever=_MockRetriever(docs),
        context_builder=ContextBuilder(),
        llm_provider=llm,
    )
    result = pipeline.answer("test")
    assert "Machine learning enables systems to learn." in result["answer"]
    citation_part = result["answer"].replace(
        "Machine learning enables systems to learn.", ""
    ).strip()
    assert citation_part == "[1] [2] [3]"


def test_normalize_citations_appends_correctly():
    answer = "Machine learning is a field of CS."
    citations = [2, 1, 2]
    result = RAGEvaluator.normalize_citations(answer, citations)
    assert result == "Machine learning is a field of CS. [1] [2]"


def test_normalize_citations_no_duplicates():
    answer = "Answer [1]"
    citations = [1, 2]
    result = RAGEvaluator.normalize_citations(answer, citations)
    assert result == "Answer [1] [2]"
    assert result.count("[1]") == 1


def test_normalize_citations_filters_invalid():
    answer = "Test answer"
    citations = [1, -1, 0, 2, "3", True, 1]
    result = RAGEvaluator.normalize_citations(answer, citations)
    assert "[1]" in result
    assert "[2]" in result
    assert "-1" not in result
    assert "[0]" not in result
