from rag_document_intelligence.evaluation import RAGEvaluator


def test_retrieval_evaluation_success():
    documents = [
        {
            "metadata": {
                "source": "doc1.pdf",
                "page": 1,
            }
        },
        {
            "metadata": {
                "source": "doc2.txt",
            }
        },
    ]

    result = RAGEvaluator.evaluate_retrieval(
        documents,
        ["doc1.pdf", "doc2.txt"],
    )

    assert result["success"] is True
    assert result["recall"] == 1.0


def test_retrieval_evaluation_partial():
    documents = [
        {
            "metadata": {
                "source": "doc1.pdf",
            }
        }
    ]

    result = RAGEvaluator.evaluate_retrieval(
        documents,
        ["doc1.pdf", "doc2.txt"],
    )

    assert result["success"] is False
    assert result["recall"] == 0.5


def test_empty_answer_is_not_grounded():
    result = RAGEvaluator.evaluate_answer(
        "",
        "Some retrieved context",
    )

    assert result["grounded"] is False


def test_context_exists():
    result = RAGEvaluator.evaluate_answer(
        "This is an answer.",
        "Some retrieved context",
    )

    assert result["grounded"] is True


def test_citation_validation_no_citations():
    docs = [
        {"rank": 1, "content": "alpha", "metadata": {"source": "a.pdf"}},
        {"rank": 2, "content": "beta", "metadata": {"source": "b.pdf"}},
    ]
    result = RAGEvaluator.evaluate_citations(
        "The answer is in the documents.", docs
    )
    assert result["cited_numbers"] == []
    assert result["has_fabricated_citations"] is False
    assert result["valid"] is True


def test_citation_validation_valid_citations():
    docs = [
        {"rank": 1, "content": "alpha", "metadata": {"source": "a.pdf"}},
        {"rank": 2, "content": "beta", "metadata": {"source": "b.pdf"}},
    ]
    result = RAGEvaluator.evaluate_citations(
        "Source [1] says alpha. Source [2] says beta.", docs
    )
    assert result["cited_numbers"] == [1, 2]
    assert result["has_fabricated_citations"] is False
    assert result["valid"] is True


def test_citation_validation_detects_fabricated():
    docs = [
        {"rank": 1, "content": "alpha", "metadata": {"source": "a.pdf"}},
    ]
    result = RAGEvaluator.evaluate_citations(
        "According to [1] and [3], the answer is alpha.", docs
    )
    assert result["cited_numbers"] == [1, 3]
    assert result["has_fabricated_citations"] is True
    assert result["fabricated_citations"] == [3]
    assert result["valid"] is False


def test_retrieval_evaluation_uses_source_name():
    documents = [
        {
            "metadata": {
                "source_name": "doc1.pdf",
                "source": "/some/path/doc1.pdf",
            }
        },
    ]
    result = RAGEvaluator.evaluate_retrieval(
        documents,
        ["doc1.pdf"],
    )
    assert result["retrieved_sources"] == ["doc1.pdf"]
    assert result["success"] is True


def test_evaluate_answer_rejects_empty_context():
    result = RAGEvaluator.evaluate_answer("An answer", "")
    assert result["grounded"] is False
    assert "context" in result["reason"].lower()


# ---------------------------------------------------------------------------
# Citation enforcement validation tests
# ---------------------------------------------------------------------------

_DOCS = [
    {"rank": 1, "content": "alpha", "metadata": {"source": "a.pdf"}},
    {"rank": 2, "content": "beta", "metadata": {"source": "b.pdf"}},
    {"rank": 3, "content": "gamma", "metadata": {"source": "c.pdf"}},
]


def test_citation_enforcement_valid_citations():
    result = RAGEvaluator.evaluate_citation_enforcement(
        "The answer is based on [1] and [2].", _DOCS
    )
    assert result["valid"] is True
    assert result["cited_numbers"] == [1, 2]
    assert result["reason"] == "valid citations"


def test_citation_enforcement_fabricated_citation():
    result = RAGEvaluator.evaluate_citation_enforcement(
        "Based on [99], the answer is X.", _DOCS
    )
    assert result["valid"] is False
    assert result["reason"] == "fabricated citations"
    assert 99 in result["fabricated_citations"]


def test_citation_enforcement_missing_citations():
    result = RAGEvaluator.evaluate_citation_enforcement(
        "The answer is X without any citations.", _DOCS
    )
    assert result["valid"] is False
    assert result["reason"] == "missing citations"


def test_citation_enforcement_empty_answer():
    result = RAGEvaluator.evaluate_citation_enforcement("", _DOCS)
    assert result["valid"] is False
    assert result["reason"] == "empty answer"


def test_citation_enforcement_refusal_is_accepted():
    result = RAGEvaluator.evaluate_citation_enforcement(
        "I could not find this information in the provided documents.",
        _DOCS,
    )
    assert result["valid"] is True
    assert "refusal" in result["reason"]


def test_citation_enforcement_whitespace_answer():
    result = RAGEvaluator.evaluate_citation_enforcement("   ", _DOCS)
    assert result["valid"] is False
    assert result["reason"] == "empty answer"


def test_is_refusal_detects_standard_phrase():
    assert RAGEvaluator.is_refusal(
        "I could not find this information in the provided documents."
    )


def test_is_refusal_detects_variants():
    assert RAGEvaluator.is_refusal(
        "I don't have enough information in the retrieved "
        "documents to answer that confidently."
    )
    assert RAGEvaluator.is_refusal("I cannot find this information")


def test_is_refusal_rejects_normal_answer():
    assert not RAGEvaluator.is_refusal("The answer is [1] based.")


def test_is_refusal_rejects_empty():
    assert not RAGEvaluator.is_refusal("")
