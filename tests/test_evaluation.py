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
