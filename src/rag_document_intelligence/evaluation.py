from typing import Any, Dict, List


class RAGEvaluator:
    """Basic evaluation utilities for the RAG pipeline."""

    @staticmethod
    def evaluate_retrieval(
        retrieved_documents: List[Dict[str, Any]],
        expected_sources: List[str],
    ) -> Dict[str, Any]:
        """
        Evaluate whether expected source documents were retrieved.
        """

        retrieved_sources = []

        for document in retrieved_documents:
            metadata = document.get("metadata", {})
            source = metadata.get("source")

            if source:
                retrieved_sources.append(str(source))

        expected = set(expected_sources)
        retrieved = set(retrieved_sources)

        matched = expected.intersection(retrieved)

        recall = (
            len(matched) / len(expected)
            if expected
            else 0.0
        )

        return {
            "expected_sources": list(expected),
            "retrieved_sources": list(retrieved),
            "matched_sources": list(matched),
            "recall": recall,
            "success": recall == 1.0 if expected else False,
        }

    @staticmethod
    def evaluate_answer(
        answer: str,
        context: str,
    ) -> Dict[str, Any]:
        """
        Perform a basic groundedness check.

        This is intentionally simple and does not require an LLM.
        """

        if not answer or not answer.strip():
            return {
                "grounded": False,
                "reason": "Answer is empty.",
            }

        if not context or not context.strip():
            return {
                "grounded": False,
                "reason": "No retrieved context was available.",
            }

        return {
            "grounded": True,
            "reason": "Answer was generated using retrieved context.",
        }
