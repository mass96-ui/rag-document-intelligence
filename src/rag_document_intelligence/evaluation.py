import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_CITATION_RE = re.compile(r"\[(\d+)\]")


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

        retrieved_sources: List[str] = []

        for document in retrieved_documents:
            metadata = document.get("metadata", {})
            source = metadata.get("source")
            source_name = metadata.get("source_name")

            if source_name:
                retrieved_sources.append(str(source_name))
            elif source:
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

    @staticmethod
    def evaluate_citations(
        answer: str,
        source_documents: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Check that citation numbers in the answer correspond to
        retrieved evidence and detect fabricated citations.
        """

        cited_numbers = sorted(
            set(int(n) for n in _CITATION_RE.findall(answer))
        )

        available_ranks: set[int] = set()
        for idx, doc in enumerate(source_documents, start=1):
            rank = doc.get("rank", idx)
            if rank is not None:
                available_ranks.add(rank)

        fabricated = [
            num for num in cited_numbers
            if num not in available_ranks
        ]

        return {
            "cited_numbers": cited_numbers,
            "available_ranks": sorted(available_ranks),
            "fabricated_citations": fabricated,
            "has_fabricated_citations": len(fabricated) > 0,
            "valid": len(fabricated) == 0,
        }
