import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_CITATION_RE = re.compile(r"\[(\d+)\]")

_REFUSAL_PHRASES = [
    "i could not find this information in the provided documents",
    "i could not answer the question because no relevant document context was retrieved",
    "i don't have enough information in the retrieved documents to answer that confidently",
    "i could not answer the question because",
    "i cannot find this information",
    "i don't know",
]


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

    @staticmethod
    def is_refusal(answer: str) -> bool:
        """Return True if the answer is a refusal to answer."""
        if not answer or not answer.strip():
            return False
        answer_lower = answer.lower().strip()
        return any(phrase in answer_lower for phrase in _REFUSAL_PHRASES)

    @staticmethod
    def evaluate_citation_enforcement(
        answer: str,
        source_documents: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Validate that an answer has proper, non-fabricated citations.

        Returns a dict with:
            - valid: bool
            - reason: str explaining the outcome
            - cited_numbers: list of cited citation numbers
            - fabricated_citations: list of fabricated numbers
        """
        if not answer or not answer.strip():
            return {
                "valid": False,
                "reason": "empty answer",
                "cited_numbers": [],
                "fabricated_citations": [],
            }

        if RAGEvaluator.is_refusal(answer):
            return {
                "valid": True,
                "reason": "refusal (acceptable without citations)",
                "cited_numbers": [],
                "fabricated_citations": [],
            }

        cite_eval = RAGEvaluator.evaluate_citations(
            answer, source_documents
        )

        if cite_eval["has_fabricated_citations"]:
            return {
                "valid": False,
                "reason": "fabricated citations",
                "cited_numbers": cite_eval["cited_numbers"],
                "fabricated_citations": cite_eval["fabricated_citations"],
            }

        if not cite_eval["cited_numbers"]:
            return {
                "valid": False,
                "reason": "missing citations",
                "cited_numbers": [],
                "fabricated_citations": [],
            }

        return {
            "valid": True,
            "reason": "valid citations",
            "cited_numbers": cite_eval["cited_numbers"],
            "fabricated_citations": [],
        }
