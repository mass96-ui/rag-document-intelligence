import logging
import re
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

_CITATION_RE = re.compile(r"\[(\d+)\]")

_REFUSAL_PHRASES = [
    "i could not find this information in the provided documents",
    "i could not find enough information in the provided documents",
    "i could not answer the question because no relevant document context was retrieved",
    "i don't have enough information in the retrieved documents to answer that confidently",
    "i don't have enough information in the provided documents",
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

    @staticmethod
    def validate_structured_response(
        answer: Any,
        citations: Any,
        source_documents: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Validate a structured LLM response.

        Checks:
        - answer is a non-empty string
        - citations is a list
        - every citation is a positive int (bool rejected)
        - duplicates removed, ordering sorted & deterministic
        - every citation exists in available source ranks
        - refusal answers may have empty citations
        - non-refusal answers require at least one valid citation

        Does NOT trust the model — the application is the authority
        on valid citations.
        """

        available_ranks: Set[int] = set()
        for idx, doc in enumerate(source_documents, start=1):
            rank = doc.get("rank", idx)
            if rank is not None:
                available_ranks.add(int(rank))

        sorted_ranks = sorted(available_ranks)

        # --- answer ---
        if not isinstance(answer, str) or not answer.strip():
            return {
                "valid": False,
                "reason": "missing or empty answer",
                "citations": [],
                "available_ranks": sorted_ranks,
                "fabricated_citations": [],
            }

        # --- citations ---
        if not isinstance(citations, list):
            return {
                "valid": False,
                "reason": "citations is not a list",
                "citations": [],
                "available_ranks": sorted_ranks,
                "fabricated_citations": [],
            }

        validated: List[int] = []
        for idx, c in enumerate(citations):
            if isinstance(c, bool):
                return {
                    "valid": False,
                    "reason": (
                        f"citation at index {idx} is a boolean, "
                        "not an integer"
                    ),
                    "citations": [],
                    "available_ranks": sorted_ranks,
                    "fabricated_citations": [],
                }
            if not isinstance(c, int):
                return {
                    "valid": False,
                    "reason": (
                        f"citation at index {idx} is not an integer "
                        f"(got {type(c).__name__})"
                    ),
                    "citations": [],
                    "available_ranks": sorted_ranks,
                    "fabricated_citations": [],
                }
            if c <= 0:
                return {
                    "valid": False,
                    "reason": (
                        f"citation at index {idx} is not positive "
                        f"(got {c})"
                    ),
                    "citations": [],
                    "available_ranks": sorted_ranks,
                    "fabricated_citations": [],
                }
            validated.append(c)

        # Deduplicate + sort for deterministic ordering.
        validated = sorted(set(validated))

        # --- fabricated check ---
        fabricated = [
            c for c in validated if c not in available_ranks
        ]
        if fabricated:
            return {
                "valid": False,
                "reason": "fabricated citations",
                "citations": validated,
                "available_ranks": sorted_ranks,
                "fabricated_citations": fabricated,
            }

        # --- refusal accepted without citations ---
        if RAGEvaluator.is_refusal(answer):
            return {
                "valid": True,
                "reason": "refusal (acceptable without citations)",
                "citations": validated,
                "available_ranks": sorted_ranks,
                "fabricated_citations": [],
            }

        # --- non-refusal requires at least one citation ---
        if not validated:
            return {
                "valid": False,
                "reason": "missing citations",
                "citations": [],
                "available_ranks": sorted_ranks,
                "fabricated_citations": [],
            }

        return {
            "valid": True,
            "reason": "valid structured response",
            "citations": validated,
            "available_ranks": sorted_ranks,
            "fabricated_citations": [],
        }

    @staticmethod
    def normalize_citations(
        answer: str,
        citations: List[int],
    ) -> str:
        """Append sorted, deduplicated citation markers to *answer*.

        Rules:
        - citations are sorted numerically and deduplicated
        - only positive integers are kept
        - existing ``[N]`` markers in *answer* are not duplicated
        - fabricated numbers (not checked here) should be filtered
          upstream by ``validate_structured_response``
        """
        seen: Set[int] = set()
        existing: Set[int] = set(
            int(n) for n in _CITATION_RE.findall(answer)
        )

        clean: List[int] = []
        for c in citations:
            if isinstance(c, bool):
                continue
            if not isinstance(c, int):
                continue
            if c <= 0:
                continue
            if c in seen:
                continue
            seen.add(c)
            clean.append(c)

        clean = sorted(clean)

        append: List[str] = []
        for c in clean:
            if c not in existing:
                append.append(f"[{c}]")

        if append:
            return f"{answer.rstrip()} {' '.join(append)}"
        return answer.rstrip()
