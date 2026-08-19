import logging
from typing import Any, Dict, List, Optional

from .config import MAX_QUERY_LENGTH
from .context_builder import ContextBuilder
from .evaluation import RAGEvaluator
from .llm import LLMProvider
from .retriever import RAGRetriever

logger = logging.getLogger(__name__)

_INSUFFICIENT_EVIDENCE = (
    "I don't have enough information in the retrieved "
    "documents to answer that confidently."
)

_SAFE_REFUSAL = (
    "I could not find this information in the provided documents."
)


class RAGPipeline:
    """Orchestrate retrieval, context construction, and answer generation."""

    MAX_REGENERATION_ATTEMPTS = 1

    def __init__(
        self,
        retriever: RAGRetriever,
        context_builder: ContextBuilder,
        llm_provider: LLMProvider,
    ):
        self.retriever = retriever
        self.context_builder = context_builder
        self.llm_provider = llm_provider

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_available_ranks(
        retrieved_docs: List[Dict[str, Any]],
    ) -> List[int]:
        """Return sorted list of citation ranks available from retrieval."""
        ranks: set[int] = set()
        for idx, doc in enumerate(retrieved_docs, start=1):
            rank = doc.get("rank", idx)
            if rank is not None:
                ranks.add(int(rank))
        return sorted(ranks)

    def _build_regeneration_context(
        self,
        context: str,
        available_ranks: List[int],
    ) -> str:
        """Append citation-enforcement instructions to the context."""
        ranks_str = ", ".join(
            f"[{r}]" for r in sorted(available_ranks)
        )
        return (
            f"{context}\n\n"
            f"---\n"
            f"REGENERATION INSTRUCTIONS (Citation Enforcement):\n"
            f"The available citation numbers from the retrieved "
            f"documents are: {ranks_str}.\n"
            f"Your previous answer contained invalid or missing "
            f"citations.\n"
            f"Re-answer using ONLY the documents in the context "
            f"above,\n"
            f"citing ONLY the available numbers listed. Do NOT "
            f"create new citation numbers.\n"
            f"If you cannot find the information, respond exactly:\n"
            f'"{_SAFE_REFUSAL}"'
        )

    def _validate_structured(
        self,
        structured: Dict[str, Any],
        retrieved_docs: List[Dict[str, Any]],
    ) -> Optional[str]:
        """Validate a structured response; return normalized text or None."""
        if not isinstance(structured, dict):
            logger.warning(
                "Structured response is not a dict: %s",
                type(structured).__name__,
            )
            return None

        answer_field = structured.get("answer")
        citations_field = structured.get("citations")

        # --- structural validation ---
        validation = RAGEvaluator.validate_structured_response(
            answer_field, citations_field, retrieved_docs,
        )

        if validation["valid"]:
            normalized = RAGEvaluator.normalize_citations(
                answer_field, validation["citations"],
            )
            return normalized

        # --- text fallback: check answer field for [N] markers ---
        if isinstance(answer_field, str) and answer_field.strip():
            text_validation = (
                RAGEvaluator.evaluate_citation_enforcement(
                    answer_field, retrieved_docs,
                )
            )
            if text_validation["valid"]:
                return answer_field

        logger.warning(
            "Structured validation failed (%s) and text fallback "
            "also failed.",
            validation["reason"],
        )
        return None

    def _validate_text(
        self,
        answer_text: str,
        retrieved_docs: List[Dict[str, Any]],
    ) -> Optional[str]:
        """Validate a text response; return it if valid, else None."""
        validation = RAGEvaluator.evaluate_citation_enforcement(
            answer_text, retrieved_docs,
        )
        if validation["valid"]:
            return answer_text
        logger.warning(
            "Text citation validation failed (%s).",
            validation["reason"],
        )
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def answer(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Execute the complete RAG pipeline with citation enforcement.

        Flow:
            User query
                -> document retrieval
                -> context construction
                -> structured LLM generation (preferred)
                -> text generation fallback
                -> citation validation
                -> valid answer OR one regeneration
                -> safe refusal if still invalid
        """

        cleaned_query = query.strip() if query else ""

        # 1. Validate input
        if not cleaned_query:
            return {
                "query": query,
                "answer": "Error: Question cannot be empty.",
                "source_documents": [],
                "context_length": 0,
            }

        if len(cleaned_query) > MAX_QUERY_LENGTH:
            return {
                "query": query,
                "answer": (
                    f"Error: Question exceeds maximum length of "
                    f"{MAX_QUERY_LENGTH} characters."
                ),
                "source_documents": [],
                "context_length": 0,
            }

        if top_k <= 0:
            return {
                "query": cleaned_query,
                "answer": (
                    f"Error: Invalid top_k ({top_k}). "
                    "Must be greater than 0."
                ),
                "source_documents": [],
                "context_length": 0,
            }

        # 2. Retrieve relevant document chunks
        try:
            if score_threshold is not None:
                retrieved_docs = self.retriever.retrieve(
                    query=cleaned_query,
                    top_k=top_k,
                    score_threshold=score_threshold,
                )
            else:
                retrieved_docs = self.retriever.retrieve(
                    query=cleaned_query,
                    top_k=top_k,
                )
        except Exception as exc:
            logger.error("Retrieval failed: %s", exc)
            return {
                "query": cleaned_query,
                "answer": f"Error during retrieval: {exc}",
                "source_documents": [],
                "context_length": 0,
            }

        # 3. Handle no retrieval results
        if not retrieved_docs:
            return {
                "query": cleaned_query,
                "answer": _INSUFFICIENT_EVIDENCE,
                "source_documents": [],
                "context_length": 0,
            }

        # 4. Build structured context
        context = self.context_builder.build_context(
            retrieved_docs
        )

        if not context:
            return {
                "query": cleaned_query,
                "answer": (
                    "Relevant documents were retrieved, but no usable "
                    "document content was available."
                ),
                "source_documents": retrieved_docs,
                "context_length": 0,
            }

        available_ranks = self._get_available_ranks(retrieved_docs)
        last_error: Optional[Exception] = None
        answer_text: Optional[str] = None

        # 5. Attempt structured generation (preferred path)
        try:
            structured = self.llm_provider.generate_structured(
                query=cleaned_query,
                context=context,
            )
            answer_text = self._validate_structured(
                structured, retrieved_docs,
            )
        except Exception as exc:
            logger.warning(
                "Structured generation failed: %s", exc,
            )

        # 8. Fallback to text generation if structured unavailable/invalid
        if not answer_text:
            try:
                text_answer = self.llm_provider.generate(
                    query=cleaned_query,
                    context=context,
                )
            except Exception as exc:
                logger.error("Generation failed: %s", exc)
                last_error = exc
            else:
                answer_text = self._validate_text(
                    text_answer, retrieved_docs,
                )

        # 10. Regenerate at most once
        if not answer_text:
            logger.warning(
                "Initial generation invalid. Attempting one "
                "regeneration.",
            )
            regen_context = self._build_regeneration_context(
                context, available_ranks,
            )

            # Try structured regeneration
            try:
                structured = self.llm_provider.generate_structured(
                    query=cleaned_query,
                    context=regen_context,
                )
                answer_text = self._validate_structured(
                    structured, retrieved_docs,
                )
            except Exception as exc:
                logger.warning(
                    "Structured regeneration failed: %s", exc,
                )

            # Try text regeneration if structured didn't yield valid answer
            if not answer_text:
                try:
                    regen_text = self.llm_provider.generate(
                        query=cleaned_query,
                        context=regen_context,
                    )
                except Exception as exc:
                    logger.error("Regeneration generation failed: %s", exc)
                    last_error = exc
                else:
                    answer_text = self._validate_text(
                        regen_text, retrieved_docs,
                    )

        # 11. Final safe refusal or error propagation
        if not answer_text:
            if last_error is not None:
                answer_text = (
                    f"Error during answer generation: {last_error}"
                )
            else:
                answer_text = _SAFE_REFUSAL

        logger.info(
            "RAG answer generated for query "
            "(context=%d chars, sources=%d)",
            len(context), len(retrieved_docs),
        )

        # 9. Return answer + evidence
        return {
            "query": cleaned_query,
            "answer": answer_text,
            "source_documents": retrieved_docs,
            "context_length": len(context),
        }
