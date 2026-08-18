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

    def _build_regeneration_context(
        self,
        context: str,
        available_ranks: List[int],
    ) -> str:
        """Append citation-enforcement instructions to the context."""
        ranks_str = ", ".join(f"[{r}]" for r in sorted(available_ranks))
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

    def _generate_safely(
        self,
        query: str,
        context: str,
    ) -> str:
        """Call the LLM provider and catch errors."""
        return self.llm_provider.generate(
            query=query,
            context=context,
        )

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
                -> LLM generation
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

        # 5. Generate answer
        try:
            answer_text = self._generate_safely(
                query=cleaned_query,
                context=context,
            )
        except Exception as exc:
            logger.error("Generation failed: %s", exc)
            return {
                "query": cleaned_query,
                "answer": f"Error during answer generation: {exc}",
                "source_documents": retrieved_docs,
                "context_length": len(context),
            }

        # 6. Validate citations
        validation = RAGEvaluator.evaluate_citation_enforcement(
            answer_text, retrieved_docs
        )

        if not validation["valid"]:
            logger.warning(
                "Citation validation failed (%s). Attempting "
                "regeneration.", validation["reason"],
            )

            available_ranks = sorted(
                set(
                    doc.get("rank", idx)
                    for idx, doc in enumerate(
                        retrieved_docs, start=1
                    )
                )
            )

            # 7. Regenerate with stricter instructions (at most once)
            try:
                regen_context = self._build_regeneration_context(
                    context, available_ranks
                )
                regenerated_text = self._generate_safely(
                    query=cleaned_query,
                    context=regen_context,
                )
            except Exception as exc:
                logger.error("Regeneration failed: %s", exc)
                return {
                    "query": cleaned_query,
                    "answer": _SAFE_REFUSAL,
                    "source_documents": retrieved_docs,
                    "context_length": len(context),
                }

            # 8. Validate regenerated answer
            regen_validation = (
                RAGEvaluator.evaluate_citation_enforcement(
                    regenerated_text, retrieved_docs
                )
            )

            if regen_validation["valid"]:
                logger.info(
                    "Regenerated answer passed citation validation."
                )
                answer_text = regenerated_text
            else:
                logger.warning(
                    "Regenerated answer still invalid (%s). "
                    "Returning safe refusal.",
                    regen_validation["reason"],
                )
                answer_text = _SAFE_REFUSAL

        # 9. Return answer + evidence
        logger.info(
            "RAG answer generated for query (context=%d chars)",
            len(context),
        )

        return {
            "query": cleaned_query,
            "answer": answer_text,
            "source_documents": retrieved_docs,
            "context_length": len(context),
        }
