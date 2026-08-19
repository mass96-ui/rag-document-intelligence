import logging
import re
from typing import Any, Dict, List, Optional

from .config import MAX_QUERY_LENGTH
from .context_builder import ContextBuilder
from .evaluation import RAGEvaluator
from .llm import LLMProvider
from .patient_context import (
    MLSessionResult,
    PatientContext,
    PatientContextProvider,
    PatientNotFoundError,
)
from .retriever import RAGRetriever

logger = logging.getLogger(__name__)

_INSUFFICIENT_EVIDENCE = (
    "I don't have enough information in the retrieved "
    "documents to answer that confidently."
)

_SAFE_REFUSAL = (
    "I could not find this information in the provided documents."
)

_MEDICAL_REFERRAL = (
    "I cannot provide a medical recommendation or prescription. "
    "This requires clinical judgment. Please consult with a "
    "qualified healthcare provider or refer to doctor-approved "
    "protocols for guidance."
)

_MEDICAL_QUERY_PATTERNS = [
    re.compile(r"how many\s+.*?\s+reps", re.IGNORECASE),
    re.compile(r"how much\s+.*?\s+reps", re.IGNORECASE),
    re.compile(r"recommended\s+.*?\s+resistance", re.IGNORECASE),
    re.compile(r"what\s+.*?\s+resistance", re.IGNORECASE),
    re.compile(r"should\s+.*?\s+(?:exercise|do|perform|do\s+exercise)", re.IGNORECASE),
    re.compile(r"prescription", re.IGNORECASE),
    re.compile(r"dosage", re.IGNORECASE),
    re.compile(r"treatment\s+plan", re.IGNORECASE),
    re.compile(r"safe\s+to\s+exercise", re.IGNORECASE),
    re.compile(r"is\s+it\s+safe", re.IGNORECASE),
    re.compile(r"medical\s+advice", re.IGNORECASE),
    re.compile(r"should\s+patient", re.IGNORECASE),
    re.compile(r"how\s+should\s+i", re.IGNORECASE),
    re.compile(r"can\s+i\s+exercise", re.IGNORECASE),
    re.compile(r"exercise\s+prescription", re.IGNORECASE),
    re.compile(r"resistance\s+level", re.IGNORECASE),
]

_DOCTOR_APPROVED_SOURCE_TYPES = {
    "doctor_document",
    "clinical_guideline",
    "rehabilitation_protocol",
}


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

    @staticmethod
    def _is_medical_prescription_query(query: str) -> bool:
        """Detect whether *query* asks for a medical recommendation."""
        query_lower = query.lower().strip()
        for pattern in _MEDICAL_QUERY_PATTERNS:
            if pattern.search(query_lower):
                return True
        return False

    @staticmethod
    def _has_doctor_approved_evidence(
        retrieved_docs: List[Dict[str, Any]],
    ) -> bool:
        """Check if any retrieved document is doctor-approved."""
        for doc in retrieved_docs:
            metadata = doc.get("metadata") or {}
            source_type = metadata.get(
                "source_type", "general_reference"
            )
            trust_level = metadata.get(
                "trust_level", "unspecified"
            )
            if source_type in _DOCTOR_APPROVED_SOURCE_TYPES:
                return True
            if trust_level in ("high", "verified", "approved"):
                source_name = metadata.get("source_name", "")
                if source_name and "doctor" in str(source_name).lower():
                    return True
        return False

    @staticmethod
    def _compute_confidence(
        citations: List[int],
        retrieved_docs: List[Dict[str, Any]],
        refused: bool,
    ) -> str:
        """Compute categorical confidence based on evidence quality."""
        if refused or not citations:
            return "unknown"

        has_doctor_approved = False
        total_cited = 0

        for doc in retrieved_docs:
            rank = doc.get("rank")
            if rank in citations or rank in citations:
                total_cited += 1
                metadata = doc.get("metadata") or {}
                source_type = metadata.get(
                    "source_type", "general_reference"
                )
                if source_type in _DOCTOR_APPROVED_SOURCE_TYPES:
                    has_doctor_approved = True

        if has_doctor_approved and len(citations) >= 2:
            return "high"
        if has_doctor_approved and len(citations) >= 1:
            return "moderate"
        if len(citations) >= 2:
            return "moderate"
        if len(citations) >= 1:
            return "low"

        return "unknown"

    @staticmethod
    def _extract_citation_numbers(answer: str) -> List[int]:
        """Extract citation numbers from a validated answer string."""
        import re as _re

        matches = _re.findall(r"\[(\d+)\]", answer)
        return sorted(set(int(m) for m in matches))

    def _get_patient_context(
        self,
        patient_id: Optional[str],
        provider: Optional[PatientContextProvider],
    ) -> Optional[PatientContext]:
        """Retrieve patient context if patient_id and provider are given."""
        if not patient_id or provider is None:
            return None

        try:
            context = provider.get_patient_context(patient_id)
            logger.info(
                "Patient context loaded for patient_id=%s",
                patient_id,
            )
            return context
        except PatientNotFoundError:
            logger.warning(
                "Patient '%s' not found in context provider.",
                patient_id,
            )
            return None
        except Exception as exc:
            logger.warning(
                "Failed to load patient context for '%s': %s",
                patient_id, exc,
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
        patient_id: Optional[str] = None,
        patient_context_provider: Optional[PatientContextProvider] = None,
        ml_result: Optional[MLSessionResult] = None,
    ) -> Dict[str, Any]:
        """
        Execute the complete RAG pipeline with citation enforcement.

        Flow:
            User query
                -> patient context retrieval (optional)
                -> medical safety boundary check
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
            return self._error_result(query, "Error: Question cannot be empty.")

        if len(cleaned_query) > MAX_QUERY_LENGTH:
            return self._error_result(
                query,
                f"Error: Question exceeds maximum length of "
                f"{MAX_QUERY_LENGTH} characters.",
            )

        if top_k <= 0:
            return self._error_result(
                query,
                f"Error: Invalid top_k ({top_k}). Must be greater than 0.",
            )

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
            return self._error_result(
                cleaned_query,
                f"Error during retrieval: {exc}",
                source_documents=[],
                context_length=0,
            )

        available_ranks = self._get_available_ranks(retrieved_docs)

        # 3. Retrieve patient context (optional)
        patient_context = self._get_patient_context(
            patient_id, patient_context_provider,
        )

        # 4. Medical safety boundary check
        if patient_context is not None:
            if self._is_medical_prescription_query(cleaned_query):
                if not self._has_doctor_approved_evidence(retrieved_docs):
                    logger.warning(
                        "Medical prescription query without "
                        "doctor-approved evidence. Query: %s",
                        cleaned_query,
                    )
                    return self._refusal_result(
                        cleaned_query,
                        _MEDICAL_REFERRAL,
                        retrieved_docs,
                        0,
                        patient_context_used=patient_context is None,
                        ml_context_used=ml_result is None,
                    )

        # 5. Handle no retrieval results
        if not retrieved_docs:
            if patient_context is not None and cleaned_query:
                logger.info(
                    "No documents retrieved but patient context "
                    "available; returning insufficient evidence."
                )
            return self._refusal_result(
                cleaned_query,
                _INSUFFICIENT_EVIDENCE,
                retrieved_docs,
                0,
                patient_context_used=patient_context is not None,
                ml_context_used=ml_result is not None,
            )

        # 6. Build structured context
        context = self.context_builder.build_context(
            retrieved_docs,
            patient_context=patient_context,
            ml_result=ml_result,
        )

        if not context:
            return self._error_result(
                cleaned_query,
                "Relevant documents were retrieved, but no usable "
                "document content was available.",
                source_documents=retrieved_docs,
                context_length=0,
            )

        last_error: Optional[Exception] = None
        answer_text: Optional[str] = None

        # 7. Attempt structured generation (preferred path)
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

        # 9. Regenerate at most once
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

        # 10. Final safe refusal or error propagation
        refused = False
        reason: Optional[str] = None

        if not answer_text:
            if last_error is not None:
                answer_text = (
                    f"Error during answer generation: {last_error}"
                )
            else:
                answer_text = _SAFE_REFUSAL
            refused = True
            reason = "safe_refusal"

        logger.info(
            "RAG answer generated for query "
            "(context=%d chars, sources=%d)",
            len(context), len(retrieved_docs),
        )

        # 11. Extract citations from final answer
        citations = self._extract_citation_numbers(
            answer_text or ""
        )

        # 12. Compute confidence
        confidence = self._compute_confidence(
            citations, retrieved_docs, refused,
        )

        # 13. Build integration result
        patient_context_used = patient_context is not None
        ml_context_used = ml_result is not None

        return {
            "query": cleaned_query,
            "answer": answer_text,
            "citations": citations,
            "confidence": confidence,
            "refused": refused,
            "reason": reason,
            "sources": [
                {
                    "rank": doc.get("rank", idx),
                    "source_name": (
                        (doc.get("metadata") or {}).get(
                            "source_name", "unknown"
                        )
                    ),
                    "page": (doc.get("metadata") or {}).get(
                        "page", "unknown"
                    ),
                    "score": doc.get("score"),
                    "distance": doc.get("distance"),
                    "source_type": (
                        (doc.get("metadata") or {}).get(
                            "source_type", "general_reference"
                        )
                    ),
                    "trust_level": (
                        (doc.get("metadata") or {}).get(
                            "trust_level", "unspecified"
                        )
                    ),
                }
                for idx, doc in enumerate(retrieved_docs, start=1)
            ],
            "patient_context_used": patient_context_used,
            "ml_context_used": ml_context_used,
            "source_documents": retrieved_docs,
            "context_length": len(context),
        }

    def _error_result(
        self,
        query: str,
        error_message: str,
        source_documents: Optional[List[Dict[str, Any]]] = None,
        context_length: int = 0,
        patient_context_used: bool = False,
        ml_context_used: bool = False,
    ) -> Dict[str, Any]:
        """Build a standardized error result."""
        return {
            "query": query,
            "answer": error_message,
            "citations": [],
            "confidence": "unknown",
            "refused": True,
            "reason": "error",
            "sources": [],
            "patient_context_used": patient_context_used,
            "ml_context_used": ml_context_used,
            "source_documents": source_documents or [],
            "context_length": context_length,
        }

    def _refusal_result(
        self,
        query: str,
        answer: str,
        source_documents: List[Dict[str, Any]],
        context_length: int,
        patient_context_used: bool = False,
        ml_context_used: bool = False,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build a standardized refusal result."""
        return {
            "query": query,
            "answer": answer,
            "citations": [],
            "confidence": "unknown",
            "refused": True,
            "reason": reason or "refusal",
            "sources": [
                {
                    "rank": doc.get("rank", idx),
                    "source_name": (
                        (doc.get("metadata") or {}).get(
                            "source_name", "unknown"
                        )
                    ),
                    "page": (doc.get("metadata") or {}).get(
                        "page", "unknown"
                    ),
                    "score": doc.get("score"),
                    "distance": doc.get("distance"),
                    "source_type": (
                        (doc.get("metadata") or {}).get(
                            "source_type", "general_reference"
                        )
                    ),
                    "trust_level": (
                        (doc.get("metadata") or {}).get(
                            "trust_level", "unspecified"
                        )
                    ),
                }
                for idx, doc in enumerate(source_documents, start=1)
            ],
            "patient_context_used": patient_context_used,
            "ml_context_used": ml_context_used,
            "source_documents": source_documents,
            "context_length": context_length,
        }
