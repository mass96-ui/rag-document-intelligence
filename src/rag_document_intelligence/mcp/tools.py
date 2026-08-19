"""MCP tool implementations for RAG Document Intelligence.

This module defines the MCP tool handlers and tool registry. The handlers
are designed to be testable without requiring a live pipeline — a
pipeline factory function can be injected for unit testing.

All handlers delegate to existing RAG components:
    - RAGPipeline.answer() for ask_rag
    - RAGRetriever.retrieve() for search_knowledge
    - PatientContextProvider.get_patient_context() for get_patient_context

No retrieval, citation, safety, or patient-validation logic is duplicated
in this module.
"""
import json
import logging
from typing import Any, Callable, Dict, List, Optional

from mcp.types import (
    CallToolRequestParams,
    CallToolResult,
    Tool,
    TextContent,
)

from ..config import MAX_QUERY_LENGTH
from ..patient_context import (
    PatientContext,
    get_patient_context_provider,
)
from ..pipeline import RAGPipeline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Input validators
# ---------------------------------------------------------------------------

def _validate_query(query: Any) -> Optional[str]:
    """Validate a query string. Returns an error message or None if valid."""
    if query is None or not isinstance(query, str) or not query.strip():
        return "Query cannot be empty."
    if len(query) > MAX_QUERY_LENGTH:
        return f"Query exceeds maximum length of {MAX_QUERY_LENGTH} characters."
    return None


def _validate_top_k(top_k: Any) -> Optional[str]:
    """Validate a top_k value. Returns an error message or None if valid."""
    try:
        top_k_int = int(top_k)
    except (TypeError, ValueError):
        return f"top_k must be an integer (got {type(top_k).__name__})."
    if top_k_int <= 0:
        return f"top_k must be greater than 0 (got {top_k_int})."
    if top_k_int > 50:
        return f"top_k must be <= 50 (got {top_k_int})."
    return None


def _validate_patient_id(patient_id: Any) -> Optional[str]:
    """Validate a patient ID. Returns an error message or None if valid."""
    if patient_id is None or not isinstance(patient_id, str):
        return "Patient ID is required."
    if not patient_id.strip():
        return "Patient ID cannot be empty."
    if len(patient_id) > 128:
        return "Patient ID is too long (max 128 characters)."
    if "/" in patient_id or "\\" in patient_id:
        return "Patient ID contains invalid characters."
    return None


# ---------------------------------------------------------------------------
# Result builders
# ---------------------------------------------------------------------------

def _error_result(message: str) -> CallToolResult:
    """Build an MCP CallToolResult with an error message."""
    return CallToolResult(
        content=[TextContent(type="text", text=message)],
        isError=True,
    )


def _json_result(data: Any) -> CallToolResult:
    """Build an MCP CallToolResult with structured JSON data."""
    if isinstance(data, str):
        return CallToolResult(
            content=[TextContent(type="text", text=data)],
            isError=False,
        )
    json_str = json.dumps(data, indent=2, default=str)
    return CallToolResult(
        content=[TextContent(type="text", text=json_str)],
        isError=False,
    )


def _truncate(text: Optional[str], max_len: int = 200) -> str:
    """Truncate text to max_len for safe output."""
    if not text:
        return ""
    return str(text)[:max_len]


# ---------------------------------------------------------------------------
# Pipeline factory (injectable for testing)
# ---------------------------------------------------------------------------

def _default_pipeline_factory() -> RAGPipeline:
    """Create a RAGPipeline from configuration.

    This is the production factory used when no custom factory is injected.
    """
    from ..cli import create_pipeline
    return create_pipeline()


def _default_patient_provider_factory():
    """Create a PatientContextProvider from configuration."""
    return get_patient_context_provider()


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

def handle_search_knowledge(
    args: Dict[str, Any],
    pipeline_factory: Callable = _default_pipeline_factory,
) -> CallToolResult:
    """search_knowledge: Search the RAG knowledge base (retrieval only)."""
    query = args.get("query")
    top_k = args.get("top_k", 5)

    err = _validate_query(query)
    if err:
        return _error_result(err)

    err = _validate_top_k(top_k)
    if err:
        return _error_result(err)

    try:
        pipeline = pipeline_factory()
    except Exception as exc:
        return _error_result(f"Failed to initialize RAG pipeline: {exc}")

    try:
        retrieved = pipeline.retriever.retrieve(
            query=query.strip(),
            top_k=int(top_k),
            score_threshold=None,
        )
    except Exception as exc:
        return _error_result(f"Retrieval failed: {exc}")

    sources = []
    for idx, doc in enumerate(retrieved, start=1):
        metadata = doc.get("metadata") or {}
        sources.append({
            "rank": idx,
            "source": metadata.get("source_name") or metadata.get("source", "unknown"),
            "page": metadata.get("page", metadata.get("pages", "unknown")),
            "score": doc.get("score"),
            "distance": doc.get("distance"),
            "source_type": metadata.get("source_type", "general_reference"),
            "trust_level": metadata.get("trust_level", "unspecified"),
            "snippet": _truncate(doc.get("content", ""), 200),
        })

    return _json_result({
        "query": query.strip(),
        "result_count": len(sources),
        "sources": sources,
    })


def handle_ask_rag(
    args: Dict[str, Any],
    pipeline_factory: Callable = _default_pipeline_factory,
    patient_provider_factory: Callable = _default_patient_provider_factory,
) -> CallToolResult:
    """ask_rag: Ask the grounded RAG question-answering pipeline."""
    query = args.get("query")
    patient_id = args.get("patient_id")
    top_k = args.get("top_k", 5)

    err = _validate_query(query)
    if err:
        return _error_result(err)

    err = _validate_top_k(top_k)
    if err:
        return _error_result(err)

    if patient_id is not None:
        err = _validate_patient_id(patient_id)
        if err:
            return _error_result(err)

    try:
        pipeline = pipeline_factory()
    except Exception as exc:
        return _error_result(f"Failed to initialize RAG pipeline: {exc}")

    patient_provider = None
    if patient_id:
        try:
            patient_provider = patient_provider_factory()
        except Exception as exc:
            return _error_result(f"Failed to initialize patient provider: {exc}")

    try:
        result = pipeline.answer(
            query=query.strip(),
            top_k=int(top_k),
            patient_id=patient_id,
            patient_context_provider=patient_provider,
        )
    except Exception as exc:
        return _error_result(f"Error during answer generation: {exc}")

    # Preserve the full structured result including:
    # - answer with [N] citations
    # - refused flag for safe refusals
    # - confidence level
    # - source documents
    return _json_result(result)


def handle_get_patient_context(
    args: Dict[str, Any],
    patient_provider_factory: Callable = _default_patient_provider_factory,
) -> CallToolResult:
    """get_patient_context: Retrieve sanitized patient context."""
    patient_id = args.get("patient_id")

    err = _validate_patient_id(patient_id)
    if err:
        return _error_result(err)

    try:
        provider = patient_provider_factory()
    except Exception as exc:
        return _error_result(f"Failed to initialize patient provider: {exc}")

    try:
        context = provider.get_patient_context(patient_id)
    except Exception as exc:
        return _error_result(f"Failed to load patient context: {exc}")

    sanitized = _sanitize_patient_context(context)
    return _json_result(sanitized)


def handle_get_ml_context_interface(
    args: Dict[str, Any],
) -> CallToolResult:
    """get_ml_context_interface: Return the ML context interface contract.

    There is currently no persistent ML session provider in the codebase.
    ML session results are injected into the RAG pipeline at call time via
    the ``ml_result`` parameter of ``RAGPipeline.answer()``.

    This tool returns the documented interface so that SoleusAI developers
    know the expected data format. When an ML session provider is added in
    the future, this tool can be extended to fetch actual data.
    """
    return _json_result({
        "available": False,
        "message": (
            "No persistent ML session provider is currently configured. "
            "ML session results are injected at pipeline invocation time "
            "via the 'ml_result' parameter of RAGPipeline.answer(). "
            "This is an explicit integration point for future ML session "
            "data sources."
        ),
        "interface": {
            "type": "MLSessionResult",
            "fields": [
                "activation_score (Optional[float])",
                "repetition_count (Optional[int])",
                "resistance (Optional[float])",
                "movement_quality (Optional[float])",
                "fatigue_score (Optional[float])",
                "timestamp (Optional[str])",
                "model_version (Optional[str])",
            ],
        },
    })


def handle_validate_medical_evidence(
    args: Dict[str, Any],
    pipeline_factory: Callable = _default_pipeline_factory,
) -> CallToolResult:
    """validate_medical_evidence: Check clinical trust requirements.

    Reuses the existing pipeline medical safety boundary logic.
    """
    query = args.get("query")
    top_k = args.get("top_k", 5)

    err = _validate_query(query)
    if err:
        return _error_result(err)

    err = _validate_top_k(top_k)
    if err:
        return _error_result(err)

    from ..pipeline import RAGPipeline

    is_clinical_query = RAGPipeline._is_medical_prescription_query(query.strip())

    try:
        pipeline = pipeline_factory()
    except Exception as exc:
        return _error_result(f"Failed to initialize RAG pipeline: {exc}")

    try:
        retrieved = pipeline.retriever.retrieve(
            query=query.strip(),
            top_k=int(top_k),
        )
    except Exception as exc:
        return _error_result(f"Retrieval failed: {exc}")

    has_doctor_approved = RAGPipeline._has_doctor_approved_evidence(retrieved)

    return _json_result({
        "query": query.strip(),
        "is_clinical_query": is_clinical_query,
        "retrieved_count": len(retrieved),
        "has_doctor_approved_evidence": has_doctor_approved,
        "safe_to_proceed": (not is_clinical_query) or has_doctor_approved,
        "recommendation": (
            "Evidence meets clinical requirements for recommendation."
            if has_doctor_approved
            else (
                "No doctor-approved evidence found. "
                "Clinical recommendation requires physician review."
                if is_clinical_query
                else "Non-clinical query. Standard evidence applies."
            )
        ),
    })


# ---------------------------------------------------------------------------
# Patient context sanitization
# ---------------------------------------------------------------------------

def _sanitize_patient_context(
    patient: PatientContext,
) -> Dict[str, Any]:
    """Return a sanitized view of patient context for MCP output.

    No internal filesystem paths or secrets are exposed.
    """
    result: Dict[str, Any] = {
        "patient_id": patient.patient_id,
        "medical_conditions": patient.medical_conditions or [],
        "diabetes_status": patient.diabetes_status,
        "contraindications": patient.contraindications or [],
        "rehabilitation_stage": patient.rehabilitation_stage,
    }

    if patient.demographics:
        result["demographics"] = {
            "age": patient.demographics.age,
            "sex": patient.demographics.sex,
            "bmi": patient.demographics.bmi,
            "height_cm": patient.demographics.height_cm,
            "weight_kg": patient.demographics.weight_kg,
        }

    result["doctor_notes"] = _truncate(patient.doctor_notes)
    result["current_medications"] = patient.current_medications or []
    result["exercise_history"] = _truncate(patient.exercise_history)
    result["relevant_medical_history"] = _truncate(
        patient.relevant_medical_history
    )

    return result


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

TOOL_HANDLERS: Dict[str, Callable[[Dict[str, Any]], CallToolResult]] = {
    "search_knowledge": handle_search_knowledge,
    "ask_rag": handle_ask_rag,
    "get_patient_context": handle_get_patient_context,
    "get_ml_context_interface": handle_get_ml_context_interface,
    "validate_medical_evidence": handle_validate_medical_evidence,
}


TOOL_DEFINITIONS: List[Tool] = [
    Tool(
        name="search_knowledge",
        description=(
            "Search the RAG knowledge base for relevant documents. "
            "Returns retrieval results with citations only — "
            "does NOT generate an answer."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query to find relevant documents.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (1-50, default 5).",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="ask_rag",
        description=(
            "Ask the grounded RAG question-answering pipeline. "
            "Generates a grounded answer with citations. "
            "May include patient context if patient_id is provided."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The question to answer.",
                },
                "patient_id": {
                    "type": "string",
                    "description": "Optional patient ID for context-aware answers.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of documents to retrieve (1-50, default 5).",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="get_patient_context",
        description=(
            "Retrieve sanitized patient context for the given patient ID. "
            "Uses the configured PatientContextProvider abstraction. "
            "No filesystem paths or secrets are exposed."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "patient_id": {
                    "type": "string",
                    "description": "The patient identifier.",
                },
            },
            "required": ["patient_id"],
        },
    ),
    Tool(
        name="get_ml_context_interface",
        description=(
            "Return the interface contract for ML session context. "
            "Currently returns the documented MLSessionResult format "
            "since no persistent ML session provider is configured. "
            "This is an explicit integration point for future ML providers."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="validate_medical_evidence",
        description=(
            "Check whether retrieved evidence meets clinical trust "
            "requirements for a medical/prescription-style question. "
            "Reuses existing RAG safety boundary logic."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The clinical query to evaluate.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of documents to retrieve (1-50, default 5).",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    ),
]


def get_tool_definitions() -> List[Tool]:
    """Return the list of MCP tool definitions."""
    return list(TOOL_DEFINITIONS)


def call_tool(name: str, arguments: Dict[str, Any]) -> CallToolResult:
    """Dispatch a tool call to the appropriate handler.

    This is the synchronous entry point used by the MCP server.
    All handlers delegate to existing RAG components — no logic is
    duplicated in this module.
    """
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        return _error_result(f"Unknown tool: {name}")

    try:
        return handler(arguments or {})
    except Exception as exc:
        logger.exception("MCP tool '%s' failed: %s", name, exc)
        return _error_result(f"Tool '{name}' failed: {exc}")
