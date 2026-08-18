import logging
import sys
from typing import Any, Dict

from .config import (
    COLLECTION_NAME,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL_NAME,
    LLM_PROVIDER,
    MAX_QUERY_LENGTH,
    VECTOR_STORE_DIR,
)
from .context_builder import ContextBuilder
from .embeddings import EmbeddingManager
from .llm import get_llm_provider
from .pipeline import RAGPipeline
from .retriever import RAGRetriever
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Configure logging for CLI usage."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stderr,
    )


def _safe_source(doc: Dict[str, Any], index: int) -> str:
    """Return a human-readable source label for a retrieved doc."""
    metadata = doc.get("metadata") or {}
    source_name = metadata.get("source_name")
    if source_name:
        return str(source_name)
    source = metadata.get("source", "unknown")
    if source and source != "unknown":
        from pathlib import Path
        return Path(str(source)).name
    return "unknown"


def _safe_page(doc: Dict[str, Any]) -> str:
    """Return a human-readable page label for a retrieved doc."""
    metadata = doc.get("metadata") or {}
    page = metadata.get(
        "page",
        metadata.get("pages", "unknown"),
    )
    if page is None:
        return "unknown"
    return str(page)


def print_response(result: Dict[str, Any]) -> None:
    """Display a RAG result in a readable format."""

    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(result["answer"])

    print("\n" + "-" * 60)
    print("RETRIEVED SOURCES")
    print("-" * 60)

    source_documents = result.get("source_documents", [])

    if not source_documents:
        print("No relevant documents found.")
    else:
        for index, doc in enumerate(source_documents, start=1):
            rank = doc.get("rank", index)
            source = _safe_source(doc, index)
            page = _safe_page(doc)
            score = doc.get("score")
            score_str = ""
            if score is not None:
                try:
                    score_str = f" | Score: {float(score):.3f}"
                except (TypeError, ValueError):
                    pass

            print(
                f"[{rank}] {source} | Page: {page}{score_str}"
            )

    print(
        f"\nRetrieved chunks: {len(source_documents)}"
    )
    print(
        f"Context length: {result.get('context_length', 0)} characters"
    )

    print("=" * 60)


def create_pipeline() -> RAGPipeline:
    """Initialize all RAG pipeline components."""

    embedding_manager = EmbeddingManager(
        model_name=EMBEDDING_MODEL_NAME
    )

    vector_store = VectorStore(
        persist_directory=VECTOR_STORE_DIR,
        collection_name=COLLECTION_NAME,
    )

    retriever = RAGRetriever(
        vector_store,
        embedding_manager,
    )

    context_builder = ContextBuilder()

    llm_provider = get_llm_provider(
        LLM_PROVIDER
    )

    return RAGPipeline(
        retriever,
        context_builder,
        llm_provider,
    )


def run_app() -> None:
    """Run the interactive RAG command-line application."""

    _configure_logging()

    print("\n" + "=" * 60)
    print("             RAG DOCUMENT INTELLIGENCE")
    print("=" * 60)

    print(f"Embedding model : {EMBEDDING_MODEL_NAME}")
    print(f"Vector database : ChromaDB")
    print(f"LLM provider    : {LLM_PROVIDER}")
    print(f"Top-K retrieval : {DEFAULT_TOP_K}")
    print(f"Max query length: {MAX_QUERY_LENGTH} characters")

    print("\nInitializing RAG pipeline...")

    logger.info("Loading embedding model and vector store...")

    try:
        pipeline = create_pipeline()
    except Exception as exc:
        logger.error("Failed to initialize RAG pipeline: %s", exc)
        print(
            f"\nA fatal error occurred during initialization: {exc}"
        )
        sys.exit(1)

    print("\nPipeline ready.")
    print("Type your question.")
    print("Type 'exit' or 'quit' to close the application.")

    while True:
        try:
            query = input("\nAsk a question: ").strip()

        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break

        if query.lower() in {"exit", "quit"}:
            print("\nGoodbye!")
            break

        if not query:
            print("Please enter a question.")
            continue

        if len(query) > MAX_QUERY_LENGTH:
            print(
                f"Question exceeds the maximum length of "
                f"{MAX_QUERY_LENGTH} characters. Please shorten it."
            )
            continue

        try:
            result = pipeline.answer(
                query,
                top_k=DEFAULT_TOP_K,
            )

            print_response(result)

        except Exception as exc:
            logger.error("Unexpected error: %s", exc)
            print(
                "\nAn error occurred while processing your "
                "request. Please try again."
            )


def main() -> None:
    """Entry point for the ``rag`` console script."""
    run_app()


if __name__ == "__main__":
    run_app()
