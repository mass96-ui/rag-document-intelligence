import sys
from typing import Any, Dict

from .config import (
    COLLECTION_NAME,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL_NAME,
    LLM_PROVIDER,
    VECTOR_STORE_DIR,
)
from .context_builder import ContextBuilder
from .embeddings import EmbeddingManager
from .llm import get_llm_provider
from .pipeline import RAGPipeline
from .retriever import RAGRetriever
from .vector_store import VectorStore


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
            metadata = doc.get("metadata") or {}

            source = metadata.get("source", "unknown")
            page = metadata.get(
                "page",
                metadata.get("pages", "unknown"),
            )

            rank = doc.get("rank", index)

            print(
                f"[{rank}] {source} | Page: {page}"
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
        persist_directory=str(VECTOR_STORE_DIR),
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

    print("\n" + "=" * 60)
    print("             RAG DOCUMENT INTELLIGENCE")
    print("=" * 60)

    print(f"Embedding model : {EMBEDDING_MODEL_NAME}")
    print(f"Vector database : ChromaDB")
    print(f"LLM provider    : {LLM_PROVIDER}")
    print(f"Top-K retrieval : {DEFAULT_TOP_K}")

    print("\nInitializing RAG pipeline...")

    try:
        pipeline = create_pipeline()
    except Exception as exc:
        print(f"\nFailed to initialize RAG pipeline: {exc}")
        sys.exit(1)

    print("\nPipeline ready.")
    print("Type your question.")
    print("Type 'exit' or 'quit' to close the application.")

    while True:
        try:
            query = input("\nAsk a question: ").strip()

            if query.lower() in {"exit", "quit"}:
                print("\nGoodbye!")
                break

            if not query:
                print("Please enter a question.")
                continue

            result = pipeline.answer(
                query,
                top_k=DEFAULT_TOP_K,
            )

            print_response(result)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break

        except Exception as exc:
            print(
                f"\nAn unexpected error occurred: {exc}"
            )


if __name__ == "__main__":
    run_app()
