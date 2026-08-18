import sys
from typing import Any, Dict

from .config import (
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    VECTOR_STORE_DIR,
    LLM_PROVIDER,
    DEFAULT_TOP_K,
)
from .context_builder import ContextBuilder
from .embeddings import EmbeddingManager
from .llm import get_llm_provider
from .pipeline import RAGPipeline
from .retriever import RAGRetriever
from .vector_store import VectorStore


def print_response(result: Dict[str, Any]):
    """Print the RAG response in a user-friendly format."""
    print("\n" + "="*50)
    print("QUESTION:", result["query"])
    print("="*50)
    print("\nANSWER:")
    print(result["answer"])
    print("\n" + "-"*50)
    
    if result["source_documents"]:
        print(f"SOURCES ({len(result['source_documents'])} chunks retrieved, context length: {result['context_length']}):")
        for doc in result["source_documents"]:
            metadata = doc.get("metadata", {})
            source = metadata.get("source", "unknown")
            page = metadata.get("page", metadata.get("pages", "unknown"))
            rank = doc.get("rank", "N/A")
            print(f"[{rank}] {source} (Page {page})")
    else:
        print("SOURCES: No relevant documents found.")
    print("="*50 + "\n")


def run_app():
    """Main CLI loop for querying the RAG pipeline."""
    print("--- RAG Document Intelligence CLI ---")
    print(f"Loading vector store from: {VECTOR_STORE_DIR}")
    print(f"Using LLM Provider: {LLM_PROVIDER}")

    # 1. Initialize the pipeline
    try:
        embedding_manager = EmbeddingManager(model_name=EMBEDDING_MODEL_NAME)
        vector_store = VectorStore(
            persist_directory=str(VECTOR_STORE_DIR),
            collection_name=COLLECTION_NAME
        )
        retriever = RAGRetriever(vector_store, embedding_manager)
        context_builder = ContextBuilder()
        llm_provider = get_llm_provider(LLM_PROVIDER)

        pipeline = RAGPipeline(retriever, context_builder, llm_provider)
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        sys.exit(1)

    print("Pipeline ready! Type your question or 'exit' to quit.")

    while True:
        try:
            query = input("\nAsk a question: ").strip()
            
            if query.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            
            if not query:
                continue

            # Run the pipeline
            result = pipeline.answer(query, top_k=DEFAULT_TOP_K)
            
            # Display results
            print_response(result)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")


if __name__ == "__main__":
    run_app()
