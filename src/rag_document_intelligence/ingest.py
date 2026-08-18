import os
import sys
from typing import Optional

from .chunking import DocumentChunker
from .config import (
    COLLECTION_NAME,
    DOCUMENTS_DIR,
    EMBEDDING_MODEL_NAME,
    VECTOR_STORE_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from .embeddings import EmbeddingManager
from .loaders import DocumentLoader
from .vector_store import VectorStore


def run_ingestion(docs_dir: Optional[str] = None):
    """
    Load documents, split them into chunks, generate embeddings,
    and store the chunks and embeddings in ChromaDB.
    """
    target_dir = docs_dir or DOCUMENTS_DIR

    print("--- Starting Document Ingestion ---")
    print(f"Scanning directory: {target_dir}")

    if not os.path.exists(target_dir):
        print(f"Error: Directory '{target_dir}' does not exist.")
        return

    # 1. Initialize components
    loader = DocumentLoader(documents_dir=target_dir)

    chunker = DocumentChunker(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    embedding_manager = EmbeddingManager(
        model_name=EMBEDDING_MODEL_NAME
    )

    vector_store = VectorStore(
        persist_directory=str(VECTOR_STORE_DIR),
        collection_name=COLLECTION_NAME,
    )

    # 2. Load documents
    print("Loading documents...")
    documents = loader.load_all_documents()

    if not documents:
        print("No documents found to ingest.")
        return

    print(f"Loaded {len(documents)} document(s).")

    # 3. Split documents into chunks
    print("Splitting documents into chunks...")
    chunks = chunker.split_documents(documents)

    if not chunks:
        print("No chunks were created.")
        return

    print(f"Created {len(chunks)} chunks.")

    # 4. Generate embeddings for each chunk
    print("Generating embeddings...")
    texts = [chunk.page_content for chunk in chunks]

    embeddings = embedding_manager.generate_embeddings(texts)

    print(f"Generated {len(embeddings)} embeddings.")

    # 5. Store chunks and embeddings in ChromaDB
    print("Adding chunks to vector store...")
    added_count = vector_store.add_documents(
        chunks,
        embeddings,
    )

    print(f"Added {added_count} new chunks.")

    print("--- Ingestion Complete ---")
    print(f"Vector store updated at: {VECTOR_STORE_DIR}")


if __name__ == "__main__":
    custom_dir = sys.argv[1] if len(sys.argv) > 1 else None
    run_ingestion(custom_dir)