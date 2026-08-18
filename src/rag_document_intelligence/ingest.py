from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document

from .chunking import DocumentChunker
from .config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DOCUMENTS_DIR,
    EMBEDDING_MODEL_NAME,
    VECTOR_STORE_DIR,
)
from .embeddings import EmbeddingManager
from .loaders import DocumentLoader
from .vector_store import VectorStore


def ingest_documents(
    documents: List[Document],
) -> int:
    """
    Run the complete ingestion pipeline for already-loaded documents.

    Flow:
        Documents
        -> Chunking
        -> Embeddings
        -> ChromaDB
    """
    if not documents:
        raise ValueError("No documents provided for ingestion.")

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

    print(f"Input documents: {len(documents)}")

    chunks = chunker.split_documents(documents)

    if not chunks:
        raise ValueError("No chunks were created from the provided documents.")

    print(f"Created {len(chunks)} chunks.")

    texts = [chunk.page_content for chunk in chunks]

    print("Generating embeddings...")
    embeddings = embedding_manager.generate_embeddings(texts)

    print(f"Generated {len(embeddings)} embeddings.")

    added_count = vector_store.add_documents(
        chunks,
        embeddings,
    )

    print(f"Added {added_count} new chunks.")
    print(f"Total documents in collection: {vector_store.collection.count()}")

    return added_count


def ingest_file(
    file_path: str | Path,
) -> int:
    """
    Ingest one supported file.

    Supported:
        PDF, TXT, MD, DOCX
    """
    loader = DocumentLoader()

    documents = loader.load_file(file_path)

    if not documents:
        raise ValueError(
            f"No readable content found in file: {file_path}"
        )

    return ingest_documents(documents)


def ingest_text(
    text: str,
    source_name: str = "user_input",
) -> int:
    """
    Ingest direct user-provided text.
    """
    loader = DocumentLoader()

    documents = loader.load_text_content(
        text,
        source_name=source_name,
    )

    return ingest_documents(documents)


def run_ingestion(
    docs_dir: Optional[str] = None,
):
    """
    Ingest every supported document from a directory.
    """
    target_dir = Path(docs_dir or DOCUMENTS_DIR)

    print("--- Starting Document Ingestion ---")
    print(f"Scanning directory: {target_dir}")

    if not target_dir.exists():
        print(f"Error: Directory '{target_dir}' does not exist.")
        return

    loader = DocumentLoader(
        documents_dir=target_dir
    )

    print("Loading documents...")
    documents = loader.load_all_documents()

    if not documents:
        print("No supported documents found to ingest.")
        return

    print(f"Loaded {len(documents)} document(s).")

    added_count = ingest_documents(documents)

    print("--- Ingestion Complete ---")
    print(f"Added {added_count} new chunks.")
    print(f"Vector store: {VECTOR_STORE_DIR}")


if __name__ == "__main__":
    import sys

    custom_dir = sys.argv[1] if len(sys.argv) > 1 else None
    run_ingestion(custom_dir)
