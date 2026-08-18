import logging
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

logger = logging.getLogger(__name__)


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
        persist_directory=VECTOR_STORE_DIR,
        collection_name=COLLECTION_NAME,
    )

    logger.info("Input documents: %d", len(documents))

    chunks = chunker.split_documents(documents)

    if not chunks:
        raise ValueError(
            "No chunks were created from the provided documents."
        )

    logger.info("Created %d chunks", len(chunks))

    texts = [chunk.page_content for chunk in chunks]

    logger.info("Generating embeddings...")
    embeddings = embedding_manager.generate_embeddings(texts)

    logger.info("Generated %d embeddings", len(embeddings))

    added_count = vector_store.add_documents(
        chunks,
        embeddings,
    )

    logger.info(
        "Ingestion complete. Added %d new chunks. "
        "Total in collection: %d",
        added_count, vector_store.count(),
    )

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
) -> int:
    """
    Ingest every supported document from a directory.

    Returns the number of newly added chunks.
    """
    target_dir = Path(docs_dir or DOCUMENTS_DIR)

    logger.info("--- Starting Document Ingestion ---")
    logger.info("Scanning directory: %s", target_dir)

    if not target_dir.exists():
        logger.warning(
            "Directory '%s' does not exist.", target_dir
        )
        return 0

    loader = DocumentLoader(
        documents_dir=target_dir
    )

    logger.info("Loading documents...")
    documents = loader.load_all_documents()

    if not documents:
        logger.info("No supported documents found to ingest.")
        return 0

    logger.info("Loaded %d document(s).", len(documents))

    added_count = ingest_documents(documents)

    logger.info("--- Ingestion Complete ---")
    logger.info("Added %d new chunks.", added_count)
    logger.info("Vector store: %s", VECTOR_STORE_DIR)

    return added_count


if __name__ == "__main__":
    import sys

    custom_dir = sys.argv[1] if len(sys.argv) > 1 else None
    run_ingestion(custom_dir)
