from pathlib import Path


# Project root:
# src/rag_document_intelligence/config.py
# parents[2] moves:
# config.py -> rag_document_intelligence -> src -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Local document directory.
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"

# Local persistent ChromaDB directory.
VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "vector_store"

# ChromaDB collection used by the RAG pipeline.
COLLECTION_NAME = "pdf_documents"

# Sentence Transformer embedding model.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Text splitting configuration.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Default number of documents returned by retrieval.
DEFAULT_TOP_K = 5
