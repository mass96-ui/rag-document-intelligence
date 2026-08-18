import os
from pathlib import Path

from dotenv import load_dotenv


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load environment variables from .env
load_dotenv(PROJECT_ROOT / ".env", override=True)


# Data directories
DATA_DIR = PROJECT_ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"


# Vector database
COLLECTION_NAME = os.getenv(
    "COLLECTION_NAME",
    "pdf_documents",
)


# Embedding configuration
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "all-MiniLM-L6-v2",
)


# RAG configuration
CHUNK_SIZE = int(
    os.getenv("CHUNK_SIZE", "500")
)

CHUNK_OVERLAP = int(
    os.getenv("CHUNK_OVERLAP", "50")
)

DEFAULT_TOP_K = int(
    os.getenv("DEFAULT_TOP_K", "5")
)


# LLM configuration
LLM_PROVIDER = os.getenv(
    "LLM_PROVIDER",
    "mock",
).lower()

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3.2",
)
