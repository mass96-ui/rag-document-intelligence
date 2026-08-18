import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load environment variables from .env (optional — does not crash if absent)
load_dotenv(PROJECT_ROOT / ".env", override=True)


# Data directories
DATA_DIR = PROJECT_ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"


def _get_int_env(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid integer for %s=%r; falling back to default %d",
            key, raw, default,
        )
        return default


def _get_float_env(key: str, default: Optional[float]) -> Optional[float]:
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid float for %s=%r; falling back to default %s",
            key, raw, default,
        )
        return default


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
CHUNK_SIZE = _get_int_env("CHUNK_SIZE", 500)
CHUNK_OVERLAP = _get_int_env("CHUNK_OVERLAP", 50)
DEFAULT_TOP_K = _get_int_env("DEFAULT_TOP_K", 5)


# Retrieval configuration
SCORE_THRESHOLD = _get_float_env("SCORE_THRESHOLD", 1.0)
MAX_QUERY_LENGTH = _get_int_env("MAX_QUERY_LENGTH", 2000)


# LLM configuration
SUPPORTED_PROVIDERS = ("mock", "ollama")

LLM_PROVIDER = os.getenv(
    "LLM_PROVIDER",
    "ollama",
).lower().strip()

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3.2",
)

OLLAMA_TIMEOUT = _get_int_env("OLLAMA_TIMEOUT", 120)


def validate_config() -> None:
    """Validate configuration values and raise on invalid settings."""

    errors: list[str] = []

    if CHUNK_SIZE <= 0:
        errors.append(
            f"CHUNK_SIZE must be greater than 0 (got {CHUNK_SIZE})"
        )

    if CHUNK_OVERLAP < 0:
        errors.append(
            f"CHUNK_OVERLAP must be >= 0 (got {CHUNK_OVERLAP})"
        )

    if CHUNK_OVERLAP >= CHUNK_SIZE:
        errors.append(
            f"CHUNK_OVERLAP ({CHUNK_OVERLAP}) must be less than "
            f"CHUNK_SIZE ({CHUNK_SIZE})"
        )

    if DEFAULT_TOP_K <= 0:
        errors.append(
            f"DEFAULT_TOP_K must be greater than 0 (got {DEFAULT_TOP_K})"
        )

    if MAX_QUERY_LENGTH <= 0:
        errors.append(
            f"MAX_QUERY_LENGTH must be greater than 0 "
            f"(got {MAX_QUERY_LENGTH})"
        )

    if OLLAMA_TIMEOUT <= 0:
        errors.append(
            f"OLLAMA_TIMEOUT must be greater than 0 "
            f"(got {OLLAMA_TIMEOUT})"
        )

    if SCORE_THRESHOLD is not None and SCORE_THRESHOLD < 0:
        errors.append(
            f"SCORE_THRESHOLD must be >= 0 (got {SCORE_THRESHOLD})"
        )

    if LLM_PROVIDER not in SUPPORTED_PROVIDERS:
        errors.append(
            f"LLM_PROVIDER must be one of "
            f"{SUPPORTED_PROVIDERS} (got '{LLM_PROVIDER}')"
        )

    if not COLLECTION_NAME.strip():
        errors.append("COLLECTION_NAME must not be empty")

    if not OLLAMA_BASE_URL.strip():
        errors.append("OLLAMA_BASE_URL must not be empty")

    if not OLLAMA_MODEL.strip():
        errors.append("OLLAMA_MODEL must not be empty")

    if errors:
        raise ValueError(
            "Invalid configuration:\n  - "
            + "\n  - ".join(errors)
        )
