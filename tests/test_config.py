"""Tests for configuration loading and validation."""

import importlib
from unittest.mock import patch

import pytest


def _reload_config(monkeypatch, **env_vars):
    """Reload config module with patched environment variables."""
    for key, value in env_vars.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, str(value))

    with patch(
        "dotenv.load_dotenv", return_value=False
    ):
        import rag_document_intelligence.config as config_module
        return importlib.reload(config_module)


def test_config_defaults(monkeypatch):
    config = _reload_config(
        monkeypatch,
        LLM_PROVIDER="ollama",
        COLLECTION_NAME="test_docs",
        EMBEDDING_MODEL_NAME="all-MiniLM-L6-v2",
        CHUNK_SIZE="500",
        CHUNK_OVERLAP="50",
        DEFAULT_TOP_K="5",
        SCORE_THRESHOLD="",
        MAX_QUERY_LENGTH="2000",
        OLLAMA_TIMEOUT="120",
        OLLAMA_BASE_URL="http://localhost:11434",
        OLLAMA_MODEL="llama3.2",
    )
    assert config.LLM_PROVIDER == "ollama"
    assert config.MAX_QUERY_LENGTH == 2000
    assert config.OLLAMA_TIMEOUT == 120
    assert config.COLLECTION_NAME == "test_docs"


def test_config_env_overrides(monkeypatch):
    config = _reload_config(
        monkeypatch,
        LLM_PROVIDER="mock",
        COLLECTION_NAME="custom_collection",
        CHUNK_SIZE="256",
        DEFAULT_TOP_K="10",
        SCORE_THRESHOLD="0.5",
        MAX_QUERY_LENGTH="500",
        OLLAMA_TIMEOUT="30",
        OLLAMA_MODEL="qwen2.5",
    )
    assert config.LLM_PROVIDER == "mock"
    assert config.COLLECTION_NAME == "custom_collection"
    assert config.CHUNK_SIZE == 256
    assert config.DEFAULT_TOP_K == 10
    assert config.SCORE_THRESHOLD == 0.5
    assert config.MAX_QUERY_LENGTH == 500
    assert config.OLLAMA_TIMEOUT == 30
    assert config.OLLAMA_MODEL == "qwen2.5"


def test_config_score_threshold_default(monkeypatch):
    config = _reload_config(
        monkeypatch,
        SCORE_THRESHOLD=None,
    )
    assert config.SCORE_THRESHOLD == 1.0


def test_config_invalid_chunk_size(monkeypatch):
    config = _reload_config(
        monkeypatch,
        CHUNK_SIZE="0",
        CHUNK_OVERLAP="0",
    )
    with pytest.raises(ValueError, match="CHUNK_SIZE"):
        config.validate_config()


def test_config_invalid_overlap(monkeypatch):
    config = _reload_config(
        monkeypatch,
        CHUNK_SIZE="100",
        CHUNK_OVERLAP="100",
    )
    with pytest.raises(ValueError, match="CHUNK_OVERLAP"):
        config.validate_config()


def test_config_invalid_top_k(monkeypatch):
    config = _reload_config(
        monkeypatch,
        DEFAULT_TOP_K="0",
    )
    with pytest.raises(ValueError, match="DEFAULT_TOP_K"):
        config.validate_config()


def test_config_invalid_provider(monkeypatch):
    config = _reload_config(
        monkeypatch,
        LLM_PROVIDER="openai",
    )
    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        config.validate_config()


def test_config_invalid_score_threshold(monkeypatch):
    config = _reload_config(
        monkeypatch,
        SCORE_THRESHOLD="-1",
    )
    with pytest.raises(ValueError, match="SCORE_THRESHOLD"):
        config.validate_config()


def test_config_invalid_timeout(monkeypatch):
    config = _reload_config(
        monkeypatch,
        OLLAMA_TIMEOUT="0",
    )
    with pytest.raises(ValueError, match="OLLAMA_TIMEOUT"):
        config.validate_config()
