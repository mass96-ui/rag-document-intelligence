"""Tests for the Ollama LLM provider (no live Ollama required)."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from rag_document_intelligence.llm import (
    MockLLMProvider,
    OllamaLLMProvider,
    get_llm_provider,
)


def _mock_response(status_code: int = 200, json_data: dict | str = None):
    """Build a MagicMock that mimics requests.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = "raw text" if json_data is None else str(json_data)

    if json_data is None:
        resp.json.side_effect = requests.JSONDecodeError(
            "No JSON", "[]", 0
        )
    elif isinstance(json_data, str):
        resp.json.return_value = json_data
    else:
        resp.json.return_value = json_data

    resp.raise_for_status.return_value = None
    return resp


def test_ollama_provider_success():
    with patch(
        "rag_document_intelligence.llm.requests.post"
    ) as mock_post:
        mock_post.return_value = _mock_response(
            json_data={"response": "The answer is 42.", "done": True}
        )
        provider = OllamaLLMProvider()
        answer = provider.generate("What is the answer?", "Context: 42")

        assert answer == "The answer is 42."
        mock_post.assert_called_once()


def test_ollama_provider_empty_context_returns_refusal():
    provider = OllamaLLMProvider()
    answer = provider.generate("test", "")
    assert "could not answer" in answer.lower()
    assert "no relevant" in answer.lower()


def test_ollama_provider_connection_refused():
    with patch(
        "rag_document_intelligence.llm.requests.post"
    ) as mock_post:
        mock_post.side_effect = requests.ConnectionError("Connection refused")
        provider = OllamaLLMProvider()

        with pytest.raises(RuntimeError, match="connect to Ollama"):
            provider.generate("test", "context")


def test_ollama_provider_timeout():
    with patch(
        "rag_document_intelligence.llm.requests.post"
    ) as mock_post:
        mock_post.side_effect = requests.Timeout("Timed out")
        provider = OllamaLLMProvider(timeout=1)

        with pytest.raises(RuntimeError, match="timed out"):
            provider.generate("test", "context")


def test_ollama_provider_http_error():
    with patch(
        "rag_document_intelligence.llm.requests.post"
    ) as mock_post:
        resp = _mock_response(status_code=500, json_data={"error": "server"})
        resp.raise_for_status.side_effect = requests.HTTPError("500 error")
        mock_post.return_value = resp
        provider = OllamaLLMProvider()

        with pytest.raises(RuntimeError, match="HTTP 500"):
            provider.generate("test", "context")


def test_ollama_provider_http_404():
    with patch(
        "rag_document_intelligence.llm.requests.post"
    ) as mock_post:
        resp = _mock_response(status_code=404, json_data={})
        resp.raise_for_status.side_effect = requests.HTTPError("404")
        mock_post.return_value = resp
        provider = OllamaLLMProvider()

        with pytest.raises(RuntimeError, match="endpoint not found"):
            provider.generate("test", "context")


def test_ollama_provider_http_400():
    with patch(
        "rag_document_intelligence.llm.requests.post"
    ) as mock_post:
        resp = _mock_response(status_code=400, json_data={})
        resp.raise_for_status.side_effect = requests.HTTPError("400")
        mock_post.return_value = resp
        provider = OllamaLLMProvider()

        with pytest.raises(RuntimeError, match="rejected"):
            provider.generate("test", "context")


def test_ollama_provider_invalid_json():
    with patch(
        "rag_document_intelligence.llm.requests.post"
    ) as mock_post:
        mock_post.return_value = _mock_response(json_data=None)
        provider = OllamaLLMProvider()

        with pytest.raises(RuntimeError, match="malformed"):
            provider.generate("test", "context")


def test_ollama_provider_missing_response_field():
    with patch(
        "rag_document_intelligence.llm.requests.post"
    ) as mock_post:
        mock_post.return_value = _mock_response(
            json_data={"done": True}
        )
        provider = OllamaLLMProvider()

        with pytest.raises(RuntimeError, match="response"):
            provider.generate("test", "context")


def test_ollama_provider_non_string_response():
    with patch(
        "rag_document_intelligence.llm.requests.post"
    ) as mock_post:
        mock_post.return_value = _mock_response(
            json_data={"response": 12345}
        )
        provider = OllamaLLMProvider()

        with pytest.raises(RuntimeError, match="non-string"):
            provider.generate("test", "context")


def test_ollama_provider_empty_response():
    with patch(
        "rag_document_intelligence.llm.requests.post"
    ) as mock_post:
        mock_post.return_value = _mock_response(
            json_data={"response": "   "}
        )
        provider = OllamaLLMProvider()

        with pytest.raises(RuntimeError, match="empty"):
            provider.generate("test", "context")


def test_ollama_provider_malformed_non_dict():
    with patch(
        "rag_document_intelligence.llm.requests.post"
    ) as mock_post:
        mock_post.return_value = _mock_response(
            json_data="not a dict"
        )
        provider = OllamaLLMProvider()

        with pytest.raises(RuntimeError, match="unexpected response"):
            provider.generate("test", "context")


def test_get_llm_provider_mock():
    provider = get_llm_provider("mock")
    assert isinstance(provider, MockLLMProvider)


def test_get_llm_provider_ollama():
    provider = get_llm_provider("ollama")
    assert isinstance(provider, OllamaLLMProvider)


def test_get_llm_provider_invalid():
    with pytest.raises(ValueError, match="Unsupported"):
        get_llm_provider("openai")
