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


# ---------------------------------------------------------------------------
# Structured generation tests
# ---------------------------------------------------------------------------


def _structured_response(json_str, status_code=200):
    """Helper: mock Ollama API returning a structured JSON response field."""
    return _mock_response(
        status_code=status_code,
        json_data={"response": json_str, "done": True},
    )


def _structured_call_kwargs(mock_post):
    """Extract the JSON payload from a mocked requests.post call."""
    call = mock_post.call_args
    if call.kwargs:
        return call.kwargs.get("json", {})
    return call[1] if len(call) > 1 else {}


def test_ollama_structured_valid_json():
    with patch(
        "rag_document_intelligence.llm.requests.post"
    ) as mock_post:
        mock_post.return_value = _structured_response(
            '{"answer": "The answer is 42.", "citations": [1, 2]}'
        )
        provider = OllamaLLMProvider()
        result = provider.generate_structured("What is the answer?", "Context: 42")

        assert result["answer"] == "The answer is 42."
        assert result["citations"] == [1, 2]

        payload = _structured_call_kwargs(mock_post)
        assert payload.get("format") == "json"


def test_ollama_structured_multiple_citations():
    with patch(
        "rag_document_intelligence.llm.requests.post"
    ) as mock_post:
        mock_post.return_value = _structured_response(
            '{"answer": "Multi-source.", "citations": [3, 1, 2]}'
        )
        provider = OllamaLLMProvider()
        result = provider.generate_structured("q", "c")
        assert result["citations"] == [1, 2, 3]


def test_ollama_structured_duplicate_citations():
    with patch(
        "rag_document_intelligence.llm.requests.post"
    ) as mock_post:
        mock_post.return_value = _structured_response(
            '{"answer": "Answer.", "citations": [1, 1, 2]}'
        )
        provider = OllamaLLMProvider()
        result = provider.generate_structured("q", "c")
        assert result["citations"] == [1, 2]


def test_ollama_structured_malformed_json():
    with patch(
        "rag_document_intelligence.llm.requests.post"
    ) as mock_post:
        mock_post.return_value = _structured_response("not json at all")
        provider = OllamaLLMProvider()
        with pytest.raises(RuntimeError, match="parsed as JSON"):
            provider.generate_structured("test", "context")


def test_ollama_structured_missing_answer():
    with patch(
        "rag_document_intelligence.llm.requests.post"
    ) as mock_post:
        mock_post.return_value = _structured_response(
            '{"citations": [1]}'
        )
        provider = OllamaLLMProvider()
        with pytest.raises(RuntimeError, match="answer"):
            provider.generate_structured("test", "context")


def test_ollama_structured_empty_answer():
    with patch(
        "rag_document_intelligence.llm.requests.post"
    ) as mock_post:
        mock_post.return_value = _structured_response(
            '{"answer": "", "citations": [1]}'
        )
        provider = OllamaLLMProvider()
        with pytest.raises(RuntimeError, match="answer"):
            provider.generate_structured("test", "context")


def test_ollama_structured_whitespace_answer():
    with patch(
        "rag_document_intelligence.llm.requests.post"
    ) as mock_post:
        mock_post.return_value = _structured_response(
            '{"answer": "   ", "citations": [1]}'
        )
        provider = OllamaLLMProvider()
        with pytest.raises(RuntimeError, match="answer"):
            provider.generate_structured("test", "context")


def test_ollama_structured_missing_citations():
    with patch(
        "rag_document_intelligence.llm.requests.post"
    ) as mock_post:
        mock_post.return_value = _structured_response(
            '{"answer": "some answer"}'
        )
        provider = OllamaLLMProvider()
        with pytest.raises(RuntimeError, match="citations"):
            provider.generate_structured("test", "context")


def test_ollama_structured_citations_not_list():
    with patch(
        "rag_document_intelligence.llm.requests.post"
    ) as mock_post:
        mock_post.return_value = _structured_response(
            '{"answer": "some answer", "citations": "not a list"}'
        )
        provider = OllamaLLMProvider()
        with pytest.raises(RuntimeError, match="citations"):
            provider.generate_structured("test", "context")


def test_ollama_structured_string_citation_rejected():
    with patch(
        "rag_document_intelligence.llm.requests.post"
    ) as mock_post:
        mock_post.return_value = _structured_response(
            '{"answer": "some answer", "citations": ["1"]}'
        )
        provider = OllamaLLMProvider()
        with pytest.raises(RuntimeError, match="not an integer"):
            provider.generate_structured("test", "context")


def test_ollama_structured_bool_citation_rejected():
    with patch(
        "rag_document_intelligence.llm.requests.post"
    ) as mock_post:
        mock_post.return_value = _structured_response(
            '{"answer": "some answer", "citations": [true]}'
        )
        provider = OllamaLLMProvider()
        with pytest.raises(RuntimeError, match="boolean|integer"):
            provider.generate_structured("test", "context")


def test_ollama_structured_negative_citation():
    with patch(
        "rag_document_intelligence.llm.requests.post"
    ) as mock_post:
        mock_post.return_value = _structured_response(
            '{"answer": "some answer", "citations": [-1]}'
        )
        provider = OllamaLLMProvider()
        with pytest.raises(RuntimeError, match="positive|not positive"):
            provider.generate_structured("test", "context")


def test_ollama_structured_zero_citation():
    with patch(
        "rag_document_intelligence.llm.requests.post"
    ) as mock_post:
        mock_post.return_value = _structured_response(
            '{"answer": "some answer", "citations": [0]}'
        )
        provider = OllamaLLMProvider()
        with pytest.raises(RuntimeError, match="positive|not positive"):
            provider.generate_structured("test", "context")


def test_ollama_structured_http_500():
    with patch(
        "rag_document_intelligence.llm.requests.post"
    ) as mock_post:
        resp = _mock_response(
            status_code=500, json_data={"error": "server"}
        )
        resp.raise_for_status.side_effect = requests.HTTPError(
            "500 error"
        )
        mock_post.return_value = resp
        provider = OllamaLLMProvider()
        with pytest.raises(RuntimeError, match="HTTP 500"):
            provider.generate_structured("test", "context")


def test_ollama_structured_http_404():
    with patch(
        "rag_document_intelligence.llm.requests.post"
    ) as mock_post:
        resp = _mock_response(status_code=404, json_data={})
        resp.raise_for_status.side_effect = requests.HTTPError("404")
        mock_post.return_value = resp
        provider = OllamaLLMProvider()
        with pytest.raises(RuntimeError, match="endpoint not found"):
            provider.generate_structured("test", "context")


def test_ollama_structured_http_400():
    with patch(
        "rag_document_intelligence.llm.requests.post"
    ) as mock_post:
        resp = _mock_response(status_code=400, json_data={})
        resp.raise_for_status.side_effect = requests.HTTPError("400")
        mock_post.return_value = resp
        provider = OllamaLLMProvider()
        with pytest.raises(RuntimeError, match="rejected"):
            provider.generate_structured("test", "context")


def test_ollama_structured_timeout():
    with patch(
        "rag_document_intelligence.llm.requests.post"
    ) as mock_post:
        mock_post.side_effect = requests.Timeout("Timed out")
        provider = OllamaLLMProvider(timeout=1)
        with pytest.raises(RuntimeError, match="timed out"):
            provider.generate_structured("test", "context")


def test_ollama_structured_connection_refused():
    with patch(
        "rag_document_intelligence.llm.requests.post"
    ) as mock_post:
        mock_post.side_effect = requests.ConnectionError(
            "Connection refused"
        )
        provider = OllamaLLMProvider()
        with pytest.raises(RuntimeError, match="connect to Ollama"):
            provider.generate_structured("test", "context")


def test_ollama_structured_empty_context():
    provider = OllamaLLMProvider()
    result = provider.generate_structured("test", "")
    assert "could not answer" in result["answer"].lower()
    assert result["citations"] == []


def test_ollama_structured_non_string_response():
    with patch(
        "rag_document_intelligence.llm.requests.post"
    ) as mock_post:
        mock_post.return_value = _mock_response(
            json_data={"response": 12345}
        )
        provider = OllamaLLMProvider()
        with pytest.raises(RuntimeError, match="non-string"):
            provider.generate_structured("test", "context")


def test_ollama_structured_empty_response():
    with patch(
        "rag_document_intelligence.llm.requests.post"
    ) as mock_post:
        mock_post.return_value = _mock_response(
            json_data={"response": "  "}
        )
        provider = OllamaLLMProvider()
        with pytest.raises(RuntimeError, match="empty"):
            provider.generate_structured("test", "context")


def test_ollama_structured_missing_response_field():
    with patch(
        "rag_document_intelligence.llm.requests.post"
    ) as mock_post:
        mock_post.return_value = _mock_response(
            json_data={"done": True}
        )
        provider = OllamaLLMProvider()
        with pytest.raises(RuntimeError, match="response"):
            provider.generate_structured("test", "context")


def test_ollama_structured_non_dict_json():
    with patch(
        "rag_document_intelligence.llm.requests.post"
    ) as mock_post:
        mock_post.return_value = _structured_response(
            '"just a string"'
        )
        provider = OllamaLLMProvider()
        with pytest.raises(RuntimeError, match="not a JSON object"):
            provider.generate_structured("test", "context")


def test_ollama_structured_json_with_markdown_fences():
    with patch(
        "rag_document_intelligence.llm.requests.post"
    ) as mock_post:
        mock_post.return_value = _structured_response(
            '```json\n{"answer": "Hello", "citations": [1]}\n```'
        )
        provider = OllamaLLMProvider()
        result = provider.generate_structured("test", "context")
        assert result["answer"] == "Hello"
        assert result["citations"] == [1]
