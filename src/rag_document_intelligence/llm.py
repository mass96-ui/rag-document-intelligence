from abc import ABC, abstractmethod
from typing import Optional

import requests

from .config import OLLAMA_BASE_URL, OLLAMA_MODEL


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(self, query: str, context: str) -> str:
        """Generate an answer using the supplied query and context."""
        raise NotImplementedError


class MockLLMProvider(LLMProvider):
    """Deterministic provider used for testing."""

    def generate(self, query: str, context: str) -> str:
        context_preview = (
            context[:100] + "..."
            if len(context) > 100
            else context
        )

        return (
            "[MOCK RESPONSE]\n"
            f"Question: {query}\n"
            f"Based on the provided context "
            f"(Preview: {context_preview}), "
            "this is a simulated answer."
        )


class OllamaLLMProvider(LLMProvider):
    """Local LLM provider using the Ollama HTTP API."""

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = OLLAMA_MODEL,
        timeout: int = 120,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(self, query: str, context: str) -> str:
        if not context.strip():
            return (
                "I could not answer the question because "
                "no relevant document context was retrieved."
            )

        prompt = f"""You are a document question-answering assistant.

Answer the user's question using ONLY the provided context.

If the answer cannot be found in the context, say:
"I could not find this information in the provided documents."

Do not invent facts.

Context:
{context}

Question:
{query}

Answer:"""

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
            },
            timeout=self.timeout,
        )

        response.raise_for_status()

        data = response.json()

        answer = data.get("response", "").strip()

        if not answer:
            raise RuntimeError(
                "Ollama returned an empty response."
            )

        return answer


def get_llm_provider(
    provider_name: Optional[str] = None,
) -> LLMProvider:
    """Create an LLM provider from configuration."""

    from .config import LLM_PROVIDER

    provider = (
        provider_name or LLM_PROVIDER
    ).lower().strip()

    if provider == "mock":
        return MockLLMProvider()

    if provider == "ollama":
        return OllamaLLMProvider()

    raise ValueError(
        f"Unsupported LLM provider: '{provider}'. "
        "Supported providers: mock, ollama."
    )
