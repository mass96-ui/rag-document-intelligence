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

        prompt = f"""You are a precise document question-answering assistant.

Your job is to answer the user's question using ONLY the provided
document context.

STRICT RULES:
1. Do not use outside knowledge.
2. Do not invent or assume facts.
3. Every factual claim must be supported by the provided context.
4. Cite the supporting source using the citation number shown in the
   context, for example [1] or [2].
5. If multiple sources support a statement, cite all relevant sources,
   for example [1][3].
6. If the answer cannot be found in the context, respond exactly:
   "I could not find this information in the provided documents."
7. Keep the answer concise and directly answer the question.
8. Do not create citations that do not exist in the context.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{query}

ANSWER WITH CITATIONS:"""

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                    },
                },
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise RuntimeError(
                "Could not connect to Ollama at "
                f"{self.base_url}. Make sure Ollama is running. "
                f"Original error: {exc}"
            ) from exc

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
