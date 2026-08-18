from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(self, query: str, context: str) -> str:
        """
        Generate an answer for the given query based on the provided context.

        Args:
            query: The user's question.
            context: The retrieved document context.

        Returns:
            The generated answer string.
        """
        pass


class MockLLMProvider(LLMProvider):
    """A mock LLM provider for testing and validation."""

    def generate(self, query: str, context: str) -> str:
        """
        Return a deterministic mock response.

        The response includes parts of the query and context to verify
        that the data was passed correctly.
        """
        # Truncate context for display in the mock answer
        context_preview = (
            context[:100] + "..." if len(context) > 100 else context
        )

        return (
            f"[MOCK RESPONSE]\n"
            f"Question: {query}\n"
            f"Based on the provided context (Preview: {context_preview}), "
            f"this is a simulated answer."
        )
