from typing import Any, Dict, List


class ContextBuilder:
    """Build structured context from retrieved RAG documents."""

    def build_context(
        self,
        retrieved_documents: List[Dict[str, Any]],
    ) -> str:
        """
        Convert retrieved document chunks into structured LLM context.

        Each retrieved document may contain:
        - content
        - metadata
        - rank
        """

        if not retrieved_documents:
            return ""

        context_parts = []

        for index, doc in enumerate(retrieved_documents, start=1):
            rank = doc.get("rank", index)
            content = str(doc.get("content", "")).strip()

            if not content:
                continue

            metadata = doc.get("metadata") or {}

            source = metadata.get("source", "unknown")
            page = metadata.get(
                "page",
                metadata.get("pages", "unknown"),
            )

            # Keep the original source header format for compatibility
            # with existing tests and downstream code.
            header = f"[{rank}] Source: {source}, Page: {page}"

            formatted_doc = (
                f"{header}\n"
                f"Content: {content}"
            )

            context_parts.append(formatted_doc)

        return "\n\n".join(context_parts)
