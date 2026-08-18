from typing import Any, Dict, List


class ContextBuilder:
    """Transform retrieved document chunks into formatted LLM context."""

    def build_context(
        self,
        retrieved_documents: List[Dict[str, Any]],
    ) -> str:
        """
        Format a list of retrieved documents into a single context string.

        Each document should include:
        - content
        - metadata (source, page)
        - rank
        """
        if not retrieved_documents:
            return ""

        context_parts = []

        for doc in retrieved_documents:
            rank = doc.get("rank", "N/A")
            content = doc.get("content", "").strip()
            metadata = doc.get("metadata", {})
            source = metadata.get("source", "unknown")
            page = metadata.get("page", metadata.get("pages", "unknown"))

            header = f"[{rank}] Source: {source}, Page: {page}"
            formatted_doc = f"{header}\nContent: {content}"
            context_parts.append(formatted_doc)

        return "\n\n".join(context_parts)
