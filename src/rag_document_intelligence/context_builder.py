from pathlib import Path
from typing import Any, Dict, List


class ContextBuilder:
    """Build structured, citation-ready context from retrieved documents."""

    @staticmethod
    def _source_name(metadata: Dict[str, Any]) -> str:
        """Return a clean human-readable source name."""

        source_name = metadata.get("source_name")

        if source_name:
            return str(source_name)

        source = metadata.get("source", "unknown")

        if source == "unknown":
            return "unknown"

        return Path(str(source)).name

    @staticmethod
    def _page_label(metadata: Dict[str, Any]) -> str:
        """Return the original document page/location value."""

        page = metadata.get(
            "page",
            metadata.get("pages"),
        )

        if page is None:
            return "unknown"

        return str(page)

    def build_context(
        self,
        retrieved_documents: List[Dict[str, Any]],
    ) -> str:
        """
        Convert retrieved document chunks into citation-ready context.

        Each source receives a citation marker such as [1], [2],
        allowing the LLM to reference the evidence used for its answer.
        """

        if not retrieved_documents:
            return ""

        context_parts = []

        for index, doc in enumerate(
            retrieved_documents,
            start=1,
        ):
            rank = doc.get("rank", index)
            content = str(
                doc.get("content", "")
            ).strip()

            if not content:
                continue

            metadata = doc.get("metadata") or {}

            source_name = self._source_name(metadata)
            page_label = self._page_label(metadata)

            header = (
                f"[{rank}] Source: {source_name}, "
                f"Page: {page_label}"
            )

            formatted_doc = (
                f"{header}\n"
                f"Content: {content}"
            )

            context_parts.append(
                formatted_doc
            )

        return "\n\n".join(context_parts)
