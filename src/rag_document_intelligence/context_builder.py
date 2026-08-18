import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ContextBuilder:
    """Build structured, citation-ready context from retrieved documents."""

    @staticmethod
    def _source_name(metadata: Dict[str, Any]) -> str:
        """Return a clean human-readable source name."""

        source_name = metadata.get("source_name")

        if source_name:
            return str(source_name)

        source = metadata.get("source")

        if not source or source == "unknown":
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

    @staticmethod
    def _format_score(score: Optional[float]) -> str:
        """Format a normalized score for display."""
        if score is None:
            return ""
        try:
            return f" (Score: {float(score):.3f})"
        except (TypeError, ValueError):
            return ""

    @staticmethod
    def _format_doc_id(doc_id: Optional[str]) -> str:
        """Format a document ID for display."""
        if not doc_id:
            return ""
        return f" (ID: {doc_id})"

    def build_context(
        self,
        retrieved_documents: List[Dict[str, Any]],
    ) -> str:
        """
        Convert retrieved document chunks into citation-ready context.

        Each source receives a citation marker such as [1], [2],
        allowing the LLM to reference the evidence used for its answer.

        The citation number is derived from the retrieval rank, ensuring
        stable, deterministic numbering that matches the context.
        """

        if not retrieved_documents:
            return ""

        context_parts: List[str] = []

        for index, doc in enumerate(
            retrieved_documents,
            start=1,
        ):
            rank = doc.get("rank", index)
            content = str(
                doc.get("content", "")
            ).strip()

            if not content:
                logger.debug(
                    "Skipping context entry with empty content "
                    "(rank=%s, id=%s)",
                    rank, doc.get("id", "unknown"),
                )
                continue

            metadata = doc.get("metadata") or {}

            source_name = self._source_name(metadata)
            page_label = self._page_label(metadata)
            score_label = self._format_score(
                doc.get("score")
            )
            doc_id_label = self._format_doc_id(
                doc.get("id")
            )

            header = (
                f"[{rank}] Source: {source_name}, "
                f"Page: {page_label}{score_label}{doc_id_label}"
            )

            formatted_doc = (
                f"{header}\n"
                f"Content: {content}"
            )

            context_parts.append(formatted_doc)

        result = "\n\n".join(context_parts)

        logger.debug(
            "Built context with %d sources (%d characters)",
            len(context_parts), len(result),
        )

        return result
