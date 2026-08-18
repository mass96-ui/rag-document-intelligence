import logging
from typing import Any, Dict, List, Optional

from .config import MAX_QUERY_LENGTH, SCORE_THRESHOLD
from .embeddings import EmbeddingManager
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class RAGRetriever:
    """Retrieve the most relevant document chunks for a user query."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_manager: EmbeddingManager,
        score_threshold: Optional[float] = None,
    ):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        self.score_threshold = (
            score_threshold
            if score_threshold is not None
            else SCORE_THRESHOLD
        )

    @staticmethod
    def _normalize_distance(distance: float) -> float:
        """Map a Chroma distance to a similarity score in [0, 1]."""
        try:
            value = float(distance)
        except (TypeError, ValueError):
            value = 0.0
        if value < 0:
            value = 0.0
        return 1.0 / (1.0 + value)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Find the most relevant document chunks for a query."""

        if not query or not query.strip():
            raise ValueError("query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        cleaned_query = query.strip()

        if len(cleaned_query) > MAX_QUERY_LENGTH:
            raise ValueError(
                f"Query exceeds maximum length of {MAX_QUERY_LENGTH} "
                f"characters."
            )

        effective_threshold = (
            score_threshold
            if score_threshold is not None
            else self.score_threshold
        )

        query_embedding = self.embedding_manager.generate_embeddings(
            [cleaned_query]
        )[0]

        if self.vector_store is None or self.vector_store.collection is None:
            raise RuntimeError(
                "Vector store collection is not initialized"
            )

        results = self.vector_store.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
        )

        retrieved_documents = self._parse_chroma_results(results)

        if effective_threshold is not None:
            retrieved_documents = [
                doc
                for doc in retrieved_documents
                if doc["distance"] <= effective_threshold
            ]

        seen_ids: set[str] = set()
        deduplicated: List[Dict[str, Any]] = []
        for doc in retrieved_documents:
            if doc["id"] in seen_ids:
                logger.debug(
                    "Suppressing duplicate retrieval for id=%s",
                    doc["id"],
                )
                continue
            seen_ids.add(doc["id"])
            deduplicated.append(doc)

        deduplicated.sort(key=lambda d: (d["distance"], d["id"]))

        for rank, doc in enumerate(deduplicated, start=1):
            doc["rank"] = rank

        logger.info(
            "Retrieved %d documents for query (threshold=%s)",
            len(deduplicated), effective_threshold,
        )

        return deduplicated

    @staticmethod
    def _parse_chroma_results(
        results: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Safely parse ChromaDB query results into a list of dicts.

        Handles missing keys, mismatched list lengths, and missing
        per-document fields without raising.
        """
        documents = results.get("documents") or [[]]
        metadatas = results.get("metadatas") or [[]]
        distances = results.get("distances") or [[]]
        ids = results.get("ids") or [[]]

        doc_list = documents[0] if documents else []
        meta_list = metadatas[0] if metadatas else []
        dist_list = distances[0] if distances else []
        id_list = ids[0] if ids else []

        count = min(
            len(doc_list), len(meta_list), len(dist_list), len(id_list)
        )

        if count == 0:
            return []

        if not (
            len(doc_list) == len(meta_list) == len(dist_list) == len(id_list)
        ):
            logger.warning(
                "ChromaDB returned mismatched result lists: "
                "ids=%d, documents=%d, metadatas=%d, distances=%d. "
                "Processing only the first %d entries.",
                len(id_list), len(doc_list), len(meta_list),
                len(dist_list), count,
            )

        parsed: List[Dict[str, Any]] = []
        for index in range(count):
            doc_id = id_list[index]
            content = doc_list[index]
            metadata = meta_list[index]
            distance = dist_list[index]

            if metadata is None:
                metadata = {}

            try:
                distance_val = float(distance)
            except (TypeError, ValueError):
                distance_val = float("inf")

            parsed.append(
                {
                    "id": doc_id,
                    "content": content if content is not None else "",
                    "metadata": metadata,
                    "distance": distance_val,
                    "score": RAGRetriever._normalize_distance(
                        distance_val
                    ),
                }
            )

        return parsed
