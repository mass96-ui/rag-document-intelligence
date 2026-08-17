from typing import Any, Dict, List, Optional

from .embeddings import EmbeddingManager
from .vector_store import VectorStore


class RAGRetriever:
    """Retrieve the most relevant document chunks for a user query."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_manager: EmbeddingManager,
    ):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

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

        query = query.strip()

        query_embedding = self.embedding_manager.generate_embeddings(
            [query]
        )[0]

        results = self.vector_store.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
        )

        retrieved_documents: List[Dict[str, Any]] = []

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        for rank, (doc_id, document, metadata, distance) in enumerate(
            zip(ids, documents, metadatas, distances),
            start=1,
        ):
            if (
                score_threshold is not None
                and distance > score_threshold
            ):
                continue

            retrieved_documents.append(
                {
                    "id": doc_id,
                    "content": document,
                    "metadata": metadata,
                    "distance": distance,
                    "rank": rank,
                }
            )

        return retrieved_documents
