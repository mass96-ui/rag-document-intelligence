from pathlib import Path
from typing import Any, List

import chromadb
import numpy as np

from .config import COLLECTION_NAME, VECTOR_STORE_DIR


class VectorStore:
    """Manage persistent document embeddings in ChromaDB."""

    def __init__(
        self,
        collection_name: str = COLLECTION_NAME,
        persist_directory: Path = VECTOR_STORE_DIR,
    ):
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)

        self.client = None
        self.collection = None

        self._initialize_store()

    def _initialize_store(self) -> None:
        """Create the ChromaDB client and collection."""
        try:
            self.persist_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory)
            )

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "description": (
                        "Document embeddings for "
                        "RAG retrieval"
                    )
                },
            )

            print(
                "Vector store initialized. "
                f"Collection: {self.collection_name}"
            )

            print(
                "Existing documents in collection: "
                f"{self.collection.count()}"
            )

        except Exception as exc:
            print(
                f"Error initializing vector store: {exc}"
            )
            raise

    @staticmethod
    def _create_document_id(
        document: Any,
    ) -> str:
        """Create a stable ID from document source, page and content."""
        import hashlib

        source = str(
            document.metadata.get(
                "source",
                "unknown",
            )
        )

        page = str(
            document.metadata.get(
                "page",
                document.metadata.get(
                    "pages",
                    "unknown",
                ),
            )
        )

        content_hash = hashlib.sha256(
            document.page_content.encode("utf-8")
        ).hexdigest()[:16]

        return f"{source}|{page}|{content_hash}"

    def add_documents(
        self,
        documents: List[Any],
        embeddings: np.ndarray,
    ) -> int:
        """Add new document chunks while preventing duplicates."""
        if self.collection is None:
            raise RuntimeError(
                "Vector store collection is not initialized"
            )

        if len(documents) != len(embeddings):
            raise ValueError(
                "Number of documents must match "
                "number of embeddings"
            )

        if not documents:
            print("No documents to add.")
            return 0

        ids = []
        metadatas = []
        document_texts = []
        embedding_list = []

        for index, (document, embedding) in enumerate(
            zip(documents, embeddings)
        ):
            document_id = self._create_document_id(
                document
            )

            metadata = dict(
                document.metadata
            )

            metadata["doc_index"] = index
            metadata["content_length"] = len(
                document.page_content
            )

            ids.append(document_id)
            metadatas.append(metadata)
            document_texts.append(
                document.page_content
            )
            embedding_list.append(
                embedding.tolist()
            )

        try:
            existing = self.collection.get(
                ids=ids,
                include=[],
            )

            existing_ids = set(
                existing["ids"]
            )

            new_indexes = [
                index
                for index, document_id in enumerate(ids)
                if document_id not in existing_ids
            ]

            if not new_indexes:
                print(
                    "No new documents to add. "
                    "All chunks already exist "
                    "in ChromaDB."
                )

                print(
                    "Total documents in collection: "
                    f"{self.collection.count()}"
                )

                return 0

            self.collection.add(
                ids=[
                    ids[index]
                    for index in new_indexes
                ],
                embeddings=[
                    embedding_list[index]
                    for index in new_indexes
                ],
                metadatas=[
                    metadatas[index]
                    for index in new_indexes
                ],
                documents=[
                    document_texts[index]
                    for index in new_indexes
                ],
            )

            added_count = len(new_indexes)
            skipped_count = (
                len(ids) - added_count
            )

            print(
                f"Added {added_count} new "
                "document chunks."
            )

            print(
                f"Skipped {skipped_count} "
                "existing chunks."
            )

            print(
                "Total documents in collection: "
                f"{self.collection.count()}"
            )

            return added_count

        except Exception as exc:
            print(
                "Error adding documents to "
                f"vector store: {exc}"
            )
            raise

    def count(self) -> int:
        """Return the number of records in the collection."""
        if self.collection is None:
            raise RuntimeError(
                "Vector store collection is not initialized"
            )

        return self.collection.count()
