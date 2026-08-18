import logging
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import CHUNK_OVERLAP, CHUNK_SIZE

logger = logging.getLogger(__name__)


class DocumentChunker:
    """Split loaded documents into smaller chunks for embedding and retrieval."""

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def split_documents(
        self,
        documents: List[Document],
    ) -> List[Document]:
        """Split documents into smaller overlapping chunks."""
        if not documents:
            return []

        chunks = self.splitter.split_documents(documents)

        chunks = [
            chunk for chunk in chunks if chunk.page_content.strip()
        ]

        logger.info(
            "Chunked %d documents into %d non-empty chunks",
            len(documents), len(chunks),
        )

        return chunks
