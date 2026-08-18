import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Generate semantic embeddings for document text."""

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
    ):
        self.model_name = model_name
        self.model: SentenceTransformer | None = None

        self._load_model()

    def _load_model(self) -> None:
        """Load the configured Sentence Transformer model."""
        try:
            logger.info("Loading embedding model: %s", self.model_name)

            self.model = SentenceTransformer(self.model_name)

            logger.info(
                "Embedding model loaded. Dimension: %d",
                self.model.get_embedding_dimension(),
            )

        except Exception as exc:
            logger.error(
                "Failed to load embedding model %s: %s",
                self.model_name, exc,
            )
            raise

    def generate_embeddings(
        self,
        texts: List[str],
    ) -> np.ndarray:
        """Convert a list of texts into numerical embeddings."""
        if self.model is None:
            raise RuntimeError("Embedding model is not loaded")

        if not texts:
            return np.empty((0, self.model.get_embedding_dimension()))

        valid_texts = [
            text if isinstance(text, str) else str(text)
            for text in texts
        ]

        logger.debug("Generating embeddings for %d texts", len(valid_texts))

        embeddings = self.model.encode(
            valid_texts,
            show_progress_bar=False,
        )

        embeddings = np.asarray(embeddings)

        logger.debug(
            "Generated embeddings with shape: %s", embeddings.shape
        )

        return embeddings
