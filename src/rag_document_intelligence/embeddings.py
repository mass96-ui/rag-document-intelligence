from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import EMBEDDING_MODEL_NAME


class EmbeddingManager:
    """Generate semantic embeddings for document text."""

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
    ):
        self.model_name = model_name
        self.model = None

        self._load_model()

    def _load_model(self) -> None:
        """Load the configured Sentence Transformer model."""
        try:
            print(f"Loading embedding model: {self.model_name}")

            self.model = SentenceTransformer(self.model_name)

            print(
                "Model loaded successfully. "
                f"Embedding dimension: "
                f"{self.model.get_embedding_dimension()}"
            )

        except Exception as exc:
            print(
                f"Error loading embedding model "
                f"{self.model_name}: {exc}"
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

        print(f"Generating embeddings for {len(texts)} texts...")

        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
        )

        embeddings = np.asarray(embeddings)

        print(
            f"Generated embeddings with shape: "
            f"{embeddings.shape}"
        )

        return embeddings
