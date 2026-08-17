from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyMuPDFLoader,
    TextLoader,
)

from .config import DOCUMENTS_DIR


class DocumentLoader:
    """Load PDF and text documents from the project's document directory."""

    def __init__(self, documents_dir: Path = DOCUMENTS_DIR):
        self.documents_dir = Path(documents_dir)

    def load_text_documents(self) -> List[Document]:
        """Load all .txt files from the document directory."""
        loader = DirectoryLoader(
            str(self.documents_dir),
            glob="*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=True,
        )

        return loader.load()

    def load_pdf_documents(self) -> List[Document]:
        """Load all PDF files from the document directory."""
        loader = DirectoryLoader(
            str(self.documents_dir),
            glob="**/*.pdf",
            loader_cls=PyMuPDFLoader,
            show_progress=False,
        )

        return loader.load()

    def load_all_documents(self) -> List[Document]:
        """Load both PDF and text documents."""
        documents = []

        documents.extend(self.load_text_documents())
        documents.extend(self.load_pdf_documents())

        return documents
