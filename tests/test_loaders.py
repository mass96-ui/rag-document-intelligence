from pathlib import Path

import pytest

from rag_document_intelligence.loaders import DocumentLoader


def test_supported_extensions():
    loader = DocumentLoader()

    assert loader.supported_extensions() == [
        ".docx",
        ".md",
        ".pdf",
        ".txt",
    ]


def test_direct_text_input():
    loader = DocumentLoader()

    documents = loader.load_text_content(
        "Diabetes is a chronic condition.",
        source_name="user_question",
    )

    assert len(documents) == 1
    assert (
        documents[0].page_content
        == "Diabetes is a chronic condition."
    )
    assert documents[0].metadata["input_type"] == "text"
    assert (
        documents[0].metadata["source_name"]
        == "user_question"
    )


def test_empty_direct_text_rejected():
    loader = DocumentLoader()

    with pytest.raises(ValueError):
        loader.load_text_content("")


def test_unsupported_file_rejected(tmp_path: Path):
    file_path = tmp_path / "example.csv"
    file_path.write_text(
        "name,value",
        encoding="utf-8",
    )

    loader = DocumentLoader()

    with pytest.raises(ValueError):
        loader.load_file(file_path)


def test_text_file_loading(tmp_path: Path):
    file_path = tmp_path / "example.txt"

    file_path.write_text(
        "This is a test document.",
        encoding="utf-8",
    )

    loader = DocumentLoader()

    documents = loader.load_file(file_path)

    assert len(documents) == 1
    assert (
        documents[0].page_content
        == "This is a test document."
    )
    assert documents[0].metadata["input_type"] == "txt"


def test_markdown_file_loading(tmp_path: Path):
    file_path = tmp_path / "example.md"

    file_path.write_text(
        "# Diabetes\n\nDiabetes is a chronic condition.",
        encoding="utf-8",
    )

    loader = DocumentLoader()

    documents = loader.load_file(file_path)

    assert len(documents) == 1
    assert "Diabetes" in documents[0].page_content
    assert documents[0].metadata["input_type"] == "md"


def test_empty_file_rejected(tmp_path: Path):
    file_path = tmp_path / "empty.txt"
    file_path.write_text("", encoding="utf-8")

    loader = DocumentLoader()

    with pytest.raises(ValueError, match="empty"):
        loader.load_file(file_path)


def test_missing_file_rejected():
    loader = DocumentLoader()

    with pytest.raises(FileNotFoundError):
        loader.load_file("/nonexistent/path/file.txt")


def test_unsupported_extension_rejected(tmp_path: Path):
    file_path = tmp_path / "data.csv"
    file_path.write_text("a,b,c", encoding="utf-8")

    loader = DocumentLoader()

    with pytest.raises(ValueError, match="Unsupported"):
        loader.load_file(file_path)


def test_path_traversal_blocked(tmp_path: Path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    loader = DocumentLoader(documents_dir=docs_dir)

    with pytest.raises(ValueError, match="traversal"):
        loader.load_file("../../etc/passwd.txt")


def test_loader_sets_source_name(tmp_path: Path):
    file_path = tmp_path / "mydoc.txt"
    file_path.write_text("Some content here.", encoding="utf-8")

    loader = DocumentLoader()
    documents = loader.load_file(file_path)

    assert documents[0].metadata["source_name"] == "mydoc.txt"
    assert documents[0].metadata["input_type"] == "txt"


def test_load_all_documents_empty_dir(tmp_path: Path):
    loader = DocumentLoader(documents_dir=tmp_path)
    documents = loader.load_all_documents()
    assert documents == []


def test_load_all_documents_nonexistent_dir(tmp_path: Path):
    loader = DocumentLoader(documents_dir=tmp_path / "nonexistent")
    documents = loader.load_all_documents()
    assert documents == []
