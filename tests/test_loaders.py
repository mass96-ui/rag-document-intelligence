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
