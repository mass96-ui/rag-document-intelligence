"""Tests for the interactive CLI behaviour."""

from unittest.mock import MagicMock

import pytest

from rag_document_intelligence.cli import run_app


def _patch_pipeline(monkeypatch, answer_result=None, answer_side_effect=None):
    """Patch create_pipeline to return a pipeline with a configurable answer."""
    pipeline = MagicMock()
    if answer_side_effect is not None:
        pipeline.answer.side_effect = answer_side_effect
    elif answer_result is not None:
        pipeline.answer.return_value = answer_result
    else:
        pipeline.answer.return_value = {
            "query": "test",
            "answer": "Mocked answer",
            "source_documents": [],
            "context_length": 0,
        }
    monkeypatch.setattr(
        "rag_document_intelligence.cli.create_pipeline",
        lambda: pipeline,
    )


def test_cli_exit_quits(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda _: "exit")
    _patch_pipeline(monkeypatch)
    run_app()
    captured = capsys.readouterr()
    assert "Goodbye" in captured.out


def test_cli_quit_quits(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda _: "quit")
    _patch_pipeline(monkeypatch)
    run_app()
    captured = capsys.readouterr()
    assert "Goodbye" in captured.out


def test_cli_empty_input_then_exit(monkeypatch, capsys):
    inputs = iter(["", "exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    _patch_pipeline(monkeypatch)
    run_app()
    captured = capsys.readouterr()
    assert "Please enter a question." in captured.out
    assert "Goodbye" in captured.out


def test_cli_whitespace_input_then_exit(monkeypatch, capsys):
    inputs = iter(["   ", "exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    _patch_pipeline(monkeypatch)
    run_app()
    captured = capsys.readouterr()
    assert "Please enter a question." in captured.out


def test_cli_eof_handled(monkeypatch, capsys):
    def _raise_eof(_prompt):
        raise EOFError()

    monkeypatch.setattr("builtins.input", _raise_eof)
    _patch_pipeline(monkeypatch)
    run_app()
    captured = capsys.readouterr()
    assert "Goodbye" in captured.out


def test_cli_keyboard_interrupt_handled(monkeypatch, capsys):
    def _raise_kbi(_prompt):
        raise KeyboardInterrupt()

    monkeypatch.setattr("builtins.input", _raise_kbi)
    _patch_pipeline(monkeypatch)
    run_app()
    captured = capsys.readouterr()
    assert "Goodbye" in captured.out


def test_cli_long_query_rejected(monkeypatch, capsys):
    long_query = "x" * 3000
    inputs = iter([long_query, "exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    _patch_pipeline(monkeypatch)
    run_app()
    captured = capsys.readouterr()
    assert "maximum length" in captured.out.lower()


def test_cli_generation_error_shown_gracefully(monkeypatch, capsys):
    error_result = {
        "query": "test",
        "answer": "Error during answer generation: Ollama is down",
        "source_documents": [],
        "context_length": 0,
    }
    inputs = iter(["test question", "exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    _patch_pipeline(monkeypatch, answer_result=error_result)
    run_app()
    captured = capsys.readouterr()
    assert "Error during answer generation" in captured.out


def test_cli_init_failure_exits(monkeypatch, capsys):
    def _fail():
        raise RuntimeError("Ollama is not running")

    monkeypatch.setattr(
        "rag_document_intelligence.cli.create_pipeline", _fail
    )
    with pytest.raises(SystemExit) as exc_info:
        run_app()
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "fatal" in captured.out.lower()
