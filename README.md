# RAG Document Intelligence

A modular, production-hardened Retrieval-Augmented Generation (RAG) system for
document ingestion, semantic retrieval, citation-aware context construction,
and grounded answer generation using a **local Ollama LLM**.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Supported Documents](#supported-documents)
4. [Installation](#installation)
5. [Ollama Setup](#ollama-setup)
6. [Configuration (`.env`)](#configuration-env)
7. [Document Ingestion](#document-ingestion)
8. [Running the CLI](#running-the-cli)
9. [Running Tests](#running-tests)
10. [Troubleshooting](#troubleshooting)
11. [Design Decisions](#design-decisions)

---

## Project Overview

RAG Document Intelligence lets you ask questions about your own documents
(PDF, TXT, Markdown, DOCX) and receive **grounded answers with citations**
— not hallucinated responses from a generic language model.

The pipeline works in five stages:

1. **Load** documents into normalized `Document` objects.
2. **Chunk** text into overlapping windows for fine-grained retrieval.
3. **Embed** chunks with `all-MiniLM-L6-v2` sentence embeddings.
4. **Store** embeddings persistently in ChromaDB with stable deduplication IDs.
5. **Retrieve** the most relevant chunks for a query, build a citation-
   numbered context, and generate a grounded answer via Ollama.

Key guarantees:

- Answers are generated **only** from retrieved context.
- Every cited source number `[n]` corresponds to a retrieved chunk.
- Insufficient evidence produces a clear refusal, not a hallucination.
- Duplicate ingestion is detected and skipped via content-hash IDs.
- The LLM layer is abstracted; tests use a `MockLLMProvider`, production
  uses the Ollama provider.

---

## Architecture

```
                        DOCUMENT INGESTION
                             |
                             v
                     +----------------+
                     |   Documents    |   (PDF / TXT / MD / DOCX)
                     +-------+--------+
                             |
                             v
                     +----------------+
                     | DocumentLoader |   (langchain loaders)
                     +-------+--------+
                             |
                             v
                     +----------------+
                     |   Chunking     |   (RecursiveCharacterTextSplitter)
                     +-------+--------+
                             |
                             v
                     +----------------+
                     | Embedding Mgr  |   (SentenceTransformer, all-MiniLM-L6-v2)
                     +-------+--------+
                             |
                             v
                     +----------------+
                     |    ChromaDB    |   (PersistentClient, stable IDs)
                     +-------+--------+
                             |
                             |
                          RETRIEVAL
                             |
                             v
                     +----------------+
                     |  RAGRetriever  |   (similarity search, top-k, dedup)
                     +-------+--------+
                             |
                             v
                     +----------------+
                     | ContextBuilder |   (citation-numbered context)
                     +-------+--------+
                             |
                             v
                     +----------------+
                     |  LLMProvider   |   (Ollama / Mock)
                     +-------+--------+
                             |
                             v
                     +----------------+
                     | Grounded Answer|   (+ source citations)
                     +----------------+
```

### Module Map

| Module             | Responsibility                                      |
|--------------------|-----------------------------------------------------|
| `config.py`        | Environment loading, defaults, validation           |
| `loaders.py`       | PDF / TXT / MD / DOCX loading, path safety          |
| `chunking.py`      | Overlapping text splitting                          |
| `embeddings.py`    | SentenceTransformer embedding generation            |
| `vector_store.py`  | ChromaDB persistence, stable IDs, dedup             |
| `retriever.py`      | Similarity search, score normalization, dedup       |
| `context_builder.py` | Citation-numbered context from retrieved chunks   |
| `llm.py`           | Provider abstraction (Mock + Ollama)                |
| `pipeline.py`      | End-to-end orchestration                             |
| `cli.py`           | Interactive command-line interface                   |
| `ingest.py`        | Ingestion entry points                               |
| `evaluation.py`    | Retrieval recall, groundedness, citation validation   |

---

## Supported Documents

| Format  | Loader          | Metadata preserved                    |
|---------|-----------------|---------------------------------------|
| PDF     | PyMuPDF         | `source_name`, `input_type`, `page`   |
| TXT     | TextLoader      | `source_name`, `input_type`           |
| MD      | TextLoader      | `source_name`, `input_type`           |
| DOCX    | python-docx     | `source_name`, `input_type`           |

Empty files, unsupported extensions, and missing files are rejected with
clear error messages.

---

## Installation

### Prerequisites

- Python 3.10+
- Ollama (see [Ollama Setup](#ollama-setup))

### Steps

```powershell
# Clone the repository
git clone <repo-url>
cd rag_project

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\Activate.ps1    # PowerShell
# or
venv\Scripts\activate        # Command Prompt

# Install the project in editable mode
pip install -e ".[dev]"
```

### Python Requirements

Runtime dependencies (declared in `pyproject.toml`):

| Package                | Purpose                          |
|------------------------|----------------------------------|
| `python-dotenv`        | `.env` loading                   |
| `requests`             | Ollama HTTP API                  |
| `python-docx`          | DOCX document loading            |
| `numpy`                | Embedding array operations       |
| `chromadb`             | Vector database                  |
| `sentence-transformers`| Embedding model                |
| `pypdf`                | PDF utilities                    |
| `pymupdf`              | PDF page extraction              |
| `langchain`            | Document abstractions            |
| `langchain-community`  | Loaders                          |
| `langchain-text-splitters` | Text chunking                |
| `langchain-chroma`     | ChromaDB integration             |

Development dependencies (`pip install -e ".[dev]"`):

- `pytest`, `jupyter`, `ipykernel`

---

## Ollama Setup

Ollama serves the local LLM that generates grounded answers.

### 1. Verify Ollama is installed

```powershell
ollama --version
```

> **If Ollama is already running** (e.g. started automatically or via a
> system service), **do not** start a second `ollama serve` process.
> Just verify the API endpoint is reachable:
>
> ```powershell
> curl http://localhost:11434/api/tags
> ```

### 2. Pull the model

```powershell
ollama pull llama3.2
```

### 3. Verify the model is available

```powershell
ollama list
```

You should see `llama3.2` in the output. The Ollama API runs by default at:

```
http://localhost:11434
```

---

## Configuration (`.env`)

All configuration is via environment variables. Copy the example and
customize:

```powershell
cp .env.example .env
```

| Variable             | Default                | Description                                  |
|----------------------|------------------------|----------------------------------------------|
| `LLM_PROVIDER`       | `ollama`               | `mock` or `ollama`                           |
| `OLLAMA_BASE_URL`    | `http://localhost:11434`| Ollama API endpoint                          |
| `OLLAMA_MODEL`       | `llama3.2`             | Model name to use                            |
| `OLLAMA_TIMEOUT`     | `120`                  | Request timeout in seconds                   |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2`   | SentenceTransformers model                   |
| `COLLECTION_NAME`    | `pdf_documents`        | ChromaDB collection name                     |
| `CHUNK_SIZE`         | `500`                  | Characters per chunk                         |
| `CHUNK_OVERLAP`      | `50`                   | Overlap between consecutive chunks           |
| `DEFAULT_TOP_K`      | `5`                    | Number of chunks to retrieve                 |
| `SCORE_THRESHOLD`    | `1.0`                  | Max L2 distance to retain (lower = stricter) |
| `MAX_QUERY_LENGTH`   | `2000`                 | Maximum characters in a user query           |

`.env` is in `.gitignore` and is never committed.

Configuration is validated at startup via `validate_config()` — invalid
values cause a clear error before any processing begins.

---

## Document Ingestion

Documents go in `data/documents/`. Supported formats: PDF, TXT, MD, DOCX.

### Ingest all documents from the default directory

```python
from rag_document_intelligence.ingest import run_ingestion

run_ingestion()
```

### Ingest a single file

```python
from rag_document_intelligence.ingest import ingest_file

ingest_file("data/documents/my_report.pdf")
```

### Ingest raw text

```python
from rag_document_intelligence.ingest import ingest_text

ingest_text(
    "Python is a versatile programming language.",
    source_name="quick_note",
)
```

Duplicate chunks (same source, page, and content hash) are automatically
skipped on re-ingestion.

---

## Running the CLI

### As a module

```powershell
python -m rag_document_intelligence.cli
```

### Via the console entry point (after `pip install -e .`)

```powershell
rag
```

### CLI commands

| Input         | Action                                     |
|---------------|---------------------------------------------|
| Any question  | Performs RAG retrieval + Ollama generation  |
| `exit`        | Quits cleanly (never sent to the LLM)       |
| `quit`        | Quits cleanly                               |
| `Ctrl+C`      | Graceful exit                               |
| `Ctrl+D` / EOF| Graceful exit                               |
| Empty / space | Prompts again (no LLM call)                 |

The CLI displays the answer, all retrieved sources with page and score,
and the context length.

---

## Running Tests

```powershell
python -m pytest -q
```

Tests cover retrieval, scoring, duplicate suppression, malformed results,
context building, Ollama provider success/failure paths, document loaders,
CLI behavior, and configuration validation.

---

## Troubleshooting

### Ollama is not running

```
RuntimeError: Could not connect to Ollama at http://localhost:11434...
```

Start Ollama and ensure it is listening:

```powershell
ollama serve      # only if not already running
ollama list       # verify llama3.2 is present
```

### Model is missing

```
RuntimeError: Ollama rejected the request (HTTP 400). Model 'llama3.2' may not be loaded.
```

Pull the model:

```powershell
ollama pull llama3.2
```

### Embedding model download fails

The first run downloads `all-MiniLM-L6-v2` from Hugging Face. Ensure you
have internet access on the first run. No API key is required for this
local model.

### ChromaDB errors

The vector store persists at `data/vector_store/`. If it becomes corrupt,
remove the directory and re-ingest:

```powershell
Remove-Item -Recurse -Force data/vector_store
```

### `.env` not being read

Ensure `.env` exists in the project root (not committed to git):

```powershell
cp .env.example .env
```

### Tests fail with import errors

Reinstall the package in editable mode:

```powershell
pip install -e ".[dev]"
```

---

## Design Decisions

### Provider abstraction

`LLMProvider` is an abstract base class. `MockLLMProvider` enables
hermetic testing. `OllamaLLMProvider` handles the local LLM over HTTP.
Adding a new provider requires only implementing `generate(query, context)
-> str`.

### Stable document IDs

Each chunk is stored with an ID of the form
`<source_name>|<page>|<sha256(content_hash)>[:16]`. This ensures:
- Re-ingestion skips existing chunks (no duplicates).
- IDs are deterministic and machine-independent (uses `source_name`,
  not the full filesystem path).

### Score threshold semantics

ChromaDB returns L2 distances (lower = more similar). `SCORE_THRESHOLD`
sets the maximum distance to retain. A distance of 1.0 approximately
corresponds to cosine similarity ≈ 0.5 for `all-MiniLM-L6-v2` embeddings.
A normalized similarity score `1 / (1 + distance)` is also reported in
`[0, 1]`.

### Grounded prompt

The system prompt is emitted **before** the retrieved context, with
explicit instructions that the context is evidence, not instructions.
This prevents untrusted document text from overriding the grounding rules.

### Citation enforcement

After the LLM generates an answer, the pipeline validates citations using
`RAGEvaluator.evaluate_citation_enforcement()`:

1. **Empty answer** — detected and rejected.
2. **Refusal phrases** — accepted without citations (e.g., "I could not find
   this information in the provided documents.").
3. **Fabricated citations** (e.g., `[99]` when only `[1]`–`[3]` exist) —
   detected and rejected.
4. **Missing citations** on a non-refusal answer — detected and rejected.

When the initial answer fails validation, the pipeline makes **at most one**
regeneration attempt. The regeneration prompt explicitly lists the available
citation numbers and instructs the model to cite only those numbers.

If the regenerated answer is still invalid, the pipeline returns a safe
refusal: "I could not find this information in the provided documents."

This ensures the pipeline **never silently returns** an answer with
fabricated, invalid, or missing citations.

### Structured Ollama generation

When the provider is `ollama`, the pipeline prefers a **structured
generation** path that leverages Ollama's native `format: "json"`
support:

1. The model is prompted to return **only valid JSON** with two fields:
   `answer` (a string) and `citations` (a list of positive integers).
2. The request is sent to `POST /api/generate` with
   `"format": "json"` and `"options": {"temperature": 0.1}`.
3. The raw response is parsed and structurally validated by
   `OllamaLLMProvider._parse_structured_response()`:
   - Malformed JSON, missing/empty `answer`, `citations` not a list,
     non-integer or non-positive citation values all raise
     `RuntimeError`.
   - Markdown code fences (```` ```json ... ``` ````) are stripped
     before parsing as a tolerance fallback.
4. `RAGEvaluator.validate_structured_response()` performs
   **application-side** validation — it does **not** trust the model:
   - Every citation must map to an available retrieval rank.
   - A factual answer with zero citations is invalid.
   - A refusal answer with zero citations is valid.
   - Duplicates are deduplicated and sorted deterministically.
5. `RAGEvaluator.normalize_citations()` appends `[N]` markers to the
   final answer text, deduplicating against any existing markers.

If structured generation fails (HTTP error, timeout, malformed output,
or invalid citations), the pipeline **falls back** to text-based
generation via `generate()`. The existing text citation enforcement
and at-most-one-regeneration logic still applies to the fallback path.

`MockLLMProvider.generate_structured()` returns
`{"answer": <text>, "citations": []}` so tests exercise the text
fallback path. Providers that only implement `generate()` (not
`generate_structured()`) work via the non-abstract default on
`LLMProvider`, which delegates to `generate()` and attempts a lenient
JSON parse.

### No credentials

No API keys or secrets are stored. All LLM inference runs locally via
Ollama. The embedding model is downloaded directly from Hugging Face
(no token required for `all-MiniLM-L6-v2`).
