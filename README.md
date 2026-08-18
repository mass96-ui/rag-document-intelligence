# RAG Document Intelligence

A modular Retrieval-Augmented Generation (RAG) system for document ingestion, semantic retrieval, context construction, and LLM-based answer generation.

This project demonstrates how the major components of a RAG system can be separated into independent, testable modules.

---

## Project Objective

The goal of this project is to build a practical RAG pipeline that can:

1. Load PDF and text documents.
2. Extract document content and metadata.
3. Split documents into smaller overlapping chunks.
4. Generate semantic embeddings.
5. Store embeddings persistently in ChromaDB.
6. Retrieve relevant document chunks for a user query.
7. Build structured context from retrieved documents.
8. Generate an answer through a provider-independent LLM interface.
9. Return the generated answer together with the retrieved source documents.

The system is designed incrementally so that each stage can be developed, tested, and replaced independently.

---

## Current Implementation

The current implementation includes:

- PDF document loading
- Text document loading
- Document chunking
- Sentence Transformer embeddings
- `all-MiniLM-L6-v2` embedding model
- 384-dimensional embeddings
- Persistent ChromaDB vector storage
- Deterministic document IDs
- Duplicate chunk prevention
- Source and page metadata preservation
- Top-K semantic retrieval
- RAG retriever
- Context builder
- Provider-independent LLM abstraction
- Mock LLM provider for testing
- End-to-end RAG pipeline orchestration
- Automated unit tests
- Python package configuration using `pyproject.toml`

The current implementation does **not require Ollama or any external LLM service** because the pipeline can be tested using the included `MockLLMProvider`.

A real LLM provider can be integrated later without changing the core retrieval and orchestration components.

---

## Architecture

```text
                         INGESTION
                            |
                            v
                    +----------------+
                    |    Documents   |
                    +-------+--------+
                            |
                            v
                    +----------------+
                    | DocumentLoader |
                    +-------+--------+
                            |
                            v
                    +----------------+
                    |    Chunking    |
                    +-------+--------+
                            |
                            v
                    +----------------+
                    |   Embeddings   |
                    +-------+--------+
                            |
                            v
                    +----------------+
                    |    ChromaDB    |
                    +-------+--------+
                            |
                            |
                         RETRIEVAL
                            |
                            v
                    +----------------+
                    |  RAGRetriever  |
                    +-------+--------+
                            |
                            v
                    +----------------+
                    | ContextBuilder |
                    +-------+--------+
                            |
                            v
                    +----------------+
                    |  LLMProvider   |
                    +-------+--------+
                            |
                            v
                    +----------------+
                    | Grounded Answer|
                    +----------------+