# RAG Document Intelligence - Code Explanation

## 1. Project Overview

RAG Document Intelligence is a modular Retrieval-Augmented Generation (RAG) pipeline for document processing, retrieval, and answer generation.

The generation layer is provider-independent through the LLMProvider abstraction, allowing different LLM implementations to be integrated without changing the core retrieval and orchestration pipeline.

The current architecture is:

Document Loading
→ Chunking
→ Embeddings
→ ChromaDB Vector Storage
→ Semantic Retrieval
→ Context Building
→ LLM Provider
→ Answer

---

## 2. Project Structure

src/rag_document_intelligence/
|-- __init__.py
|-- config.py
|-- loaders.py
|-- chunking.py
|-- embeddings.py
|-- vector_store.py
|-- retriever.py
|-- context_builder.py
|-- llm.py
`-- pipeline.py

tests/
|-- test_rag_pipeline.py
`-- test_llm_pipeline.py

docs/
`-- CODE_EXPLANATION.md

---

## 3. Architecture

The pipeline is divided into independent, modular components.

### 3.1 Configuration

File: `src/rag_document_intelligence/config.py`

Centralizes configuration for the pipeline, including directories, model names, chunking parameters, and retrieval settings.

---

## 4. Document Loading

File: `src/rag_document_intelligence/loaders.py`

Handles loading of PDF and TXT documents using LangChain loaders.

---

## 5. Document Chunking

File: `src/rag_document_intelligence/chunking.py`

Splits documents into smaller overlapping chunks using `RecursiveCharacterTextSplitter`.

---

## 6. Embedding Generation

File: `src/rag_document_intelligence/embeddings.py`

Generates semantic embeddings using Sentence Transformers (default: `all-MiniLM-L6-v2`).

---

## 7. Vector Storage

File: `src/rag_document_intelligence/vector_store.py`

Manages persistent storage and duplicate prevention using ChromaDB.

---

## 8. Semantic Retrieval

File: `src/rag_document_intelligence/retriever.py`

Performs similarity searches in ChromaDB to find the most relevant document chunks for a query.

---

## 9. Context Building

File: `src/rag_document_intelligence/context_builder.py`

Transforms retrieved document chunks into a single formatted string for the LLM.

It preserves:
- Retrieval rank
- Source document name
- Page numbers
- Document content

This ensures the LLM has the necessary context and metadata to generate grounded answers with potential citations.

---

## 10. LLM Abstraction

File: `src/rag_document_intelligence/llm.py`

Provides a provider-independent interface for Large Language Models.

- **LLMProvider (ABC)**: Abstract base class defining the `generate(query, context)` interface.
- **MockLLMProvider**: A mock implementation used for testing and validation without requiring external APIs or local models like Ollama.

This design allows for easy integration of various providers (Ollama, OpenAI, Gemini, etc.) in future stages.

---

## 11. RAG Pipeline Orchestration

File: `src/rag_document_intelligence/pipeline.py`

The `RAGPipeline` class orchestrates the complete end-to-end flow:
1. **Retrieve**: Get relevant chunks via `RAGRetriever`.
2. **Build Context**: Format chunks via `ContextBuilder`.
3. **Generate**: Get the final answer via an `LLMProvider`.

---

## 12. Complete Data Flow

```text
                    DOCUMENT INGESTION
Documents -> Loader -> Chunker -> EmbeddingManager -> VectorStore -> ChromaDB

                    QUERY & RESPONSE
User Query -> EmbeddingManager -> ChromaDB -> RAGRetriever -> ContextBuilder -> LLMProvider -> Answer
```

---

## 13. Current Project Scope

The current implementation provides a complete, modular RAG foundation:
1. Full ingestion pipeline (PDF/TXT).
2. Persistent semantic search.
3. Modular context construction.
4. Provider-independent LLM interface.
5. Automated testing with mock providers.

Ollama is NOT required for the current implementation, but the system is designed to support it as a provider in the next phase.

---

## 14. Testing

The project contains two test suites:
- `tests/test_rag_pipeline.py`: Validates the retrieval foundation.
- `tests/test_llm_pipeline.py`: Validates context building, LLM abstraction, and pipeline orchestration.

Total Tests: 10 passed.

---

## 15. Next Development Stage

The next stage will involve:
1. Manual installation of Ollama.
2. Implementation of an `OllamaProvider`.
3. End-to-end testing with real local models (e.g., Qwen2.5-Coder).
4. Refinement of prompt templates.
