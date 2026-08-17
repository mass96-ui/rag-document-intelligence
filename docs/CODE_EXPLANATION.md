# RAG Document Intelligence - Code Explanation

## 1. Project Overview

RAG Document Intelligence is a Retrieval-Augmented Generation (RAG) document processing and retrieval pipeline.

The current implementation focuses on the retrieval layer. It processes documents, converts them into semantic vector representations, stores those vectors in ChromaDB, and retrieves the most relevant document chunks for a user query.

The current pipeline is:

Documents
-> Document Loading
-> Document Chunking
-> Embedding Generation
-> Vector Storage
-> Query Embedding
-> Semantic Retrieval
-> Relevant Document Chunks

The LLM generation layer will be added in a later stage.

---

## 2. Project Structure

src/rag_document_intelligence/
|-- __init__.py
|-- config.py
|-- loaders.py
|-- chunking.py
|-- embeddings.py
|-- vector_store.py
`-- retriever.py

tests/
`-- test_rag_pipeline.py

docs/
`-- CODE_EXPLANATION.md

---

## 3. Architecture

The pipeline is divided into independent components.

### 3.1 Configuration

File:

src/rag_document_intelligence/config.py

This module centralizes configuration used by the RAG pipeline.

It defines:

- Project root directory
- Source document directory
- Persistent ChromaDB directory
- ChromaDB collection name
- Sentence Transformer embedding model
- Chunk size
- Chunk overlap
- Default retrieval count

Current defaults:

- Embedding model: all-MiniLM-L6-v2
- Chunk size: 500 characters
- Chunk overlap: 50 characters
- Default Top-K: 5

Centralizing configuration makes the system easier to maintain and modify.

---

## 4. Document Loading

File:

src/rag_document_intelligence/loaders.py

The DocumentLoader component is responsible for loading source documents from the configured document directory.

Currently supported document types:

- TXT files
- PDF files

TXT files are loaded using TextLoader.

PDF files are loaded using PyMuPDFLoader.

The loader returns LangChain Document objects containing:

- Document content
- Source metadata
- Page metadata where available

The main methods are:

- load_text_documents()
- load_pdf_documents()
- load_all_documents()

---

## 5. Document Chunking

File:

src/rag_document_intelligence/chunking.py

Large documents cannot always be passed directly into an embedding model or retrieval system.

The DocumentChunker component splits documents into smaller overlapping chunks using RecursiveCharacterTextSplitter.

Current configuration:

- Chunk size: 500 characters
- Chunk overlap: 50 characters

The overlap helps preserve contextual information between neighboring chunks.

The component also validates its configuration and rejects invalid values such as:

- Zero or negative chunk size
- Negative chunk overlap
- Chunk overlap greater than or equal to chunk size

Main method:

- split_documents()

---

## 6. Embedding Generation

File:

src/rag_document_intelligence/embeddings.py

The EmbeddingManager converts text into numerical vector representations.

The current model is:

all-MiniLM-L6-v2

The model is provided through Sentence Transformers.

The embedding process is:

Text
-> Sentence Transformer
-> Numerical Vector

The same embedding model is used for both:

- Document chunks
- User queries

This allows semantic similarity to be calculated between a user's question and stored document chunks.

Main method:

- generate_embeddings()

---

## 7. Vector Storage

File:

src/rag_document_intelligence/vector_store.py

The VectorStore component manages persistent vector storage using ChromaDB.

It is responsible for:

- Initializing the persistent ChromaDB client
- Creating or reusing the document collection
- Storing document chunks
- Storing embeddings
- Storing document metadata
- Generating deterministic document IDs
- Preventing duplicate chunks from being inserted

The vector store is persisted locally under:

data/vector_store/

The collection name is:

pdf_documents

### Deterministic Document IDs

Each document chunk receives an ID based on:

- Document source
- Page information
- Content hash

The content hash is generated using SHA-256.

This allows the same chunk to produce the same ID across repeated ingestion operations and prevents duplicate records from being inserted.

Main methods:

- add_documents()
- count()

---

## 8. Semantic Retrieval

File:

src/rag_document_intelligence/retriever.py

The RAGRetriever component retrieves document chunks that are semantically relevant to a user's query.

The retrieval process is:

User Query
-> Query Embedding
-> ChromaDB Similarity Search
-> Top-K Results
-> Optional Distance Filtering
-> Retrieved Documents

For each retrieved result, the system returns:

- Document ID
- Document content
- Metadata
- Distance
- Retrieval rank

The retriever also validates:

- Empty queries
- Invalid Top-K values

Main method:

- retrieve()

---

## 9. Complete Data Flow

The complete current flow is:

                    DOCUMENT INGESTION

Documents
    |
    v
DocumentLoader
    |
    v
Loaded Documents
    |
    v
DocumentChunker
    |
    v
Document Chunks
    |
    v
EmbeddingManager
    |
    v
Document Embeddings
    |
    v
VectorStore
    |
    v
ChromaDB


                    QUERY RETRIEVAL

User Query
    |
    v
EmbeddingManager
    |
    v
Query Embedding
    |
    v
ChromaDB Similarity Search
    |
    v
RAGRetriever
    |
    v
Relevant Document Chunks

---

## 10. Current Project Scope

The current implementation covers the retrieval side of a RAG system:

1. Document ingestion
2. PDF and TXT document loading
3. Document chunking
4. Semantic embedding generation
5. Persistent vector storage
6. Duplicate prevention
7. Semantic similarity retrieval
8. Retrieval distance filtering
9. Automated unit testing

The LLM generation and response synthesis layer is intentionally not included in the current stage.

Future stages can add:

- LLM-based answer generation
- Retrieval-Augmented answer synthesis
- Prompt management
- Citation generation
- Reranking
- Evaluation metrics
- Conversational memory
- API integration

---

## 11. Testing

The project currently contains 6 automated tests.

The test suite covers:

- TXT document loading
- Document chunking
- Invalid chunk configuration
- Deterministic vector-store document IDs
- Empty retrieval queries
- Invalid retrieval parameters

Current validation result:

6 passed

The tests are designed to validate core behavior without requiring a full external LLM generation pipeline.

---

## 12. Dependencies

The main technologies currently used are:

- Python
- LangChain
- LangChain Core
- LangChain Community
- LangChain Text Splitters
- ChromaDB
- PyMuPDF
- Sentence Transformers
- NumPy
- Pytest

The runtime and development dependencies are maintained in:

requirements.txt

---

## 13. Current Limitations

The current implementation is intentionally focused on document retrieval.

Current limitations include:

- No LLM response generation
- No REST API layer
- No user authentication
- No conversational memory
- No production database
- No advanced retrieval reranking
- No retrieval evaluation framework
- Local ChromaDB persistence only
- LangChain Community document loaders are currently used

These limitations are expected because the current stage establishes the core RAG retrieval foundation.

---

## 14. Next Development Stage

The next stage can extend the current retrieval pipeline into a complete RAG system:

Document
-> Chunk
-> Embed
-> Store
-> Retrieve
-> Rerank
-> Prompt Construction
-> LLM
-> Generated Answer
-> Source Citations

The existing retrieval components are designed to provide the foundation for this future generation layer.