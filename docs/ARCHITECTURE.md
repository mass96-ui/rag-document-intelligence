\# System Architecture



\## Overview



The system follows a modular Retrieval-Augmented Generation architecture.



The current implementation is centered around document ingestion, embedding generation, persistent vector storage, and semantic retrieval.



The final architecture will extend this pipeline with an LLM generation layer.



\---



\## High-Level Architecture



```text

&#x20;                       DOCUMENT INGESTION

&#x20;                              │

&#x20;                              ▼

&#x20;                   ┌────────────────────┐

&#x20;                   │  Document Loaders  │

&#x20;                   │                    │

&#x20;                   │ PDF / TXT          │

&#x20;                   └─────────┬──────────┘

&#x20;                             │

&#x20;                             ▼

&#x20;                   ┌────────────────────┐

&#x20;                   │ Document Processing │

&#x20;                   │                    │

&#x20;                   │ Extraction         │

&#x20;                   │ Chunking           │

&#x20;                   │ Metadata           │

&#x20;                   └─────────┬──────────┘

&#x20;                             │

&#x20;                             ▼

&#x20;                   ┌────────────────────┐

&#x20;                   │ Embedding Manager  │

&#x20;                   │                    │

&#x20;                   │ all-MiniLM-L6-v2   │

&#x20;                   └─────────┬──────────┘

&#x20;                             │

&#x20;                             ▼

&#x20;                   ┌────────────────────┐

&#x20;                   │     ChromaDB       │

&#x20;                   │                    │

&#x20;                   │ Persistent Vectors │

&#x20;                   └─────────┬──────────┘

&#x20;                             │

&#x20;                        USER QUERY

&#x20;                             │

&#x20;                             ▼

&#x20;                   ┌────────────────────┐

&#x20;                   │   RAG Retriever    │

&#x20;                   │                    │

&#x20;                   │ Query Embedding    │

&#x20;                   │ Similarity Search  │

&#x20;                   │ Top-K Results      │

&#x20;                   └─────────┬──────────┘

&#x20;                             │

&#x20;                             ▼

&#x20;                   ┌────────────────────┐

&#x20;                   │ Retrieved Context  │

&#x20;                   │ + Metadata         │

&#x20;                   └─────────┬──────────┘

&#x20;                             │

&#x20;                             ▼

&#x20;                   ┌────────────────────┐

&#x20;                   │    LLM Layer       │

&#x20;                   │   Planned Stage    │

&#x20;                   └─────────┬──────────┘

&#x20;                             │

&#x20;                             ▼

&#x20;                   ┌────────────────────┐

&#x20;                   │ Grounded Response  │

&#x20;                   └────────────────────┘

