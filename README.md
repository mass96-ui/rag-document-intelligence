\# RAG Project



A document-based Retrieval-Augmented Generation (RAG) system for loading documents, generating semantic embeddings, storing those embeddings in ChromaDB, and retrieving relevant information for user queries.



The project is currently focused on building and validating a reliable document ingestion and retrieval pipeline before adding the final LLM response-generation and application layers.



\---



\## Project Objective



The objective of this project is to build a practical RAG pipeline that can:



1\. Load documents such as PDF and text files.

2\. Extract document content.

3\. Split documents into smaller chunks.

4\. Generate semantic embeddings.

5\. Store embeddings persistently in ChromaDB.

6\. Retrieve relevant document chunks for user queries.

7\. Preserve document metadata such as source and page.

8\. Use retrieved context as the foundation for grounded LLM responses.



The system is being developed incrementally so that each stage can be implemented and validated independently.



\---



\## Current Implementation



The current implementation includes:



\- PDF document loading

\- Text document loading

\- Document chunking

\- Sentence Transformers embeddings

\- `all-MiniLM-L6-v2` embedding model

\- 384-dimensional embeddings

\- Persistent ChromaDB vector storage

\- Stable document IDs

\- Duplicate document prevention

\- Source and page metadata preservation

\- Top-K semantic retrieval

\- RAG retriever implementation

\- Notebook-based development and validation



The current validation corpus includes an internship report PDF and sample text documents.



The current ingestion pipeline produces 61 document chunks.



\---



\## Technology Stack



| Component | Technology |

|---|---|

| Programming Language | Python |

| Document Processing | LangChain Community |

| PDF Processing | PyMuPDF |

| Embeddings | Sentence Transformers |

| Embedding Model | all-MiniLM-L6-v2 |

| Vector Database | ChromaDB |

| Development Environment | Jupyter Notebook |

| Environment Management | Python virtual environment |



\---



\## Architecture



```text

Documents

&#x20;   |

&#x20;   v

Document Loaders

&#x20;   |

&#x20;   v

Document Chunking

&#x20;   |

&#x20;   v

Sentence Transformer

&#x20;   |

&#x20;   v

Semantic Embeddings

&#x20;   |

&#x20;   v

ChromaDB

&#x20;   |

&#x20;   v

RAG Retriever

&#x20;   |

&#x20;   v

Relevant Context

&#x20;   |

&#x20;   v

LLM Response Generation

&#x20;   |

&#x20;   v

Grounded Answer

