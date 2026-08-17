\# Project Information



\## Project Name



RAG Document Intelligence System



\## Purpose



This project is a document-based Retrieval-Augmented Generation (RAG) system designed to retrieve relevant information from a collection of documents using semantic search.



The current implementation focuses on building and validating the retrieval layer before introducing the final LLM response-generation layer.



The system is intended to provide grounded answers by retrieving relevant document context rather than relying only on the language model's internal knowledge.



\---



\## Problem Statement



Traditional keyword-based document search can struggle when the user's query and the document use different wording.



For example, a user may ask:



"Who is Mahesh Vijay Kakade?"



while the relevant document may contain the person's name inside a certificate, acknowledgement, or internship section.



A semantic retrieval system can identify these relationships based on meaning rather than exact keyword matching.



\---



\## Project Objective



The primary objective is to build a reliable RAG foundation capable of:



\- Loading documents

\- Extracting document content

\- Splitting documents into manageable chunks

\- Generating semantic embeddings

\- Persisting embeddings in a vector database

\- Retrieving relevant document chunks

\- Preserving document metadata

\- Preventing duplicate document records

\- Providing retrieved context to a future LLM layer



\---



\## Current Scope



The current implementation covers the retrieval pipeline:



```text

Document

&#x20;   ↓

Document Loader

&#x20;   ↓

Document Chunks

&#x20;   ↓

Embedding Generation

&#x20;   ↓

ChromaDB

&#x20;   ↓

Semantic Retrieval

&#x20;   ↓

Relevant Context

