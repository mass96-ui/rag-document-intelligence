"""MCP (Model Context Protocol) server for RAG Document Intelligence.

This module exposes the RAG pipeline's capabilities as MCP tools, enabling
integration with the future SoleusAI AI agent architecture.

Exposed tools:
    - search_knowledge: Search the RAG knowledge base (retrieval only, no answer generation)
    - ask_rag: Ask the grounded RAG question-answering pipeline
    - get_patient_context: Retrieve patient context through the provider abstraction
    - get_latest_ml_context: Retrieve the latest ML session result (if available)
    - validate_medical_evidence: Check if retrieved evidence meets clinical trust requirements

The MCP layer is thin and stateless. It delegates to existing RAG components
without duplicating retrieval logic, patient validation, citation validation,
or medical safety rules.
"""
