from typing import Any, Dict

from .context_builder import ContextBuilder
from .llm import LLMProvider
from .retriever import RAGRetriever


class RAGPipeline:
    """Orchestrate retrieval, context construction, and answer generation."""

    def __init__(
        self,
        retriever: RAGRetriever,
        context_builder: ContextBuilder,
        llm_provider: LLMProvider,
    ):
        self.retriever = retriever
        self.context_builder = context_builder
        self.llm_provider = llm_provider

    def answer(
        self,
        query: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Execute the complete RAG pipeline.

        Flow:
            User query
                -> document retrieval
                -> context construction
                -> LLM generation
                -> grounded answer
        """

        cleaned_query = query.strip() if query else ""

        # 1. Validate input
        if not cleaned_query:
            return {
                "query": query,
                "answer": "Error: Question cannot be empty.",
                "source_documents": [],
                "context_length": 0,
            }

        if top_k <= 0:
            return {
                "query": cleaned_query,
                "answer": (
                    f"Error: Invalid top_k ({top_k}). "
                    "Must be greater than 0."
                ),
                "source_documents": [],
                "context_length": 0,
            }

        # 2. Retrieve relevant document chunks
        try:
            retrieved_docs = self.retriever.retrieve(
                query=cleaned_query,
                top_k=top_k,
            )
        except Exception as exc:
            return {
                "query": cleaned_query,
                "answer": f"Error during retrieval: {exc}",
                "source_documents": [],
                "context_length": 0,
            }

        # 3. Handle no retrieval results
        if not retrieved_docs:
            return {
                "query": cleaned_query,
                "answer": (
                    "I could not find relevant information in the "
                    "provided documents to answer your question."
                ),
                "source_documents": [],
                "context_length": 0,
            }

        # 4. Build structured context
        context = self.context_builder.build_context(
            retrieved_docs
        )

        if not context:
            return {
                "query": cleaned_query,
                "answer": (
                    "Relevant documents were retrieved, but no usable "
                    "document content was available."
                ),
                "source_documents": retrieved_docs,
                "context_length": 0,
            }

        # 5. Generate answer
        try:
            answer_text = self.llm_provider.generate(
                query=cleaned_query,
                context=context,
            )
        except Exception as exc:
            return {
                "query": cleaned_query,
                "answer": f"Error during answer generation: {exc}",
                "source_documents": retrieved_docs,
                "context_length": len(context),
            }

        # 6. Return answer + evidence
        return {
            "query": cleaned_query,
            "answer": answer_text,
            "source_documents": retrieved_docs,
            "context_length": len(context),
        }
