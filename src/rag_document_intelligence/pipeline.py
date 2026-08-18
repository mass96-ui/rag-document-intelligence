from typing import Any, Dict, List

from .context_builder import ContextBuilder
from .llm import LLMProvider
from .retriever import RAGRetriever


class RAGPipeline:
    """Orchestrate the complete RAG query-to-answer flow."""

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
        Perform the full RAG process: retrieve, build context, and generate answer.

        Args:
            query: The user's question.
            top_k: Number of chunks to retrieve.

        Returns:
            A dictionary containing the answer and source documents.
        """
        # 1. Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(
            query=query,
            top_k=top_k,
        )

        # 2. Build context string
        context = self.context_builder.build_context(
            retrieved_docs
        )

        # 3. Generate answer using the LLM provider
        answer_text = self.llm_provider.generate(
            query=query,
            context=context,
        )

        return {
            "query": query,
            "answer": answer_text,
            "source_documents": retrieved_docs,
            "context_length": len(context),
        }
