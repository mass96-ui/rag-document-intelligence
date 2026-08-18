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
        # 1. Basic Validation
        if not query or not query.strip():
            return {
                "query": query,
                "answer": "Error: Question cannot be empty.",
                "source_documents": [],
                "context_length": 0,
            }

        if top_k <= 0:
            return {
                "query": query,
                "answer": f"Error: Invalid top_k ({top_k}). Must be greater than 0.",
                "source_documents": [],
                "context_length": 0,
            }

        # 2. Retrieve relevant documents
        try:
            retrieved_docs = self.retriever.retrieve(
                query=query.strip(),
                top_k=top_k,
            )
        except Exception as e:
            return {
                "query": query,
                "answer": f"Error during retrieval: {str(e)}",
                "source_documents": [],
                "context_length": 0,
            }

        if not retrieved_docs:
            return {
                "query": query,
                "answer": "I could not find any relevant information in the documents to answer your question.",
                "source_documents": [],
                "context_length": 0,
            }

        # 3. Build context string
        context = self.context_builder.build_context(
            retrieved_docs
        )

        # 4. Generate answer using the LLM provider
        try:
            answer_text = self.llm_provider.generate(
                query=query.strip(),
                context=context,
            )
        except Exception as e:
            return {
                "query": query,
                "answer": f"Error during generation: {str(e)}",
                "source_documents": retrieved_docs,
                "context_length": len(context),
            }

        return {
            "query": query.strip(),
            "answer": answer_text,
            "source_documents": retrieved_docs,
            "context_length": len(context),
        }
