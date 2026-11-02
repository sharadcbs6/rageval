"""LangChain RAG pipeline using Groq LLM + local embeddings."""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_groq import ChatGroq

from src.config import get_settings
from src.vectorstore.store import get_vector_store
from src.logger import get_logger

logger = get_logger(__name__)

_RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful assistant. Answer the question using ONLY the
provided context. If the context does not contain enough information,
say "I don't have enough information to answer that."

Context:
{context}
""",
        ),
        ("human", "{question}"),
    ]
)

_GRAPH_RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful assistant with access to both document excerpts
and a structured knowledge graph. Use ALL provided context to answer accurately.
If the context does not contain enough information, say so.

Document context:
{vector_context}

{graph_context}
""",
        ),
        ("human", "{question}"),
    ]
)


def _format_docs(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def _docs_to_strings(docs: list[Document]) -> list[str]:
    return [doc.page_content for doc in docs]


class RAGPipeline:
    """Standard retrieval-augmented generation pipeline."""

    def __init__(self, top_k: int = 5) -> None:
        settings = get_settings()
        self._llm = ChatGroq(
            model=settings.groq_model,
            temperature=0,
            api_key=settings.groq_api_key,
        )
        self._top_k = top_k
        self._retriever = get_vector_store().as_retriever(k=top_k)

    def _build_vector_chain(self):
        return (
            RunnableParallel(
                context=(self._retriever | _format_docs),
                question=RunnablePassthrough(),
            )
            | _RAG_PROMPT
            | self._llm
            | StrOutputParser()
        )

    def _build_graph_chain(self):
        from src.graph.retriever import GraphRetriever

        graph_retriever = GraphRetriever()

        def get_graph_context(question: str) -> str:
            return graph_retriever.get_context(question)

        return (
            RunnableParallel(
                vector_context=(self._retriever | _format_docs),
                graph_context=(RunnablePassthrough() | get_graph_context),
                question=RunnablePassthrough(),
            )
            | _GRAPH_RAG_PROMPT
            | self._llm
            | StrOutputParser()
        )

    def query(
        self,
        question: str,
        use_graph: bool = False,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        if top_k and top_k != self._top_k:
            self._retriever = get_vector_store().as_retriever(k=top_k)

        docs: list[Document] = self._retriever.invoke(question)
        contexts = _docs_to_strings(docs)

        chain = self._build_graph_chain() if use_graph else self._build_vector_chain()
        answer: str = chain.invoke(question)

        logger.info("rag_query", question=question[:80], use_graph=use_graph, docs_retrieved=len(docs))

        return {
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "source_documents": [{"content": d.page_content, **d.metadata} for d in docs],
        }
