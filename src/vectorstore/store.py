"""ChromaDB vector store wrapper with LangChain integration.

Uses local sentence-transformers for embeddings — no API key needed.
"""

from __future__ import annotations

from typing import Any

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import get_settings
from src.logger import get_logger

logger = get_logger(__name__)


class ChromaVectorStore:
    """Thin wrapper around LangChain's Chroma integration."""

    def __init__(self) -> None:
        settings = get_settings()

        # Local sentence-transformers model — downloaded once, cached on disk
        self._embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self._collection_name = settings.chroma_collection

        self._http_client = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
        )

        self._store = Chroma(
            client=self._http_client,
            collection_name=self._collection_name,
            embedding_function=self._embeddings,
        )
        logger.info(
            "chroma_connected",
            host=settings.chroma_host,
            port=settings.chroma_port,
            collection=self._collection_name,
            embedding_model=settings.embedding_model,
        )

    # ── ingestion ─────────────────────────────────────────────────────────────

    def add_texts(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        metadatas = metadatas or [{} for _ in texts]
        ids = self._store.add_texts(texts=texts, metadatas=metadatas)
        logger.info("chroma_add_texts", count=len(texts))
        return ids

    def add_documents(self, documents: list[Document]) -> list[str]:
        ids = self._store.add_documents(documents)
        logger.info("chroma_add_documents", count=len(documents))
        return ids

    # ── retrieval ─────────────────────────────────────────────────────────────

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        docs = self._store.similarity_search(query, k=k, filter=filter)
        logger.debug("similarity_search", query=query[:80], results=len(docs))
        return docs

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        return self._store.similarity_search_with_score(query, k=k, filter=filter)

    def mmr_search(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
    ) -> list[Document]:
        docs = self._store.max_marginal_relevance_search(
            query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult
        )
        logger.debug("mmr_search", query=query[:80], results=len(docs))
        return docs

    def as_retriever(self, search_type: str = "similarity", k: int = 5):
        return self._store.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k},
        )

    # ── collection management ─────────────────────────────────────────────────

    def count(self) -> int:
        return self._store._collection.count()  # type: ignore[attr-defined]

    def delete_collection(self) -> None:
        self._http_client.delete_collection(self._collection_name)
        logger.warning("chroma_collection_deleted", collection=self._collection_name)

    def health_check(self) -> bool:
        try:
            self._http_client.heartbeat()
            return True
        except Exception:
            return False


# ── module-level singleton ─────────────────────────────────────────────────────

_store: ChromaVectorStore | None = None


def get_vector_store() -> ChromaVectorStore:
    global _store
    if _store is None:
        _store = ChromaVectorStore()
    return _store
