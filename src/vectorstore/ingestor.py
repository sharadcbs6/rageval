"""Document ingestion pipeline.

Handles chunking raw text with LangChain's text splitters,
optionally enriching metadata, and writing to both the vector store
and the Neo4j knowledge graph.
"""

from __future__ import annotations

from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.vectorstore.store import get_vector_store
from src.logger import get_logger

logger = get_logger(__name__)

# Default chunking parameters — tune per use-case
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64


class DocumentIngestor:
    """Split, embed, and store documents; optionally build the knowledge graph."""

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
        )

    # ── internal helpers ──────────────────────────────────────────────────────

    def _build_documents(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None,
    ) -> list[Document]:
        metadatas = metadatas or [{} for _ in texts]
        raw_docs = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(texts, metadatas)
        ]
        chunks = self._splitter.split_documents(raw_docs)
        logger.info("chunks_created", raw_docs=len(raw_docs), chunks=len(chunks))
        return chunks

    # ── public API ────────────────────────────────────────────────────────────

    def ingest(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        build_graph: bool = True,
    ) -> dict[str, Any]:
        """
        Ingest *texts* into the vector store and optionally into Neo4j.

        Returns a summary dict with counts.
        """
        chunks = self._build_documents(texts, metadatas)

        # 1. Vector store
        vs = get_vector_store()
        ids = vs.add_documents(chunks)
        logger.info("vector_store_ingested", ids=len(ids))

        # 2. Knowledge graph (optional — import lazily to avoid circular deps)
        graph_stats: dict[str, int] = {}
        if build_graph:
            from src.graph.extractor import KnowledgeGraphExtractor  # noqa: PLC0415

            extractor = KnowledgeGraphExtractor()
            total_nodes = 0
            total_rels = 0
            for text in texts:
                stats = extractor.extract_and_store(text)
                total_nodes += stats.get("nodes_created", 0)
                total_rels += stats.get("relations_created", 0)
            graph_stats = {"nodes_created": total_nodes, "relations_created": total_rels}
            logger.info("graph_ingested", **graph_stats)

        return {
            "chunks_ingested": len(ids),
            "collection": vs._collection_name,
            **graph_stats,
        }

    def ingest_documents(
        self,
        documents: list[Document],
        build_graph: bool = False,
    ) -> dict[str, Any]:
        """Ingest pre-built LangChain Document objects."""
        vs = get_vector_store()
        chunks = self._splitter.split_documents(documents)
        ids = vs.add_documents(chunks)
        logger.info("documents_ingested", count=len(ids))

        graph_stats: dict[str, int] = {}
        if build_graph:
            from src.graph.extractor import KnowledgeGraphExtractor  # noqa: PLC0415

            extractor = KnowledgeGraphExtractor()
            total_nodes = total_rels = 0
            for doc in documents:
                stats = extractor.extract_and_store(doc.page_content)
                total_nodes += stats.get("nodes_created", 0)
                total_rels += stats.get("relations_created", 0)
            graph_stats = {"nodes_created": total_nodes, "relations_created": total_rels}

        return {
            "chunks_ingested": len(ids),
            "collection": vs._collection_name,
            **graph_stats,
        }
