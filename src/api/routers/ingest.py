"""Ingestion router — POST /ingest."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from src.models.schemas import IngestRequest, IngestResponse
from src.vectorstore.ingestor import DocumentIngestor
from src.logger import get_logger

router = APIRouter(prefix="/ingest", tags=["Ingestion"])
logger = get_logger(__name__)


@router.post(
    "",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest documents into the vector store (and optionally Neo4j).",
)
async def ingest_documents(payload: IngestRequest) -> IngestResponse:
    """
    Chunk, embed, and store raw text documents.

    - Splits texts with a recursive character splitter (512 tokens, 64 overlap).
    - Embeds chunks with OpenAI and stores them in ChromaDB.
    - When `build_graph=true`, also extracts entity/relation triples into Neo4j.
    """
    try:
        ingestor = DocumentIngestor()
        stats = ingestor.ingest(
            texts=payload.texts,
            metadatas=payload.metadatas,
            build_graph=payload.build_graph,
        )
        return IngestResponse(
            ingested=stats["chunks_ingested"],
            collection=stats["collection"],
            graph_nodes_created=stats.get("nodes_created", 0),
        )
    except Exception as exc:
        logger.error("ingest_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {exc}",
        ) from exc
