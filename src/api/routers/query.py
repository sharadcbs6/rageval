"""Query router — POST /query (simple pipeline) and POST /query/workflow (LangGraph)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from src.models.schemas import QueryRequest, QueryResponse
from src.rag.pipeline import RAGPipeline
from src.rag.workflow import run_workflow
from src.logger import get_logger

router = APIRouter(prefix="/query", tags=["RAG Query"])
logger = get_logger(__name__)


@router.post(
    "",
    response_model=QueryResponse,
    summary="Answer a question using the LangChain RAG pipeline.",
)
async def query_pipeline(payload: QueryRequest) -> QueryResponse:
    """
    Retrieve relevant chunks from ChromaDB (and optionally Neo4j graph context)
    then generate an answer with GPT-4o via the LCEL pipeline.
    """
    try:
        pipeline = RAGPipeline(top_k=payload.top_k)
        result = pipeline.query(
            question=payload.question,
            use_graph=payload.use_graph,
            top_k=payload.top_k,
        )
        return QueryResponse(**result)
    except Exception as exc:
        logger.error("query_pipeline_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {exc}",
        ) from exc


@router.post(
    "/workflow",
    response_model=QueryResponse,
    summary="Answer a question using the full LangGraph agentic workflow.",
)
async def query_workflow(payload: QueryRequest) -> QueryResponse:
    """
    Routes the question through a LangGraph state machine:
    route → retrieve → [graph_retrieve] → grade_docs → generate → check_answer.

    Includes automatic hallucination detection and retry (up to 2 retries).
    """
    try:
        result = run_workflow(
            question=payload.question,
            use_graph=payload.use_graph,
        )
        return QueryResponse(
            question=result["question"],
            answer=result["answer"],
            contexts=result["contexts"],
            source_documents=result["source_documents"],
        )
    except Exception as exc:
        logger.error("query_workflow_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Workflow failed: {exc}",
        ) from exc
