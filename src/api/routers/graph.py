"""Graph router — Neo4j knowledge graph endpoints.

Endpoints
---------
POST /graph/query         — run a raw Cypher query
GET  /graph/health        — Neo4j connectivity check
GET  /graph/search        — natural-language entity search
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status

from src.models.schemas import GraphQueryRequest, GraphQueryResponse
from src.graph.client import get_neo4j_client
from src.graph.retriever import GraphRetriever
from src.logger import get_logger

router = APIRouter(prefix="/graph", tags=["Knowledge Graph"])
logger = get_logger(__name__)


@router.post(
    "/query",
    response_model=GraphQueryResponse,
    summary="Execute a Cypher query against Neo4j.",
)
async def run_cypher(payload: GraphQueryRequest) -> GraphQueryResponse:
    """
    Run an arbitrary Cypher statement and return matching records.
    Use with care — write queries are allowed.
    """
    try:
        client = get_neo4j_client()
        records = client.run_query(payload.cypher, payload.params)
        return GraphQueryResponse(records=records, count=len(records))
    except Exception as exc:
        logger.error("cypher_query_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cypher query failed: {exc}",
        ) from exc


@router.get(
    "/search",
    response_model=GraphQueryResponse,
    summary="Find graph entities related to a natural-language question.",
)
async def graph_search(
    question: str = Query(..., description="Natural-language question or keyword phrase."),
    raw: bool = Query(False, description="Return raw Neo4j records instead of formatted string."),
) -> GraphQueryResponse:
    """
    Extract key entities from the question, query the knowledge graph,
    and return the 1-hop subgraph neighbourhood as records.
    """
    try:
        retriever = GraphRetriever()
        records = retriever.get_raw_records(question)
        return GraphQueryResponse(records=records, count=len(records))
    except Exception as exc:
        logger.error("graph_search_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Graph search failed: {exc}",
        ) from exc


@router.get(
    "/health",
    summary="Check Neo4j connectivity.",
)
async def graph_health() -> dict[str, bool]:
    """Returns `{"neo4j": true}` when the database is reachable."""
    client = get_neo4j_client()
    return {"neo4j": client.health_check()}
