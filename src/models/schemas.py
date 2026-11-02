"""Shared Pydantic schemas used across the API and evaluation layers."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ── Ingestion ────────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    """Payload for ingesting documents into the vector store and/or knowledge graph."""

    texts: list[str] = Field(..., min_length=1, description="Raw text chunks to ingest.")
    metadatas: list[dict[str, Any]] | None = Field(
        default=None,
        description="Optional metadata dicts aligned with texts.",
    )
    build_graph: bool = Field(
        default=True,
        description="Whether to also extract entities/relations into Neo4j.",
    )


class IngestResponse(BaseModel):
    ingested: int
    collection: str
    graph_nodes_created: int = 0


# ── RAG Query ────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    use_graph: bool = Field(
        default=False,
        description="Augment retrieval with Neo4j graph context.",
    )


class QueryResponse(BaseModel):
    question: str
    answer: str
    contexts: list[str]
    source_documents: list[dict[str, Any]] = []


# ── Evaluation ───────────────────────────────────────────────────────────────

class EvalSample(BaseModel):
    """A single QA sample for evaluation."""

    question: str
    answer: str
    contexts: list[str]
    ground_truth: str | None = None


class RagasEvalRequest(BaseModel):
    samples: list[EvalSample] = Field(..., min_length=1)
    metrics: list[str] = Field(
        default=["faithfulness", "answer_relevancy", "context_precision"],
        description="Ragas metric names to compute.",
    )


class DeepEvalRequest(BaseModel):
    samples: list[EvalSample] = Field(..., min_length=1)
    metrics: list[str] = Field(
        default=["answer_relevancy", "faithfulness", "hallucination"],
        description="DeepEval metric names to compute.",
    )


class MetricScore(BaseModel):
    metric: str
    score: float | None
    reason: str | None = None


class EvalResponse(BaseModel):
    framework: str
    results: list[dict[str, Any]]
    aggregate: dict[str, float] = {}


# ── Graph ────────────────────────────────────────────────────────────────────

class GraphQueryRequest(BaseModel):
    cypher: str = Field(..., description="Cypher query to run against Neo4j.")
    params: dict[str, Any] = Field(default_factory=dict)


class GraphQueryResponse(BaseModel):
    records: list[dict[str, Any]]
    count: int
