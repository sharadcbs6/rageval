"""Evaluation router.

Endpoints
---------
POST /evaluate/ragas        — run Ragas metrics
POST /evaluate/deepeval     — run DeepEval metrics
POST /evaluate/compare      — run both frameworks and return side-by-side
GET  /evaluate/metrics      — list available metrics for each framework
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status

from src.models.schemas import (
    DeepEvalRequest,
    EvalResponse,
    EvalSample,
    RagasEvalRequest,
)
from src.evaluation.ragas_eval import RagasEvaluator
from src.evaluation.deepeval_eval import DeepEvaluator
from src.logger import get_logger

router = APIRouter(prefix="/evaluate", tags=["Evaluation"])
logger = get_logger(__name__)


def _samples_to_dicts(samples: list[EvalSample]) -> list[dict[str, Any]]:
    return [s.model_dump() for s in samples]


# ── Ragas ─────────────────────────────────────────────────────────────────────

@router.post(
    "/ragas",
    response_model=EvalResponse,
    summary="Evaluate QA samples with Ragas.",
)
async def evaluate_ragas(payload: RagasEvalRequest) -> EvalResponse:
    """
    Compute Ragas metrics over the provided samples.

    Available metrics: faithfulness, answer_relevancy, context_precision,
    context_recall, context_entity_recall, answer_correctness, answer_similarity.
    """
    try:
        evaluator = RagasEvaluator()
        result = evaluator.evaluate(
            samples=_samples_to_dicts(payload.samples),
            metrics=payload.metrics,
        )
        return EvalResponse(**result)
    except Exception as exc:
        logger.error("ragas_eval_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ragas evaluation failed: {exc}",
        ) from exc


# ── DeepEval ──────────────────────────────────────────────────────────────────

@router.post(
    "/deepeval",
    response_model=EvalResponse,
    summary="Evaluate QA samples with DeepEval.",
)
async def evaluate_deepeval(payload: DeepEvalRequest) -> EvalResponse:
    """
    Compute DeepEval metrics over the provided samples.

    Available metrics: answer_relevancy, faithfulness, hallucination,
    contextual_precision, contextual_recall, contextual_relevancy, bias, toxicity.
    """
    try:
        evaluator = DeepEvaluator()
        result = evaluator.evaluate(
            samples=_samples_to_dicts(payload.samples),
            metrics=payload.metrics,
        )
        return EvalResponse(**result)
    except Exception as exc:
        logger.error("deepeval_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"DeepEval evaluation failed: {exc}",
        ) from exc


# ── Compare ───────────────────────────────────────────────────────────────────

class CompareRequest(RagasEvalRequest):
    """Reuse RagasEvalRequest shape; metrics apply to both frameworks."""
    deepeval_metrics: list[str] = ["answer_relevancy", "faithfulness", "hallucination"]


@router.post(
    "/compare",
    summary="Run both Ragas and DeepEval and return results side-by-side.",
    response_model=dict,
)
async def evaluate_compare(payload: CompareRequest) -> dict[str, Any]:
    """Run both frameworks on the same samples and return a unified comparison."""
    samples_dicts = _samples_to_dicts(payload.samples)
    try:
        ragas_result = RagasEvaluator().evaluate(
            samples=samples_dicts, metrics=payload.metrics
        )
    except Exception as exc:
        logger.error("compare_ragas_failed", error=str(exc))
        ragas_result = {"error": str(exc)}

    try:
        deepeval_result = DeepEvaluator().evaluate(
            samples=samples_dicts, metrics=payload.deepeval_metrics
        )
    except Exception as exc:
        logger.error("compare_deepeval_failed", error=str(exc))
        deepeval_result = {"error": str(exc)}

    return {
        "ragas": ragas_result,
        "deepeval": deepeval_result,
    }


# ── Metrics catalogue ─────────────────────────────────────────────────────────

@router.get(
    "/metrics",
    summary="List all available metrics for each framework.",
)
async def list_metrics() -> dict[str, list[str]]:
    return {
        "ragas": RagasEvaluator().available_metrics(),
        "deepeval": DeepEvaluator().available_metrics(),
    }
