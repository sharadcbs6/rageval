"""Ragas evaluation module — uses Groq LLM + local HuggingFace embeddings."""

from __future__ import annotations

import os
from typing import Any

os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

import pandas as pd
from datasets import Dataset
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from ragas import evaluate as ragas_evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from src.config import get_settings
from src.logger import get_logger

logger = get_logger(__name__)

DEFAULT_METRICS = ["faithfulness", "answer_relevancy", "context_precision"]

_METRIC_MAP = {
    "faithfulness":       faithfulness,
    "answer_relevancy":   answer_relevancy,
    "context_precision":  context_precision,
    "context_recall":     context_recall,
    "answer_correctness": answer_correctness,
}


class RagasEvaluator:
    """Run Ragas metrics using Groq LLM and local sentence-transformer embeddings."""

    def __init__(self) -> None:
        settings = get_settings()

        lc_llm = ChatGroq(
            model=settings.groq_model,
            temperature=0,
            api_key=settings.groq_api_key,
        )
        lc_emb = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self._llm = LangchainLLMWrapper(lc_llm)
        self._embeddings = LangchainEmbeddingsWrapper(lc_emb)

        # Wire LLM + embeddings into each singleton metric
        for metric in _METRIC_MAP.values():
            if hasattr(metric, "llm"):
                metric.llm = self._llm
            if hasattr(metric, "embeddings"):
                metric.embeddings = self._embeddings

    @staticmethod
    def _build_dataset(samples: list[dict[str, Any]]) -> Dataset:
        rows = []
        for s in samples:
            row: dict[str, Any] = {
                "question": s["question"],
                "answer":   s["answer"],
                "contexts": s["contexts"],
            }
            if s.get("ground_truth"):
                row["ground_truth"] = s["ground_truth"]
            rows.append(row)
        return Dataset.from_list(rows)

    def _resolve_metrics(self, names: list[str]) -> list[Any]:
        resolved = []
        for name in names:
            m = _METRIC_MAP.get(name.lower())
            if m is None:
                logger.warning("unknown_ragas_metric", name=name)
            else:
                resolved.append(m)
        return resolved or [_METRIC_MAP[m] for m in DEFAULT_METRICS]

    def evaluate(
        self,
        samples: list[dict[str, Any]],
        metrics: list[str] | None = None,
    ) -> dict[str, Any]:
        metric_names = metrics or DEFAULT_METRICS
        metric_objects = self._resolve_metrics(metric_names)
        dataset = self._build_dataset(samples)

        logger.info("ragas_eval_start", samples=len(samples), metrics=metric_names)

        # ragas uses asyncio internally; uvloop (from uvicorn) is incompatible.
        # Force a fresh default event loop in a thread to avoid the uvloop conflict.
        import asyncio  # noqa: PLC0415
        import threading  # noqa: PLC0415

        result_holder: dict[str, Any] = {}
        exc_holder: list[Exception] = []

        def _run_ragas():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result_holder["result"] = ragas_evaluate(
                    dataset=dataset, metrics=metric_objects
                )
                loop.close()
            except Exception as e:
                exc_holder.append(e)

        t = threading.Thread(target=_run_ragas)
        t.start()
        t.join()

        if exc_holder:
            raise exc_holder[0]
        result = result_holder["result"]

        scores_df: pd.DataFrame = result.to_pandas()
        per_sample = scores_df.to_dict(orient="records")
        numeric_cols = scores_df.select_dtypes(include="number").columns.tolist()
        aggregate = {
            col: round(float(scores_df[col].mean(skipna=True)), 4)
            for col in numeric_cols
        }

        logger.info("ragas_eval_complete", aggregate=aggregate)
        return {"framework": "ragas", "results": per_sample, "aggregate": aggregate}

    def available_metrics(self) -> list[str]:
        return list(_METRIC_MAP.keys())
