"""DeepEval evaluation module — uses Groq LLM via LangChain adapter."""

from __future__ import annotations

import os
from typing import Any

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from deepeval import evaluate as deepeval_evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    HallucinationMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    BiasMetric,
    ToxicityMetric,
)
from deepeval.test_case import LLMTestCase
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_groq import ChatGroq

from src.config import get_settings
from src.logger import get_logger

logger = get_logger(__name__)

DEFAULT_METRICS = ["answer_relevancy", "faithfulness", "hallucination"]


class _GroqAdapter(DeepEvalBaseLLM):
    """Wraps ChatGroq so DeepEval can use it as its LLM backend."""

    def __init__(self, chat_model: ChatGroq) -> None:
        self._model = chat_model

    def load_model(self):
        return self._model

    def generate(self, prompt: str) -> str:
        response = self._model.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)

    async def a_generate(self, prompt: str) -> str:
        response = await self._model.ainvoke(prompt)
        return response.content if hasattr(response, "content") else str(response)

    def get_model_name(self) -> str:
        return getattr(self._model, "model_name", "groq-model")


def _build_metric(name: str, model: _GroqAdapter, threshold: float = 0.5) -> Any:
    kw = {"model": model, "threshold": threshold}
    mapping = {
        "answer_relevancy":     lambda: AnswerRelevancyMetric(**kw),
        "faithfulness":         lambda: FaithfulnessMetric(**kw),
        "hallucination":        lambda: HallucinationMetric(**kw),
        "contextual_precision": lambda: ContextualPrecisionMetric(**kw),
        "contextual_recall":    lambda: ContextualRecallMetric(**kw),
        "contextual_relevancy": lambda: ContextualRelevancyMetric(**kw),
        "bias":                 lambda: BiasMetric(**kw),
        "toxicity":             lambda: ToxicityMetric(**kw),
    }
    factory = mapping.get(name.lower())
    if factory is None:
        logger.warning("unknown_deepeval_metric", name=name)
        return None
    return factory()


class DeepEvaluator:
    """Run DeepEval metrics using Groq as the LLM backend."""

    def __init__(self, threshold: float = 0.5) -> None:
        settings = get_settings()
        lc_llm = ChatGroq(
            model=settings.groq_model,
            temperature=0,
            api_key=settings.groq_api_key,
        )
        self._model = _GroqAdapter(lc_llm)
        self._threshold = threshold

    @staticmethod
    def _build_test_cases(samples: list[dict[str, Any]]) -> list[LLMTestCase]:
        return [
            LLMTestCase(
                input=s["question"],
                actual_output=s["answer"],
                # HallucinationMetric requires `context`; others use `retrieval_context`
                context=s.get("contexts", []),
                retrieval_context=s.get("contexts", []),
                expected_output=s.get("ground_truth"),
            )
            for s in samples
        ]

    def _resolve_metrics(self, names: list[str]) -> list[Any]:
        metrics = [
            m for name in names
            if (m := _build_metric(name, self._model, self._threshold)) is not None
        ]
        return metrics or [
            _build_metric(n, self._model, self._threshold) for n in DEFAULT_METRICS
        ]

    def evaluate(
        self,
        samples: list[dict[str, Any]],
        metrics: list[str] | None = None,
    ) -> dict[str, Any]:
        metric_names = metrics or DEFAULT_METRICS
        metric_objects = self._resolve_metrics(metric_names)
        test_cases = self._build_test_cases(samples)

        logger.info("deepeval_eval_start", samples=len(samples), metrics=metric_names)

        from deepeval.evaluate.configs import AsyncConfig, DisplayConfig  # noqa: PLC0415

        eval_results = deepeval_evaluate(
            test_cases=test_cases,
            metrics=metric_objects,
            async_config=AsyncConfig(run_async=False),
            display_config=DisplayConfig(
                show_indicator=False,
                print_results=False,
                verbose_mode=False,
            ),
        )

        per_sample: list[dict[str, Any]] = []
        aggregate_acc: dict[str, list[float]] = {n: [] for n in metric_names}

        for tc in eval_results.test_results:
            row: dict[str, Any] = {
                "question": tc.input,
                "answer":   tc.actual_output,
                "success":  tc.success,
            }
            for md in tc.metrics_data:
                key = md.name.lower().replace(" ", "_")
                row[key] = md.score
                row[f"{key}_reason"] = md.reason
                if key in aggregate_acc and md.score is not None:
                    aggregate_acc[key].append(md.score)
            per_sample.append(row)

        aggregate = {
            name: round(sum(vals) / len(vals), 4)
            for name, vals in aggregate_acc.items() if vals
        }

        logger.info("deepeval_eval_complete", aggregate=aggregate)
        return {"framework": "deepeval", "results": per_sample, "aggregate": aggregate}

    def available_metrics(self) -> list[str]:
        return [
            "answer_relevancy", "faithfulness", "hallucination",
            "contextual_precision", "contextual_recall",
            "contextual_relevancy", "bias", "toxicity",
        ]
