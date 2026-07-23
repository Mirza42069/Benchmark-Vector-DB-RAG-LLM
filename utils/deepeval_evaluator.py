"""DeepEval RAG metric helpers."""

from __future__ import annotations

import logging
from typing import Any

from langchain_ollama import ChatOllama
from pydantic import BaseModel

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase

logger = logging.getLogger(__name__)


class OllamaDeepEvalModel(DeepEvalBaseLLM):
    def __init__(self, model_name: str, temperature: float = 0.0):
        self.model_name = model_name
        self.model = ChatOllama(model=model_name, temperature=temperature)

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema: type[BaseModel] | None = None) -> str | BaseModel:
        if schema is not None:
            structured_model = self.model.with_structured_output(schema)
            return structured_model.invoke(prompt)
        return self.model.invoke(prompt).content

    async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None) -> str | BaseModel:
        if schema is not None:
            structured_model = self.model.with_structured_output(schema)
            return await structured_model.ainvoke(prompt)
        response = await self.model.ainvoke(prompt)
        return response.content

    def get_model_name(self) -> str:
        return f"Ollama/{self.model_name}"


def build_reference_answer(query: str, keywords: list[str]) -> str:
    keyword_text = ", ".join(keywords)
    return f"Jawaban ideal untuk pertanyaan '{query}' harus mencakup fakta kunci berikut: {keyword_text}."


def evaluate_deepeval_rag_metrics(
    query: str,
    actual_output: str,
    retrieval_context: list[str],
    expected_output: str,
    judge_model: OllamaDeepEvalModel,
    threshold: float = 0.5,
) -> dict[str, Any]:
    test_case = LLMTestCase(
        input=query,
        actual_output=actual_output or "",
        expected_output=expected_output,
        retrieval_context=retrieval_context,
    )
    metrics = [
        AnswerRelevancyMetric(threshold=threshold, model=judge_model, include_reason=True),
        FaithfulnessMetric(threshold=threshold, model=judge_model, include_reason=True),
        ContextualRelevancyMetric(threshold=threshold, model=judge_model, include_reason=True),
        ContextualPrecisionMetric(threshold=threshold, model=judge_model, include_reason=True),
        ContextualRecallMetric(threshold=threshold, model=judge_model, include_reason=True),
    ]

    scores: dict[str, Any] = {}
    for metric in metrics:
        metric_key = metric.__class__.__name__.replace("Metric", "")
        try:
            metric.measure(test_case)
            scores[f"{metric_key}_score"] = metric.score
            scores[f"{metric_key}_reason"] = metric.reason
            scores[f"{metric_key}_success"] = metric.is_successful()
        except Exception as e:
            logger.exception("DeepEval metric failed: %s", metric_key)
            scores[f"{metric_key}_score"] = 0.0
            scores[f"{metric_key}_reason"] = str(e)
            scores[f"{metric_key}_success"] = False
    return scores
