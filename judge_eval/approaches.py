from __future__ import annotations

from typing import Any

from openai import OpenAI

from .llm_client import LLMCallResult, complete_chat, parse_verdict_json
from .prompts import (
    FEW_SHOT_USER,
    REFERENCE_SYSTEM,
    ZERO_SHOT_SYSTEM,
    user_block_zero_few,
)
from .schemas import DatasetExample, JudgeVerdict


def _messages_zero(example: DatasetExample) -> list[dict[str, str]]:
    user = user_block_zero_few(example.model_dump(), include_reference_instruction=False)
    return [
        {"role": "system", "content": ZERO_SHOT_SYSTEM},
        {"role": "user", "content": user},
    ]


def _messages_few(example: DatasetExample) -> list[dict[str, str]]:
    user = FEW_SHOT_USER + "\n\n" + user_block_zero_few(example.model_dump(), include_reference_instruction=True)
    return [
        {"role": "system", "content": ZERO_SHOT_SYSTEM},
        {"role": "user", "content": user},
    ]


def _messages_reference(example: DatasetExample) -> list[dict[str, str]]:
    user = user_block_zero_few(example.model_dump(), include_reference_instruction=True)
    return [
        {"role": "system", "content": REFERENCE_SYSTEM},
        {"role": "user", "content": user},
    ]


def run_approach(
    client: OpenAI,
    api_model: str,
    approach: str,
    example: DatasetExample,
    hybrid_thresholds: dict[str, float],
) -> tuple[JudgeVerdict, LLMCallResult, dict[str, Any]]:
    """
    Возвращает (verdict, last_llm_call, meta).
    Для hybrid meta содержит hybrid_path и опционально второй вызов.
    """
    meta: dict[str, Any] = {"approach": approach}

    if approach == "zero_shot":
        messages = _messages_zero(example)
        r = complete_chat(client, api_model, messages)
        v = parse_verdict_json(r.text, example.task_type)
        meta["hybrid_path"] = "primary"
        return v, r, meta

    if approach == "few_shot":
        messages = _messages_few(example)
        r = complete_chat(client, api_model, messages)
        v = parse_verdict_json(r.text, example.task_type)
        meta["hybrid_path"] = "primary"
        return v, r, meta

    if approach == "reference_based":
        messages = _messages_reference(example)
        r = complete_chat(client, api_model, messages)
        v = parse_verdict_json(r.text, example.task_type)
        meta["hybrid_path"] = "primary"
        return v, r, meta

    if approach == "hybrid":
        low = hybrid_thresholds.get("ambiguous_low", 0.35)
        high = hybrid_thresholds.get("ambiguous_high", 0.65)
        # Шаг 1: reference-based если есть reference; иначе zero-shot
        if example.reference and example.task_type == "hallucination":
            messages1 = _messages_reference(example)
        else:
            messages1 = _messages_zero(example)
        r1 = complete_chat(client, api_model, messages1)
        v1 = parse_verdict_json(r1.text, example.task_type)
        ambiguous = low <= v1.confidence <= high
        if not ambiguous:
            meta["hybrid_path"] = "primary"
            meta["first_confidence"] = v1.confidence
            meta["latency_ms_total"] = r1.latency_ms
            return v1, r1, meta
        messages2 = _messages_few(example)
        r2 = complete_chat(client, api_model, messages2)
        v2 = parse_verdict_json(r2.text, example.task_type)
        meta["hybrid_path"] = "escalated"
        meta["first_confidence"] = v1.confidence
        meta["second_confidence"] = v2.confidence
        # Суммируем latency и tokens для лога на уровне runner
        meta["latency_ms_total"] = r1.latency_ms + r2.latency_ms
        meta["tokens"] = _sum_usage(r1, r2)
        return v2, r2, meta

    raise ValueError(f"Unknown approach: {approach}")


def _sum_usage(a: LLMCallResult, b: LLMCallResult) -> tuple[int | None, int | None, int | None]:
    if a.prompt_tokens is None and b.prompt_tokens is None:
        return None, None, None
    pt = (a.prompt_tokens or 0) + (b.prompt_tokens or 0)
    ct = (a.completion_tokens or 0) + (b.completion_tokens or 0)
    tt = (a.total_tokens or 0) + (b.total_tokens or 0)
    return pt, ct, tt
