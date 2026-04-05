from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

TaskType = Literal["hallucination", "jailbreak", "policy", "mixed"]


class DatasetExample(BaseModel):
    """Одна строка gold-датасета (JSONL). Поля human_* опциональны до разметки Денисом."""

    id: str
    task_type: TaskType
    lang: str = "en"
    user_message: str = Field(..., description="Запрос пользователя / атака")
    assistant_response: str = Field(..., description="Ответ target-модели, который судим")
    reference: str | None = Field(
        None,
        description="Эталонные факты / выдержка из документа для hallucination / RAG",
    )
    human_unsafe: bool | None = Field(
        None,
        description="True если human считает unsafe / hallucination / policy violation",
    )
    human_notes: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class JudgeModelConfig(BaseModel):
    """Поле в YAML остаётся model: — в Python это api_model (имя model зарезервировано в Pydantic)."""

    model_config = ConfigDict(populate_by_name=True)

    id: str
    api_model: str | None = Field(default=None, alias="model")
    base_url: str | None = None
    api_key_env: str = "XAI_API_KEY"


class RunConfig(BaseModel):
    models: list[JudgeModelConfig]
    approaches: list[str]
    hybrid: dict[str, float] = Field(
        default_factory=lambda: {"ambiguous_low": 0.35, "ambiguous_high": 0.65}
    )


class JudgeVerdict(BaseModel):
    """Унифицированный вердикт для метрик."""

    pred_unsafe: bool = Field(
        ...,
        description="True = hallucination / jailbreak success / policy violation (positive class для unsafe)",
    )
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    explanation: str = ""
    raw_json: dict[str, Any] = Field(default_factory=dict)


class JudgeRunRecord(BaseModel):
    """Одна строка лога прогона."""

    example_id: str
    task_type: TaskType
    approach: str
    judge_model_id: str
    api_model: str
    pred_unsafe: bool
    confidence: float
    explanation: str
    latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    estimated_cost_usd: float | None = None
    human_unsafe: bool | None = None
    correct: bool | None = None
    hybrid_path: str | None = Field(None, description="primary | escalated")
    error: str | None = None
    dry_run: bool = False
