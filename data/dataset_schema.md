# Формат gold-датасета (JSONL)

Одна строка = один JSON-объект.

## Обязательные поля

| Поле | Тип | Описание |
|------|-----|----------|
| `id` | string | Стабильный id |
| `task_type` | string | `hallucination` \| `jailbreak` \| `policy` \| `mixed` |
| `user_message` | string | Запрос / атака |
| `assistant_response` | string | Ответ target-модели |

## Рекомендуемые поля

| Поле | Тип | Описание |
|------|-----|----------|
| `lang` | string | `ru`, `en`, `mixed` |
| `reference` | string \| null | Эталон для hallucination / RAG |
| `human_unsafe` | bool \| null | Эталон разметчика: `true` = unsafe / hallucination / jailbreak success / violation |
| `human_notes` | string \| null | Короткое обоснование |
| `extra` | object | Произвольные метаданные |

При заполненном `human_unsafe` скрипт `summarize_runs.py` считает accuracy, F1 по классу unsafe и Cohen’s κ относительно разметки.
