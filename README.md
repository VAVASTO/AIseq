# Safety Judge Eval

Прогоняет **evaluator** (LLM-as-a-Judge) по датасету: для каждой пары «judge-модель × подход» вызывает API и пишет унифицированный лог. Target-модель не запускается — судятся уже готовые `user_message` + `assistant_response` (и опционально `reference`).

## Компоненты

- **`run_eval.py`** — читает JSONL и `judge_config.yaml`, для каждого примера и каждой комбинации модели и подхода записывает строку в `judges_raw.jsonl` и сохраняет `run_manifest.json` в выходную директорию.
- **`judge_eval/`** — OpenAI-compatible клиент (`chat.completions`, JSON-ответ), четыре подхода: `zero_shot`, `few_shot`, `reference_based`, `hybrid` (для `hallucination` с `reference` — сначала reference-based иначе zero-shot; при уверенности в «серой зоне» — второй вызов few-shot).
- **`summarize_runs.py`** — агрегирует `judges_raw.jsonl` по `(judge_model_id, approach)`: accuracy, precision/recall/F1 по классу unsafe (если в примерах есть `human_unsafe`), Cohen’s κ с человеком, средняя и макс. latency, сумма `estimated_cost_usd` (оценка в `judge_eval/llm_client.py`).
- **`data/dataset_schema.md`** — описание полей JSONL.
- **`data/examples_sample.jsonl`** — минимальный пример датасета.

## Конфигурация

**Переменные окружения** — как в `academiy_test_llm_safety/backend/app/config.py` (см. `.env.example`):

- `XAI_API_KEY`
- `XAI_BASE_URL` (по умолчанию `https://api.x.ai/v1`)
- `XAI_MODEL` (по умолчанию `grok-4.20-non-reasoning`)

Вызов API: `OpenAI(base_url=XAI_BASE_URL, api_key=XAI_API_KEY).chat.completions.create(model=..., messages=..., temperature=...)` — без `response_format`, как в `academiy_test_llm_safety/backend/app/llm.py`.

**`judge_config.yaml`**: список `models` с полями `id`, опционально `model` (если нет — берётся `XAI_MODEL`), опционально `base_url` и `api_key_env` (по умолчанию `XAI_API_KEY`); список `approaches`; блок `hybrid`.

## Запуск

```bash
python3 -m venv .venv && source .venv/bin/activate  # или аналог под вашу ОС
pip install -r requirements.txt
cp .env.example .env   # XAI_API_KEY и при необходимости XAI_BASE_URL / XAI_MODEL
```

Без вызовов API (`pred_unsafe` фиктивный, в логе `dry_run: true`; сводка метрик пропускает такие строки):

```bash
python run_eval.py --dataset data/examples_sample.jsonl --out logs/dry --dry-run
python summarize_runs.py --run-dir logs/dry
```

С API:

```bash
python run_eval.py --dataset <путь к.jsonl> --config judge_config.yaml --out <директория_вывода>
python run_eval.py ... --limit N   # только первые N примеров
python summarize_runs.py --run-dir <та_же_директория>
```

## Выходные файлы

- **`judges_raw.jsonl`** — на строку: `example_id`, `task_type`, `approach`, `judge_model_id`, `api_model`, `pred_unsafe`, `confidence`, `explanation`, `latency_ms`, опционально токены и `estimated_cost_usd`, `human_unsafe`, `correct`, `hybrid_path`, `error`, `dry_run`.
- **`run_manifest.json`** — время запуска, число примеров, конфиг моделей и подходов.
- **`metrics_summary.json`** — таблица метрик по группам (пишет `summarize_runs.py`).

## Ограничения

- Стоимость считается грубо по таблице в `llm_client.py` для известных имён моделей.
- Ответ судьи парсится как JSON (промпт требует JSON); при обёртке в markdown извлекается первый объект `{...}`.
- Другой ключ на запись: в YAML укажите `api_key_env` (имя переменной окружения).
