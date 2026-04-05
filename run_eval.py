#!/usr/bin/env python3
"""
Единая точка входа: прогон датасета по всем judge-моделям и подходам из judge_config.yaml.

Пример (из корня этого проекта):
  python3 -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  cp .env.example .env   # заполнить ключи
  python run_eval.py --dataset data/examples_sample.jsonl --out logs/run_smoke
  python summarize_runs.py --run-dir logs/run_smoke
"""
from __future__ import annotations

import argparse
from pathlib import Path

from judge_eval.runner import load_dataset, load_run_config, run_matrix


def main() -> None:
    p = argparse.ArgumentParser(description="Safety judge eval matrix runner")
    p.add_argument("--dataset", type=Path, required=True, help="JSONL с примерами (см. data/dataset_schema.md)")
    p.add_argument("--config", type=Path, default=Path("judge_config.yaml"))
    p.add_argument("--out", type=Path, required=True, help="Папка для judges_raw.jsonl и run_manifest.json")
    p.add_argument("--dry-run", action="store_true", help="Без API; проверка формата и пайплайна")
    p.add_argument("--limit", type=int, default=0, help="Ограничить число примеров (0 = все)")
    p.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Дополнительный .env (например ../academiy_test_llm_safety/.env); переопределяет переменные",
    )
    args = p.parse_args()

    examples = load_dataset(args.dataset)
    if args.limit and args.limit > 0:
        examples = examples[: args.limit]
    cfg = load_run_config(args.config)
    log_path = run_matrix(
        examples,
        cfg,
        args.out,
        dry_run=args.dry_run,
        extra_env_file=args.env_file,
    )
    print(f"Wrote {log_path}")


if __name__ == "__main__":
    main()
