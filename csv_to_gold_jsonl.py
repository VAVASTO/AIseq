#!/usr/bin/env python3
"""Конвертация dataset.csv (команда) → data/gold.jsonl для run_eval.py."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def map_task_type(raw: str) -> str:
    if raw == "ambiguous":
        return "mixed"
    if raw == "multilingual":
        return "mixed"
    return raw


def human_label_to_unsafe(task_type_orig: str, label: str) -> bool:
    label = (label or "").strip().lower()
    if label in ("unsafe", "hallucination"):
        return True
    if label in ("safe", "grounded"):
        return False
    raise ValueError(f"Unknown human_label: {label!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path("dataset.csv"))
    ap.add_argument("--out", type=Path, default=Path("data/gold.jsonl"))
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.csv.open(encoding="utf-8", newline="") as fp_in, args.out.open("w", encoding="utf-8") as fp_out:
        reader = csv.DictReader(fp_in)
        for row in reader:
            tid = (row.get("id") or "").strip()
            if not tid:
                continue
            task_orig = (row.get("task_type") or "").strip()
            task = map_task_type(task_orig)
            if task not in ("hallucination", "jailbreak", "policy", "mixed"):
                raise ValueError(f"id={tid}: bad task_type {task_orig!r} -> {task!r}")

            lang = (row.get("language") or "").strip() or "ru"
            ctx = row.get("context") or ""
            if ctx.strip() == "":
                ctx = None

            rec = {
                "id": tid,
                "task_type": task,
                "lang": lang,
                "user_message": row.get("prompt") or "",
                "assistant_response": row.get("model_response") or "",
                "reference": ctx,
                "human_unsafe": human_label_to_unsafe(task_orig, row.get("human_label") or ""),
                "human_notes": (row.get("explanation") or "").strip() or None,
                "extra": {
                    "source_task_type": task_orig,
                    "is_borderline": row.get("is_borderline", "").lower() == "true",
                    "annotator_id": row.get("annotator_id"),
                    "annotator_confidence": row.get("confidence"),
                },
            }
            fp_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"Wrote {n} rows to {args.out}")


if __name__ == "__main__":
    main()
