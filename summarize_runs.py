#!/usr/bin/env python3
"""Агрегация judges_raw.jsonl: accuracy, F1, κ, latency, cost по (judge_model_id × approach)."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def cohen_kappa(y_true: list[bool], y_pred: list[bool]) -> float:
    n = len(y_true)
    if n == 0:
        return 0.0
    agree = sum(1 for a, b in zip(y_true, y_pred) if a == b) / n
    p_yes = sum(y_true) / n
    p_pred_yes = sum(y_pred) / n
    pe = p_yes * p_pred_yes + (1 - p_yes) * (1 - p_pred_yes)
    if pe >= 1.0:
        return 1.0 if agree == 1.0 else 0.0
    return (agree - pe) / (1 - pe)


def prf1(y_true: list[bool], y_pred: list[bool], positive: bool = True) -> dict[str, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == positive and p == positive)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != positive and p == positive)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == positive and p != positive)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true) if y_true else 0.0
    return {"accuracy": acc, "precision_unsafe": prec, "recall_unsafe": rec, "f1_unsafe": f1}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    args = ap.parse_args()
    raw_path = args.run_dir / "judges_raw.jsonl"
    rows = []
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))

    usable = [r for r in rows if not r.get("error") and not r.get("dry_run")]
    if not usable:
        print("No non-dry rows to summarize (all dry_run or errors).")
        (args.run_dir / "metrics_summary.json").write_text("[]", encoding="utf-8")
        return

    by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in usable:
        by_key[(r["judge_model_id"], r["approach"])].append(r)

    summary = []
    for (mid, appr), grp in sorted(by_key.items()):
        labeled = [r for r in grp if r.get("human_unsafe") is not None]
        latencies = [r["latency_ms"] for r in grp]
        costs = [r["estimated_cost_usd"] for r in grp if r.get("estimated_cost_usd") is not None]
        item: dict = {
            "judge_model_id": mid,
            "approach": appr,
            "n_total": len(grp),
            "n_labeled": len(labeled),
            "latency_ms_mean": sum(latencies) / len(latencies) if latencies else 0,
            "latency_ms_max": max(latencies) if latencies else 0,
            "cost_usd_sum": sum(costs) if costs else None,
        }
        if labeled:
            yt = [bool(x["human_unsafe"]) for x in labeled]
            yp = [bool(x["pred_unsafe"]) for x in labeled]
            m = prf1(yt, yp, positive=True)
            item.update(m)
            item["cohen_kappa"] = cohen_kappa(yt, yp)
        summary.append(item)

    out = args.run_dir / "metrics_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out}")

    # Простая таблица в stdout
    print("\nmodel_id\tapproach\tacc\tF1(unsafe)\tκ\tlat_mean_ms\tcost_sum_usd")
    for item in summary:
        print(
            f"{item['judge_model_id']}\t{item['approach']}\t"
            f"{item.get('accuracy', '—')}\t{item.get('f1_unsafe', '—')}\t"
            f"{item.get('cohen_kappa', '—')}\t"
            f"{item['latency_ms_mean']:.1f}\t"
            f"{item.get('cost_usd_sum', '—')}"
        )


if __name__ == "__main__":
    main()
