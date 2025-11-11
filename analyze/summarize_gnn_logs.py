#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="汇总 log/GNN-* 目录下的 GNN 训练结果（ndcg@k）。"
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=Path("log") / "GNN-G3",
        help="包含 run.log/results.json 的目录，默认 log/GNN-G3。",
    )
    parser.add_argument(
        "--sort-by",
        choices=("avg_ndcg", "ndcg@1", "ndcg@3", "ndcg@5"),
        default="avg_ndcg",
        help="结果排序依据（默认 Average NDCG）。",
    )
    parser.add_argument(
        "--ascending",
        action="store_true",
        help="按升序排序，默认按降序排列（指标高在前）。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="只展示前 N 条记录，默认展示全部。",
    )
    return parser.parse_args()


@dataclass
class RunRecord:
    run_dir: Path
    params: Dict[str, str]
    metrics: Dict[str, float]


AVERAGE_METRIC_NAMES = ("ndcg@1", "ndcg@3", "ndcg@5")


def parse_params_from_name(run_dir: Path) -> Dict[str, str]:
    """解析诸如 lr=1e-4+bs=32+epoch=1+layers=2 的目录名。"""
    params: Dict[str, str] = {}
    for chunk in run_dir.name.split("+"):
        if "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        params[key.strip()] = value.strip()
    return params


def load_results_json(results_path: Path) -> Optional[Dict[str, float]]:
    try:
        payload = json.loads(results_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    for split in ("eval", "test"):
        metrics = payload.get(split)
        if isinstance(metrics, dict):
            try:
                return {k.lower(): float(v) for k, v in metrics.items()}
            except (TypeError, ValueError):
                return None
    return None


def collect_runs(log_root: Path) -> List[RunRecord]:
    runs: List[RunRecord] = []
    for child in sorted(log_root.iterdir()):
        if not child.is_dir():
            continue
        results_path = child / "results.json"
        if not results_path.exists():
            continue
        metrics = load_results_json(results_path)
        if not metrics:
            continue
        runs.append(
            RunRecord(
                run_dir=child,
                params=parse_params_from_name(child),
                metrics=metrics,
            )
        )
    return runs


def format_metric(metrics: Dict[str, float], name: str) -> str:
    value = metrics.get(name.lower())
    return f"{value:.4f}" if value is not None else "-"


def format_value(value: Optional[float]) -> str:
    return f"{value:.4f}" if value is not None else "-"


def average_ndcg(metrics: Dict[str, float]) -> Optional[float]:
    values = []
    for metric_name in AVERAGE_METRIC_NAMES:
        metric_value = metrics.get(metric_name)
        if metric_value is None:
            return None
        values.append(metric_value)
    return sum(values) / len(values)


def resolve_sort_metric(record: RunRecord, metric_name: str) -> Optional[float]:
    if metric_name == "avg_ndcg":
        return average_ndcg(record.metrics)
    return record.metrics.get(metric_name)


def render_table(records: List[RunRecord], sort_by: str, ascending: bool, limit: int) -> None:
    if not records:
        print("未找到可用的 GNN 训练结果（缺少 results.json）。", file=sys.stderr)
        sys.exit(1)

    sort_key = sort_by.lower()

    def sort_value(record: RunRecord) -> float:
        metric_value = resolve_sort_metric(record, sort_key)
        return metric_value if metric_value is not None else float("-inf")

    records.sort(key=sort_value, reverse=not ascending)

    if limit > 0:
        records = records[:limit]

    headers = [
        "Rank",
        "LR",
        "Layers",
        "Epochs",
        "ndcg@1",
        "ndcg@3",
        "ndcg@5",
        "Avg NDCG",
        "Run Directory",
    ]
    rows: List[List[str]] = []
    for idx, record in enumerate(records, start=1):
        params = record.params
        rows.append(
            [
                str(idx),
                params.get("lr", "?"),
                params.get("layers", "?"),
                params.get("epoch", params.get("epochs", "?")),
                format_metric(record.metrics, "ndcg@1"),
                format_metric(record.metrics, "ndcg@3"),
                format_metric(record.metrics, "ndcg@5"),
                format_value(average_ndcg(record.metrics)),
                str(record.run_dir),
            ]
        )

    col_widths = [max(len(row[col]) for row in ([headers] + rows)) for col in range(len(headers))]

    def print_row(values: List[str]) -> None:
        print(" | ".join(val.ljust(col_widths[i]) for i, val in enumerate(values)))

    print_row(headers)
    print("-+-".join("-" * w for w in col_widths))
    for row in rows:
        print_row(row)

    best = records[0]
    key_metric = resolve_sort_metric(best, sort_key)
    if key_metric is not None:
        print(
            f"\nTop run by {sort_by}: {best.run_dir.name} "
            f"({sort_by}={key_metric:.4f}, layers={best.params.get('layers', '?')}, "
            f"lr={best.params.get('lr', '?')}, epochs={best.params.get('epoch', best.params.get('epochs', '?'))})"
        )


def main() -> None:
    args = parse_args()
    log_root = args.log_root
    if not log_root.exists():
        print(f"指定的日志目录不存在: {log_root}", file=sys.stderr)
        sys.exit(1)

    records = collect_runs(log_root)
    render_table(records, sort_by=args.sort_by.lower(), ascending=args.ascending, limit=args.limit)


if __name__ == "__main__":
    main()
