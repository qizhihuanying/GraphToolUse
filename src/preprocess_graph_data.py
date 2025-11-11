from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, List


def _setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def _load_instruction_records(query_file: Path) -> List[dict]:
    with query_file.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, list):
        raise ValueError(f"{query_file} 的内容必须是list")
    logging.info("读取 %s，样本数=%d", query_file, len(data))
    normalized = []
    dataset_name = query_file.stem.split("_")[0]
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            logging.debug("跳过非dict记录: %s", item)
            continue
        record = dict(item)
        qid = record.get("query_id")
        if qid is None:
            qid = f"{dataset_name}_{idx}"
        record["query_id"] = str(qid)
        record.setdefault("api_list", [])
        record.setdefault("relevant APIs", [])
        record.setdefault("query", "")
        normalized.append(record)
    return normalized


def _load_eval_ids(test_query_file: Path | None) -> set:
    if test_query_file is None:
        logging.warning("未提供test_query_ids文件，默认所有样本用于训练。")
        return set()
    if not test_query_file.exists():
        logging.warning("测试ID文件不存在: %s，默认所有样本用于训练。", test_query_file)
        return set()
    with test_query_file.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    eval_ids = set()
    if isinstance(payload, dict):
        eval_ids.update(str(key) for key in payload.keys())
    elif isinstance(payload, list):
        eval_ids.update(str(item) for item in payload)
    else:
        raise ValueError(f"无法解析测试ID文件: {test_query_file}")
    logging.info("加载评估ID %s，数量=%d", test_query_file, len(eval_ids))
    return eval_ids


def _write_jsonl(records: Iterable[dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    logging.info("写入 %s，样本数=%d", output_path, count)


def parse_args():
    parser = argparse.ArgumentParser(description="预处理图神经网络所需的指令数据。")
    parser.add_argument("--query_file", type=str, required=True, help="指令数据文件，如 data/instruction/G1_query.json。")
    parser.add_argument("--test_query_ids", type=str, default=None,
                        help="测试/评估 query_id 文件，可选。如果缺失则所有样本划入训练。")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录，例如 data/retrieval_graph/G1。")
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="数据集名称（如G1）。若未提供则根据query_file推断。")
    parser.add_argument("--verbose", action="store_true", help="输出更多日志。")
    return parser.parse_args()


def main():
    args = parse_args()
    _setup_logging(args.verbose)

    query_file = Path(args.query_file)
    test_query_file = Path(args.test_query_ids) if args.test_query_ids else None
    output_dir = Path(args.output_dir)
    dataset_name = args.dataset_name or query_file.stem.split("_")[0]

    records = _load_instruction_records(query_file)
    eval_ids = _load_eval_ids(test_query_file)

    train_records = []
    eval_records = []
    for record in records:
        qid = str(record.get("query_id"))
        if qid in eval_ids:
            eval_records.append(record)
        else:
            train_records.append(record)

    logging.info("%s: train=%d, eval=%d", dataset_name, len(train_records), len(eval_records))

    _write_jsonl(train_records, output_dir / "train.jsonl")
    if eval_records:
        _write_jsonl(eval_records, output_dir / "eval.jsonl")
    else:
        logging.warning("%s 未生成eval文件，因为评估集合为空。", dataset_name)


if __name__ == "__main__":
    main()

