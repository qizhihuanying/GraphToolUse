from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sentence_transformers import InputExample

from utils import process_retrieval_ducoment


def _read_queries(file_path: Path) -> Dict[str, str]:
    df = pd.read_csv(file_path, sep="\t", names=["qid", "query"])
    return {row.qid: row.query for row in df.itertuples()}


def _read_qrels(file_path: Path) -> pd.DataFrame:
    return pd.read_csv(file_path, sep="\t", names=["qid", "useless", "docid", "label"])


def _build_relevant_docs(labels_df: pd.DataFrame) -> Dict[str, set]:
    relevant_docs: Dict[str, set] = {}
    for row in labels_df.itertuples():
        relevant_docs.setdefault(row.qid, set()).add(row.docid)
    return relevant_docs


def prepare_training_data(
    data_path: str | Path,
) -> Tuple[List[InputExample], Dict[str, str], Dict[str, str], Dict[str, set]]:
    data_dir = Path(data_path)
    documents_df = pd.read_csv(data_dir / "corpus.tsv", sep="\t")
    ir_corpus, _ = process_retrieval_ducoment(documents_df)

    train_queries = _read_queries(data_dir / "train.query.txt")
    test_queries = _read_queries(data_dir / "test.query.txt")

    train_labels_df = _read_qrels(data_dir / "qrels.train.tsv")
    train_samples: List[InputExample] = []
    for row in train_labels_df.itertuples():
        sample = InputExample(texts=[train_queries[row.qid], ir_corpus[row.docid]], label=row.label)
        train_samples.append(sample)

    test_labels_df = _read_qrels(data_dir / "qrels.test.tsv")
    relevant_docs = _build_relevant_docs(test_labels_df)
    return train_samples, test_queries, ir_corpus, relevant_docs


def load_evaluation_data(data_path: str | Path):
    data_dir = Path(data_path)
    documents_df = pd.read_csv(data_dir / "corpus.tsv", sep="\t")
    ir_corpus, _ = process_retrieval_ducoment(documents_df)
    test_queries = _read_queries(data_dir / "test.query.txt")
    test_labels_df = _read_qrels(data_dir / "qrels.test.tsv")
    relevant_docs = _build_relevant_docs(test_labels_df)
    return test_queries, ir_corpus, relevant_docs
