from __future__ import annotations

import json
import logging
import os
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Optional, Sequence, Set

import numpy as np
from sklearn.metrics import ndcg_score
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from transformers.trainer_callback import PrinterCallback, ProgressCallback

from models.bert_sentence_transformer import BertSentenceTransformer
from models.qwen3 import Qwen3Retriever

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


_TRAINER_CALLBACK_PATCHED = False


def _patch_trainer_logging_callbacks():
    global _TRAINER_CALLBACK_PATCHED
    if _TRAINER_CALLBACK_PATCHED:
        return

    original_printer_on_log = PrinterCallback.on_log
    original_progress_on_log = ProgressCallback.on_log

    def patched_printer_on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        if getattr(state, "is_local_process_zero", True):
            logs_copy = dict(logs)
            logs_copy.pop("total_flos", None)
            logging.info(json.dumps(logs_copy, ensure_ascii=False))

    def patched_progress_on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return original_progress_on_log(self, args, state, control, logs, **kwargs)
        logs_copy = dict(logs)
        result = original_progress_on_log(self, args, state, control, logs, **kwargs)
        if not getattr(args, "disable_tqdm", False) and getattr(state, "is_world_process_zero", True):
            logs_copy.pop("total_flos", None)
            logging.info(json.dumps(logs_copy, ensure_ascii=False))
        return result

    PrinterCallback.on_log = patched_printer_on_log
    ProgressCallback.on_log = patched_progress_on_log
    _TRAINER_CALLBACK_PATCHED = True


def get_model(model_config):
    model_type = model_config.get("model_type", "sentence_transformer")
    device = model_config.get("device")
    if model_type == "sentence_transformer":
        return BertSentenceTransformer(
            model_name_or_path=model_config["model_name_or_path"],
            max_seq_length=model_config.get("max_seq_length", 256),
            device=device,
        )
    if model_type == "qwen3":
        return Qwen3Retriever(
            model_name_or_path=model_config.get("model_name_or_path", "Qwen/Qwen3-0.6B"),
            max_seq_length=model_config.get("max_seq_length", 512),
            torch_dtype=model_config.get("torch_dtype", "auto"),
            device=device,
        )
    raise ValueError(f"Unsupported model_type: {model_type}")


def compute_ndcg_for_query(query_tuple):
    _, query_id, top_hits, relevant_docs, corpus_ids, k = query_tuple
    query_relevant_docs = relevant_docs[query_id]

    true_relevance = np.zeros(len(corpus_ids))
    predicted_scores = np.zeros(len(corpus_ids))

    for hit in top_hits:
        idx = corpus_ids.index(hit["corpus_id"])
        predicted_scores[idx] = hit["score"]
        if hit["corpus_id"] in query_relevant_docs:
            true_relevance[idx] = 1

    return ndcg_score([true_relevance], [predicted_scores], k=k)


class APIEvaluator:
    """SentenceTransformer-compatible evaluator for retrieval tasks."""

    def __init__(
        self,
        queries: Dict[str, str],
        corpus: Dict[str, str],
        relevant_docs: Dict[str, Set[str]],
        corpus_chunk_size: int = 5,
        show_progress_bar: bool = True,
        batch_size: int = 1,
        score_function=None,
    ):
        from sentence_transformers.util import cos_sim

        self.queries_id = list(queries.keys())
        self.queries = [queries[qid] for qid in self.queries_id]
        self.corpus_ids = list(corpus.keys())
        self.corpus = [corpus[cid] for cid in self.corpus_ids]
        self.relevant_docs = relevant_docs
        self.corpus_chunk_size = corpus_chunk_size
        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size
        self.score_function = score_function or cos_sim

    def __call__(self, model, output_path: str | None = None, epoch: int = -1, steps: int = -1, *args, **kwargs) -> float:
        out_txt = ":"
        if epoch != -1:
            out_txt = f" after epoch {epoch}:" if steps == -1 else f" in epoch {epoch} after {steps} steps:"
        logging.info("Information Retrieval Evaluation%s", out_txt)

        ndcg_scores = self.compute_metrices(model)
        logging.info("Average NDCG@1: %.2f", ndcg_scores[0] * 100)
        logging.info("Average NDCG@3: %.2f", ndcg_scores[1] * 100)
        logging.info("Average NDCG@5: %.2f", ndcg_scores[2] * 100)
        return float(min(ndcg_scores))

    def compute_metrices(self, model) -> Dict[int, float]:
        query_embeddings = model.encode(
            self.queries,
            show_progress_bar=self.show_progress_bar,
            batch_size=self.batch_size,
            convert_to_tensor=True,
        )

        queries_result_list = [[] for _ in range(len(query_embeddings))]

        for corpus_start_idx in trange(
            0,
            len(self.corpus),
            self.corpus_chunk_size,
            desc="Corpus Chunks",
            disable=not self.show_progress_bar,
        ):
            corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(self.corpus))
            sub_corpus_embeddings = model.encode(
                self.corpus[corpus_start_idx:corpus_end_idx],
                show_progress_bar=False,
                batch_size=self.batch_size,
                convert_to_tensor=True,
            )
            pair_scores = self.score_function(query_embeddings, sub_corpus_embeddings)
            pair_scores_list = pair_scores.cpu().tolist()

            for query_itr in range(len(query_embeddings)):
                for sub_corpus_id, score in enumerate(pair_scores_list[query_itr]):
                    corpus_id = self.corpus_ids[corpus_start_idx + sub_corpus_id]
                    queries_result_list[query_itr].append({"corpus_id": corpus_id, "score": score})

        scores = []
        k_list = [1, 3, 5]
        for k in k_list:
            query_tuples = []
            for query_itr in range(len(queries_result_list)):
                query_id = self.queries_id[query_itr]
                top_hits = sorted(queries_result_list[query_itr], key=lambda x: x["score"], reverse=True)
                query_tuples.append((query_itr, query_id, top_hits, self.relevant_docs, self.corpus_ids, k))

            ndcg_scores = []
            with Pool() as pool:
                iterable = pool.imap(compute_ndcg_for_query, query_tuples)
                iterator = enumerate(iterable)
                if self.show_progress_bar:
                    iterator = tqdm(iterator, total=len(query_tuples))
                for _, score in iterator:
                    ndcg_scores.append(score)
            scores.append(float(np.mean(ndcg_scores)))

        return scores


class Trainer:
    def __init__(
        self,
        *,
        model,
        dataloader=None,
        evaluator: Optional[APIEvaluator] = None,
        test_evaluator: Optional[APIEvaluator] = None,
        output_path: str,
        learning_rate: float,
        num_epochs: int,
        warmup_steps: int,
        tensorboard_dir: Path,
        results_path: Optional[Path] = None,
    ):
        self.model = model
        self.dataloader = dataloader
        self.evaluator = evaluator
        self.test_evaluator = test_evaluator
        self.output_path = output_path
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.tensorboard_dir = tensorboard_dir
        self.results_path = results_path
        self.results = {}

    def train_epoch(self):
        if self.num_epochs <= 0:
            logging.info("Skipping training because num_epochs <= 0.")
            return
        if not getattr(self.model, "is_trainable", False):
            logging.info("Skipping training because model is not trainable.")
            return
        if self.dataloader is None:
            logging.info("Skipping training because no dataloader was provided.")
            return

        _patch_trainer_logging_callbacks()
        train_loss = self.model.get_loss()
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        logs_writer = SummaryWriter(self.tensorboard_dir)

        def log_callback(train_ix, global_step, training_steps, current_lr, loss_value):
            logs_writer.add_scalar("train_loss", loss_value, global_step)
            logs_writer.add_scalar("lr", current_lr[0], global_step)

        self.model.fit(
            train_dataloader=self.dataloader,
            train_loss=train_loss,
            evaluator=self.evaluator,
            epochs=self.num_epochs,
            warmup_steps=self.warmup_steps,
            learning_rate=self.learning_rate,
            output_path=self.output_path,
            callback=log_callback,
        )
        logs_writer.close()

    def eval_epoch(self):
        return self._run_evaluator(self.evaluator, "eval")

    def test_epoch(self):
        return self._run_evaluator(self.test_evaluator, "test")

    def run(self):
        self.train_epoch()
        self.eval_epoch()
        self.test_epoch()

    def _run_evaluator(self, evaluator, split: str):
        if evaluator is None:
            logging.info("Skipping %s epoch because no evaluator is provided.", split)
            return None
        scores = evaluator.compute_metrices(self.model)
        metrics = {
            "ndcg@1": float(scores[0]),
            "ndcg@3": float(scores[1]),
            "ndcg@5": float(scores[2]),
        }
        logging.info("%s metrics: %s", split.capitalize(), json.dumps(metrics))
        self._record_results(split, metrics)
        return metrics

    def _record_results(self, split: str, metrics: dict):
        if self.results_path is None:
            return
        self.results[split] = metrics
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        with self.results_path.open("w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
