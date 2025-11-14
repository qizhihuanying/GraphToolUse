from __future__ import annotations

import json
import logging
import os
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import numpy as np
import torch
from sklearn.metrics import ndcg_score
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from transformers.trainer_callback import PrinterCallback, ProgressCallback

from models.bert_sentence_transformer import BertSentenceTransformer
from models.gnn import QueryAwareGNN
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
            return
        if not getattr(args, "disable_tqdm", False) and getattr(state, "is_world_process_zero", True):
            logs_copy = dict(logs)
            logs_copy.pop("total_flos", None)
            logging.info(json.dumps(logs_copy, ensure_ascii=False))

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
    if model_type == "gnn":
        return QueryAwareGNN(
            input_dim=model_config["input_dim"],
            hidden_dim=model_config.get("hidden_dim", 768),
            gnn_layers=model_config.get("gnn_layers", 2),
            dropout_rate=model_config.get("dropout_rate", 0.1),
            pos_weight=model_config.get("pos_weight", 1.0),
            device=device,
            weight_decay=model_config.get("weight_decay", 0.01),
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
    """兼容SentenceTransformer与图模型的统一Evaluator，用于输出NDCG@K。"""

    def __init__(
        self,
        queries: Optional[Dict[str, str]] = None,
        corpus: Optional[Dict[str, str]] = None,
        relevant_docs: Optional[Dict[str, Set[str]]] = None,
        corpus_chunk_size: int = 5,
        show_progress_bar: bool = True,
        batch_size: int = 1,
        score_function=None,
        graph_dataset=None,
        graph_batch_size: int = 1,
        device: Optional[str] = None,
        top_k: Sequence[int] = (1, 3, 5),
    ):
        self.mode = "graph" if graph_dataset is not None else "retrieval"
        self.top_k = tuple(top_k)
        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size
        self.corpus_chunk_size = corpus_chunk_size

        if self.mode == "graph":
            try:
                from torch_geometric.loader import DataLoader as GraphDataLoader
            except ImportError as err:  # pragma: no cover - 仅在缺少PyG时触发
                raise ImportError("请先安装torch_geometric，再运行gnn评估。") from err

            self.graph_dataset = graph_dataset
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.graph_loader = GraphDataLoader(
                graph_dataset,
                batch_size=graph_batch_size,
                shuffle=False,
            )
            self.graph_batch_size = graph_batch_size
            self.relevant_docs = None
            self.corpus = None
            self.corpus_ids = None
            self.queries = None
            self.queries_id = None
            self.score_function = None
            return

        if queries is None or corpus is None or relevant_docs is None:
            raise ValueError("检索评估必须提供queries/corpus/relevant_docs。")

        from sentence_transformers.util import cos_sim

        self.queries_id = list(queries.keys())
        self.queries = [queries[qid] for qid in self.queries_id]
        self.corpus_ids = list(corpus.keys())
        self.corpus = [corpus[cid] for cid in self.corpus_ids]
        self.relevant_docs = relevant_docs
        self.corpus_chunk_size = corpus_chunk_size
        self.score_function = score_function or cos_sim

    def __call__(self, model, output_path: str | None = None, epoch: int = -1, steps: int = -1, *args, **kwargs) -> float:
        out_txt = ":"
        if epoch != -1:
            out_txt = f" after epoch {epoch}:" if steps == -1 else f" in epoch {epoch} after {steps} steps:"
        task_name = "Graph Tool Selection" if self.mode == "graph" else "Information Retrieval"
        logging.info("%s Evaluation%s", task_name, out_txt)

        ndcg_scores = self.compute_metrices(model)
        logging.info("Average NDCG@1: %.2f", ndcg_scores[0] * 100)
        logging.info("Average NDCG@3: %.2f", ndcg_scores[1] * 100)
        logging.info("Average NDCG@5: %.2f", ndcg_scores[2] * 100)
        return float(min(ndcg_scores))

    def compute_metrices(self, model) -> Dict[int, float]:
        if self.mode == "graph":
            return self._compute_graph_metrics(model)
        return self._compute_retrieval_metrics(model)

    def _compute_graph_metrics(self, model) -> Dict[int, float]:
        model.eval()
        ndcg_lists: Dict[int, List[float]] = {k: [] for k in self.top_k}

        last_detail_message = None
        with torch.no_grad():
            for batch in self.graph_loader:
                batch = batch.to(self.device)
                tool_logits, query_logits, tool_batch_index, tool_mask = model(batch)
                labels = batch.y[tool_mask]
                tool_scores = tool_logits.detach().cpu()
                labels_cpu = labels.detach().cpu()
                batch_indices = tool_batch_index.detach().cpu()

                unique_graphs = batch_indices.unique(sorted=True).tolist()
                for graph_id in unique_graphs:
                    graph_mask = batch_indices == graph_id
                    if not torch.any(graph_mask):
                        continue
                    y_true = labels_cpu[graph_mask].numpy()
                    y_pred = tool_scores[graph_mask].numpy()
                    if y_true.size == 0:
                        continue
                    for k in self.top_k:
                        score = ndcg_score([y_true], [y_pred], k=k)
                        ndcg_lists[k].append(float(score))

                if hasattr(model, "describe_last_graph"):
                    detail = model.describe_last_graph(
                        batch,
                        tool_logits,
                        labels,
                        query_logits,
                        tool_batch_index,
                    )
                    if detail:
                        last_detail_message = detail

        if last_detail_message:
            logging.info("Graph eval sample detail: %s", last_detail_message)

        return [float(np.mean(ndcg_lists[k])) if ndcg_lists[k] else 0.0 for k in self.top_k]

    def _compute_retrieval_metrics(self, model) -> Dict[int, float]:
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
        for k in self.top_k:
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
        self.results_path = Path(results_path) if results_path else None
        self.results = {
            "per_epoch_eval": [],
            "best_eval": None,
            "final_metrics": {},
        }

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

        def log_callback(*callback_args):
            if len(callback_args) == 5:
                _, global_step, _, current_lr, loss_value = callback_args
                logs_writer.add_scalar("train_loss", loss_value, global_step)
                logs_writer.add_scalar("lr", current_lr[0], global_step)
            elif len(callback_args) == 3:
                value, epoch, steps = callback_args
                if isinstance(value, dict):
                    metrics = value.get("metrics") or {}
                    aggregate = value.get("aggregate")
                    avg_ndcg = value.get("avg_ndcg", aggregate)
                    entry = {
                        "epoch": (epoch + 1) if epoch is not None else None,
                        "step": steps,
                        "split": value.get("split", "eval"),
                        "metrics": metrics,
                        "avg_ndcg": avg_ndcg,
                    }
                    if metrics:
                        for name, metric_value in metrics.items():
                            logs_writer.add_scalar(f"eval/{name}", metric_value, steps)
                    if aggregate is not None:
                        logs_writer.add_scalar("eval_metric", aggregate, steps)
                    self._record_epoch_metrics(entry)
                else:
                    logs_writer.add_scalar("eval_metric", value, steps)
            else:
                logging.debug("Unexpected callback args: %s", callback_args)

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
        self.results.setdefault("final_metrics", {})[split] = metrics
        self._write_results_file()

    def _record_epoch_metrics(self, entry: dict):
        self.results.setdefault("per_epoch_eval", []).append(entry)
        self._update_best_eval()
        self._write_results_file()

    def _update_best_eval(self):
        history = self.results.get("per_epoch_eval") or []
        best_entry = None
        best_value = float("-inf")
        for item in history:
            avg = item.get("avg_ndcg")
            if avg is None:
                continue
            if avg > best_value:
                best_value = avg
                best_entry = item
        self.results["best_eval"] = best_entry

    def _write_results_file(self):
        if self.results_path is None:
            return
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        with self.results_path.open("w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
