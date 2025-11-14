from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv
from transformers.optimization import get_linear_schedule_with_warmup
from tqdm.auto import tqdm

from models.base import RetrievalModel


class QueryAwareGNN(RetrievalModel, nn.Module):
    """查询感知的GCN，实现端到端的多标签工具选择。"""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int = 768,
        gnn_layers: int = 2,
        dropout_rate: float = 0.1,
        pos_weight: float = 1.0,
        device: Optional[str] = None,
        weight_decay: float = 0.01,
    ):
        RetrievalModel.__init__(self, device=device)
        nn.Module.__init__(self)

        if input_dim is None or input_dim <= 0:
            raise ValueError("input_dim必须是正整数")
        if gnn_layers <= 0:
            raise ValueError("gnn_layers必须>=1")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gnn_layers = gnn_layers
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay

        self.embedding_aligner = nn.Linear(input_dim, hidden_dim)
        self.gnn_layers_list = nn.ModuleList(
            [GCNConv(hidden_dim, hidden_dim) for _ in range(gnn_layers)]
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.output_mlp = nn.Linear(hidden_dim, 1)

        self.register_buffer("pos_weight", torch.tensor(float(pos_weight), dtype=torch.float32))
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @property
    def is_trainable(self) -> bool:  # type: ignore[override]
        return True

    def get_loss(self):  # 与Trainer接口保持一致
        return None

    def encode(  # type: ignore[override]
        self,
        sentences,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_tensor: bool = True,
        **kwargs,
    ):
        raise NotImplementedError("QueryAwareGNN 不支持 encode 操作，请通过predict接口进行推理。")

    def forward(self, batch: Batch):
        if batch.x is None:
            raise ValueError("batch.x不能为空")

        x = batch.x.to(self.device)
        edge_index = batch.edge_index.to(self.device)
        edge_weight = getattr(batch, "edge_weight", None)
        if edge_weight is not None:
            edge_weight = edge_weight.to(self.device)

        projected = self.embedding_aligner(x)
        x = projected

        for conv in self.gnn_layers_list:
            residual = x
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            x = self.dropout(x)
            if residual.shape == x.shape:
                x = x + residual

        node_type = getattr(batch, "node_type", None)
        if node_type is not None:
            node_type = node_type.to(x.device)
            query_mask = node_type == 0
            tool_mask = node_type == 1
        else:
            query_mask, tool_mask = self._create_masks_from_batch(batch)

        tool_embeddings = x[tool_mask]
        query_embeddings = x[query_mask]

        tool_logits = self.output_mlp(tool_embeddings).squeeze(-1)
        query_logits = self.output_mlp(query_embeddings).squeeze(-1)
        tool_batch_index = batch.batch[tool_mask]

        return tool_logits, query_logits, tool_batch_index, tool_mask

    def _create_masks_from_batch(self, batch: Batch):
        batch_idx = batch.batch
        ptr = getattr(batch, "ptr", None)
        if ptr is None:
            raise ValueError("Batch缺少ptr信息，无法推断查询节点位置。")
        query_mask = torch.zeros_like(batch_idx, dtype=torch.bool)
        query_mask[ptr[:-1]] = True
        tool_mask = ~query_mask
        return query_mask, tool_mask

    def predict(self, batch: Batch, threshold: float = 0.5, use_dynamic_threshold: bool = False):
        self.eval()
        with torch.no_grad():
            batch = batch.to(self.device)
            tool_logits, query_logits, tool_batch_index, _ = self.forward(batch)
            tool_probs = torch.sigmoid(tool_logits)
            if use_dynamic_threshold and len(query_logits) > 0:
                query_probs = torch.sigmoid(query_logits)
                thresholds = query_probs[tool_batch_index]
                predictions = (tool_probs > thresholds).float()
            else:
                predictions = (tool_probs > threshold).float()
        return predictions

    def describe_last_graph(
        self,
        batch: Batch,
        tool_logits: torch.Tensor,
        labels: Optional[torch.Tensor],
        query_logits: torch.Tensor,
        tool_batch_index: torch.Tensor,
    ) -> Optional[str]:
        summary = self._collect_last_graph_summary(batch, tool_logits, labels, query_logits, tool_batch_index)
        if summary is None:
            return None

        probs = summary["probabilities"]
        preview_count = 5
        formatted_probs = ", ".join(f"{score:.3f}" for score in probs[:preview_count])
        if len(probs) > preview_count:
            formatted_probs += ", ..."

        parts = [
            f"apis={summary['num_tools']}",
            f"pred_probs=[{formatted_probs}]",
            f"threshold={summary['threshold']:.3f}",
            f"pred_pos={summary['predicted_positive']}",
        ]
        hit_count = summary.get("hit_count")
        positive_total = summary.get("positive_total")
        if positive_total is not None:
            parts.append(f"gold_pos={positive_total}")
        if hit_count is not None:
            parts.append(f"tp={hit_count}")
        return " ".join(parts)

    def _collect_last_graph_summary(
        self,
        batch: Batch,
        tool_logits: torch.Tensor,
        labels: Optional[torch.Tensor],
        query_logits: torch.Tensor,
        tool_batch_index: torch.Tensor,
    ) -> Optional[dict]:
        if batch.num_graphs == 0:
            return None
        last_graph_id = int(batch.num_graphs - 1)

        tool_batch_cpu = tool_batch_index.detach().cpu()
        selection = tool_batch_cpu == last_graph_id
        if not torch.any(selection):
            return None

        logits_cpu = tool_logits.detach().cpu()
        logits_graph = logits_cpu[selection]
        probabilities = torch.sigmoid(logits_graph)

        threshold = self._extract_threshold_for_graph(batch, query_logits, last_graph_id)
        predictions = (probabilities > threshold).float()
        summary = {
            "num_tools": int(probabilities.shape[0]),
            "probabilities": probabilities.tolist(),
            "threshold": float(threshold),
            "predicted_positive": int(predictions.sum().item()),
        }

        if labels is not None:
            labels_cpu = labels.detach().cpu()
            labels_graph = labels_cpu[selection]
            positive_mask = labels_graph > 0.5
            summary["positive_total"] = int(positive_mask.sum().item())
            summary["hit_count"] = int(((predictions == 1.0) & positive_mask).sum().item())

        return summary

    def _extract_threshold_for_graph(self, batch: Batch, query_logits: torch.Tensor, graph_id: int) -> float:
        if query_logits is None or query_logits.numel() == 0:
            return 0.5
        query_mask = self._get_query_mask(batch)
        query_indices = batch.batch[query_mask].detach().cpu()
        query_probs = torch.sigmoid(query_logits.detach().cpu())
        mask = query_indices == graph_id
        if not torch.any(mask):
            return 0.5
        return float(query_probs[mask][-1].item())

    def _get_query_mask(self, batch: Batch) -> torch.Tensor:
        node_type = getattr(batch, "node_type", None)
        if node_type is not None:
            return node_type == 0
        query_mask, _ = self._create_masks_from_batch(batch)
        return query_mask

    def _compute_loss(self, tool_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        pos_weight = self.pos_weight.to(tool_logits.device)
        return F.binary_cross_entropy_with_logits(tool_logits, labels, pos_weight=pos_weight)

    def fit(  # type: ignore[override]
        self,
        *,
        train_dataloader,
        train_loss,
        evaluator,
        epochs: int,
        warmup_steps: int,
        learning_rate: float,
        output_path: str,
        callback=None,
    ):
        if train_dataloader is None:
            raise ValueError("训练GNN需要提供图DataLoader。")
        self.to(self.device)
        optimizer = AdamW(self.parameters(), lr=learning_rate, weight_decay=self.weight_decay)
        total_steps = epochs * len(train_dataloader)
        scheduler = None
        if total_steps > 0:
            scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        global_step = 0
        best_metric = float('-inf')
        best_state = None

        for epoch in range(epochs):
            self.train()
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch in progress_bar:
                batch = batch.to(self.device)
                tool_logits, query_logits, tool_batch_index, tool_mask = self.forward(batch)
                labels = batch.y[tool_mask].to(self.device)
                loss = self._compute_loss(tool_logits, labels)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                global_step += 1
                if global_step % 50 == 0:
                    detail_msg = self.describe_last_graph(
                        batch,
                        tool_logits,
                        labels,
                        query_logits,
                        tool_batch_index,
                    )
                    log_msg = f"epoch={epoch + 1} step={global_step} loss={loss.item():.4f}"
                    if detail_msg:
                        log_msg = f"{log_msg} | {detail_msg}"
                    logging.info(log_msg)
                if callback is not None:
                    current_lr = scheduler.get_last_lr() if scheduler is not None else [learning_rate]
                    callback(None, global_step, None, current_lr, loss.item())

            if evaluator is not None:
                scores = evaluator.compute_metrices(self)
                metric_pairs: List[tuple] = []
                metric_dict = {}
                if isinstance(scores, Sequence):
                    top_k = getattr(evaluator, "top_k", tuple(range(1, len(scores) + 1)))
                    for idx, k in enumerate(top_k):
                        if idx >= len(scores):
                            break
                        score_value = float(scores[idx])
                        metric_pairs.append((k, score_value))
                        metric_dict[f"ndcg@{k}"] = score_value
                    if metric_pairs:
                        metric_text = ", ".join(
                            f"ndcg@{k}={score:.4f}" for k, score in metric_pairs
                        )
                        logging.info("Epoch %d eval metrics: %s", epoch + 1, metric_text)
                aggregate_metric = (
                    float(np.mean(list(metric_dict.values()))) if metric_dict else float(scores)
                )
                if aggregate_metric > best_metric:
                    best_metric = aggregate_metric
                    best_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                    logging.info("New best metric %.4f at epoch %d", aggregate_metric, epoch + 1)
                if callback is not None:
                    callback(
                        {
                            "metrics": metric_dict,
                            "aggregate": aggregate_metric,
                            "avg_ndcg": aggregate_metric,
                            "split": "eval",
                        },
                        epoch,
                        global_step,
                    )

        # Save final model
        self.save(output_path)

        if best_state is not None:
            best_dir = Path(output_path) / "best_model"
            best_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": best_state, "config": self._export_config()}, best_dir / "query_aware_gnn.pt")

    def save(self, output_path: str):
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.state_dict(),
            "config": self._export_config(),
        }
        torch.save(payload, output_dir / "query_aware_gnn.pt")

    def _export_config(self):
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "gnn_layers": self.gnn_layers,
            "dropout_rate": self.dropout_rate,
            "weight_decay": self.weight_decay,
            "pos_weight": float(self.pos_weight.item()),
        }
