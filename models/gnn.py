from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv
from transformers.optimization import get_linear_schedule_with_warmup

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
        for epoch in range(epochs):
            self.train()
            for batch in train_dataloader:
                batch = batch.to(self.device)
                tool_logits, _, _, tool_mask = self.forward(batch)
                labels = batch.y[tool_mask].to(self.device)
                loss = self._compute_loss(tool_logits, labels)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                global_step += 1
                if callback is not None:
                    current_lr = scheduler.get_last_lr() if scheduler is not None else [learning_rate]
                    callback(None, global_step, None, current_lr, loss.item())

            if evaluator is not None:
                scores = evaluator.compute_metrices(self)
                metric = float(min(scores)) if isinstance(scores, Sequence) else float(scores)
                if callback is not None:
                    callback(metric, epoch, global_step)

        self.save(output_path)

    def save(self, output_path: str):
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.state_dict(),
            "config": {
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "gnn_layers": self.gnn_layers,
                "dropout_rate": self.dropout_rate,
                "weight_decay": self.weight_decay,
                "pos_weight": float(self.pos_weight.item()),
            },
        }
        torch.save(payload, output_dir / "query_aware_gnn.pt")
