from __future__ import annotations

from typing import Sequence

import logging
import torch
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer

from models.base import RetrievalModel


def _mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class Qwen3Retriever(RetrievalModel):
    """Inference-only retriever backed by HuggingFace Qwen models."""

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen3-0.6B",
        max_seq_length: int = 512,
        torch_dtype: str = "auto",
        device: str | None = None,
    ):
        super().__init__(device=device)
        self.is_sentence_transformer = False
        self.st_model = None
        try:
            self.st_model = SentenceTransformer(model_name_or_path, device=self.device)
            self.st_model.to(self.device)
            self.is_sentence_transformer = True
            self.max_seq_length = self.st_model.max_seq_length or max_seq_length
        except Exception as err:
            logging.info(
                "Failed to load %s via SentenceTransformer (%s). Falling back to AutoModel + mean pooling.",
                model_name_or_path,
                err,
            )
            dtype = self._resolve_dtype(torch_dtype)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            dtype=dtype,
        )
            self.model.to(self.device)
            self.model.eval()
            self.max_seq_length = max_seq_length

    def _resolve_dtype(self, torch_dtype: str):
        if torch_dtype == "auto":
            if torch.cuda.is_available():
                return torch.float16
            return torch.float32
        if torch_dtype == "fp16":
            return torch.float16
        if torch_dtype == "bf16":
            return torch.bfloat16
        return torch.float32

    def encode(
        self,
        sentences: Sequence[str],
        batch_size: int = 8,
        show_progress_bar: bool = False,
        convert_to_tensor: bool = True,
        **kwargs,
    ):
        if self.is_sentence_transformer:
            return self.st_model.encode(
                sentences,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                convert_to_tensor=convert_to_tensor,
                device=self.device,
            )

        embeddings = []
        iterator = range(0, len(sentences), batch_size)
        if show_progress_bar:
            try:
                from tqdm import tqdm

                iterator = tqdm(iterator, desc="Encoding", total=(len(sentences) + batch_size - 1) // batch_size)
            except ImportError:  # pragma: no cover - tqdm optional
                pass

        with torch.no_grad():
            for start in iterator:
                batch_texts = sentences[start : start + batch_size]
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_seq_length,
                    return_tensors="pt",
                ).to(self.device)
                outputs = self.model(**inputs)
                hidden_states = outputs.last_hidden_state
                pooled = _mean_pooling(hidden_states, inputs["attention_mask"]).to(torch.float32)
                embeddings.append(pooled.cpu())

        stacked = torch.cat(embeddings, dim=0)
        if convert_to_tensor:
            return stacked
        return stacked.numpy()
