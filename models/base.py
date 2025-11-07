from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import torch


class RetrievalModel(ABC):
    """Abstract interface for retrieval models used by training and evaluation pipelines."""

    def __init__(self, device: str | None = None):
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def device(self) -> str:
        return self._device

    @property
    def is_trainable(self) -> bool:
        return False

    @abstractmethod
    def encode(
        self,
        sentences: Sequence[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_tensor: bool = True,
    ):
        """Return embeddings for the provided sentences."""

    def encode_queries(
        self,
        queries: Sequence[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_tensor: bool = True,
    ):
        return self.encode(
            sentences=queries,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_tensor=convert_to_tensor,
        )

    def encode_corpus(
        self,
        corpus_texts: Sequence[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_tensor: bool = True,
    ):
        return self.encode(
            sentences=corpus_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_tensor=convert_to_tensor,
        )
