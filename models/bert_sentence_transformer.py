from __future__ import annotations

from typing import Sequence

from sentence_transformers import SentenceTransformer, losses

from models.base import RetrievalModel


class BertSentenceTransformer(RetrievalModel):
    """SentenceTransformer-backed retriever used for both training and evaluation."""

    def __init__(
        self,
        model_name_or_path: str,
        max_seq_length: int = 256,
        device: str | None = None,
    ):
        super().__init__(device=device)
        self._model = SentenceTransformer(model_name_or_path)
        if max_seq_length is not None:
            self._model.max_seq_length = max_seq_length
        self._model.to(self.device)

    @property
    def sentence_transformer(self) -> SentenceTransformer:
        return self._model

    @property
    def is_trainable(self) -> bool:
        return True

    def get_loss(self):
        return losses.MultipleNegativesRankingLoss(self._model)

    def fit(
        self,
        train_dataloader,
        train_loss,
        evaluator,
        epochs: int,
        warmup_steps: int,
        learning_rate: float,
        output_path: str,
        callback=None,
    ):
        self._model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=epochs,
            warmup_steps=warmup_steps,
            optimizer_params={"lr": learning_rate},
            output_path=output_path,
            callback=callback,
        )

    def encode(
        self,
        sentences: Sequence[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_tensor: bool = True,
        **kwargs,
    ):
        return self._model.encode(
            sentences,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_tensor=convert_to_tensor,
        )
