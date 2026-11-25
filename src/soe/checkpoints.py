"""Checkpoint selection and processed-triplet threshold mapping."""

from __future__ import annotations

import torch

from .results import CheckpointMapping, EvaluationRecord


class BestCheckpoint:
    """Keep an independent snapshot of the earliest minimum-loss checkpoint."""

    def __init__(self, record: EvaluationRecord, embeddings: torch.Tensor) -> None:
        self.record = record
        self.embeddings = embeddings.detach().clone()

    def consider(self, record: EvaluationRecord, embeddings: torch.Tensor) -> None:
        if record.selection_loss < self.record.selection_loss:
            self.record = record
            self.embeddings = embeddings.detach().clone()


def map_completed_thresholds(
    next_threshold: int,
    interval: int,
    processed_triplets: int,
    evaluation_index: int,
) -> tuple[list[CheckpointMapping], int]:
    """Map crossed thresholds to the same completed optimizer step."""
    mappings = []
    while next_threshold <= processed_triplets:
        mappings.append(
            CheckpointMapping(
                requested_processed_triplets=next_threshold,
                actual_processed_triplets=processed_triplets,
                evaluation_index=evaluation_index,
            )
        )
        next_threshold += interval
    return mappings, next_threshold
