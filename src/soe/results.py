"""Records and independently owned embedding snapshots from SOE training."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .evaluation import EvaluationMetrics


@dataclass(frozen=True)
class StepRecord:
    """Loss evaluated on one batch immediately before its optimizer update."""

    epoch: int
    optimizer_step: int
    processed_triplets: int
    batch_size: int
    training_step_loss: float
    training_runtime_seconds: float


@dataclass(frozen=True)
class EvaluationRecord:
    """Complete fixed-state metrics at a shared evaluation checkpoint."""

    epoch: int
    optimizer_step: int
    processed_triplets: int
    training_runtime_seconds: float
    total_runtime_seconds: float
    evaluation_runtime_seconds: float
    training: EvaluationMetrics
    validation: EvaluationMetrics | None
    selection_loss: float


@dataclass(frozen=True)
class CheckpointMapping:
    """Map a requested processed-triplet threshold to one actual evaluation."""

    requested_processed_triplets: int
    actual_processed_triplets: int
    evaluation_index: int


@dataclass(frozen=True)
class TrainResult:
    initial_embeddings: torch.Tensor
    best_embeddings: torch.Tensor
    final_embeddings: torch.Tensor
    best_epoch: int
    best_optimizer_step: int
    optimizer_steps: int
    processed_triplets: int
    training_runtime_seconds: float
    evaluation_runtime_seconds: float
    total_runtime_seconds: float
    step_history: tuple[StepRecord, ...]
    evaluation_history: tuple[EvaluationRecord, ...]
    checkpoint_mappings: tuple[CheckpointMapping, ...]
