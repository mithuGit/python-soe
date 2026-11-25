"""Triplet-based Soft Ordinal Embedding."""

from .evaluation import EvaluationMetrics, evaluate_embedding, ordinal_error
from .losses import triplet_hinge_loss
from .results import CheckpointMapping, EvaluationRecord, StepRecord, TrainResult
from .training import train_soe

__all__ = [
    "CheckpointMapping",
    "EvaluationMetrics",
    "EvaluationRecord",
    "StepRecord",
    "TrainResult",
    "evaluate_embedding",
    "ordinal_error",
    "train_soe",
    "triplet_hinge_loss",
]
