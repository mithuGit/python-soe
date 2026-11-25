"""Shared loss and ordinal-error evaluation for SOE embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch

from .validation import prepare_embeddings, prepare_triplets, validate_loss_options
from .losses import _triplet_hinge_values


OrdinalMode = Literal["ties_as_errors", "reference"]


@dataclass(frozen=True)
class EvaluationMetrics:
    """Complete-dataset metrics computed on one fixed embedding state."""

    loss: float
    ordinal_error: float
    n_triplets: int


def _ordinal_error_prepared(
    embeddings: torch.Tensor,
    triplets: torch.Tensor,
    mode: OrdinalMode,
) -> torch.Tensor:
    points = embeddings[triplets]
    near_squared = torch.sum((points[:, 0] - points[:, 1]) ** 2, dim=1)
    far_squared = torch.sum((points[:, 0] - points[:, 2]) ** 2, dim=1)
    if mode == "ties_as_errors":
        errors = near_squared >= far_squared
    elif mode == "reference":
        # evaluate-OE uses d(i,k) < d(i,j), so exact ties are not errors.
        errors = far_squared < near_squared
    else:
        raise ValueError("mode must be 'ties_as_errors' or 'reference'")
    return errors.to(dtype=torch.float64).mean()


def ordinal_error(
    embeddings: Any,
    triplets: Any,
    *,
    mode: OrdinalMode = "ties_as_errors",
) -> float:
    """Return the fraction of violated ordinal comparisons.

    The default treats ties as errors: ``d(i,j) >= d(i,k)``. The ``reference``
    mode matches evaluate-OE's strict comparison and treats ties as correct.
    Squared distances are used because they preserve Euclidean ordering.
    """

    embedding_tensor = prepare_embeddings(embeddings)
    triplet_tensor = prepare_triplets(
        triplets,
        n_objects=embedding_tensor.shape[0],
        device=embedding_tensor.device,
    )
    with torch.no_grad():
        return _ordinal_error_prepared(embedding_tensor, triplet_tensor, mode).item()


def evaluate_embedding(
    embeddings: Any,
    triplets: Any,
    *,
    margin: float = 1.0,
    reduction: str = "mean",
    ordinal_mode: OrdinalMode = "ties_as_errors",
) -> EvaluationMetrics:
    """Compute complete loss and ordinal error on a single embedding state."""

    validate_loss_options(margin, reduction)
    embedding_tensor = prepare_embeddings(embeddings)
    triplet_tensor = prepare_triplets(
        triplets,
        n_objects=embedding_tensor.shape[0],
        device=embedding_tensor.device,
    )
    with torch.no_grad():
        values = _triplet_hinge_values(embedding_tensor, triplet_tensor, margin)
        loss = values.mean() if reduction == "mean" else values.sum()
        error = _ordinal_error_prepared(embedding_tensor, triplet_tensor, ordinal_mode)
    return EvaluationMetrics(
        loss=loss.item(),
        ordinal_error=error.item(),
        n_triplets=triplet_tensor.shape[0],
    )
