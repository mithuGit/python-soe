"""Loss functions for triplet-based Soft Ordinal Embedding."""

from __future__ import annotations

from typing import Any

import torch

from .validation import prepare_embeddings, prepare_triplets, validate_loss_options


def _triplet_hinge_values(
    embeddings: torch.Tensor,
    triplets: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    points = embeddings[triplets]
    near_distances = torch.linalg.vector_norm(points[:, 0] - points[:, 1], dim=1)
    far_distances = torch.linalg.vector_norm(points[:, 0] - points[:, 2], dim=1)
    return torch.clamp_min(margin + near_distances - far_distances, 0.0)


def triplet_hinge_loss(
    embeddings: Any,
    triplets: Any,
    *,
    margin: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute the unsquared SOE triplet hinge loss.

    For each ``(i, j, k)`` row, the loss is
    ``max(0, margin + ||x_i-x_j||_2 - ||x_i-x_k||_2)``.
    No epsilon or smoothing is introduced, so this function implements the
    stated mathematical objective exactly. PyTorch chooses a zero subgradient
    for the Euclidean norm at coinciding points. At an exact hinge boundary,
    ``torch.clamp_min`` uses derivative 1, i.e. the active-side subgradient.
    """

    validate_loss_options(margin, reduction)
    embedding_tensor = prepare_embeddings(embeddings)
    triplet_tensor = prepare_triplets(
        triplets,
        n_objects=embedding_tensor.shape[0],
        device=embedding_tensor.device,
    )
    values = _triplet_hinge_values(embedding_tensor, triplet_tensor, margin)
    return values.mean() if reduction == "mean" else values.sum()
