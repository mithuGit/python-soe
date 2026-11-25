"""Shared input validation for the public SOE functions."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}


def resolve_device(device: str | torch.device) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if resolved.type not in {"cpu", "cuda"}:
        raise ValueError("device must be a CPU or CUDA device")
    return resolved


def prepare_embeddings(
    embeddings: Any,
    *,
    device: torch.device | None = None,
    name: str = "embeddings",
) -> torch.Tensor:
    if isinstance(embeddings, torch.Tensor):
        tensor = embeddings if device is None else embeddings.to(device)
    else:
        tensor = torch.as_tensor(embeddings, device=device)

    if not tensor.is_floating_point():
        raise TypeError(f"{name} must contain floating-point values")
    if tensor.ndim != 2:
        raise ValueError(f"{name} must have shape (n_objects, embedding_dim)")
    if tensor.shape[0] < 1 or tensor.shape[1] < 1:
        raise ValueError(f"{name} must have non-zero object and dimension axes")
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"{name} must contain only finite values")
    return tensor


def as_triplet_tensor(triplets: Any, *, name: str = "triplets") -> torch.Tensor:
    """Check triplet shape and dtype before making an owned training copy."""
    source = triplets.detach() if isinstance(triplets, torch.Tensor) else torch.as_tensor(triplets)
    if source.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"{name} must contain integer indices")
    if source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(f"{name} must have shape (n_triplets, 3)")
    if source.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one triplet")
    return source


def prepare_triplets(
    triplets: Any,
    *,
    n_objects: int,
    device: torch.device,
    name: str = "triplets",
) -> torch.Tensor:
    source = as_triplet_tensor(triplets, name=name)
    # Own the training tensor even if the source already has the target dtype and device.
    tensor = source.to(device=device, dtype=torch.long).clone()
    if torch.any(tensor < 0).item() or torch.any(tensor >= n_objects).item():
        raise IndexError(f"{name} contains an object index outside [0, {n_objects})")

    repeated = (
        (tensor[:, 0] == tensor[:, 1])
        | (tensor[:, 0] == tensor[:, 2])
        | (tensor[:, 1] == tensor[:, 2])
    )
    if torch.any(repeated).item():
        raise ValueError(f"{name} requires three distinct object indices per row")
    return tensor


def validate_loss_options(margin: float, reduction: str) -> None:
    if not np.isfinite(margin) or margin < 0:
        raise ValueError("margin must be finite and non-negative")
    if reduction not in {"mean", "sum"}:
        raise ValueError("reduction must be 'mean' or 'sum'")


def validate_training_options(
    *,
    optimizer: str,
    learning_rate: float,
    epochs: int,
    evaluation_interval: int,
    evaluation_interval_triplets: int | None,
    shuffle_seed: int | None,
    adam_betas: tuple[float, float],
    adam_eps: float,
) -> None:
    if optimizer not in {"sgd", "adam"}:
        raise ValueError("optimizer must be 'sgd' or 'adam'")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if not isinstance(evaluation_interval, int) or evaluation_interval < 1:
        raise ValueError("evaluation_interval must be a positive integer")
    if evaluation_interval_triplets is not None and (
        not isinstance(evaluation_interval_triplets, int)
        or evaluation_interval_triplets < 1
    ):
        raise ValueError("evaluation_interval_triplets must be a positive integer or None")
    if shuffle_seed is not None and (
        not isinstance(shuffle_seed, int) or shuffle_seed < 0
    ):
        raise ValueError("shuffle_seed must be a non-negative integer or None")
    if (
        not isinstance(adam_betas, tuple)
        or len(adam_betas) != 2
        or any(not np.isfinite(value) or value < 0 or value >= 1 for value in adam_betas)
    ):
        raise ValueError("adam_betas must contain two finite values in [0, 1)")
    if not np.isfinite(adam_eps) or adam_eps <= 0:
        raise ValueError("adam_eps must be finite and positive")
