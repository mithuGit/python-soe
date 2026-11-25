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


def prepare_triplets(
    triplets: Any,
    *,
    n_objects: int,
    device: torch.device,
    name: str = "triplets",
) -> torch.Tensor:
    if isinstance(triplets, torch.Tensor):
        source = triplets.detach()
    elif isinstance(triplets, np.ndarray):
        source = torch.as_tensor(triplets)
    else:
        source = torch.as_tensor(triplets)

    if source.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"{name} must contain integer indices")
    if source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(f"{name} must have shape (n_triplets, 3)")
    if source.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one triplet")

    # Clone before any device transfer so caller-owned arrays and tensors can
    # never be changed by shuffling or other internal operations.
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
