"""Reproducible embedding initialization."""

from __future__ import annotations

from typing import Any

import torch

from .validation import prepare_embeddings


def initialize_embeddings(
    n_objects: int,
    embedding_dim: int,
    *,
    seed: int,
    device: torch.device,
    initial_embeddings: Any | None = None,
) -> torch.Tensor:
    """Create float32 embeddings without changing caller-owned inputs."""

    if not isinstance(n_objects, int) or n_objects < 1:
        raise ValueError("n_objects must be a positive integer")
    if not isinstance(embedding_dim, int) or embedding_dim < 1:
        raise ValueError("embedding_dim must be a positive integer")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    if initial_embeddings is None:
        # Generate on CPU for a stable, device-independent initialization path,
        # then transfer to the requested device.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        values = torch.rand(
            (n_objects, embedding_dim),
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        ).to(device)
    else:
        try:
            values = torch.as_tensor(initial_embeddings, dtype=torch.float32, device=device)
        except (TypeError, ValueError) as exc:
            raise TypeError("initial_embeddings must be numeric") from exc
        values = prepare_embeddings(values, name="initial_embeddings")
        if values.shape != (n_objects, embedding_dim):
            raise ValueError(
                "initial_embeddings must have shape "
                f"({n_objects}, {embedding_dim})"
            )
        values = values.detach().clone()

    if not torch.isfinite(values).all().item():
        raise ValueError("initial_embeddings must contain only finite values")
    return values
