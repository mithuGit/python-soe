"""Validation, dtype, shape, and device edge cases found during code review."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from soe import evaluate_embedding, train_soe, triplet_hinge_loss


VALID_TRIPLETS = torch.tensor([[0, 1, 2]])
VALID_EMBEDDINGS = torch.tensor([[0.0], [1.0], [2.0]])


@pytest.mark.parametrize(
    ("triplets", "error"),
    [
        (torch.empty((0, 3), dtype=torch.long), ValueError),
        (torch.tensor([[0, -1, 2]]), IndexError),
        (torch.tensor([[0, 1, 3]]), IndexError),
        (torch.tensor([[0, 0, 2]]), ValueError),
        (torch.tensor([[0.0, 1.0, 2.0]]), TypeError),
    ],
)
def test_public_loss_rejects_invalid_triplets(
    triplets: torch.Tensor, error: type[Exception]
) -> None:
    with pytest.raises(error):
        triplet_hinge_loss(VALID_EMBEDDINGS, triplets)


@pytest.mark.parametrize(
    "embeddings",
    [
        torch.tensor([[0], [1], [2]], dtype=torch.int64),
        torch.tensor([0.0, 1.0, 2.0]),
        torch.empty((0, 1)),
        torch.tensor([[0.0], [float("nan")], [2.0]]),
        torch.tensor([[0.0], [float("inf")], [2.0]]),
    ],
)
def test_public_loss_rejects_invalid_embeddings(embeddings: torch.Tensor) -> None:
    with pytest.raises((TypeError, ValueError)):
        triplet_hinge_loss(embeddings, VALID_TRIPLETS)


def test_public_loss_preserves_float64_dtype_and_gradient_connection() -> None:
    embeddings = torch.tensor(
        [[0.0], [2.0], [1.0]], dtype=torch.float64, requires_grad=True
    )
    loss = triplet_hinge_loss(embeddings, VALID_TRIPLETS)
    assert loss.dtype == torch.float64
    loss.backward()
    assert embeddings.grad is not None
    assert embeddings.grad.dtype == torch.float64


@pytest.mark.parametrize(
    "kwargs",
    [
        {"margin": -0.1},
        {"margin": float("nan")},
        {"reduction": "none"},
        {"optimizer": "rmsprop"},
        {"learning_rate": 0.0},
        {"learning_rate": float("inf")},
        {"epochs": 0},
        {"evaluation_interval": 0},
        {"device": "mps"},
    ],
)
def test_training_rejects_invalid_options(kwargs: dict) -> None:
    with pytest.raises((TypeError, ValueError)):
        train_soe(VALID_TRIPLETS, n_objects=3, **kwargs)


@pytest.mark.parametrize(
    "initial",
    [
        np.zeros((3,), dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32),
        np.array([[0.0], [1.0], [np.nan]], dtype=np.float32),
    ],
)
def test_training_rejects_invalid_initial_embedding(initial: np.ndarray) -> None:
    with pytest.raises(ValueError):
        train_soe(
            VALID_TRIPLETS,
            n_objects=3,
            embedding_dim=1,
            initial_embeddings=initial,
        )


def test_training_rejects_invalid_validation_triplets() -> None:
    with pytest.raises(ValueError, match="validation_triplets"):
        train_soe(
            VALID_TRIPLETS,
            n_objects=3,
            validation_triplets=torch.tensor([[0, 1, 1]]),
        )


def test_evaluation_rejects_nonfinite_embedding() -> None:
    with pytest.raises(ValueError, match="finite"):
        evaluate_embedding(
            torch.tensor([[0.0], [1.0], [float("inf")]]), VALID_TRIPLETS
        )


def test_external_initialization_is_intentionally_converted_to_float32() -> None:
    initial = np.array([[0.0], [2.0], [1.0]], dtype=np.float64)
    result = train_soe(
        VALID_TRIPLETS,
        n_objects=3,
        embedding_dim=1,
        initial_embeddings=initial,
        epochs=1,
        shuffle=False,
    )
    assert result.initial_embeddings.dtype == torch.float32
