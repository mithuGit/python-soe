"""Analytical and finite-difference gradient verification."""

from __future__ import annotations

import torch

from soe import triplet_hinge_loss


TRIPLET = torch.tensor([[0, 1, 2]], dtype=torch.long)


def _analytical_active_gradient(embeddings: torch.Tensor) -> torch.Tensor:
    x_i, x_j, x_k = embeddings
    d_ij = torch.linalg.vector_norm(x_i - x_j)
    d_ik = torch.linalg.vector_norm(x_i - x_k)
    grad_i = (x_i - x_j) / d_ij - (x_i - x_k) / d_ik
    grad_j = (x_j - x_i) / d_ij
    grad_k = (x_i - x_k) / d_ik
    return torch.stack((grad_i, grad_j, grad_k))


def _central_difference(
    embeddings: torch.Tensor,
    *,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    numerical = torch.empty_like(embeddings)
    for row in range(embeddings.shape[0]):
        for column in range(embeddings.shape[1]):
            plus = embeddings.detach().clone()
            minus = embeddings.detach().clone()
            plus[row, column] += epsilon
            minus[row, column] -= epsilon
            f_plus = triplet_hinge_loss(plus, TRIPLET).item()
            f_minus = triplet_hinge_loss(minus, TRIPLET).item()
            numerical[row, column] = (f_plus - f_minus) / (2.0 * epsilon)
    return numerical


def test_active_hinge_matches_independent_analytical_gradient() -> None:
    embeddings = torch.tensor(
        [[0.2, -0.3], [1.1, 0.4], [-0.5, 0.7]],
        dtype=torch.float64,
        requires_grad=True,
    )
    loss = triplet_hinge_loss(embeddings, TRIPLET, margin=1.2)
    assert loss.item() > 0.0
    loss.backward()
    expected = _analytical_active_gradient(embeddings.detach())
    torch.testing.assert_close(embeddings.grad, expected, rtol=1e-12, atol=1e-12)


def test_active_hinge_matches_central_finite_differences() -> None:
    embeddings = torch.tensor(
        [[0.2, -0.3], [1.1, 0.4], [-0.5, 0.7]],
        dtype=torch.float64,
        requires_grad=True,
    )
    loss = triplet_hinge_loss(embeddings, TRIPLET, margin=1.2)
    loss.backward()
    numerical = _central_difference(embeddings.detach())
    torch.testing.assert_close(embeddings.grad, numerical, rtol=2e-6, atol=2e-7)


def test_inactive_hinge_has_zero_analytical_and_numerical_gradient() -> None:
    embeddings = torch.tensor(
        [[0.0, 0.0], [0.2, 0.1], [3.0, 1.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    loss = triplet_hinge_loss(embeddings, TRIPLET, margin=0.5)
    assert loss.item() == 0.0
    loss.backward()
    numerical = _central_difference(embeddings.detach())
    torch.testing.assert_close(embeddings.grad, torch.zeros_like(embeddings))
    torch.testing.assert_close(numerical, torch.zeros_like(embeddings), atol=1e-12, rtol=0)
