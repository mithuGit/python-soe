"""Mathematical correctness tests for the SOE triplet hinge loss."""

from __future__ import annotations

import pytest
import torch

from soe import triplet_hinge_loss


def test_manual_active_loss_value() -> None:
    embeddings = torch.tensor([[0.0, 0.0], [3.0, 4.0], [0.0, 8.0]])
    triplets = torch.tensor([[0, 1, 2]])
    # 1 + 5 - 8 = -2, so use the reverse comparison for an active value 4.
    reverse = torch.tensor([[0, 2, 1]])
    assert triplet_hinge_loss(embeddings, triplets).item() == 0.0
    assert triplet_hinge_loss(embeddings, reverse).item() == 4.0


def test_custom_margin_exact_value() -> None:
    embeddings = torch.tensor([[0.0], [2.0], [3.5]])
    triplets = torch.tensor([[0, 1, 2]])
    assert triplet_hinge_loss(embeddings, triplets, margin=2.0).item() == 0.5


def test_satisfied_and_violated_triplets() -> None:
    embeddings = torch.tensor([[0.0], [0.5], [3.0]])
    assert triplet_hinge_loss(embeddings, torch.tensor([[0, 1, 2]])).item() == 0.0
    assert triplet_hinge_loss(embeddings, torch.tensor([[0, 2, 1]])).item() == 3.5


def test_sum_and_mean_reductions_for_multiple_triplets() -> None:
    embeddings = torch.tensor([[0.0], [0.5], [2.0], [4.0]])
    triplets = torch.tensor([[0, 1, 2], [0, 2, 1], [1, 2, 3]])
    # Per-triplet values: 0, 2.5, 0.
    assert triplet_hinge_loss(embeddings, triplets, reduction="sum").item() == 2.5
    assert triplet_hinge_loss(embeddings, triplets, reduction="mean").item() == pytest.approx(2.5 / 3.0)


@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_single_triplet_reductions_agree(reduction: str) -> None:
    embeddings = torch.tensor([[0.0], [2.0], [1.0]])
    triplets = torch.tensor([[0, 1, 2]])
    assert triplet_hinge_loss(embeddings, triplets, reduction=reduction).item() == 2.0


def test_non_contiguous_embeddings() -> None:
    backing = torch.tensor(
        [
            [0.0, 99.0, 0.0, 99.0],
            [1.0, 99.0, 0.0, 99.0],
            [0.0, 99.0, 3.0, 99.0],
        ]
    )
    embeddings = backing[:, ::2]
    assert not embeddings.is_contiguous()
    actual = triplet_hinge_loss(embeddings, torch.tensor([[0, 1, 2]]))
    assert actual.item() == 0.0


def test_coinciding_points_preserve_exact_objective() -> None:
    embeddings = torch.zeros((3, 2), dtype=torch.float64, requires_grad=True)
    loss = triplet_hinge_loss(embeddings, torch.tensor([[0, 1, 2]]))
    assert loss.item() == 1.0
    loss.backward()
    torch.testing.assert_close(embeddings.grad, torch.zeros_like(embeddings))
    assert torch.isfinite(embeddings.grad).all()


def test_one_zero_distance_has_finite_gradient() -> None:
    embeddings = torch.tensor(
        [[0.0, 0.0], [0.0, 0.0], [0.5, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    loss = triplet_hinge_loss(embeddings, torch.tensor([[0, 1, 2]]))
    assert loss.item() == 0.5
    loss.backward()
    assert torch.isfinite(embeddings.grad).all()
    # The coincident near-distance term contributes PyTorch's zero norm
    # subgradient, so the near point receives no gradient.
    torch.testing.assert_close(embeddings.grad[1], torch.zeros(2, dtype=torch.float64))


def test_hinge_boundary_uses_active_side_subgradient() -> None:
    embeddings = torch.tensor(
        [[0.0], [1.0], [2.0]], dtype=torch.float64, requires_grad=True
    )
    loss = triplet_hinge_loss(embeddings, torch.tensor([[0, 1, 2]]))
    assert loss.item() == 0.0
    loss.backward()
    assert torch.isfinite(embeddings.grad).all()
    # torch.clamp_min uses derivative 1 at its lower boundary. The expected
    # active-side gradient is therefore [0, +1, -1] for this configuration.
    torch.testing.assert_close(
        embeddings.grad,
        torch.tensor([[0.0], [1.0], [-1.0]], dtype=torch.float64),
    )


def test_translation_invariance() -> None:
    embeddings = torch.tensor(
        [[-0.2, 0.5], [1.0, 0.4], [0.3, 2.1], [-1.0, -0.5]],
        dtype=torch.float64,
    )
    triplets = torch.tensor([[0, 1, 2], [3, 0, 1], [2, 1, 3]])
    translation = torch.tensor([11.3, -7.8], dtype=torch.float64)
    original = triplet_hinge_loss(embeddings, triplets, reduction="sum")
    translated = triplet_hinge_loss(
        embeddings + translation, triplets, reduction="sum"
    )
    torch.testing.assert_close(original, translated, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize(
    "transform",
    [
        torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float64),
        torch.tensor([[-1.0, 0.0], [0.0, 1.0]], dtype=torch.float64),
    ],
)
def test_orthogonal_rotation_and_reflection_invariance(transform: torch.Tensor) -> None:
    embeddings = torch.tensor(
        [[-0.2, 0.5], [1.0, 0.4], [0.3, 2.1], [-1.0, -0.5]],
        dtype=torch.float64,
    )
    triplets = torch.tensor([[0, 1, 2], [3, 0, 1], [2, 1, 3]])
    original = triplet_hinge_loss(embeddings, triplets, reduction="sum")
    transformed = triplet_hinge_loss(
        embeddings @ transform.T, triplets, reduction="sum"
    )
    torch.testing.assert_close(original, transformed, rtol=1e-14, atol=1e-14)


def test_fixed_margin_breaks_scale_invariance() -> None:
    embeddings = torch.tensor([[0.0], [1.0], [1.5]], dtype=torch.float64)
    triplets = torch.tensor([[0, 1, 2]])
    original = triplet_hinge_loss(embeddings, triplets)
    scaled = triplet_hinge_loss(2.0 * embeddings, triplets)
    assert original.item() == 0.5
    assert scaled.item() == 0.0
