"""Focused smoke tests for the Phase 2 core SOE implementation."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from soe import evaluate_embedding, train_soe, triplet_hinge_loss


TRIPLETS_FIVE = np.array(
    [[0, 1, 2], [0, 1, 3], [1, 2, 3], [2, 3, 0], [3, 0, 1]],
    dtype=np.int64,
)


def test_batch_size_one_runs() -> None:
    result = train_soe(
        TRIPLETS_FIVE,
        n_objects=4,
        optimizer="sgd",
        batch_size=1,
        epochs=1,
        seed=10,
    )
    assert result.optimizer_steps == 5
    assert result.processed_triplets == 5
    assert torch.isfinite(result.final_embeddings).all()


def test_incomplete_final_batch_processes_all_five_triplets() -> None:
    result = train_soe(
        TRIPLETS_FIVE,
        n_objects=4,
        optimizer="sgd",
        batch_size=2,
        epochs=1,
        shuffle=False,
    )
    assert result.optimizer_steps == 3
    assert result.processed_triplets == 5
    assert [record.batch_size for record in result.step_history] == [2, 2, 1]


@pytest.mark.parametrize("optimizer", ["sgd", "adam"])
def test_sgd_and_adam_produce_finite_embeddings(optimizer: str) -> None:
    result = train_soe(
        TRIPLETS_FIVE,
        n_objects=4,
        optimizer=optimizer,
        batch_size=2,
        epochs=3,
        seed=12,
        amsgrad=optimizer == "adam",
    )
    assert torch.isfinite(result.best_embeddings).all()
    assert torch.isfinite(result.final_embeddings).all()


def test_fixed_seed_reproduces_cpu_result() -> None:
    kwargs = dict(
        triplets=TRIPLETS_FIVE,
        n_objects=4,
        optimizer="adam",
        batch_size=2,
        epochs=3,
        seed=123,
        device="cpu",
        amsgrad=True,
    )
    first = train_soe(**kwargs)
    second = train_soe(**kwargs)

    torch.testing.assert_close(first.initial_embeddings, second.initial_embeddings)
    torch.testing.assert_close(first.best_embeddings, second.best_embeddings)
    torch.testing.assert_close(first.final_embeddings, second.final_embeddings)
    assert [r.training_step_loss for r in first.step_history] == [
        r.training_step_loss for r in second.step_history
    ]
    assert [r.selection_loss for r in first.evaluation_history] == [
        r.selection_loss for r in second.evaluation_history
    ]


def test_supplied_initialization_is_respected_and_input_is_not_mutated() -> None:
    initial = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=np.float32,
    )
    initial_before = initial.copy()
    triplets = TRIPLETS_FIVE.copy()
    triplets_before = triplets.copy()

    result = train_soe(
        triplets,
        n_objects=4,
        embedding_dim=2,
        initial_embeddings=initial,
        optimizer="sgd",
        batch_size=2,
        epochs=2,
        shuffle=True,
    )

    np.testing.assert_array_equal(initial, initial_before)
    np.testing.assert_array_equal(triplets, triplets_before)
    torch.testing.assert_close(result.initial_embeddings, torch.from_numpy(initial_before))


def test_complete_training_loss_is_recomputed_on_final_state() -> None:
    result = train_soe(
        TRIPLETS_FIVE,
        n_objects=4,
        optimizer="sgd",
        batch_size=2,
        epochs=2,
        shuffle=False,
        seed=5,
    )
    expected = triplet_hinge_loss(
        result.final_embeddings,
        TRIPLETS_FIVE,
        margin=1.0,
        reduction="mean",
    ).item()
    final_evaluation = result.evaluation_history[-1]
    assert final_evaluation.optimizer_step == result.optimizer_steps
    assert final_evaluation.training.n_triplets == 5
    assert final_evaluation.training.loss == pytest.approx(expected)


def test_sum_reduction_matches_evaluate_oe_reference() -> None:
    reference_dir = Path(os.environ.get("EVALUATE_OE_DIR", "external/evaluate-OE"))
    if not (reference_dir / "lib" / "soe.py").is_file():
        pytest.skip("evaluate-OE checkout not available for compatibility check")

    sys.path.insert(0, str(reference_dir.resolve()))
    from lib import soe as reference_soe

    embeddings = torch.tensor(
        [[0.0, 0.0], [0.5, 0.1], [1.2, 0.3], [-0.4, 0.8]],
        dtype=torch.float32,
    )
    triplets = torch.tensor([[0, 1, 2], [3, 1, 0], [2, 1, 3]])
    points = embeddings[triplets]

    expected = reference_soe.soe_loss(points[:, 0], points[:, 1], points[:, 2])
    actual = triplet_hinge_loss(embeddings, triplets, reduction="sum")
    torch.testing.assert_close(actual, expected)


def test_best_embedding_is_snapshot_independent_of_later_updates() -> None:
    initial = np.array([[0.0], [2.0], [1.0]], dtype=np.float32)
    training = np.array([[0, 1, 2]], dtype=np.int64)
    validation = np.array([[0, 2, 1]], dtype=np.int64)

    result = train_soe(
        training,
        n_objects=3,
        embedding_dim=1,
        initial_embeddings=initial,
        validation_triplets=validation,
        optimizer="sgd",
        learning_rate=0.1,
        batch_size=1,
        epochs=2,
        shuffle=False,
    )

    assert result.best_epoch == 0
    torch.testing.assert_close(result.best_embeddings, torch.from_numpy(initial))
    assert not torch.allclose(result.final_embeddings, result.best_embeddings)
    final_before = result.final_embeddings.clone()
    result.final_embeddings.add_(100.0)
    torch.testing.assert_close(result.best_embeddings, torch.from_numpy(initial))
    assert not torch.allclose(result.final_embeddings, final_before)


def test_coinciding_points_have_finite_pytorch_subgradient() -> None:
    embeddings = torch.tensor(
        [[0.0, 0.0], [0.0, 0.0], [0.5, 0.0]],
        requires_grad=True,
    )
    triplets = torch.tensor([[0, 1, 2]])
    loss = triplet_hinge_loss(embeddings, triplets)
    loss.backward()

    assert torch.isfinite(loss)
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()


def test_default_and_reference_tie_policies_are_distinct() -> None:
    embeddings = torch.tensor([[0.0], [1.0], [-1.0]])
    triplets = torch.tensor([[0, 1, 2]])
    default = evaluate_embedding(embeddings, triplets)
    reference = evaluate_embedding(embeddings, triplets, ordinal_mode="reference")
    assert default.ordinal_error == 1.0
    assert reference.ordinal_error == 0.0


@pytest.mark.parametrize(
    ("triplets", "error"),
    [
        (np.array([0, 1, 2]), ValueError),
        (np.array([[0, 1, 4]]), IndexError),
        (np.array([[0, 1, 1]]), ValueError),
        (np.array([[0.0, 1.0, 2.0]]), TypeError),
    ],
)
def test_triplet_validation(triplets: np.ndarray, error: type[Exception]) -> None:
    with pytest.raises(error):
        train_soe(triplets, n_objects=4, epochs=1)


def test_training_option_and_embedding_validation() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        train_soe(TRIPLETS_FIVE, n_objects=4, batch_size=0)
    with pytest.raises(ValueError, match="finite"):
        train_soe(
            TRIPLETS_FIVE,
            n_objects=4,
            initial_embeddings=np.array(
                [[0.0], [1.0], [2.0], [np.inf]], dtype=np.float32
            ),
            embedding_dim=1,
        )
