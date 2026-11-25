"""Ordinal evaluation and best-state selection tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from soe import evaluate_embedding, ordinal_error, train_soe


def test_manual_ordinal_errors_and_tie_policies() -> None:
    embeddings = torch.tensor([[0.0], [1.0], [3.0], [-1.0]])
    triplets = torch.tensor(
        [
            [0, 1, 2],  # correct: 1 < 3
            [0, 2, 1],  # error: 3 > 1
            [0, 1, 3],  # tie: 1 == 1
        ]
    )
    assert ordinal_error(embeddings, triplets) == pytest.approx(2.0 / 3.0)
    assert ordinal_error(embeddings, triplets, mode="reference") == pytest.approx(1.0 / 3.0)


def test_evaluation_returns_manual_loss_error_and_count() -> None:
    embeddings = torch.tensor([[0.0], [0.5], [2.0], [-0.5]])
    triplets = torch.tensor([[0, 1, 2], [0, 2, 1], [0, 1, 3]])
    metrics = evaluate_embedding(embeddings, triplets, reduction="sum")
    # Losses are 0, 2.5, 1.0; ordinal errors are false, true, true (tie).
    assert metrics.loss == pytest.approx(3.5)
    assert metrics.ordinal_error == pytest.approx(2.0 / 3.0)
    assert metrics.n_triplets == 3


def test_squared_and_euclidean_distances_have_same_ordering() -> None:
    generator = torch.Generator().manual_seed(29)
    embeddings = torch.randn((8, 3), generator=generator, dtype=torch.float64)
    triplets = torch.tensor(
        [[0, 1, 2], [3, 4, 5], [6, 1, 7], [2, 5, 0], [7, 3, 4]]
    )
    points = embeddings[triplets]
    near = torch.linalg.vector_norm(points[:, 0] - points[:, 1], dim=1)
    far = torch.linalg.vector_norm(points[:, 0] - points[:, 2], dim=1)
    torch.testing.assert_close(near.square(), torch.sum((points[:, 0] - points[:, 1]) ** 2, dim=1))
    assert torch.equal(near >= far, near.square() >= far.square())
    expected = (near >= far).to(torch.float64).mean().item()
    assert ordinal_error(embeddings, triplets) == expected


def test_validation_loss_alone_selects_best_embedding() -> None:
    initial = torch.tensor([[0.0], [2.0], [1.0]], dtype=torch.float32)
    training = torch.tensor([[0, 1, 2]])
    validation = torch.tensor([[0, 2, 1]])
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

    initial_record = result.evaluation_history[0]
    final_record = result.evaluation_history[-1]
    assert final_record.training.loss < initial_record.training.loss
    assert final_record.validation is not None
    assert initial_record.validation is not None
    assert final_record.validation.loss > initial_record.validation.loss
    assert all(
        record.selection_loss == record.validation.loss
        for record in result.evaluation_history
        if record.validation is not None
    )
    assert result.best_epoch == 0
    torch.testing.assert_close(result.best_embeddings, initial, rtol=0, atol=0)


def test_training_loss_selects_best_only_without_validation() -> None:
    initial = torch.tensor([[0.0], [2.0], [1.0]], dtype=torch.float32)
    training = torch.tensor([[0, 1, 2]])
    result = train_soe(
        training,
        n_objects=3,
        embedding_dim=1,
        initial_embeddings=initial,
        optimizer="sgd",
        learning_rate=0.1,
        batch_size=1,
        epochs=2,
        shuffle=False,
    )
    assert all(record.validation is None for record in result.evaluation_history)
    assert all(
        record.selection_loss == record.training.loss
        for record in result.evaluation_history
    )
    assert result.best_epoch == 2
    torch.testing.assert_close(
        result.best_embeddings, result.final_embeddings, rtol=0, atol=0
    )


def test_initial_checkpoint_remains_best_on_equal_selection_losses() -> None:
    initial = torch.tensor([[0.0], [0.2], [3.0]], dtype=torch.float32)
    result = train_soe(
        torch.tensor([[0, 1, 2]]),
        n_objects=3,
        embedding_dim=1,
        initial_embeddings=initial,
        optimizer="sgd",
        epochs=3,
        shuffle=False,
    )
    assert all(record.selection_loss == 0.0 for record in result.evaluation_history)
    assert result.best_epoch == 0
    assert result.best_optimizer_step == 0
    torch.testing.assert_close(result.best_embeddings, initial, rtol=0, atol=0)


def test_evaluation_frequency_does_not_change_optimizer_result() -> None:
    initial = torch.tensor(
        [[0.0, 0.0], [1.0, 0.2], [0.1, 1.1], [1.3, 1.0]],
        dtype=torch.float32,
    )
    triplets = torch.tensor(
        [[0, 1, 2], [1, 2, 3], [2, 0, 3], [3, 1, 0]], dtype=torch.long
    )
    common = dict(
        triplets=triplets,
        n_objects=4,
        embedding_dim=2,
        initial_embeddings=initial,
        optimizer="adam",
        amsgrad=True,
        learning_rate=0.02,
        batch_size=2,
        epochs=4,
        shuffle=False,
    )
    every_epoch = train_soe(**common, evaluation_interval=1)
    final_only = train_soe(**common, evaluation_interval=4)
    torch.testing.assert_close(
        every_epoch.final_embeddings, final_only.final_embeddings, rtol=0, atol=0
    )


def test_validation_input_and_embedding_are_not_changed_by_evaluation() -> None:
    embeddings = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float64)
    triplets = np.array([[0, 1, 2]], dtype=np.int64)
    embeddings_before = embeddings.clone()
    triplets_before = triplets.copy()
    evaluate_embedding(embeddings, triplets)
    torch.testing.assert_close(embeddings, embeddings_before, rtol=0, atol=0)
    np.testing.assert_array_equal(triplets, triplets_before)


def test_invalid_ordinal_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="mode"):
        ordinal_error(
            torch.tensor([[0.0], [1.0], [2.0]]),
            torch.tensor([[0, 1, 2]]),
            mode="invalid",  # type: ignore[arg-type]
        )
