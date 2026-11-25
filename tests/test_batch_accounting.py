"""Batch coverage, counters, histories, and loss-accounting tests."""

from __future__ import annotations

from math import ceil

import numpy as np
import pytest
import torch

import soe.training as training_module
from soe import train_soe, triplet_hinge_loss


TRIPLETS = torch.tensor(
    [[0, 1, 2], [0, 1, 3], [1, 2, 3], [2, 3, 0], [3, 0, 1]],
    dtype=torch.long,
)


@pytest.mark.parametrize(
    ("batch_size", "steps_per_epoch"),
    [(1, 5), (2, 3), (5, 1), (8, 1)],
)
def test_batch_sizes_counters_and_final_batch(
    batch_size: int, steps_per_epoch: int
) -> None:
    result = train_soe(
        TRIPLETS,
        n_objects=4,
        optimizer="sgd",
        batch_size=batch_size,
        epochs=2,
        shuffle=False,
    )
    assert result.optimizer_steps == 2 * steps_per_epoch
    assert result.processed_triplets == 2 * len(TRIPLETS)
    assert len(result.step_history) == result.optimizer_steps
    assert result.step_history[-1].processed_triplets == result.processed_triplets


@pytest.mark.parametrize("shuffle", [False, True])
def test_each_triplet_is_processed_exactly_once_per_epoch(
    monkeypatch: pytest.MonkeyPatch, shuffle: bool
) -> None:
    captured: list[torch.Tensor] = []
    original = training_module._triplet_hinge_values

    def recording_values(embeddings, batch, margin):
        captured.append(batch.detach().cpu().clone())
        return original(embeddings, batch, margin)

    monkeypatch.setattr(training_module, "_triplet_hinge_values", recording_values)
    epochs = 3
    batch_size = 2
    train_soe(
        TRIPLETS,
        n_objects=4,
        optimizer="sgd",
        batch_size=batch_size,
        epochs=epochs,
        shuffle=shuffle,
        seed=91,
    )

    batches_per_epoch = ceil(len(TRIPLETS) / batch_size)
    assert len(captured) == epochs * batches_per_epoch
    expected_multiset = sorted(map(tuple, TRIPLETS.tolist()))
    for epoch in range(epochs):
        epoch_batches = captured[
            epoch * batches_per_epoch : (epoch + 1) * batches_per_epoch
        ]
        observed = torch.cat(epoch_batches, dim=0)
        assert sorted(map(tuple, observed.tolist())) == expected_multiset
        if not shuffle:
            torch.testing.assert_close(observed, TRIPLETS, rtol=0, atol=0)


def test_seeded_shuffle_order_is_reproducible(monkeypatch: pytest.MonkeyPatch) -> None:
    original = training_module._triplet_hinge_values

    def capture_run(seed: int) -> list[tuple[int, int, int]]:
        observed: list[tuple[int, int, int]] = []

        def recording_values(embeddings, batch, margin):
            observed.extend(map(tuple, batch.detach().cpu().tolist()))
            return original(embeddings, batch, margin)

        with monkeypatch.context() as context:
            context.setattr(training_module, "_triplet_hinge_values", recording_values)
            train_soe(
                TRIPLETS,
                n_objects=4,
                batch_size=2,
                epochs=2,
                shuffle=True,
                seed=seed,
            )
        return observed

    assert capture_run(123) == capture_run(123)
    assert capture_run(123) != capture_run(124)


def test_complete_mean_is_weighted_by_all_triplets_not_batch_means() -> None:
    initial = torch.tensor([[0.0], [0.2], [1.0], [3.0]], dtype=torch.float32)
    triplets = torch.tensor(
        [[0, 1, 2], [0, 2, 1], [1, 0, 3], [2, 1, 3], [3, 2, 0]]
    )
    full_mean = triplet_hinge_loss(initial, triplets, reduction="mean").item()
    batch_means = [
        triplet_hinge_loss(initial, triplets[start : start + 2]).item()
        for start in range(0, len(triplets), 2)
    ]
    naive_mean_of_batch_means = float(np.mean(batch_means))
    assert full_mean == pytest.approx(0.4)
    assert naive_mean_of_batch_means == pytest.approx(1.0 / 3.0)
    assert full_mean != pytest.approx(naive_mean_of_batch_means)

    result = train_soe(
        triplets,
        n_objects=4,
        embedding_dim=1,
        initial_embeddings=initial,
        optimizer="sgd",
        batch_size=2,
        epochs=1,
        shuffle=False,
    )
    assert result.evaluation_history[0].training.loss == pytest.approx(full_mean)


def test_sum_evaluation_uses_complete_current_embedding() -> None:
    initial = torch.tensor([[0.0], [2.0], [0.5], [-1.0]], dtype=torch.float32)
    result = train_soe(
        TRIPLETS,
        n_objects=4,
        embedding_dim=1,
        initial_embeddings=initial,
        optimizer="sgd",
        learning_rate=0.03,
        batch_size=2,
        epochs=2,
        reduction="sum",
        shuffle=False,
    )
    expected = triplet_hinge_loss(
        result.final_embeddings, TRIPLETS, reduction="sum"
    ).item()
    assert result.evaluation_history[-1].training.loss == pytest.approx(expected)


def test_training_step_loss_is_pre_update_and_distinct_from_final_evaluation() -> None:
    initial = torch.tensor([[0.0], [2.0], [0.5]], dtype=torch.float32)
    triplet = torch.tensor([[0, 1, 2]])
    result = train_soe(
        triplet,
        n_objects=3,
        embedding_dim=1,
        initial_embeddings=initial,
        optimizer="sgd",
        learning_rate=0.1,
        batch_size=1,
        epochs=1,
        shuffle=False,
    )
    initial_loss = result.evaluation_history[0].training.loss
    step_loss = result.step_history[0].training_step_loss
    final_loss = result.evaluation_history[-1].training.loss
    assert step_loss == pytest.approx(initial_loss)
    assert final_loss < step_loss


def test_history_counters_are_monotonic_and_exact() -> None:
    result = train_soe(
        TRIPLETS,
        n_objects=4,
        batch_size=2,
        epochs=2,
        shuffle=False,
        evaluation_interval=1,
    )
    assert [record.optimizer_step for record in result.step_history] == list(range(1, 7))
    assert [record.processed_triplets for record in result.step_history] == [2, 4, 5, 7, 9, 10]
    assert [record.epoch for record in result.evaluation_history] == [0, 1, 2]
    assert [record.optimizer_step for record in result.evaluation_history] == [0, 3, 6]
    assert [record.processed_triplets for record in result.evaluation_history] == [0, 5, 10]


def test_final_epoch_is_evaluated_when_interval_does_not_divide_epochs() -> None:
    result = train_soe(
        TRIPLETS,
        n_objects=4,
        epochs=3,
        batch_size=5,
        evaluation_interval=2,
        shuffle=False,
    )
    assert [record.epoch for record in result.evaluation_history] == [0, 2, 3]
    assert result.evaluation_history[-1].optimizer_step == result.optimizer_steps
    assert result.evaluation_history[-1].processed_triplets == result.processed_triplets
