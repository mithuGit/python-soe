"""Processed-triplet checkpoint scheduling tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from soe import train_soe


TRIPLETS = np.array(
    [[0, 1, 2], [0, 1, 3], [1, 2, 3], [2, 3, 0], [3, 0, 1]],
    dtype=np.int64,
)


def test_epoch_evaluation_behavior_is_preserved_when_triplet_interval_is_none() -> None:
    result = train_soe(
        TRIPLETS,
        n_objects=4,
        epochs=4,
        batch_size=2,
        evaluation_interval=2,
        evaluation_interval_triplets=None,
        shuffle=False,
    )
    assert [record.epoch for record in result.evaluation_history] == [0, 2, 4]
    assert [record.processed_triplets for record in result.evaluation_history] == [0, 10, 20]
    assert result.checkpoint_mappings == ()


def test_processed_schedule_uses_first_completed_step_at_or_after_threshold() -> None:
    result = train_soe(
        TRIPLETS,
        n_objects=4,
        epochs=2,
        batch_size=2,
        evaluation_interval_triplets=3,
        shuffle=False,
    )
    assert [record.processed_triplets for record in result.evaluation_history] == [0, 4, 7, 9, 10]
    assert [mapping.requested_processed_triplets for mapping in result.checkpoint_mappings] == [
        0,
        3,
        6,
        9,
    ]
    assert [mapping.actual_processed_triplets for mapping in result.checkpoint_mappings] == [
        0,
        4,
        7,
        9,
    ]
    assert [mapping.evaluation_index for mapping in result.checkpoint_mappings] == [0, 1, 2, 3]


def test_full_batch_overshoot_maps_multiple_thresholds_to_one_evaluation() -> None:
    result = train_soe(
        TRIPLETS,
        n_objects=4,
        epochs=2,
        batch_size=5,
        evaluation_interval_triplets=2,
        shuffle=False,
    )
    assert [record.processed_triplets for record in result.evaluation_history] == [0, 5, 10]
    assert [mapping.requested_processed_triplets for mapping in result.checkpoint_mappings] == [
        0,
        2,
        4,
        6,
        8,
        10,
    ]
    assert [mapping.actual_processed_triplets for mapping in result.checkpoint_mappings] == [
        0,
        5,
        5,
        10,
        10,
        10,
    ]
    assert [mapping.evaluation_index for mapping in result.checkpoint_mappings] == [
        0,
        1,
        1,
        2,
        2,
        2,
    ]


def test_final_state_is_evaluated_once_when_no_threshold_reaches_it() -> None:
    result = train_soe(
        TRIPLETS,
        n_objects=4,
        epochs=1,
        batch_size=2,
        evaluation_interval_triplets=100,
        shuffle=False,
    )
    assert [record.processed_triplets for record in result.evaluation_history] == [0, 5]
    assert [mapping.requested_processed_triplets for mapping in result.checkpoint_mappings] == [0]


def test_exact_final_threshold_does_not_duplicate_final_record() -> None:
    result = train_soe(
        TRIPLETS,
        n_objects=4,
        epochs=1,
        batch_size=2,
        evaluation_interval_triplets=5,
        shuffle=False,
    )
    assert [record.processed_triplets for record in result.evaluation_history] == [0, 5]
    assert len({record.optimizer_step for record in result.evaluation_history}) == 2
    assert result.checkpoint_mappings[-1].actual_processed_triplets == 5
    assert result.checkpoint_mappings[-1].evaluation_index == 1


def test_actual_processed_counts_are_monotonic_step_boundaries() -> None:
    result = train_soe(
        TRIPLETS,
        n_objects=4,
        epochs=3,
        batch_size=3,
        evaluation_interval_triplets=4,
        shuffle=False,
    )
    step_positions = {0, *(record.processed_triplets for record in result.step_history)}
    positions = [record.processed_triplets for record in result.evaluation_history]
    assert positions == sorted(set(positions))
    assert set(positions) <= step_positions
    assert positions[-1] == result.processed_triplets == 15


def test_separate_shuffle_seed_does_not_change_initial_state() -> None:
    initial_seed = 71
    first = train_soe(
        TRIPLETS,
        n_objects=4,
        epochs=1,
        batch_size=1,
        seed=initial_seed,
        shuffle_seed=100,
    )
    second = train_soe(
        TRIPLETS,
        n_objects=4,
        epochs=1,
        batch_size=1,
        seed=initial_seed,
        shuffle_seed=200,
    )
    torch.testing.assert_close(first.initial_embeddings, second.initial_embeddings, rtol=0, atol=0)


def test_evaluation_runtime_is_recorded_separately() -> None:
    result = train_soe(
        TRIPLETS,
        n_objects=4,
        epochs=1,
        evaluation_interval_triplets=2,
    )
    assert result.evaluation_runtime_seconds >= 0
    assert result.evaluation_runtime_seconds == pytest.approx(
        sum(record.evaluation_runtime_seconds for record in result.evaluation_history)
    )
    assert result.total_runtime_seconds >= result.training_runtime_seconds


@pytest.mark.parametrize("value", [0, -1, 1.5, "5"])
def test_invalid_processed_triplet_interval_is_rejected(value) -> None:
    with pytest.raises(ValueError, match="evaluation_interval_triplets"):
        train_soe(
            TRIPLETS,
            n_objects=4,
            epochs=1,
            evaluation_interval_triplets=value,
        )


@pytest.mark.parametrize("value", [-1, 1.5, "5"])
def test_invalid_shuffle_seed_is_rejected(value) -> None:
    with pytest.raises(ValueError, match="shuffle_seed"):
        train_soe(TRIPLETS, n_objects=4, epochs=1, shuffle_seed=value)
