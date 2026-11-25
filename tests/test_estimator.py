"""Tests for the public SOE estimator and its delegation to the trainer."""

from __future__ import annotations

import numpy as np
import pytest

from soe import SOE, train_soe


TRIPLETS = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 0], [3, 0, 1], [0, 2, 3]])


@pytest.mark.parametrize(
    ("requested", "count", "resolved", "step_sizes"),
    [
        (None, 5, 5, [5]),
        (None, 129, 128, [128, 1]),
        ("full", 129, 129, [129]),
        (1, 5, 1, [1, 1, 1, 1, 1]),
        (2, 5, 2, [2, 2, 1]),
        (5, 5, 5, [5]),
    ],
)
def test_batch_modes(requested, count, resolved, step_sizes) -> None:
    triplets = np.resize(TRIPLETS, (count, 3))
    model = SOE(learning_rate=0.1, batch_size=requested, max_epochs=1)
    assert model.fit(triplets, n_objects=4) is model
    assert model.batch_size_ == resolved
    assert model.batch_size == requested
    assert model.processed_triplets_ == count
    assert model.optimizer_steps_ == len(step_sizes)
    assert [record.batch_size for record in model.step_history_] == step_sizes


def test_explicit_oversized_batch_is_rejected() -> None:
    with pytest.raises(ValueError, match="batch_size cannot exceed"):
        SOE(learning_rate=0.1, batch_size=128).fit(TRIPLETS, n_objects=4)


def test_outer_api_requires_rate_and_object_count_and_has_no_test_input() -> None:
    with pytest.raises(TypeError, match="learning_rate"):
        SOE()
    model = SOE(learning_rate=0.1, max_epochs=1)
    with pytest.raises(AttributeError):
        _ = model.embedding_
    with pytest.raises(TypeError, match="n_objects"):
        model.fit(TRIPLETS)
    with pytest.raises(TypeError, match="test_triplets"):
        model.fit(TRIPLETS, n_objects=4, test_triplets=TRIPLETS)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"learning_rate": None}, "learning_rate"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"learning_rate": float("inf")}, "learning_rate"),
        ({"optimizer": "other"}, "optimizer"),
        ({"amsgrad": True}, "amsgrad"),
        ({"batch_size": True}, "batch_size"),
        ({"batch_size": "auto"}, "batch_size"),
        ({"batch_size": 0}, "batch_size"),
        ({"n_components": 0}, "n_components"),
        ({"max_epochs": 0}, "max_epochs"),
        ({"margin": -1.0}, "margin"),
        ({"reduction": "median"}, "reduction"),
        ({"random_state": -1}, "random_state"),
        ({"shuffle_random_state": 2**64}, "shuffle_random_state"),
        ({"evaluation_interval_triplets": 0}, "evaluation_interval_triplets"),
        ({"ordinal_mode": "invalid"}, "ordinal_mode"),
        ({"device": "cuda"}, "cpu"),
    ],
)
def test_invalid_options_are_rejected_at_construction(options, message) -> None:
    with pytest.raises(ValueError, match=message):
        SOE(**{"learning_rate": 0.1, **options})


def test_bad_indices_fail_with_unmodified_input() -> None:
    triplets = np.array([[0, 1, 4]])
    with pytest.raises(IndexError, match="outside"):
        SOE(learning_rate=0.1, max_epochs=1).fit(triplets, n_objects=4)
    np.testing.assert_array_equal(triplets, [[0, 1, 4]])


def test_validation_selection_and_embedding_ownership() -> None:
    training = np.array([[0, 1, 2]])
    validation = np.array([[0, 2, 1]])
    initial = np.array([[0.0], [2.0], [1.0], [9.0]], dtype=np.float32)
    model = SOE(learning_rate=0.1, max_epochs=2, n_components=1)

    selected = model.fit_transform(
        training, n_objects=4, validation_triplets=validation, initial_embedding=initial
    )
    assert selected.shape == (4, 1)
    assert selected.dtype == np.float32
    assert model.n_objects_ == 4
    assert model.n_observed_objects_ == 3
    assert model.n_validation_triplets_ == 1
    assert model.selection_source_ == "validation"
    assert model.best_epoch_ == model.best_optimizer_step_ == 0
    assert model.best_checkpoint_ is model.history_[0]
    assert model.history_[-1].training.loss < model.history_[0].training.loss
    np.testing.assert_array_equal(selected, initial)
    assert not np.array_equal(selected, model.final_embedding_)
    assert model.final_embedding_[3, 0] == initial[3, 0]

    selected[:] = -1
    np.testing.assert_array_equal(model.embedding_, initial)
    model.embedding_[:] = -2
    np.testing.assert_array_equal(model.best_embedding_, initial)
    np.testing.assert_array_equal(model.initial_embedding_, initial)
    np.testing.assert_array_equal(model.result_.best_embeddings.numpy(), initial)
    np.testing.assert_array_equal(training, [[0, 1, 2]])
    np.testing.assert_array_equal(validation, [[0, 2, 1]])
    np.testing.assert_array_equal(initial, [[0.0], [2.0], [1.0], [9.0]])


def test_refit_replaces_state_and_uses_training_selection_without_validation() -> None:
    model = SOE(learning_rate=0.1, max_epochs=2, n_components=1)
    initial = [[0.0], [2.0], [1.0]]
    model.fit(
        [[0, 1, 2]], n_objects=3, validation_triplets=[[0, 2, 1]], initial_embedding=initial
    )
    first_result = model.result_
    first_history = model.history_
    assert model.best_epoch_ == 0

    model.fit([[0, 1, 2]], n_objects=3, initial_embedding=initial)
    assert model.result_ is not first_result
    assert model.history_ is not first_history
    assert model.selection_source_ == "training"
    assert model.n_validation_triplets_ == 0
    assert model.best_epoch_ == 2
    assert model.optimizer_steps_ == 2
    assert all(record.validation is None for record in model.history_)
    np.testing.assert_array_equal(model.embedding_, model.final_embedding_)


def test_adam_options_and_resolved_seeds_match_the_trainer() -> None:
    options = dict(
        optimizer="adam", amsgrad=True, learning_rate=0.02, batch_size=2,
        evaluation_interval_triplets=3, reduction="sum",
    )
    model = SOE(
        max_epochs=2, random_state=17, shuffle_random_state=29, **options
    ).fit(TRIPLETS, n_objects=4)
    expected = train_soe(TRIPLETS, n_objects=4, epochs=2, seed=17, shuffle_seed=29, **options)
    np.testing.assert_array_equal(model.final_embedding_, expected.final_embeddings.numpy())
    np.testing.assert_array_equal(model.embedding_, expected.best_embeddings.numpy())
    assert model.checkpoint_mappings_ == expected.checkpoint_mappings
    assert model.reproducibility_["seed"] == 17
    assert model.reproducibility_["shuffle_seed"] == 29
    assert model.reproducibility_["amsgrad"] is True
    assert model.optimizer_ == "adam"
    assert model.learning_rate_ == 0.02
    assert model.device_ == "cpu"
