"""CPU reproducibility tests excluding runtime measurements."""

from __future__ import annotations

import torch

from soe import train_soe
from soe.initialization import initialize_embeddings


TRIPLETS = torch.tensor(
    [[0, 1, 2], [0, 1, 3], [1, 2, 3], [2, 3, 0], [3, 0, 1]],
    dtype=torch.long,
)


def _step_signature(result) -> list[tuple[int, int, int, int, float]]:
    return [
        (
            record.epoch,
            record.optimizer_step,
            record.processed_triplets,
            record.batch_size,
            record.training_step_loss,
        )
        for record in result.step_history
    ]


def _evaluation_signature(result) -> list[tuple]:
    return [
        (
            record.epoch,
            record.optimizer_step,
            record.processed_triplets,
            record.training.loss,
            record.training.ordinal_error,
            None if record.validation is None else record.validation.loss,
            None if record.validation is None else record.validation.ordinal_error,
            record.selection_loss,
        )
        for record in result.evaluation_history
    ]


def test_random_initialization_is_seeded_and_does_not_use_global_rng() -> None:
    device = torch.device("cpu")
    torch.manual_seed(999)
    global_before = torch.random.get_rng_state().clone()
    first = initialize_embeddings(5, 2, seed=17, device=device)
    global_after = torch.random.get_rng_state().clone()
    second = initialize_embeddings(5, 2, seed=17, device=device)
    different = initialize_embeddings(5, 2, seed=18, device=device)

    torch.testing.assert_close(global_before, global_after, rtol=0, atol=0)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert not torch.equal(first, different)


def test_random_initialization_training_is_exactly_reproducible_on_cpu() -> None:
    kwargs = dict(
        triplets=TRIPLETS,
        n_objects=4,
        embedding_dim=2,
        optimizer="adam",
        amsgrad=True,
        learning_rate=0.02,
        batch_size=2,
        epochs=5,
        shuffle=True,
        seed=73,
        device="cpu",
        validation_triplets=torch.tensor([[0, 2, 3], [1, 3, 0]]),
    )
    first = train_soe(**kwargs)
    second = train_soe(**kwargs)

    torch.testing.assert_close(first.initial_embeddings, second.initial_embeddings, rtol=0, atol=0)
    torch.testing.assert_close(first.best_embeddings, second.best_embeddings, rtol=0, atol=0)
    torch.testing.assert_close(first.final_embeddings, second.final_embeddings, rtol=0, atol=0)
    assert _step_signature(first) == _step_signature(second)
    assert _evaluation_signature(first) == _evaluation_signature(second)
    assert first.best_epoch == second.best_epoch
    assert first.best_optimizer_step == second.best_optimizer_step


def test_external_initialization_training_is_exactly_reproducible_on_cpu() -> None:
    initial = torch.tensor(
        [[0.0, 0.0], [1.0, 0.2], [0.1, 1.1], [1.3, 1.0]],
        dtype=torch.float32,
    )
    kwargs = dict(
        triplets=TRIPLETS,
        n_objects=4,
        embedding_dim=2,
        initial_embeddings=initial,
        optimizer="sgd",
        learning_rate=0.04,
        batch_size=2,
        epochs=5,
        shuffle=True,
        seed=31,
        device="cpu",
    )
    first = train_soe(**kwargs)
    second = train_soe(**kwargs)

    torch.testing.assert_close(first.initial_embeddings, initial, rtol=0, atol=0)
    torch.testing.assert_close(first.final_embeddings, second.final_embeddings, rtol=0, atol=0)
    assert _step_signature(first) == _step_signature(second)
    assert _evaluation_signature(first) == _evaluation_signature(second)


def test_runtime_fields_are_finite_but_excluded_from_reproducibility_contract() -> None:
    result = train_soe(
        TRIPLETS,
        n_objects=4,
        epochs=2,
        batch_size=2,
        seed=4,
    )
    assert result.training_runtime_seconds >= 0.0
    assert result.total_runtime_seconds >= result.training_runtime_seconds
    assert all(record.training_runtime_seconds >= 0.0 for record in result.step_history)
    assert all(record.total_runtime_seconds >= 0.0 for record in result.evaluation_history)
