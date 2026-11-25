"""Exact optimizer-update and state-handling tests."""

from __future__ import annotations

import copy

import pytest
import torch

from soe import evaluate_embedding, train_soe, triplet_hinge_loss


def test_one_sgd_step_matches_manual_analytical_update() -> None:
    initial = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 0.5]], dtype=torch.float32
    )
    triplets = torch.tensor([[0, 1, 2]])
    learning_rate = 0.1
    expected_gradient = torch.tensor(
        [[-1.0, 1.0], [1.0, 0.0], [0.0, -1.0]], dtype=torch.float32
    )
    expected = initial - learning_rate * expected_gradient

    result = train_soe(
        triplets,
        n_objects=3,
        embedding_dim=2,
        optimizer="sgd",
        learning_rate=learning_rate,
        batch_size=1,
        epochs=1,
        shuffle=False,
        initial_embeddings=initial,
    )
    torch.testing.assert_close(result.final_embeddings, expected, rtol=0, atol=1e-7)


def _direct_adam_reference(
    initial: torch.Tensor,
    triplets: torch.Tensor,
    *,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    amsgrad: bool,
) -> tuple[torch.Tensor, dict]:
    parameter = torch.nn.Parameter(initial.clone())
    optimizer = torch.optim.Adam(
        [parameter], learning_rate, amsgrad=amsgrad
    )
    for _ in range(epochs):
        for start in range(0, len(triplets), batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = triplet_hinge_loss(
                parameter, triplets[start : start + batch_size], reduction="mean"
            )
            loss.backward()
            optimizer.step()
    return parameter.detach().clone(), optimizer.state_dict()


@pytest.mark.parametrize("amsgrad", [False, True])
def test_adam_successive_steps_match_torch_reference(amsgrad: bool) -> None:
    initial = torch.tensor(
        [[0.0, 0.0], [1.0, 0.1], [0.2, 0.8], [1.4, 1.1]],
        dtype=torch.float32,
    )
    triplets = torch.tensor(
        [[0, 1, 2], [1, 2, 3], [2, 0, 3], [3, 1, 0]], dtype=torch.long
    )
    expected, state = _direct_adam_reference(
        initial,
        triplets,
        learning_rate=0.03,
        epochs=3,
        batch_size=2,
        amsgrad=amsgrad,
    )
    result = train_soe(
        triplets,
        n_objects=4,
        embedding_dim=2,
        optimizer="adam",
        amsgrad=amsgrad,
        learning_rate=0.03,
        epochs=3,
        batch_size=2,
        shuffle=False,
        initial_embeddings=initial,
    )

    torch.testing.assert_close(result.final_embeddings, expected, rtol=0, atol=1e-7)
    # Six successive steps must have accumulated in one persistent state.
    only_state = next(iter(state["state"].values()))
    assert only_state["step"].item() == 6
    assert ("max_exp_avg_sq" in only_state) is amsgrad


def test_gradient_is_reset_between_sgd_batches() -> None:
    initial = torch.tensor(
        [[0.0], [2.0], [0.5], [-1.0]], dtype=torch.float32
    )
    triplets = torch.tensor([[0, 1, 2], [3, 2, 1]], dtype=torch.long)

    parameter = torch.nn.Parameter(initial.clone())
    optimizer = torch.optim.SGD([parameter], lr=0.05, momentum=0.0)
    for batch in triplets.split(1):
        optimizer.zero_grad(set_to_none=True)
        loss = triplet_hinge_loss(parameter, batch)
        loss.backward()
        optimizer.step()

    result = train_soe(
        triplets,
        n_objects=4,
        embedding_dim=1,
        optimizer="sgd",
        learning_rate=0.05,
        epochs=1,
        batch_size=1,
        shuffle=False,
        initial_embeddings=initial,
    )
    torch.testing.assert_close(
        result.final_embeddings, parameter.detach(), rtol=0, atol=1e-7
    )


@pytest.mark.parametrize("optimizer", ["sgd", "adam"])
def test_inactive_triplet_causes_no_parameter_update(optimizer: str) -> None:
    initial = torch.tensor([[0.0], [0.2], [3.0]], dtype=torch.float32)
    result = train_soe(
        torch.tensor([[0, 1, 2]]),
        n_objects=3,
        embedding_dim=1,
        optimizer=optimizer,
        amsgrad=True,
        learning_rate=0.1,
        epochs=3,
        batch_size=1,
        shuffle=False,
        initial_embeddings=initial,
    )
    torch.testing.assert_close(result.final_embeddings, initial, rtol=0, atol=0)


def _clone_optimizer_state(state_dict: dict) -> dict:
    cloned = copy.deepcopy(state_dict)
    for state in cloned["state"].values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.clone()
    return cloned


def _assert_optimizer_states_equal(first: dict, second: dict) -> None:
    assert first["param_groups"] == second["param_groups"]
    assert first["state"].keys() == second["state"].keys()
    for key in first["state"]:
        assert first["state"][key].keys() == second["state"][key].keys()
        for name, first_value in first["state"][key].items():
            second_value = second["state"][key][name]
            if isinstance(first_value, torch.Tensor):
                torch.testing.assert_close(first_value, second_value, rtol=0, atol=0)
            else:
                assert first_value == second_value


def test_evaluation_does_not_modify_parameters_or_adam_state() -> None:
    parameter = torch.nn.Parameter(
        torch.tensor([[0.0], [2.0], [0.5]], dtype=torch.float32)
    )
    triplets = torch.tensor([[0, 1, 2]])
    optimizer = torch.optim.Adam([parameter], lr=0.03, amsgrad=True)
    optimizer.zero_grad(set_to_none=True)
    triplet_hinge_loss(parameter, triplets).backward()
    optimizer.step()

    parameter_before = parameter.detach().clone()
    state_before = _clone_optimizer_state(optimizer.state_dict())
    evaluate_embedding(parameter, triplets)
    state_after = _clone_optimizer_state(optimizer.state_dict())

    torch.testing.assert_close(parameter.detach(), parameter_before, rtol=0, atol=0)
    _assert_optimizer_states_equal(state_before, state_after)


def test_user_supplied_tensor_is_not_mutated_or_reused_as_parameter() -> None:
    initial = torch.tensor([[0.0], [2.0], [0.5]], dtype=torch.float32)
    before = initial.clone()
    result = train_soe(
        torch.tensor([[0, 1, 2]]),
        n_objects=3,
        embedding_dim=1,
        optimizer="sgd",
        epochs=2,
        initial_embeddings=initial,
        shuffle=False,
    )
    torch.testing.assert_close(initial, before, rtol=0, atol=0)
    assert result.final_embeddings.data_ptr() != initial.data_ptr()
