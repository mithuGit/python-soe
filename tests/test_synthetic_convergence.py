"""Cheap deterministic end-to-end optimization tests."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest
import torch

from soe import ordinal_error, train_soe


def _known_geometry() -> tuple[np.ndarray, np.ndarray]:
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, 0.0],
            [0.0, 2.0],
            [2.0, 2.0],
        ],
        dtype=np.float32,
    )
    triplets: list[tuple[int, int, int]] = []
    for anchor in range(len(points)):
        others = [index for index in range(len(points)) if index != anchor]
        for first, second in combinations(others, 2):
            first_distance = np.linalg.norm(points[anchor] - points[first])
            second_distance = np.linalg.norm(points[anchor] - points[second])
            if np.isclose(first_distance, second_distance):
                continue
            if first_distance < second_distance:
                triplets.append((anchor, first, second))
            else:
                triplets.append((anchor, second, first))
    return points, np.asarray(triplets, dtype=np.int64)


@pytest.mark.parametrize(
    ("optimizer", "learning_rate", "epochs", "amsgrad"),
    [
        ("sgd", 0.1, 120, False),
        ("adam", 0.05, 80, True),
    ],
)
def test_tiny_known_geometry_converges_reproducibly(
    optimizer: str,
    learning_rate: float,
    epochs: int,
    amsgrad: bool,
) -> None:
    ground_truth, triplets = _known_geometry()
    assert len(triplets) == 50
    assert ordinal_error(torch.from_numpy(ground_truth), triplets) == 0.0

    kwargs = dict(
        triplets=triplets,
        n_objects=len(ground_truth),
        embedding_dim=2,
        optimizer=optimizer,
        amsgrad=amsgrad,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=16,
        shuffle=True,
        seed=17,
        evaluation_interval=epochs,
    )
    first = train_soe(**kwargs)
    second = train_soe(**kwargs)

    initial = first.evaluation_history[0].training
    final = first.evaluation_history[-1].training
    assert final.loss < 0.15 * initial.loss
    assert final.ordinal_error < initial.ordinal_error
    assert final.ordinal_error <= 0.05
    assert torch.isfinite(first.final_embeddings).all()
    torch.testing.assert_close(
        first.final_embeddings, second.final_embeddings, rtol=0, atol=0
    )
    assert [record.training_step_loss for record in first.step_history] == [
        record.training_step_loss for record in second.step_history
    ]
