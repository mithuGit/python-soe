"""Shared mini-batch training loop for SGD and Adam SOE optimization."""

from __future__ import annotations

from time import perf_counter
from typing import Any, Literal

import torch

from .checkpoints import BestCheckpoint, map_completed_thresholds
from .evaluation import OrdinalMode, evaluate_embedding
from .initialization import initialize_embeddings
from .losses import _triplet_hinge_values
from .results import CheckpointMapping, EvaluationRecord, StepRecord, TrainResult
from .validation import (
    prepare_triplets,
    resolve_device,
    validate_loss_options,
    validate_training_options,
)


OptimizerName = Literal["sgd", "adam"]


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_soe(
    triplets: Any,
    n_objects: int,
    embedding_dim: int = 2,
    *,
    optimizer: OptimizerName = "sgd",
    batch_size: int = 128,
    learning_rate: float = 0.01,
    epochs: int = 100,
    margin: float = 1.0,
    reduction: str = "mean",
    shuffle: bool = True,
    seed: int = 42,
    shuffle_seed: int | None = None,
    device: str | torch.device = "cpu",
    initial_embeddings: Any | None = None,
    validation_triplets: Any | None = None,
    amsgrad: bool = False,
    adam_betas: tuple[float, float] = (0.9, 0.999),
    adam_eps: float = 1e-8,
    evaluation_interval: int = 1,
    evaluation_interval_triplets: int | None = None,
    ordinal_mode: OrdinalMode = "ties_as_errors",
) -> TrainResult:
    """Train a triplet-based SOE embedding with SGD or Adam.

    ``training_step_loss`` is a pre-update loss for one mini-batch and may mix
    different embedding states across a run. Evaluation records instead
    recompute the complete training and validation losses on one fixed current
    embedding. If validation triplets are supplied, only validation loss is
    used for best-state selection; otherwise complete training loss is used.

    Returned initial, best, and final embeddings are independent CPU tensors.
    This function has no test-triplet argument, preventing accidental use of a
    test set for model selection.

    If ``evaluation_interval_triplets`` is set, it replaces epoch-based
    scheduling. Threshold mappings record requested versus actual processed
    counts, while evaluation records contain only real completed-step states.
    """

    validate_loss_options(margin, reduction)
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    validate_training_options(
        optimizer=optimizer,
        learning_rate=learning_rate,
        epochs=epochs,
        evaluation_interval=evaluation_interval,
        evaluation_interval_triplets=evaluation_interval_triplets,
        shuffle_seed=shuffle_seed,
        adam_betas=adam_betas,
        adam_eps=adam_eps,
    )
    resolved_device = resolve_device(device)
    initial_values = initialize_embeddings(
        n_objects,
        embedding_dim,
        seed=seed,
        device=resolved_device,
        initial_embeddings=initial_embeddings,
    )
    train_triplets = prepare_triplets(
        triplets,
        n_objects=n_objects,
        device=resolved_device,
        name="triplets",
    )
    validation_tensor = None
    if validation_triplets is not None:
        validation_tensor = prepare_triplets(
            validation_triplets,
            n_objects=n_objects,
            device=resolved_device,
            name="validation_triplets",
        )

    embeddings = torch.nn.Parameter(initial_values.detach().clone())
    if optimizer == "sgd":
        torch_optimizer: torch.optim.Optimizer = torch.optim.SGD(
            [embeddings], lr=learning_rate, momentum=0.0
        )
    else:
        torch_optimizer = torch.optim.Adam(
            [embeddings], lr=learning_rate, betas=adam_betas,
            eps=adam_eps, amsgrad=amsgrad
        )

    shuffle_generator = torch.Generator(device="cpu")
    shuffle_generator.manual_seed(seed if shuffle_seed is None else shuffle_seed)
    optimizer_steps = 0
    processed_triplets = 0
    training_runtime = 0.0
    evaluation_runtime = 0.0
    step_history: list[StepRecord] = []
    evaluation_history: list[EvaluationRecord] = []
    checkpoint_mappings: list[CheckpointMapping] = []

    _synchronize(resolved_device)
    total_start = perf_counter()

    def checkpoint(epoch: int) -> EvaluationRecord:
        nonlocal evaluation_runtime
        _synchronize(resolved_device)
        evaluation_start = perf_counter()
        training_metrics = evaluate_embedding(
            embeddings,
            train_triplets,
            margin=margin,
            reduction=reduction,
            ordinal_mode=ordinal_mode,
        )
        validation_metrics = None
        if validation_tensor is not None:
            validation_metrics = evaluate_embedding(
                embeddings,
                validation_tensor,
                margin=margin,
                reduction=reduction,
                ordinal_mode=ordinal_mode,
            )
        _synchronize(resolved_device)
        current_evaluation_runtime = perf_counter() - evaluation_start
        evaluation_runtime += current_evaluation_runtime
        selection_loss = (
            validation_metrics.loss
            if validation_metrics is not None
            else training_metrics.loss
        )
        return EvaluationRecord(
            epoch=epoch,
            optimizer_step=optimizer_steps,
            processed_triplets=processed_triplets,
            training_runtime_seconds=training_runtime,
            total_runtime_seconds=perf_counter() - total_start,
            evaluation_runtime_seconds=current_evaluation_runtime,
            training=training_metrics,
            validation=validation_metrics,
            selection_loss=selection_loss,
        )

    initial_record = checkpoint(epoch=0)
    evaluation_history.append(initial_record)
    if evaluation_interval_triplets is not None:
        checkpoint_mappings.append(
            CheckpointMapping(
                requested_processed_triplets=0,
                actual_processed_triplets=0,
                evaluation_index=0,
            )
        )
    best = BestCheckpoint(initial_record, embeddings)

    n_triplets = train_triplets.shape[0]
    next_triplet_threshold = evaluation_interval_triplets
    for epoch in range(1, epochs + 1):
        if shuffle:
            order_cpu = torch.randperm(n_triplets, generator=shuffle_generator)
            order = order_cpu.to(resolved_device)
        else:
            order = torch.arange(n_triplets, device=resolved_device)

        for start in range(0, n_triplets, batch_size):
            batch_indices = order[start : start + batch_size]
            batch = train_triplets[batch_indices]

            _synchronize(resolved_device)
            step_start = perf_counter()
            torch_optimizer.zero_grad(set_to_none=True)
            values = _triplet_hinge_values(embeddings, batch, margin)
            step_loss = values.mean() if reduction == "mean" else values.sum()
            if not torch.isfinite(step_loss).item():
                raise FloatingPointError("training loss became NaN or infinite")
            step_loss.backward()
            torch_optimizer.step()
            if not torch.isfinite(embeddings).all().item():
                raise FloatingPointError("embedding update produced NaN or infinity")
            _synchronize(resolved_device)
            training_runtime += perf_counter() - step_start

            optimizer_steps += 1
            processed_triplets += batch.shape[0]
            step_history.append(
                StepRecord(
                    epoch=epoch,
                    optimizer_step=optimizer_steps,
                    processed_triplets=processed_triplets,
                    batch_size=batch.shape[0],
                    training_step_loss=step_loss.detach().item(),
                    training_runtime_seconds=training_runtime,
                )
            )

            if (
                evaluation_interval_triplets is not None
                and next_triplet_threshold is not None
                and processed_triplets >= next_triplet_threshold
            ):
                record = checkpoint(epoch)
                evaluation_history.append(record)
                mappings, next_triplet_threshold = map_completed_thresholds(
                    next_triplet_threshold,
                    evaluation_interval_triplets,
                    processed_triplets,
                    len(evaluation_history) - 1,
                )
                checkpoint_mappings.extend(mappings)
                best.consider(record, embeddings)

        if evaluation_interval_triplets is None and (
            epoch % evaluation_interval == 0 or epoch == epochs
        ):
            record = checkpoint(epoch)
            evaluation_history.append(record)
            best.consider(record, embeddings)

    if evaluation_history[-1].processed_triplets != processed_triplets:
        record = checkpoint(epochs)
        evaluation_history.append(record)
        best.consider(record, embeddings)

    _synchronize(resolved_device)
    total_runtime = perf_counter() - total_start
    return TrainResult(
        initial_embeddings=initial_values.detach().cpu().clone(),
        best_embeddings=best.embeddings.detach().cpu().clone(),
        final_embeddings=embeddings.detach().cpu().clone(),
        best_epoch=best.record.epoch,
        best_optimizer_step=best.record.optimizer_step,
        optimizer_steps=optimizer_steps,
        processed_triplets=processed_triplets,
        training_runtime_seconds=training_runtime,
        evaluation_runtime_seconds=evaluation_runtime,
        total_runtime_seconds=total_runtime,
        step_history=tuple(step_history),
        evaluation_history=tuple(evaluation_history),
        checkpoint_mappings=tuple(checkpoint_mappings),
    )
