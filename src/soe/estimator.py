"""Public estimator for CPU triplet-based Soft Ordinal Embedding."""

from __future__ import annotations

from numbers import Integral, Real
import platform
from typing import Any, Literal

import numpy as np
import torch

from .evaluation import OrdinalMode
from .training import OptimizerName, train_soe
from .validation import as_triplet_tensor, validate_loss_options, validate_training_options


_DEFAULT_BATCH_SIZE = 128


class SOE:
    """Fit a Euclidean embedding to anchored triplets `(i, j, k)`.

    The learning rate is required. On CPU, the default batch size is capped at
    the number of training triplets. Use a positive integer for a fixed batch
    size or `"full"` for one update per epoch. An explicit integer cannot
    exceed the number of training triplets.
    """

    def __init__(
        self,
        *,
        learning_rate: float,
        n_components: int = 2,
        optimizer: OptimizerName = "sgd",
        batch_size: int | Literal["full"] | None = None,
        max_epochs: int = 100,
        margin: float = 1.0,
        reduction: Literal["mean", "sum"] = "mean",
        shuffle: bool = True,
        random_state: int = 42,
        shuffle_random_state: int | None = None,
        device: Literal["cpu"] = "cpu",
        amsgrad: bool = False,
        adam_betas: tuple[float, float] = (0.9, 0.999),
        adam_eps: float = 1e-8,
        evaluation_interval_triplets: int | None = None,
        ordinal_mode: OrdinalMode = "ties_as_errors",
    ) -> None:
        self.learning_rate = learning_rate
        self.n_components = n_components
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.margin = margin
        self.reduction = reduction
        self.shuffle = shuffle
        self.random_state = random_state
        self.shuffle_random_state = shuffle_random_state
        self.device = device
        self.amsgrad = amsgrad
        self.adam_betas = adam_betas
        self.adam_eps = adam_eps
        self.evaluation_interval_triplets = evaluation_interval_triplets
        self.ordinal_mode = ordinal_mode
        self._validate_options()

    def _validate_options(self) -> None:
        if self.device != "cpu":
            raise ValueError("SOE currently supports only device='cpu'")
        if isinstance(self.learning_rate, bool) or not isinstance(self.learning_rate, Real):
            raise ValueError("learning_rate must be a finite positive number")
        _positive_integer(self.n_components, "n_components")
        if self.batch_size is not None and self.batch_size != "full":
            _positive_integer(self.batch_size, "batch_size")
        if not isinstance(self.shuffle, bool) or not isinstance(self.amsgrad, bool):
            raise ValueError("shuffle and amsgrad must be booleans")
        if self.amsgrad and self.optimizer != "adam":
            raise ValueError("amsgrad=True requires optimizer='adam'")
        if self.ordinal_mode not in ("ties_as_errors", "reference"):
            raise ValueError("ordinal_mode must be 'ties_as_errors' or 'reference'")
        _validate_seed(self.random_state, "random_state")
        if self.shuffle_random_state is not None:
            _validate_seed(self.shuffle_random_state, "shuffle_random_state")
        _positive_integer(self.max_epochs, "max_epochs")
        if self.evaluation_interval_triplets is not None:
            _positive_integer(self.evaluation_interval_triplets, "evaluation_interval_triplets")
        validate_loss_options(self.margin, self.reduction)
        validate_training_options(
            optimizer=self.optimizer,
            learning_rate=self.learning_rate,
            epochs=int(self.max_epochs),
            evaluation_interval=1,
            evaluation_interval_triplets=(
                int(self.evaluation_interval_triplets)
                if self.evaluation_interval_triplets is not None
                else None
            ),
            shuffle_seed=None,
            adam_betas=self.adam_betas,
            adam_eps=self.adam_eps,
        )

    def _resolve_batch_size(self, n_triplets: int) -> int:
        if self.batch_size is None:
            return min(_DEFAULT_BATCH_SIZE, n_triplets)
        if self.batch_size == "full":
            return n_triplets
        if self.batch_size > n_triplets:
            raise ValueError("batch_size cannot exceed the number of training triplets")
        return int(self.batch_size)

    def fit(
        self,
        triplets: Any,
        *,
        n_objects: int,
        validation_triplets: Any | None = None,
        initial_embedding: Any | None = None,
    ) -> SOE:
        """Fit from a fresh start and return this estimator.

        Every object has one embedding row, including objects without observed
        triplets. The completed fit replaces the previous fitted attributes.
        The best checkpoint minimizes the complete validation loss when
        validation triplets are supplied, or the complete training loss otherwise.
        """
        self._validate_options()
        _positive_integer(n_objects, "n_objects")
        training_triplets = as_triplet_tensor(triplets)
        batch_size = self._resolve_batch_size(training_triplets.shape[0])
        seed = int(self.random_state)
        shuffle_seed = seed if self.shuffle_random_state is None else int(self.shuffle_random_state)
        interval = self.evaluation_interval_triplets
        options = {
            "optimizer": self.optimizer,
            "batch_size": batch_size,
            "learning_rate": float(self.learning_rate),
            "epochs": int(self.max_epochs),
            "margin": self.margin,
            "reduction": self.reduction,
            "shuffle": self.shuffle,
            "seed": seed,
            "shuffle_seed": shuffle_seed,
            "device": self.device,
            "amsgrad": self.amsgrad,
            "adam_betas": self.adam_betas,
            "adam_eps": self.adam_eps,
            "evaluation_interval_triplets": None if interval is None else int(interval),
            "ordinal_mode": self.ordinal_mode,
        }
        result = train_soe(
            training_triplets,
            n_objects=int(n_objects),
            embedding_dim=int(self.n_components),
            validation_triplets=validation_triplets,
            initial_embeddings=initial_embedding,
            **options,
        )
        history = result.evaluation_history
        best_checkpoint = next(
            record for record in history if record.optimizer_step == result.best_optimizer_step
        )
        n_observed_objects = int(torch.unique(training_triplets.to("cpu")).numel())

        self.embedding_ = result.best_embeddings.numpy().copy()
        self.best_embedding_ = self.embedding_.copy()
        self.final_embedding_ = result.final_embeddings.numpy().copy()
        self.initial_embedding_ = result.initial_embeddings.numpy().copy()
        self.result_ = result
        self.history_ = self.evaluation_history_ = history
        self.step_history_ = result.step_history
        self.checkpoint_mappings_ = result.checkpoint_mappings
        self.best_checkpoint_ = best_checkpoint
        self.best_epoch_ = result.best_epoch
        self.best_optimizer_step_ = result.best_optimizer_step
        self.optimizer_steps_ = result.optimizer_steps
        self.processed_triplets_ = result.processed_triplets
        self.training_runtime_seconds_ = result.training_runtime_seconds
        self.evaluation_runtime_seconds_ = result.evaluation_runtime_seconds
        self.total_runtime_seconds_ = result.total_runtime_seconds
        self.n_objects_ = int(n_objects)
        self.n_observed_objects_ = n_observed_objects
        self.n_training_triplets_ = history[0].training.n_triplets
        validation = history[0].validation
        self.n_validation_triplets_ = 0 if validation is None else validation.n_triplets
        self.selection_source_ = "training" if validation is None else "validation"
        self.batch_size_ = batch_size
        self.optimizer_ = self.optimizer
        self.learning_rate_ = float(self.learning_rate)
        self.device_ = self.device
        self.reproducibility_ = {
            **options,
            "n_objects": int(n_objects),
            "n_components": int(self.n_components),
            "initialization": "random" if initial_embedding is None else "provided",
            "dtype": str(result.initial_embeddings.dtype),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "torch_version": str(torch.__version__),
            "platform": platform.platform(),
        }
        return self

    def fit_transform(
        self,
        triplets: Any,
        *,
        n_objects: int,
        validation_triplets: Any | None = None,
        initial_embedding: Any | None = None,
    ) -> np.ndarray:
        """Fit and return an independent NumPy copy of the selected embedding."""
        self.fit(
            triplets,
            n_objects=n_objects,
            validation_triplets=validation_triplets,
            initial_embedding=initial_embedding,
        )
        return self.embedding_.copy()


def _positive_integer(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _validate_seed(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value < 2**64:
        raise ValueError(f"{name} must be an integer in [0, 2**64)")
