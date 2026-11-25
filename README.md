# Python SOE

A Python package project for triplet-based Soft Ordinal Embedding (SOE).

## Status

The package provides a CPU estimator for comparisons of the form “object i is closer to j than to k.” It uses the unsquared triplet hinge loss and a shared mini-batch training path for SGD and Adam.

The core and its regression tests were migrated from the separate `soe-minibatch-study` repository at commit `7cf8055936cc41b6f21218ed48bba897a2bdb9c3`. Experimental datasets and benchmark tooling remain there.

## Development setup

Use Python 3.10 or newer. From the repository root:

```sh
python -m pip install '.[dev]'
python -m pytest --import-mode=importlib
```

The distribution name is provisionally `python-soe`. The import name is `soe`.
Set `EVALUATE_OE_DIR` to an existing evaluate-OE reference checkout to include the sum-reduction compatibility test.

## Estimator

Import `SOE` from `soe` and pass an explicit `learning_rate`. The default
optimizer is SGD without momentum. Choose a rate using validation data.

`fit(triplets, n_objects=..., validation_triplets=..., initial_embedding=...)`
returns the fitted estimator. The validation triplets and initial embedding
are optional. `fit_transform` takes the same arguments and returns a NumPy
copy of `embedding_`, the best evaluated checkpoint. Selection uses complete
validation loss when available and complete training loss otherwise.
`final_embedding_` stores the last training state.

With `batch_size=None`, training uses up to 128 triplets per batch. A positive
integer requests a fixed batch size no larger than the training set.
`batch_size="full"` uses every training triplet in each update. The resolved
size is available as `batch_size_`.

## Layout

```text
python-soe/
├── pyproject.toml
├── README.md
├── src/soe/
│   ├── estimator.py
│   ├── validation.py
│   ├── initialization.py
│   ├── losses.py
│   ├── evaluation.py
│   ├── training.py
│   ├── checkpoints.py
│   └── results.py
└── tests/
```

The next milestone is a runnable CPU example and a clean-install check.
Package naming and licensing will be settled before public distribution.
