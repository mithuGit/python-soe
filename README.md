# Python SOE

A Python package project for triplet-based Soft Ordinal Embedding (SOE).

## Status

The package provides a CPU estimator for comparisons of the form “object i is closer to j than to k.” It uses the unsquared triplet hinge loss and a shared mini-batch training path for SGD and Adam.

The core and its regression tests were migrated from the separate `soe-minibatch-study` repository at commit `7cf8055936cc41b6f21218ed48bba897a2bdb9c3`. Experimental datasets and benchmark tooling remain there.

## Install and run

Use Python 3.10 or newer. From the repository root on macOS or Linux:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
python examples/minimal_cpu.py
```

On Windows PowerShell, create the environment with `python -m venv .venv`
and activate it with `.\.venv\Scripts\Activate.ps1`.

The example prints the resolved batch size, the selected checkpoint, a
`(6, 2)` embedding and the ordinal error on held-out triplets. The learning
rate of `0.1` is a starting point for this example.

To build the source distribution and wheel from the checkout:

```sh
python -m pip install build
python -m build
```

In a fresh environment, install the wheel from `dist/` and run the example
from the checkout or the unpacked source distribution:

```sh
python -m pip install dist/python_soe-0.0.1.dev0-py3-none-any.whl
python examples/minimal_cpu.py
```

The distribution name is provisionally `python-soe`. The import name is `soe`.

## Development setup

From the repository root in an active virtual environment:

```sh
python -m pip install '.[dev]'
python -m pytest --import-mode=importlib
```

Set `EVALUATE_OE_DIR` to an existing evaluate-OE reference checkout to include the sum-reduction compatibility test.

## Estimator

Import `SOE` from `soe` and pass an explicit `learning_rate`. The default
optimizer is SGD without momentum. Choose a rate using validation data.

Each row `(i, j, k)` contains three distinct integer indices in
`[0, n_objects)` and means that object `i` is closer to `j` than to `k`.
Pass the total object count as `n_objects`, including any objects without
observed training triplets.

`fit(triplets, n_objects=..., validation_triplets=..., initial_embedding=...)`
returns the fitted estimator. The validation triplets and initial embedding
are optional. `fit_transform` takes the same arguments and returns a NumPy
copy of `embedding_`, the best evaluated checkpoint. Selection uses complete
validation loss when available and complete training loss otherwise.
`final_embedding_` stores the last training state.

Training triplets drive optimizer updates. Validation triplets select the
checkpoint and can guide learning-rate tuning. Keep test triplets separate
and pass them to `evaluate_embedding` after fitting and model selection.
By default, equal distances count as ordinal errors.

With `batch_size=None`, training uses up to 128 triplets per batch. A positive
integer requests a fixed batch size no larger than the training set.
`batch_size="full"` uses every training triplet in each update. The resolved
size is available as `batch_size_`.

## Layout

```text
python-soe/
├── pyproject.toml
├── README.md
├── MANIFEST.in
├── examples/
│   └── minimal_cpu.py
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

Package naming and licensing will be settled before public distribution.
