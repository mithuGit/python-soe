# Python SOE

A Python package project for triplet-based Soft Ordinal Embedding (SOE).

## Status

The package contains the numerical core for comparisons of the form “object i is closer to j than to k.” It uses the unsquared triplet hinge loss and a shared CPU training path for mini-batch SGD and Adam. The low-level trainer retains the defaults from the research implementation while the estimator API is pending.

The core and its regression tests were migrated from the separate `soe-minibatch-study` repository at commit `7cf8055936cc41b6f21218ed48bba897a2bdb9c3`. Experimental datasets and benchmark tooling remain there.

## Development setup

Use Python 3.10 or newer. From the repository root:

```sh
python -m pip install '.[dev]'
python -m pytest --import-mode=importlib
```

The distribution name is provisionally `python-soe`. The import name is `soe`.
Set `EVALUATE_OE_DIR` to an existing evaluate-OE reference checkout to include the sum-reduction compatibility test.

## Layout

```text
python-soe/
├── pyproject.toml
├── README.md
├── src/soe/
│   ├── validation.py
│   ├── initialization.py
│   ├── losses.py
│   ├── evaluation.py
│   ├── training.py
│   ├── checkpoints.py
│   └── results.py
└── tests/
```

The next milestone is the `SOE` estimator. Package naming and licensing will be settled before public distribution.
