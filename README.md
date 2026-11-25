# Python SOE

A Python package project for triplet-based Soft Ordinal Embedding (SOE).

## Status

This initial commit contains an installable package scaffold. The research implementation and estimator API will be added in later commits.

The planned first version will embed objects from comparisons of the form “object i is closer to j than to k.” It will use the unsquared triplet hinge loss and a shared CPU training path for mini-batch SGD and Adam. The existing research project remains the source for experiments and correctness tests.

## Development setup

Use Python 3.10 or newer. From the repository root:

```sh
python -m pip install -e .
python -c "import soe; print(soe.__file__)"
```

The distribution name is provisionally `python-soe`. The import name is `soe`.

## Layout

```text
python-soe/
├── pyproject.toml
├── README.md
└── src/
    └── soe/
        └── __init__.py
```

The next milestone is to migrate the validated numerical core and verify its behavior from an installed wheel. Package naming and licensing will be settled before public distribution.
