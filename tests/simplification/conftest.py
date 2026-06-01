from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from msynth.simplification.oracle import SimplificationOracle


@pytest.fixture
def write_empty_oracle(tmp_path: Path):
    """
    Factory fixture that materialises an empty :class:`SimplificationOracle`
    on disk and returns its path.

    Lets a test request a freshly-built empty oracle with a chosen variable
    count and sample size. ``num_variables`` must be at least the maximum
    number of unified terminals expected in any subtree the simplifier
    visits, since the oracle evaluates expressions on input vectors of that
    width.
    """

    def make(num_variables: int = 3, num_samples: int = 8) -> Path:
        oracle = SimplificationOracle.__new__(SimplificationOracle)
        oracle.num_variables = num_variables
        oracle.num_samples = num_samples
        oracle.inputs = [
            [(s * 17 + v * 3 + 1) & 0xFF for v in range(num_variables)]
            for s in range(num_samples)
        ]
        oracle.oracle_map = {}

        path = tmp_path / f"empty_oracle_v{num_variables}_s{num_samples}.pkl"
        with open(path, "wb") as f:
            pickle.dump(oracle, f)
        return path

    return make
