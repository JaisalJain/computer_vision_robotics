"""Shared pytest fixtures (CPU only, no network)."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)
