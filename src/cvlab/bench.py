"""Latency benchmarking with proper CUDA synchronisation."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np


def benchmark(
    fn: Callable[..., Any],
    *args: Any,
    warmup: int = 10,
    iters: int = 50,
    cuda: bool | None = None,
    **kwargs: Any,
) -> dict[str, float]:
    """Time ``fn(*args, **kwargs)``.

    GPU work is asynchronous, so with ``cuda=True`` every iteration is bracketed by
    ``torch.cuda.synchronize()`` to measure true wall-clock latency.

    Returns:
        ``mean_ms``, ``std_ms``, ``p50_ms``, ``p90_ms``, ``min_ms`` and ``fps`` (1000/mean).
    """
    sync: Callable[[], None] = lambda: None  # noqa: E731
    if cuda is None or cuda:
        try:
            import torch

            if torch.cuda.is_available() and cuda is not False:
                sync = torch.cuda.synchronize
        except ImportError:  # pragma: no cover
            pass
    for _ in range(warmup):
        fn(*args, **kwargs)
    sync()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        sync()
        times.append((time.perf_counter() - t0) * 1000.0)
    t = np.asarray(times)
    return {
        "mean_ms": float(t.mean()),
        "std_ms": float(t.std()),
        "p50_ms": float(np.percentile(t, 50)),
        "p90_ms": float(np.percentile(t, 90)),
        "min_ms": float(t.min()),
        "fps": float(1000.0 / t.mean()),
    }


class Timer:
    """``with Timer() as t: ...; t.ms`` - simple wall-clock timer."""

    def __enter__(self) -> Timer:
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.ms = (time.perf_counter() - self._t0) * 1000.0
