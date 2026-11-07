"""Device selection, seeding and GPU-memory bookkeeping.

Every notebook starts with :func:`setup`, which seeds all RNGs, picks the best
available device and prints a short environment summary so that results in the
committed notebook outputs are traceable to the software/hardware that produced them.
"""

from __future__ import annotations

import os
import platform
import random
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import torch


def seed_everything(seed: int = 0, deterministic: bool = False) -> None:
    """Seed Python, NumPy and (if installed) PyTorch RNGs.

    Args:
        seed: Seed value shared by all generators.
        deterministic: Also force deterministic cuDNN kernels (slower).
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - seeds legacy global RNG used by many libraries
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is optional for pure-numpy helpers
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(prefer: str = "cuda") -> torch.device:
    """Return ``cuda`` when available (and preferred), otherwise ``cpu``."""
    import torch

    if prefer.startswith("cuda") and torch.cuda.is_available():
        return torch.device(prefer)
    return torch.device("cpu")


def env_summary() -> dict[str, Any]:
    """Collect versions of the core stack and the GPU name/VRAM."""
    info: dict[str, Any] = {"python": platform.python_version(), "numpy": np.__version__}
    try:
        import cv2

        info["opencv"] = cv2.__version__
    except ImportError:  # pragma: no cover
        info["opencv"] = None
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu"] = props.name
            info["vram_gb"] = round(props.total_memory / 2**30, 2)
            info["compute_capability"] = f"sm_{props.major}{props.minor}"
    except ImportError:  # pragma: no cover
        info["torch"] = None
    try:
        import torchvision

        info["torchvision"] = torchvision.__version__
    except ImportError:  # pragma: no cover
        pass
    return info


def setup(
    seed: int = 0,
    verbose: bool = True,
    deterministic: bool = False,
    quiet_warnings: bool = True,
) -> torch.device:
    """Standard first call of every notebook.

    Seeds every RNG, selects the device and prints the environment summary.

    Args:
        seed: RNG seed.
        verbose: Print the environment summary.
        deterministic: Force deterministic cuDNN kernels.
        quiet_warnings: Silence ``FutureWarning``/``DeprecationWarning`` noise from
            third-party libraries so notebook outputs stay readable.

    Returns:
        The selected ``torch.device``.
    """
    if quiet_warnings:
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    seed_everything(seed, deterministic=deterministic)
    device = get_device()
    if verbose:
        info = env_summary()
        gpu = info.get("gpu", "none")
        vram = info.get("vram_gb")
        print(
            f"python {info['python']} | torch {info.get('torch')} (CUDA {info.get('cuda')}) | "
            f"torchvision {info.get('torchvision')} | numpy {info['numpy']} | "
            f"opencv {info.get('opencv')}"
        )
        if vram is not None:
            print(f"device: {device} | GPU: {gpu} ({vram} GB, {info.get('compute_capability')})")
        else:
            print(f"device: {device} | GPU: {gpu}")
    return device


def reset_peak_vram() -> None:
    """Reset CUDA peak-memory statistics (no-op without CUDA)."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def peak_vram_mb() -> float:
    """Peak memory allocated by PyTorch on the current CUDA device, in MiB."""
    import torch

    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 2**20


class track_vram:  # noqa: N801 - used as a lowercase context manager on purpose
    """Context manager that records peak CUDA memory of the enclosed block.

    Example::

        with track_vram() as mem:
            model(x)
        print(f"peak VRAM {mem.peak_mb:.0f} MiB")
    """

    def __init__(self) -> None:
        self.peak_mb = 0.0

    def __enter__(self) -> track_vram:
        reset_peak_vram()
        return self

    def __exit__(self, *exc: object) -> None:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.peak_mb = peak_vram_mb()
