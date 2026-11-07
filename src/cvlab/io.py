"""Repository paths and image/video/JSON I/O helpers.

All paths are resolved relative to the repository root so notebooks work no
matter which directory Jupyter was started from.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

_MARKER = "pyproject.toml"


def repo_root() -> Path:
    """Return the repository root.

    Resolution order: ``$CVLAB_ROOT``, the source checkout containing this file
    (editable install), then the first parent of the current directory that
    contains ``pyproject.toml``.
    """
    env = os.environ.get("CVLAB_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / _MARKER).is_file() and (parent / "src" / "cvlab").is_dir():
            return parent
    cwd = Path.cwd().resolve()
    for parent in (cwd, *cwd.parents):
        if (parent / _MARKER).is_file():
            return parent
    raise FileNotFoundError("Could not locate the repository root (set CVLAB_ROOT).")


def _sub(name: str, *parts: str, create: bool = False) -> Path:
    path = repo_root().joinpath(name, *parts)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def data_dir(*parts: str, create: bool = False) -> Path:
    """``<repo>/data/...`` (gitignored datasets)."""
    return _sub("data", *parts, create=create)


def weights_dir(*parts: str, create: bool = False) -> Path:
    """``<repo>/weights/...`` (gitignored checkpoints)."""
    return _sub("weights", *parts, create=create)


def runs_dir(*parts: str, create: bool = False) -> Path:
    """``<repo>/runs/...`` (gitignored training runs)."""
    return _sub("runs", *parts, create=create)


def outputs_dir(*parts: str, create: bool = False) -> Path:
    """``<repo>/outputs/...`` (gitignored large outputs: videos, .ply, engines)."""
    return _sub("outputs", *parts, create=create)


def assets_dir(project: str, create: bool = True) -> Path:
    """``<repo>/assets/<project>/`` (small committed figures used by READMEs)."""
    return _sub("assets", project, create=create)


def project_dir(project: str) -> Path:
    """``<repo>/projects/<project>/``."""
    return _sub("projects", project)


def read_image(path: str | Path, rgb: bool = True, max_side: int | None = None) -> np.ndarray:
    """Read an image as ``uint8`` HxWx3 (RGB by default, BGR if ``rgb=False``).

    Args:
        path: Image file.
        rgb: Convert OpenCV's BGR order to RGB.
        max_side: Optionally downscale so that ``max(H, W) <= max_side``.
    """
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if max_side is not None:
        img = resize_max(img, max_side)
    return img


def write_image(path: str | Path, img: np.ndarray, rgb: bool = True) -> Path:
    """Write an RGB (default) or BGR ``uint8`` image; parent dirs are created."""
    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if (rgb and img.ndim == 3) else img
    if not cv2.imwrite(str(path), out):
        raise OSError(f"Failed to write image: {path}")
    return path


def resize_max(img: np.ndarray, max_side: int) -> np.ndarray:
    """Downscale (never upscale) so the longest side is at most ``max_side`` pixels."""
    import cv2

    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return img
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def iter_video_frames(
    path: str | Path,
    stride: int = 1,
    max_frames: int | None = None,
    max_side: int | None = None,
    rgb: bool = True,
) -> Iterator[np.ndarray]:
    """Yield frames from a video file (every ``stride``-th frame)."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    idx = emitted = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                if rgb:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if max_side is not None:
                    frame = resize_max(frame, max_side)
                yield frame
                emitted += 1
                if max_frames is not None and emitted >= max_frames:
                    break
            idx += 1
    finally:
        cap.release()


def read_video_frames(path: str | Path, **kwargs: Any) -> list[np.ndarray]:
    """List version of :func:`iter_video_frames`."""
    return list(iter_video_frames(path, **kwargs))


class _NumpyEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def save_json(obj: Any, path: str | Path, indent: int = 2) -> Path:
    """Save JSON (NumPy scalars/arrays and Paths are converted)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=indent, cls=_NumpyEncoder) + "\n")
    return path


def load_json(path: str | Path) -> Any:
    """Load a JSON file."""
    return json.loads(Path(path).read_text())
