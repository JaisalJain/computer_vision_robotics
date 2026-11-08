"""Monocular depth models and the TUM RGB-D evaluation protocol of project 08.

The notebooks, ``eval_tum.py`` and the exercise solutions all import this module, so every
number in the README comes from the same code path.

Model families
--------------
* **Depth Anything V2** (Hugging Face ``transformers``): the relative model predicts
  *affine-invariant inverse depth* (disparity up to an unknown scale **and** shift); the
  Metric-Indoor model predicts metres directly (Hypersim fine-tune, ``max_depth = 20``).
* **Depth Anything 3** (pinned clone in ``third_party/depth-anything-3``): the any-view models
  (DA3-SMALL / DA3-BASE) predict *depth up to scale* plus a camera (intrinsics, pose);
  DA3METRIC-LARGE predicts a canonical depth that becomes metres via
  ``z = f_px * net_output / 300`` (``f_px`` = mean focal length at the processed resolution).

Usage::

    import sys; sys.path.insert(0, "projects/08_monocular_depth/scripts")
    from mono_depth import DepthModel, load_tum, evaluate_frame
"""

from __future__ import annotations

import os
import sys
import time
import types
import warnings
import weakref
from dataclasses import dataclass, field

import numpy as np

from cvlab.io import add_third_party, data_dir
from cvlab.metrics import align_depth_scale, align_depth_scale_shift, depth_metrics
from cvlab.tum import TumSequence, load_sequence

PROJECT = "08_monocular_depth"

# Ground-truth pixels used for evaluation: Kinect v1 depth in metres (0 = no measurement).
MIN_DEPTH, MAX_DEPTH = 0.1, 10.0
# Edges of the GT-depth bins used to study error vs distance (metres).
DEPTH_BINS = np.array([0.0, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 10.0])
# Canonical focal length of DA3METRIC-LARGE (README of the DA3 repository).
DA3_METRIC_CANONICAL_FOCAL = 300.0


# ------------------------------------------------------------------------------- models
@dataclass(frozen=True)
class ModelSpec:
    """Static description of one depth model."""

    key: str
    label: str
    repo: str
    family: str  # "da2" (transformers) | "da3" (third-party clone)
    output: str  # "disparity" | "depth" (up to scale) | "metric" (metres)
    license: str


MODELS: dict[str, ModelSpec] = {
    s.key: s
    for s in [
        ModelSpec("da2-small", "DA-V2 Small", "depth-anything/Depth-Anything-V2-Small-hf",
                  "da2", "disparity", "Apache-2.0"),
        ModelSpec("da3-small", "DA3-Small", "depth-anything/DA3-SMALL", "da3", "depth",
                  "Apache-2.0"),
        ModelSpec("da3-base", "DA3-Base", "depth-anything/DA3-BASE", "da3", "depth",
                  "Apache-2.0"),
        ModelSpec("da2-metric-indoor-small", "DA-V2 Metric-Indoor Small",
                  "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf", "da2", "metric",
                  "Apache-2.0 (original checkpoint); -hf repo has no license tag"),
        ModelSpec("da3metric-large", "DA3Metric-Large", "depth-anything/DA3METRIC-LARGE",
                  "da3", "metric", "Apache-2.0"),
    ]
}  # fmt: skip

# Alignment protocols evaluated for each output type (see ``align_prediction``).
PROTOCOLS: dict[str, list[str]] = {
    "disparity": ["lsq"],  # scale + shift fitted in inverse-depth space
    "depth": ["median", "lsq"],  # median scale; scale + shift in depth space
    "metric": ["none", "median"],  # raw metres; median scale (isolates the scale error)
}
PROTOCOL_LABELS = {
    "none": "raw metric",
    "median": "median scale",
    "lsq": "LSQ scale+shift",
}


def stub_moviepy() -> bool:
    """Register an empty ``moviepy.editor`` module if moviepy is not installed.

    ``depth_anything_3.api`` imports ``moviepy.editor`` at import time, but only uses it to
    write Gaussian-splatting videos. moviepy 1.0.3 (the last version with ``moviepy.editor``)
    would downgrade ``decorator`` in the shared environment, so it is not installed; the stub
    lets the depth API import. Returns ``True`` when a stub was inserted.
    """
    try:
        import moviepy.editor  # noqa: F401

        return False
    except ImportError:
        pass
    moviepy = types.ModuleType("moviepy")
    editor = types.ModuleType("moviepy.editor")
    moviepy.editor = editor
    sys.modules.setdefault("moviepy", moviepy)
    sys.modules.setdefault("moviepy.editor", editor)
    return True


def import_da3():
    """Import ``DepthAnything3`` from the pinned clone (quiet logger, moviepy stub)."""
    os.environ.setdefault("DA3_LOG_LEVEL", "ERROR")  # DA3's own logger prints to stdout
    warnings.filterwarnings("ignore", message=".*register_new_resolver.*")
    add_third_party("depth-anything-3", "src")
    stub_moviepy()
    from depth_anything_3.api import DepthAnything3

    return DepthAnything3


@dataclass
class DepthPrediction:
    """Output of :meth:`DepthModel.predict` at the input image resolution."""

    values: np.ndarray  # HxW float32: disparity, depth up to scale, or metres (see ``output``)
    output: str  # "disparity" | "depth" | "metric"
    proc_hw: tuple[int, int]  # network input resolution (H, W)
    raw: np.ndarray | None = None  # network output before metric conversion (DA3 metric only)
    conf: np.ndarray | None = None  # HxW confidence (DA3 any-view models)
    focal_px: float | None = None  # focal length predicted by DA3 any-view (input resolution)
    extra: dict = field(default_factory=dict)


class DepthModel:
    """Uniform wrapper: ``DepthModel(key).predict(rgb, K)`` -> :class:`DepthPrediction`.

    All models run with bf16 autocast on CUDA (DA3 does this internally; the transformers
    models are wrapped here) and the output is resized bilinearly to the input resolution.
    """

    def __init__(self, key: str, device: str = "cuda", process_res: int = 504) -> None:
        import torch

        self.spec = MODELS[key]
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.process_res = process_res  # DA3 only: longest side fed to the network
        if self.spec.family == "da2":
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            from transformers.utils import logging as hf_logging

            hf_logging.set_verbosity_error()
            hf_logging.disable_progress_bar()
            self.processor = AutoImageProcessor.from_pretrained(self.spec.repo)
            self.net = AutoModelForDepthEstimation.from_pretrained(self.spec.repo)
        else:
            self.net = import_da3().from_pretrained(self.spec.repo)
        self.net = self.net.to(self.device).eval()
        self.n_params = sum(p.numel() for p in self.net.parameters())
        # Time the network forward alone (no pre/post-processing) with hooks on the core module.
        # The hooks only hold a weak reference, so `del model` frees the GPU memory at once.
        core = self.net.model if self.spec.family == "da3" else self.net
        ref = weakref.ref(self)
        core.register_forward_pre_hook(lambda *_: ref() and ref()._tic())
        core.register_forward_hook(lambda *_: ref() and ref()._toc())
        self.last_network_ms = float("nan")

    def _sync(self) -> None:
        import torch

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _tic(self) -> None:
        self._sync()
        self._t0 = time.perf_counter()

    def _toc(self) -> None:
        self._sync()
        self.last_network_ms = (time.perf_counter() - self._t0) * 1000.0

    @property
    def output(self) -> str:
        return self.spec.output

    def predict(self, rgb: np.ndarray, K: np.ndarray | None = None) -> DepthPrediction:
        """Predict on one RGB uint8 HxWx3 image.

        Args:
            rgb: Input image (RGB order).
            K: 3x3 intrinsics at the input resolution. Required by DA3METRIC-LARGE, whose
                output is metric only once multiplied by the focal length.
        """
        import torch
        import torch.nn.functional as F

        h, w = rgb.shape[:2]
        out = DepthPrediction(values=np.empty(0), output=self.spec.output, proc_hw=(0, 0))
        with torch.inference_mode():
            if self.spec.family == "da2":
                from PIL import Image

                inputs = self.processor(images=Image.fromarray(rgb), return_tensors="pt")
                pix = inputs["pixel_values"].to(self.device)
                with torch.autocast(self.device.type, dtype=torch.bfloat16,
                                    enabled=self.device.type == "cuda"):  # fmt: skip
                    raw = self.net(pixel_values=pix).predicted_depth  # (1, h', w')
                raw = raw.float()
            else:
                pred = self.net.inference([rgb], process_res=self.process_res)
                raw = torch.from_numpy(np.ascontiguousarray(pred.depth[:1])).float()
                if pred.conf is not None:
                    conf = torch.from_numpy(np.ascontiguousarray(pred.conf[:1])).float()
                    conf = F.interpolate(conf[None], size=(h, w), mode="bilinear")[0, 0]
                    out.conf = conf.numpy()
                if pred.intrinsics is not None and self.spec.output != "metric":
                    k_proc = np.asarray(pred.intrinsics[0], dtype=np.float64)
                    sx, sy = w / raw.shape[-1], h / raw.shape[-2]
                    out.focal_px = float(0.5 * (k_proc[0, 0] * sx + k_proc[1, 1] * sy))
            out.proc_hw = (int(raw.shape[-2]), int(raw.shape[-1]))
            # Depth / disparity values do not change when the map is resized.
            full = F.interpolate(raw[None].to("cpu"), size=(h, w), mode="bilinear",
                                 align_corners=False)[0, 0].numpy()  # fmt: skip
        if self.spec.family == "da3" and self.spec.output == "metric":
            if K is None:
                raise ValueError("DA3METRIC-LARGE needs the intrinsics K to output metres")
            out.raw = full.astype(np.float32)
            full = full * da3_metric_focal(K, (h, w), out.proc_hw) / DA3_METRIC_CANONICAL_FOCAL
        out.values = full.astype(np.float32)
        out.extra["network_ms"] = self.last_network_ms
        return out


def free_cuda() -> None:
    """Release cached GPU memory after deleting models (collects garbage first)."""
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def da3_metric_focal(K: np.ndarray, image_hw: tuple[int, int], proc_hw: tuple[int, int]) -> float:
    """Mean focal length (pixels) at the DA3 processing resolution."""
    sx, sy = proc_hw[1] / image_hw[1], proc_hw[0] / image_hw[0]
    return float(0.5 * (K[0, 0] * sx + K[1, 1] * sy))


# ------------------------------------------------------------------------------ data
def load_tum(name: str) -> TumSequence:
    """Load an associated TUM sequence by short name, e.g. ``"fr1_desk"``."""
    cam, seq = name.split("_", 1)
    root = data_dir("tum", name, f"rgbd_dataset_freiburg{cam[-1]}_{seq}")
    if not root.is_dir():
        raise FileNotFoundError(f"{name} missing - run `make data-{PROJECT}`")
    return load_sequence(root)


def select_frames(n_total: int, n: int) -> np.ndarray:
    """``n`` evenly spaced frame indices covering the whole sequence."""
    return np.unique(np.linspace(0, n_total - 1, n).round().astype(int))


# ------------------------------------------------------------------------- alignment
def valid_mask(gt: np.ndarray, min_depth: float = MIN_DEPTH, max_depth: float = MAX_DEPTH):
    """Pixels with a usable sensor measurement."""
    return np.isfinite(gt) & (gt > min_depth) & (gt < max_depth)


def align_prediction(
    pred: np.ndarray, output: str, gt: np.ndarray, mask: np.ndarray, protocol: str
) -> tuple[np.ndarray, float, float]:
    """Turn a raw prediction into metric depth with one alignment protocol.

    * ``"none"``   - use the prediction as metres (metric models only).
    * ``"median"`` - ``z = s * pred`` with ``s = median(gt) / median(pred)``; valid for
      outputs that are depth up to scale (DA3 any-view, metric models).
    * ``"lsq"``    - least-squares scale **and** shift in the model's native space: depth
      space for depth outputs, inverse-depth space for disparity outputs
      (``1/z = s * d + t``), the protocol of MiDaS / Depth Anything.

    Returns ``(depth_m HxW, scale, shift)``; GT is only used on ``mask`` pixels.
    """
    pred = np.asarray(pred, dtype=np.float64)
    if protocol == "none":
        if output != "metric":
            raise ValueError("protocol 'none' only makes sense for metric outputs")
        return pred, 1.0, 0.0
    if output == "disparity":
        if protocol != "lsq":
            raise ValueError("disparity is affine-invariant: only 'lsq' alignment is valid")
        inv_gt = np.where(mask, 1.0 / np.where(mask, gt, 1.0), 0.0)
        s, t = align_depth_scale_shift(pred, inv_gt, mask)
        disp = np.maximum(s * pred + t, 1.0 / MAX_DEPTH)  # avoid z <= 0 / huge depths
        return 1.0 / disp, s, t
    if protocol == "median":
        s = align_depth_scale(pred, gt, mask)
        return s * pred, s, 0.0
    if protocol == "lsq":
        s, t = align_depth_scale_shift(pred, gt, mask)
        return np.maximum(s * pred + t, 1e-3), s, t
    raise ValueError(f"unknown protocol: {protocol}")


def evaluate_frame(
    pred: np.ndarray, output: str, gt: np.ndarray, protocols: list[str] | None = None
) -> dict[str, dict]:
    """Align + score one prediction with every protocol valid for its output type.

    Returns ``{protocol: {"depth": aligned HxW, "scale", "shift", **depth_metrics}}``.
    """
    mask = valid_mask(gt)
    res = {}
    for proto in protocols or PROTOCOLS[output]:
        depth, s, t = align_prediction(pred, output, gt, mask, proto)
        res[proto] = {"depth": depth, "scale": s, "shift": t, **depth_metrics(depth, gt, mask)}
    return res


def abs_rel_map(depth: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per-pixel ``|z - z*| / z*`` (NaN where GT is invalid)."""
    m = valid_mask(gt)
    out = np.full(gt.shape, np.nan)
    out[m] = np.abs(depth[m] - gt[m]) / gt[m]
    return out


def binned_errors(
    depth: np.ndarray, gt: np.ndarray, bins: np.ndarray = DEPTH_BINS
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per GT-depth bin: sum of AbsRel, sum of squared error and pixel count.

    Sums (not means) so that several frames can be accumulated before dividing.
    """
    m = valid_mask(gt)
    g, p = gt[m], depth[m]
    idx = np.clip(np.digitize(g, bins) - 1, 0, len(bins) - 2)
    n_bins = len(bins) - 1
    abs_rel = np.bincount(idx, weights=np.abs(p - g) / g, minlength=n_bins)
    sq = np.bincount(idx, weights=(p - g) ** 2, minlength=n_bins)
    count = np.bincount(idx, minlength=n_bins).astype(np.float64)
    return abs_rel, sq, count


# ----------------------------------------------------------------------- point clouds
def nn_distances(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Distance from every point of ``src`` (Nx3) to its nearest neighbour in ``dst``."""
    from scipy.spatial import cKDTree

    d, _ = cKDTree(np.asarray(dst, dtype=np.float64)).query(np.asarray(src, dtype=np.float64))
    return d


def chamfer_stats(pred_pts: np.ndarray, gt_pts: np.ndarray, thresh: float = 0.05) -> dict:
    """Accuracy (pred -> GT), completeness (GT -> pred), Chamfer mean and F-score@thresh."""
    acc = nn_distances(pred_pts, gt_pts)
    comp = nn_distances(gt_pts, pred_pts)
    precision, recall = float(np.mean(acc < thresh)), float(np.mean(comp < thresh))
    f = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "accuracy_mean_m": float(acc.mean()),
        "accuracy_median_m": float(np.median(acc)),
        "completeness_mean_m": float(comp.mean()),
        "chamfer_m": float(0.5 * (acc.mean() + comp.mean())),
        f"precision@{thresh * 100:.0f}cm": precision,
        f"recall@{thresh * 100:.0f}cm": recall,
        f"fscore@{thresh * 100:.0f}cm": f,
    }
