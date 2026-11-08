"""CoTracker3 point tracking helpers for project 06.

The CoTracker code is the pinned clone ``third_party/co-tracker`` (commit 82e02e8, made by
``scripts/setup_third_party.sh``); weights come from the Hugging Face repo
``facebook/cotracker3``. Building the predictor ourselves (instead of ``torch.hub.load``)
avoids cloning the moving ``main`` branch at runtime.

Conventions: ``frames`` is a list of HxWx3 uint8 RGB images; tracks are returned as NumPy
arrays ``(T, N, 2)`` in pixel coordinates of the input frames, visibility as ``(T, N)``
booleans.
"""

from __future__ import annotations

import time

import numpy as np

from cvlab.download import hf_download
from cvlab.io import add_third_party

WEIGHTS = {"offline": "scaled_offline.pth", "online": "scaled_online.pth"}


def load_cotracker(mode: str = "offline", device: str = "cuda"):
    """Build a CoTracker3 predictor (``mode`` = ``"offline"`` or ``"online"``) with HF weights."""
    import torch

    add_third_party("co-tracker")
    from cotracker.predictor import CoTrackerOnlinePredictor, CoTrackerPredictor

    if mode == "offline":  # whole clip at once, bidirectional attention over time
        model = CoTrackerPredictor(checkpoint=None, window_len=60, v2=False)
    elif mode == "online":  # causal sliding window of 16 frames, step 8
        model = CoTrackerOnlinePredictor(checkpoint=None, window_len=16, v2=False)
    else:
        raise ValueError(mode)
    state = torch.load(hf_download("facebook/cotracker3", WEIGHTS[mode]), map_location="cpu")
    model.model.load_state_dict(state)
    return model.to(device).eval()


def _video(frames: list[np.ndarray], device) -> torch.Tensor:  # noqa: F821
    import torch

    v = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)[None].float()  # (1, T, 3, H, W)
    return v.to(device)


def _queries(queries: np.ndarray | None, device):
    import torch

    if queries is None:
        return None
    return torch.as_tensor(np.asarray(queries, np.float32))[None].to(device)  # (1, N, 3) t,x,y


def _sync() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def track_offline(
    model,
    frames: list[np.ndarray],
    grid_size: int = 0,
    queries: np.ndarray | None = None,
    segm_mask: np.ndarray | None = None,
    backward_tracking: bool = False,
) -> dict:
    """Offline CoTracker3 on a clip: a ``grid_size`` x ``grid_size`` grid on frame 0, or
    explicit ``queries`` rows ``(t, x, y)``; ``segm_mask`` keeps grid points inside a mask."""
    import torch

    from cvlab.device import track_vram

    dev = next(model.parameters()).device
    video = _video(frames, dev)
    mask = None
    if segm_mask is not None:
        mask = torch.from_numpy(np.asarray(segm_mask, np.float32))[None, None].to(dev)
    with track_vram() as mem:
        t0 = time.perf_counter()
        tracks, vis = model(video, queries=_queries(queries, dev), grid_size=grid_size,
                            segm_mask=mask, backward_tracking=backward_tracking)  # fmt: skip
        _sync()
        seconds = time.perf_counter() - t0
    return {"tracks": tracks[0].cpu().numpy(), "visible": vis[0].cpu().numpy().astype(bool),
            "seconds": seconds, "peak_vram_mb": mem.peak_mb}  # fmt: skip


def track_online(model, frames: list[np.ndarray], grid_size: int = 10,
                 queries: np.ndarray | None = None) -> dict:  # fmt: skip
    """Online (streaming) CoTracker3: frames arrive one by one and every ``step`` frames the
    last ``2 * step`` frames are processed (the loop of the official ``online_demo.py``).
    Only one window lives on the GPU at a time, so memory does not grow with clip length."""
    from cvlab.device import track_vram

    dev = next(model.parameters()).device
    step, window, first = model.step, [], True
    q = _queries(queries, dev)

    def process(chunk: list[np.ndarray], is_first: bool):
        return model(_video(chunk, dev), is_first_step=is_first, queries=q, grid_size=grid_size)

    with track_vram() as mem:
        t0 = time.perf_counter()
        i = 0
        for i, f in enumerate(frames):
            if i % step == 0 and i != 0:
                process(window[-step * 2 :], first)
                first = False
            window.append(f)
        tracks, vis = process(window[-(i % step) - step - 1 :], first)
        _sync()
        seconds = time.perf_counter() - t0
    return {"tracks": tracks[0].cpu().numpy(), "visible": vis[0].cpu().numpy().astype(bool),
            "seconds": seconds, "peak_vram_mb": mem.peak_mb}  # fmt: skip


def forward_backward_error(model, frames: list[np.ndarray], fwd: dict) -> np.ndarray:
    """Cycle consistency without ground truth: track the last-frame positions of ``fwd`` back
    to frame 0 (on the time-reversed clip) and return the distance to the start points.

    Points not visible in the last frame get ``NaN``.
    """
    tracks, vis = fwd["tracks"], fwd["visible"]
    q = np.column_stack([np.zeros(tracks.shape[1]), tracks[-1]])  # (N, 3): t=0 of reversed clip
    back = track_offline(model, frames[::-1], queries=q)
    err = np.linalg.norm(back["tracks"][-1] - tracks[0], axis=1)
    err[~vis[-1]] = np.nan
    return err


def ego_motion_residuals(tracks: np.ndarray, visible: np.ndarray, ref: int = 0,
                         ransac_px: float = 3.0) -> tuple[np.ndarray, list]:  # fmt: skip
    """Explain point motion by the camera: fit a homography ``H_t`` (frame ``ref`` -> ``t``)
    with RANSAC on all points visible in both frames and return per-point residuals.

    For a rotating camera or a distant / planar scene, static points follow ``x_t ~ H_t x_ref``
    exactly; independently moving objects (people) do not, so their residual is large.

    Returns:
        residuals (T, N) in pixels (NaN where not visible) and the list of ``H_t``.
    """
    import cv2

    T, N, _ = tracks.shape
    res = np.full((T, N), np.nan)
    Hs = []
    p0 = tracks[ref].astype(np.float32)
    for t in range(T):
        both = visible[ref] & visible[t]
        if both.sum() < 8:
            Hs.append(None)
            continue
        H, _ = cv2.findHomography(p0[both], tracks[t][both].astype(np.float32), cv2.RANSAC,
                                  ransac_px)  # fmt: skip
        Hs.append(H)
        if H is None:
            continue
        proj = cv2.perspectiveTransform(p0[None], H)[0]
        r = np.linalg.norm(proj - tracks[t], axis=1)
        r[~both] = np.nan
        res[t] = r
    return res, Hs


def points_in_boxes(points: np.ndarray, boxes_xywh: np.ndarray) -> np.ndarray:
    """(N,) bool: is each ``(x, y)`` point inside any ``(x, y, w, h)`` box?"""
    p = np.asarray(points)[:, None, :]
    b = np.asarray(boxes_xywh, dtype=np.float64).reshape(1, -1, 4)
    inside = (p[..., 0] >= b[..., 0]) & (p[..., 0] <= b[..., 0] + b[..., 2])
    inside &= (p[..., 1] >= b[..., 1]) & (p[..., 1] <= b[..., 1] + b[..., 3])
    return inside.any(axis=1)
