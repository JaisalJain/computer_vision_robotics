"""Compact 3D Gaussian Splatting toolkit built on gsplat's public API.

The pieces a 3DGS pipeline needs, small enough to read in one sitting:

* :func:`load_colmap_scene` - a COLMAP model (``sparse/0``, official ``pycolmap``) plus an image
  folder -> world-to-camera poses, intrinsics at the image resolution, sparse points and a
  train/test split (every 8th image held out, the Mip-NeRF 360 convention).
* :func:`init_points` - initial Gaussian centres: COLMAP sparse points, uniform random points,
  or a feed-forward point map (Depth Anything 3, made by ``make_da3_init.py``).
* :func:`create_splats` - the parameter set (means, log-scales, quaternions, logit-opacities,
  spherical-harmonic colour) with one Adam optimiser per parameter group.
* :func:`render` - a thin wrapper around :func:`gsplat.rasterization`.
* :func:`train` - the 3DGS optimisation loop: L1 + D-SSIM loss, SH degree schedule,
  adaptive density control by :class:`gsplat.strategy.DefaultStrategy`.
* :func:`evaluate` - PSNR / SSIM / LPIPS on held-out views (torchmetrics).
* :func:`save_checkpoint`, :func:`load_checkpoint`, :func:`export_ply` (standard 3DGS PLY).
* camera-path and geometry helpers for novel-view rendering and teaching
  (:func:`orbit_path`, :func:`closest_point_to_rays`, :func:`covariance_from_scale_rot`,
  :func:`project_covariance`).

Conventions: poses are OpenCV (x right, y down, z forward); ``w2c`` maps world -> camera,
exactly what COLMAP calls ``cam_from_world`` and gsplat calls ``viewmats``. Quaternions are
``wxyz`` (gsplat), images are float RGB in ``[0, 1]``.

Heavy dependencies (gsplat, pycolmap, torchmetrics) are imported inside the functions that
need them, so the pure-numpy helpers stay importable in CPU-only environments.
"""

from __future__ import annotations

import math
import os
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

PROJECT = "11_gaussian_splatting"


def ensure_gsplat_build_env() -> None:
    """Put the conda CUDA toolkit headers/libraries on the compiler search paths.

    gsplat has no prebuilt wheel for torch 2.11 / CUDA 12.8, so its CUDA extension is
    JIT-compiled on first use (cached in ``~/.cache/torch_extensions``). nvcc finds the
    toolkit by itself, but the host C++ compiler needs ``cuda_runtime.h`` from
    ``$CONDA_PREFIX/targets/x86_64-linux/include``. Setting these variables does not change
    the build flags, so a cached build is reused as is.
    """
    inc = Path(sys.prefix) / "targets" / "x86_64-linux" / "include"
    lib = Path(sys.prefix) / "lib"
    if not inc.is_dir():
        return
    for var, path in (("CPATH", inc), ("LIBRARY_PATH", lib)):
        parts = [p for p in os.environ.get(var, "").split(os.pathsep) if p]
        if str(path) not in parts:
            os.environ[var] = os.pathsep.join([str(path), *parts])


ensure_gsplat_build_env()
SH_C0 = 0.28209479177387814  # Y_0^0 = 1 / (2 sqrt(pi))
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


# ----------------------------------------------------------------------------- colour / SH
def rgb_to_sh(rgb):
    """RGB in [0, 1] -> degree-0 SH coefficient (works on numpy arrays and torch tensors)."""
    return (rgb - 0.5) / SH_C0


def sh_to_rgb(sh):
    """Degree-0 SH coefficient -> RGB (inverse of :func:`rgb_to_sh`)."""
    return sh * SH_C0 + 0.5


# ---------------------------------------------------------------------------- scene loading
@dataclass
class ColmapScene:
    """A posed image collection from a COLMAP model, resized to the training resolution."""

    scene_dir: Path
    names: list[str]  # COLMAP image names, sorted
    image_paths: list[Path]  # files actually loaded (possibly a downscaled copy)
    w2c: np.ndarray  # (N, 4, 4) world -> camera (OpenCV)
    K: np.ndarray  # (N, 3, 3) intrinsics at the loaded image resolution
    widths: np.ndarray  # (N,) pixels
    heights: np.ndarray  # (N,) pixels
    points: np.ndarray  # (M, 3) sparse SfM points (world frame)
    points_rgb: np.ndarray  # (M, 3) uint8
    points_track_len: np.ndarray  # (M,) number of images observing each point
    points_error: np.ndarray  # (M,) mean reprojection error [px at COLMAP resolution]
    train_idx: np.ndarray
    test_idx: np.ndarray
    camera_model: str
    colmap_size: tuple[int, int]  # (width, height) of the COLMAP camera

    @property
    def n_images(self) -> int:
        return len(self.names)

    @property
    def c2w(self) -> np.ndarray:
        """(N, 4, 4) camera -> world poses."""
        return np.linalg.inv(self.w2c)

    @property
    def centers(self) -> np.ndarray:
        """(N, 3) camera centres in the world frame."""
        return self.c2w[:, :3, 3]

    @property
    def scene_scale(self) -> float:
        """1.1 x the largest camera distance from the mean camera centre (as in gsplat)."""
        c = self.centers
        return float(1.1 * np.linalg.norm(c - c.mean(0), axis=1).max())

    def load_images(self, idx=None, max_side: int | None = None) -> list[np.ndarray]:
        """Read images (RGB uint8) for the given indices (default: all)."""
        from cvlab.io import read_image

        idx = range(self.n_images) if idx is None else idx
        return [read_image(self.image_paths[i], max_side=max_side) for i in idx]


def _match_image_files(image_dir: Path, names: list[str]) -> list[Path]:
    """Map COLMAP image names to files in ``image_dir`` (exact name first, then same stem)."""
    by_stem: dict[str, Path] = {}
    for p in image_dir.iterdir():
        if p.suffix in IMAGE_SUFFIXES:
            by_stem.setdefault(p.stem, p)
    out = []
    for n in names:
        exact = image_dir / n
        if exact.is_file():
            out.append(exact)
        elif Path(n).stem in by_stem:
            out.append(by_stem[Path(n).stem])
        else:
            raise FileNotFoundError(f"no image for COLMAP entry {n!r} in {image_dir}")
    return out


def load_colmap_scene(
    scene_dir: str | Path,
    images: str = "images_4",
    sparse: str = "sparse/0",
    test_every: int = 8,
) -> ColmapScene:
    """Load a COLMAP reconstruction and match it to an image folder of any resolution.

    Intrinsics are rescaled from the COLMAP camera size to the actual image size
    (``fx' = fx * W_img / W_colmap``, same for ``cx``; COLMAP and gsplat both place pixel
    centres at +0.5, so a plain scale is exact). Images are sorted by name and every
    ``test_every``-th one (index 0, 8, 16, ...) is held out for evaluation.

    Only undistorted pinhole cameras (``PINHOLE``, ``SIMPLE_PINHOLE``) are accepted; run
    ``pycolmap.undistort_images`` first otherwise (``sfm_from_images.py`` does this).
    """
    import pycolmap
    from PIL import Image

    scene_dir = Path(scene_dir)
    rec = pycolmap.Reconstruction(str(scene_dir / sparse))
    imgs = sorted(rec.images.values(), key=lambda im: im.name)
    names = [im.name for im in imgs]
    paths = _match_image_files(scene_dir / images, names)

    w2c, Ks, ws, hs, models = [], [], [], [], set()
    colmap_size = (0, 0)
    for im, path in zip(imgs, paths, strict=True):
        cam = rec.cameras[im.camera_id]
        models.add(cam.model.name)
        if cam.model.name not in ("PINHOLE", "SIMPLE_PINHOLE"):
            raise ValueError(
                f"camera model {cam.model.name} has distortion - undistort the images first"
            )
        pose = im.cam_from_world()
        T = np.eye(4)
        T[:3, :4] = pose.matrix()
        with Image.open(path) as pil:
            w, h = pil.size
        sx, sy = w / cam.width, h / cam.height
        K = np.asarray(cam.calibration_matrix(), dtype=np.float64).copy()
        K[0, :] *= sx
        K[1, :] *= sy
        w2c.append(T)
        Ks.append(K)
        ws.append(w)
        hs.append(h)
        colmap_size = (int(cam.width), int(cam.height))

    pts = list(rec.points3D.values())
    xyz = np.array([p.xyz for p in pts], dtype=np.float64).reshape(-1, 3)
    rgb = np.array([p.color for p in pts], dtype=np.uint8).reshape(-1, 3)
    track = np.array([p.track.length() for p in pts], dtype=np.int64)
    err = np.array([p.error for p in pts], dtype=np.float64)

    n = len(names)
    all_idx = np.arange(n)
    test = all_idx[all_idx % test_every == 0] if test_every > 0 else all_idx[:0]
    train = np.setdiff1d(all_idx, test)
    return ColmapScene(
        scene_dir=scene_dir, names=names, image_paths=paths,
        w2c=np.stack(w2c), K=np.stack(Ks), widths=np.array(ws), heights=np.array(hs),
        points=xyz, points_rgb=rgb, points_track_len=track, points_error=err,
        train_idx=train, test_idx=test, camera_model="/".join(sorted(models)),
        colmap_size=colmap_size,
    )  # fmt: skip


# --------------------------------------------------------------------------- initialisation
def init_points(
    scene: ColmapScene,
    kind: str,
    n_random: int = 100_000,
    random_extent: float = 3.0,
    da3_path: str | Path | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Initial Gaussian centres and colours (``float`` RGB in [0, 1]).

    * ``colmap``: the SfM sparse points with their colours.
    * ``random``: ``n_random`` points uniform in a cube of half-size
      ``random_extent * scene_scale`` around the mean camera centre, random colours
      (gsplat's ``init_type="random"``).
    * ``da3``: a feed-forward Depth Anything 3 point map, Sim(3)-aligned to the COLMAP
      cameras (``.npz`` with ``points`` and ``colors`` written by ``make_da3_init.py``).
    """
    rng = np.random.default_rng(seed)
    if kind == "colmap":
        return scene.points.astype(np.float32), scene.points_rgb.astype(np.float32) / 255.0
    if kind == "random":
        center = scene.centers.mean(0)
        half = random_extent * scene.scene_scale
        xyz = center + half * (rng.random((n_random, 3)) * 2.0 - 1.0)
        return xyz.astype(np.float32), rng.random((n_random, 3)).astype(np.float32)
    if kind == "da3":
        if da3_path is None:
            raise ValueError("kind='da3' needs da3_path (run make_da3_init.py)")
        data = np.load(da3_path)
        return data["points"].astype(np.float32), data["colors"].astype(np.float32)
    raise ValueError(f"unknown init kind: {kind}")


def knn_mean_dist(points: np.ndarray, k: int = 3) -> np.ndarray:
    """Mean distance of every point to its ``k`` nearest neighbours (excluding itself)."""
    from scipy.spatial import cKDTree

    d, _ = cKDTree(points).query(points, k=k + 1)
    return np.sqrt((d[:, 1:] ** 2).mean(1))


# ------------------------------------------------------------------------------ parameters
@dataclass
class TrainConfig:
    """Hyper-parameters (defaults follow gsplat's simple_trainer / the 3DGS paper)."""

    steps: int = 7000
    sh_degree: int = 3
    sh_degree_interval: int = 1000  # raise the SH degree by one every N steps
    ssim_lambda: float = 0.2  # loss = (1 - l) * L1 + l * (1 - SSIM)
    init_opacity: float = 0.1
    init_scale: float = 1.0  # initial scale = init_scale * mean distance to 3 neighbours
    lr_means: float = 1.6e-4  # multiplied by scene_scale, decays exponentially to 1%
    lr_scales: float = 5e-3
    lr_quats: float = 1e-3
    lr_opacities: float = 5e-2
    lr_sh0: float = 2.5e-3
    lr_shN: float = 2.5e-3 / 20
    packed: bool = False
    rasterize_mode: str = "classic"
    near_plane: float = 0.01
    far_plane: float = 1e10
    # adaptive density control (gsplat DefaultStrategy)
    prune_opa: float = 0.005
    grow_grad2d: float = 0.0002
    refine_start_iter: int = 500
    refine_stop_iter: int = 15000
    refine_every: int = 100
    reset_every: int = 3000
    absgrad: bool = False
    max_gaussians: int | None = 3_000_000  # stop densifying above this (8 GB safety cap)
    # bookkeeping
    log_every: int = 50
    eval_every: int = 1000  # test-set PSNR every N steps (0 = only at the end)
    seed: int = 0


def create_splats(
    points: np.ndarray,
    rgb: np.ndarray,
    scene_scale: float,
    cfg: TrainConfig,
    device: str = "cuda",
):
    """Build the Gaussian parameters and one Adam optimiser per group.

    Each Gaussian starts isotropic with a radius equal to the mean distance to its three
    nearest neighbours, a random rotation, opacity ``init_opacity`` and its point colour in
    the DC spherical-harmonic band (higher bands start at zero).
    """
    import torch

    g = torch.Generator().manual_seed(cfg.seed)
    pts = torch.as_tensor(np.asarray(points), dtype=torch.float32)
    cols = torch.as_tensor(np.asarray(rgb), dtype=torch.float32)
    dist = torch.as_tensor(knn_mean_dist(np.asarray(points, np.float64)), dtype=torch.float32)
    scales = torch.log(dist.clamp_min(1e-7) * cfg.init_scale)[:, None].repeat(1, 3)
    n = pts.shape[0]
    quats = torch.rand((n, 4), generator=g)
    opacities = torch.logit(torch.full((n,), cfg.init_opacity))
    sh = torch.zeros((n, (cfg.sh_degree + 1) ** 2, 3))
    sh[:, 0, :] = rgb_to_sh(cols)
    lrs = {
        "means": cfg.lr_means * scene_scale, "scales": cfg.lr_scales, "quats": cfg.lr_quats,
        "opacities": cfg.lr_opacities, "sh0": cfg.lr_sh0, "shN": cfg.lr_shN,
    }  # fmt: skip
    splats = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(pts),
            "scales": torch.nn.Parameter(scales),
            "quats": torch.nn.Parameter(quats),
            "opacities": torch.nn.Parameter(opacities),
            "sh0": torch.nn.Parameter(sh[:, :1, :].clone()),
            "shN": torch.nn.Parameter(sh[:, 1:, :].clone()),
        }
    ).to(device)
    optimizers = {
        k: torch.optim.Adam([{"params": splats[k], "lr": lr, "name": k}], eps=1e-15)
        for k, lr in lrs.items()
    }
    return splats, optimizers


def render(
    splats,
    viewmats,
    Ks,
    width: int,
    height: int,
    sh_degree: int,
    render_mode: str = "RGB",
    packed: bool = False,
    absgrad: bool = False,
    rasterize_mode: str = "classic",
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    background=None,
):
    """Rasterise the splats into ``C`` cameras.

    Args:
        viewmats: (C, 4, 4) world -> camera; Ks: (C, 3, 3).
        render_mode: ``"RGB"``, ``"RGB+ED"`` (colour + expected depth), ``"ED"``, ...

    Returns:
        ``(renders (C, H, W, 3|4), alphas (C, H, W, 1), info)`` from :func:`gsplat.rasterization`.
    """
    import torch
    from gsplat import rasterization

    colors = torch.cat([splats["sh0"], splats["shN"]], dim=1)
    return rasterization(
        means=splats["means"],
        quats=splats["quats"],  # normalised inside gsplat
        scales=torch.exp(splats["scales"]),
        opacities=torch.sigmoid(splats["opacities"]),
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=int(width),
        height=int(height),
        sh_degree=sh_degree,
        packed=packed,
        absgrad=absgrad,
        rasterize_mode=rasterize_mode,
        render_mode=render_mode,
        near_plane=near_plane,
        far_plane=far_plane,
        backgrounds=background,
    )


# ---------------------------------------------------------------------------------- losses
def _gauss_1d(size: int, sigma: float, device=None, dtype=None):
    import torch

    x = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    g = torch.exp(-(x**2) / (2 * sigma**2))
    return g / g.sum()


def ssim(x, y, data_range: float = 1.0, size: int = 11, sigma: float = 1.5):
    """Mean SSIM of two ``(B, C, H, W)`` tensors over the valid region (no padding).

    Same definition as ``torchmetrics`` (11x11 Gaussian window, sigma 1.5, K1=0.01,
    K2=0.03) but differentiable and lightweight enough for the training loss.
    """
    import torch.nn.functional as F

    c = x.shape[1]
    g = _gauss_1d(size, sigma, x.device, x.dtype)
    wx = g.view(1, 1, 1, size).expand(c, 1, 1, size)
    wy = g.view(1, 1, size, 1).expand(c, 1, size, 1)

    def blur(t):
        return F.conv2d(F.conv2d(t, wx, groups=c), wy, groups=c)

    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mu_x, mu_y = blur(x), blur(y)
    s_xx = blur(x * x) - mu_x**2
    s_yy = blur(y * y) - mu_y**2
    s_xy = blur(x * y) - mu_x * mu_y
    num = (2 * mu_x * mu_y + c1) * (2 * s_xy + c2)
    den = (mu_x**2 + mu_y**2 + c1) * (s_xx + s_yy + c2)
    return (num / den).mean()


# --------------------------------------------------------------------------------- training
@dataclass
class TrainResult:
    """Everything a run produces besides the parameters themselves."""

    history: list[dict] = field(default_factory=list)  # every log_every steps
    eval_history: list[dict] = field(default_factory=list)  # every eval_every steps
    train_time_s: float = 0.0  # optimisation wall time, evaluation passes excluded
    peak_vram_mb: float = 0.0
    n_init: int = 0
    densify_capped_at_step: int | None = None


def images_to_tensors(images: list[np.ndarray]):
    """uint8 HxWx3 numpy images -> list of pinned CPU uint8 tensors (fast H2D copies)."""
    import torch

    out = []
    for im in images:
        t = torch.from_numpy(np.ascontiguousarray(im))
        out.append(t.pin_memory() if torch.cuda.is_available() else t)
    return out


def train(
    scene: ColmapScene,
    images: list,
    points: np.ndarray,
    rgb: np.ndarray,
    cfg: TrainConfig,
    device: str = "cuda",
    log: Callable[[str], None] | None = print,
    eval_images: list | None = None,
):
    """Optimise 3D Gaussians on the training views of ``scene``.

    Args:
        images: per-image uint8 tensors (HxWx3), indexed like ``scene.names``; only the
            training indices are used for optimisation.
        points, rgb: initial centres (M, 3) and colours in [0, 1].
        eval_images: if given, test-set PSNR is logged every ``cfg.eval_every`` steps.

    Returns:
        ``(splats, TrainResult)``.
    """
    import torch
    from gsplat.strategy import DefaultStrategy

    torch.manual_seed(cfg.seed)
    scale = scene.scene_scale
    splats, optimizers = create_splats(points, rgb, scale, cfg, device)
    strategy = DefaultStrategy(
        prune_opa=cfg.prune_opa, grow_grad2d=cfg.grow_grad2d,
        refine_start_iter=cfg.refine_start_iter, refine_stop_iter=cfg.refine_stop_iter,
        refine_every=cfg.refine_every, reset_every=cfg.reset_every, absgrad=cfg.absgrad,
        verbose=False,
    )  # fmt: skip
    strategy.check_sanity(splats, optimizers)
    state = strategy.initialize_state(scene_scale=scale)
    sched = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"], gamma=0.01 ** (1.0 / cfg.steps)
    )
    viewmats = torch.as_tensor(scene.w2c, dtype=torch.float32, device=device)
    Ks = torch.as_tensor(scene.K, dtype=torch.float32, device=device)
    res = TrainResult(n_init=int(points.shape[0]))
    rng = np.random.default_rng(cfg.seed)
    order: list[int] = []

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()
    t_eval = 0.0
    for step in range(cfg.steps):
        if not order:
            order = list(rng.permutation(scene.train_idx))
        i = int(order.pop())
        gt = images[i].to(device, non_blocking=True).float() / 255.0  # (H, W, 3)
        sh_deg = min(step // cfg.sh_degree_interval, cfg.sh_degree)
        renders, _, info = render(
            splats, viewmats[i : i + 1], Ks[i : i + 1], scene.widths[i], scene.heights[i],
            sh_deg, packed=cfg.packed, absgrad=cfg.absgrad, rasterize_mode=cfg.rasterize_mode,
            near_plane=cfg.near_plane, far_plane=cfg.far_plane,
        )  # fmt: skip
        pred = renders[0]
        strategy.step_pre_backward(splats, optimizers, state, step, info)
        l1 = (pred - gt).abs().mean()
        d_ssim = 1.0 - ssim(pred.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None])
        loss = (1.0 - cfg.ssim_lambda) * l1 + cfg.ssim_lambda * d_ssim
        loss.backward()
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        sched.step()
        strategy.step_post_backward(splats, optimizers, state, step, info, packed=cfg.packed)

        n_g = splats["means"].shape[0]
        if (
            cfg.max_gaussians is not None and n_g > cfg.max_gaussians
            and res.densify_capped_at_step is None
        ):  # fmt: skip
            strategy.refine_stop_iter = step  # no more growth (pruning stops too)
            res.densify_capped_at_step = step
            if log:
                log(f"step {step}: {n_g} Gaussians > cap {cfg.max_gaussians}, densification off")
        last = step == cfg.steps - 1
        if step % cfg.log_every == 0 or last:
            with torch.no_grad():
                mse = ((pred.detach().clamp(0, 1) - gt) ** 2).mean().item()
            res.history.append(
                {
                    "step": step + 1,
                    "loss": float(loss.item()),
                    "l1": float(l1.item()),
                    "train_psnr": float(-10 * math.log10(max(mse, 1e-12))),
                    "n_gaussians": int(n_g),
                    "sh_degree": sh_deg,
                    "time_s": time.perf_counter() - t_start - t_eval,
                    "vram_mb": torch.cuda.max_memory_allocated() / 2**20
                    if torch.cuda.is_available() else 0.0,
                }
            )  # fmt: skip
        do_eval = eval_images is not None and cfg.eval_every > 0
        if do_eval and ((step + 1) % cfg.eval_every == 0 or last):
            t0 = time.perf_counter()
            m = evaluate(splats, scene, eval_images, scene.test_idx, sh_deg, device, lpips=False)
            res.eval_history.append(
                {"step": step + 1, "psnr": m["psnr"], "ssim": m["ssim"], "n_gaussians": n_g}
            )
            t_eval += time.perf_counter() - t0
            if log:
                h = res.history[-1]
                log(
                    f"step {step + 1:>6} | loss {h['loss']:.4f} | test PSNR {m['psnr']:.2f} dB"
                    f" | {n_g:>8} Gaussians | {h['time_s']:.0f} s"
                )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    res.train_time_s = time.perf_counter() - t_start - t_eval
    res.peak_vram_mb = (
        torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0
    )
    return splats, res


# ------------------------------------------------------------------------------- evaluation
_LPIPS: dict[str, Any] = {}


def lpips_metric(device: str = "cuda"):
    """LPIPS (AlexNet, inputs in [0, 1]); backbone weights cached in ``weights/torch_hub``."""
    import torch
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    from cvlab.io import weights_dir

    if device not in _LPIPS:
        torch.hub.set_dir(str(weights_dir("torch_hub", create=True)))
        m = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True)
        _LPIPS[device] = m.to(device).eval()
    return _LPIPS[device]


def evaluate(
    splats,
    scene: ColmapScene,
    images: list,
    idx,
    sh_degree: int = 3,
    device: str = "cuda",
    lpips: bool = True,
    return_renders: bool = False,
) -> dict[str, Any]:
    """Render the views ``idx`` and compare with the photos (mean of per-image metrics)."""
    import torch
    from torchmetrics.functional.image import (
        peak_signal_noise_ratio,
        structural_similarity_index_measure,
    )

    viewmats = torch.as_tensor(scene.w2c, dtype=torch.float32, device=device)
    Ks = torch.as_tensor(scene.K, dtype=torch.float32, device=device)
    lp = lpips_metric(device) if lpips else None
    per, renders = [], []
    with torch.no_grad():
        for i in idx:
            i = int(i)
            out, _, _ = render(splats, viewmats[i : i + 1], Ks[i : i + 1], scene.widths[i],
                               scene.heights[i], sh_degree)  # fmt: skip
            pred = out[0].clamp(0, 1).permute(2, 0, 1)[None]  # (1, 3, H, W)
            gt = (images[i].to(device).float() / 255.0).permute(2, 0, 1)[None]
            row = {
                "index": i,
                "psnr": float(peak_signal_noise_ratio(pred, gt, data_range=1.0)),
                "ssim": float(structural_similarity_index_measure(pred, gt, data_range=1.0)),
            }
            if lp is not None:
                row["lpips"] = float(lp(pred, gt))
            per.append(row)
            if return_renders:
                renders.append((pred[0].permute(1, 2, 0).cpu().numpy() * 255).round()
                               .astype(np.uint8))  # fmt: skip
    out: dict[str, Any] = {
        k: float(np.mean([r[k] for r in per])) for k in per[0] if k != "index"
    }
    out["per_image"] = per
    if return_renders:
        out["renders"] = renders
    return out


# --------------------------------------------------------------------------- persistence
def save_checkpoint(path: str | Path, splats, cfg: TrainConfig, extra: dict | None = None) -> Path:
    """Save the raw parameters (+ config) with ``torch.save``."""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in splats.items()}
    torch.save({"splats": state, "config": asdict(cfg), "extra": extra or {}}, path)
    return path


def load_checkpoint(path: str | Path, device: str = "cuda"):
    """Load a checkpoint written by :func:`save_checkpoint` -> ``(splats, config_dict)``."""
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    splats = torch.nn.ParameterDict(
        {k: torch.nn.Parameter(v, requires_grad=False) for k, v in ckpt["splats"].items()}
    ).to(device)
    return splats, ckpt["config"]


def export_ply(splats, path: str | Path) -> Path:
    """Write the standard 3DGS PLY (x,y,z, f_dc_*, f_rest_*, opacity, scale_*, rot_*).

    Values are stored raw, as the reference implementation and the common web viewers
    (SuperSplat, antimatter15, PlayCanvas) expect: log-scales, logit-opacities, SH coeffs.
    """
    from gsplat import export_splats

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    export_splats(
        means=splats["means"].detach(), scales=splats["scales"].detach(),
        quats=splats["quats"].detach(), opacities=splats["opacities"].detach(),
        sh0=splats["sh0"].detach(), shN=splats["shN"].detach(), format="ply", save_to=str(path),
    )  # fmt: skip
    return path


def splat_stats(splats) -> dict[str, float]:
    """Summary statistics of a trained model (count, opacity and size distribution)."""
    import torch

    with torch.no_grad():
        op = torch.sigmoid(splats["opacities"]).float().cpu().numpy()
        sc = torch.exp(splats["scales"]).max(dim=1).values.float().cpu().numpy()
    return {
        "n_gaussians": int(op.size),
        "median_opacity": float(np.median(op)),
        "frac_opacity_gt_0.5": float((op > 0.5).mean()),
        "median_max_scale": float(np.median(sc)),
    }


# ------------------------------------------------------------------- geometry / camera paths
def quat_to_rotmat_wxyz(q: np.ndarray) -> np.ndarray:
    """(…, 4) quaternions ``wxyz`` (any norm) -> (…, 3, 3) rotation matrices (gsplat order)."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    R = np.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    )  # fmt: skip
    return R.reshape(*q.shape[:-1], 3, 3)


def covariance_from_scale_rot(scales: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """3D covariance ``Sigma = R S S^T R^T`` (always symmetric positive semi-definite)."""
    R = quat_to_rotmat_wxyz(quat_wxyz)
    S = np.diag(np.asarray(scales, dtype=np.float64))
    M = R @ S
    return M @ M.T


def project_covariance(
    mean_w: np.ndarray, cov_w: np.ndarray, w2c: np.ndarray, K: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """EWA splatting: project a 3D Gaussian to the image (Zwicker et al. 2002).

    ``Sigma_2D = J W Sigma W^T J^T`` with ``W`` the world->camera rotation and ``J`` the
    Jacobian of the perspective projection at the Gaussian centre ``(x, y, z)`` (camera frame):

    ``J = [[fx/z, 0, -fx x / z^2], [0, fy/z, -fy y / z^2]]``.

    Returns:
        ``(mean_2d (2,), cov_2d (2, 2))`` in pixels (no low-pass dilation added).
    """
    W = np.asarray(w2c, dtype=np.float64)[:3, :3]
    t = np.asarray(w2c, dtype=np.float64)[:3, 3]
    x, y, z = W @ np.asarray(mean_w, dtype=np.float64) + t
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    J = np.array([[fx / z, 0.0, -fx * x / z**2], [0.0, fy / z, -fy * y / z**2]])
    cov2d = J @ W @ np.asarray(cov_w, dtype=np.float64) @ W.T @ J.T
    mean2d = np.array([fx * x / z + cx, fy * y / z + cy])
    return mean2d, cov2d


def closest_point_to_rays(origins: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """Least-squares point closest to a bundle of 3D lines (the cameras' look-at centre).

    Minimises ``sum_i ||(I - d_i d_i^T)(p - o_i)||^2``  ->  ``(sum A_i) p = sum A_i o_i``.
    """
    o = np.asarray(origins, dtype=np.float64).reshape(-1, 3)
    d = np.asarray(dirs, dtype=np.float64).reshape(-1, 3)
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    A = np.eye(3)[None] - d[:, :, None] * d[:, None, :]
    return np.linalg.solve(A.sum(0), (A @ o[:, :, None]).sum(0)[:, 0])


def fit_up_vector(c2w: np.ndarray) -> np.ndarray:
    """World 'up' estimated from the cameras: minus the mean camera y axis (OpenCV y = down)."""
    up = -np.asarray(c2w)[:, :3, 1].mean(0)
    return up / np.linalg.norm(up)


def orbit_path(
    scene: ColmapScene,
    n_frames: int = 60,
    radius_scale: float = 1.0,
    height_offset: float = 0.0,
) -> np.ndarray:
    """Novel camera poses on a circle around the scene centre, looking at it.

    The centre is the point closest to all training optical axes, the circle lies in the
    plane orthogonal to the estimated up vector at the mean camera height, and its radius
    is the mean horizontal camera distance (times ``radius_scale``).

    Returns:
        (n_frames, 4, 4) world -> camera matrices (ready for :func:`render`).
    """
    from cvlab.geometry import invert_T, look_at

    c2w = scene.c2w
    centers = c2w[:, :3, 3]
    target = closest_point_to_rays(centers, c2w[:, :3, 2])
    up = fit_up_vector(c2w)
    rel = centers - target
    h = rel @ up  # signed heights along up
    horiz = rel - h[:, None] * up
    radius = np.linalg.norm(horiz, axis=1).mean() * radius_scale
    e1 = horiz[0] / np.linalg.norm(horiz[0])
    e2 = np.cross(up, e1)
    poses = []
    for a in np.linspace(0.0, 2 * np.pi, n_frames, endpoint=False):
        eye = target + radius * (np.cos(a) * e1 + np.sin(a) * e2) + (h.mean() + height_offset) * up
        poses.append(invert_T(look_at(eye, target, up=up)))  # y axis = -up (OpenCV)
    return np.stack(poses)
