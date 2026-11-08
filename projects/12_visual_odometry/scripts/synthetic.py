"""Tiny ray-cast RGB-D sequence with exact ground truth, for unit tests and sanity checks.

A pinhole camera moves inside a textured box (walls, floor, ceiling). Every pixel ray is
intersected with the five planes analytically, so the depth map and the poses are exact and
any error left after running a VO pipeline comes from the pipeline itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from cvlab.geometry import PinholeCamera, look_at, make_T, so3_exp

# planes as (axis, offset, normal sign): the camera looks along +z of the world
_PLANES = [(2, 6.0), (1, 1.2), (1, -1.6), (0, -2.5), (0, 2.5)]


@dataclass
class SyntheticSequence:
    """Duck-typed stand-in for :class:`cvlab.tum.TumSequence` (rgb paths, depth, GT)."""

    rgb_paths: list[Path]
    depths: list[np.ndarray]
    timestamps: np.ndarray
    gt_poses: np.ndarray
    camera: PinholeCamera
    dist: np.ndarray = field(default_factory=lambda: np.zeros(5))

    def __len__(self) -> int:
        return len(self.timestamps)

    def depth(self, i: int) -> np.ndarray:
        return self.depths[i]

    def rgb(self, i: int) -> np.ndarray:
        return cv2.cvtColor(cv2.imread(str(self.rgb_paths[i])), cv2.COLOR_BGR2RGB)


def _texture(rng: np.random.Generator, size: int = 1024) -> np.ndarray:
    """Multi-scale random blobs: corners at every scale for ORB / SIFT."""
    tex = np.zeros((size, size), np.float32)
    for cells, weight in ((8, 0.4), (32, 0.3), (128, 0.2), (512, 0.1)):
        small = rng.random((cells, cells)).astype(np.float32)
        tex += weight * cv2.resize(small, (size, size), interpolation=cv2.INTER_CUBIC)
    for _ in range(300):  # random dark/bright discs give distinctive, non-repetitive corners
        cx, cy = rng.integers(0, size, 2)
        cv2.circle(tex, (int(cx), int(cy)), int(rng.integers(3, 14)), float(rng.random()), -1)
    tex = cv2.GaussianBlur(tex, (0, 0), 1.0)
    return (255 * (tex - tex.min()) / (np.ptp(tex) + 1e-6)).astype(np.float32)


def render(T_wc: np.ndarray, cam: PinholeCamera, textures: list[np.ndarray], px_per_m: float):
    """Ray-cast the textured box -> ``(gray uint8 HxW, depth float32 HxW in metres)``."""
    u, v = np.meshgrid(np.arange(cam.width), np.arange(cam.height))
    rays_c = np.stack([(u - cam.cx) / cam.fx, (v - cam.cy) / cam.fy, np.ones_like(u, float)], -1)
    rays_w = rays_c @ T_wc[:3, :3].T  # direction in world, z_cam component = 1 per unit lambda
    c = T_wc[:3, 3]
    lam = np.full(u.shape, np.inf)
    plane_id = np.full(u.shape, -1)
    for k, (axis, off) in enumerate(_PLANES):
        with np.errstate(divide="ignore", invalid="ignore"):
            lk = (off - c[axis]) / rays_w[..., axis]
        better = (lk > 1e-6) & (lk < lam)
        lam[better] = lk[better]
        plane_id[better] = k
    pts = c + rays_w * lam[..., None]
    img = np.zeros(u.shape, np.float32)
    size = textures[0].shape[0]
    for k, (axis, _) in enumerate(_PLANES):
        sel = plane_id == k
        a, b = [ax for ax in range(3) if ax != axis]
        mu = (pts[..., a] * px_per_m + size / 2).astype(np.float32)
        mv = (pts[..., b] * px_per_m + size / 2).astype(np.float32)
        warped = cv2.remap(textures[k], mu, mv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        img[sel] = warped[sel]
    depth = lam.astype(np.float32)  # lambda along a ray with z_cam = 1 is the z-depth
    return np.clip(img, 0, 255).astype(np.uint8), depth


def make_sequence(
    out_dir: str | Path,
    n_frames: int = 40,
    width: int = 320,
    height: int = 240,
    fps: float = 30.0,
    seed: int = 0,
) -> SyntheticSequence:
    """Render a smooth hand-held-like trajectory and write the frames as PNG files."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    cam = PinholeCamera.from_fov(width, height, 65.0)
    textures = [_texture(rng) for _ in _PLANES]
    poses, paths, depths = [], [], []
    for k in range(n_frames):
        s = k / fps
        eye = np.array([0.8 * np.sin(1.3 * s), 0.2 * np.sin(2.1 * s), 0.8 * s])
        target = eye + np.array([0.15 * np.sin(0.9 * s), 0.05, 4.0])
        T_wc = look_at(eye, target, up=(0.0, -1.0, 0.0))
        T_wc = T_wc @ make_T(so3_exp([0.0, 0.0, 0.05 * np.sin(1.7 * s)]), np.zeros(3))  # roll
        gray, depth = render(T_wc, cam, textures, px_per_m=96.0)
        path = out_dir / f"{k:04d}.png"
        cv2.imwrite(str(path), gray)
        poses.append(T_wc)
        paths.append(path)
        depths.append(depth)
    return SyntheticSequence(
        rgb_paths=paths,
        depths=depths,
        timestamps=np.arange(n_frames) / fps,
        gt_poses=np.stack(poses),
        camera=cam,
    )
