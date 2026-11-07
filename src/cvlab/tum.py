"""TUM RGB-D benchmark helpers: file lists, timestamp association, depth, intrinsics, GT poses.

Dataset: Sturm et al., "A Benchmark for the Evaluation of RGB-D SLAM Systems", IROS 2012
(https://cvg.cit.tum.de/data/datasets/rgbd-dataset, CC BY 4.0).

* ``rgb/*.png``: 640x480 colour; ``depth/*.png``: 16-bit, registered to the RGB camera,
  ``depth_m = value / 5000`` (0 = missing).
* ``groundtruth.txt``: ``timestamp tx ty tz qx qy qz qw`` = camera-to-world pose of the RGB
  optical centre (motion-capture frame), quaternion in (x, y, z, w) order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .geometry import PinholeCamera, make_T, quat_to_R

DEPTH_SCALE = 5000.0

# Calibrated RGB intrinsics (fx, fy, cx, cy) and OpenCV distortion (k1, k2, p1, p2, k3).
# Source: https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats
TUM_INTRINSICS = {
    "fr1": ((517.3, 516.5, 318.6, 255.3), (0.2624, -0.9531, -0.0054, 0.0026, 1.1633)),
    "fr2": ((520.9, 521.0, 325.1, 249.7), (0.2312, -0.7849, -0.0033, -0.0001, 0.9172)),
    "fr3": ((535.4, 539.2, 320.1, 247.6), (0.0, 0.0, 0.0, 0.0, 0.0)),
    "ros_default": ((525.0, 525.0, 319.5, 239.5), (0.0, 0.0, 0.0, 0.0, 0.0)),
}


def camera_for(sequence: str | Path) -> tuple[PinholeCamera, np.ndarray]:
    """Intrinsics and distortion for a sequence name such as ``rgbd_dataset_freiburg1_xyz``."""
    name = Path(sequence).name
    key = next((k for k in ("fr1", "fr2", "fr3") if f"freiburg{k[-1]}" in name), "ros_default")
    (fx, fy, cx, cy), dist = TUM_INTRINSICS[key]
    return PinholeCamera(fx, fy, cx, cy, 640, 480), np.array(dist)


def read_file_list(path: str | Path) -> dict[float, list[str]]:
    """Parse a TUM list file (``timestamp data...``), skipping ``#`` comments."""
    out: dict[float, list[str]] = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        out[float(parts[0])] = parts[1:]
    return out


def associate(
    first: dict[float, list[str]] | list[float],
    second: dict[float, list[str]] | list[float],
    max_dt: float = 0.02,
    offset: float = 0.0,
) -> list[tuple[float, float]]:
    """Greedy one-to-one timestamp matching (same algorithm as TUM's ``associate.py``)."""
    a = sorted(first)
    b = sorted(second)
    b_arr = np.asarray(b)
    candidates = []
    for ta in a:
        lo, hi = np.searchsorted(b_arr, [ta + offset - max_dt, ta + offset + max_dt])
        for tb in b_arr[lo:hi]:
            candidates.append((abs(ta - (tb - offset)), ta, float(tb)))
    candidates.sort()
    used_a, used_b, matches = set(), set(), []
    for _, ta, tb in candidates:
        if ta not in used_a and tb not in used_b:
            used_a.add(ta)
            used_b.add(tb)
            matches.append((ta, tb))
    return sorted(matches)


def read_depth(path: str | Path) -> np.ndarray:
    """Read a TUM depth PNG as float32 metres (0 = invalid)."""
    import cv2

    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(path)
    return raw.astype(np.float32) / DEPTH_SCALE


def read_trajectory(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a TUM trajectory file -> ``(timestamps N, poses Nx4x4 camera-to-world)``."""
    data = read_file_list(path)
    ts = np.array(sorted(data))
    poses = np.empty((len(ts), 4, 4))
    for i, t in enumerate(ts):
        tx, ty, tz, qx, qy, qz, qw = map(float, data[t][:7])
        poses[i] = make_T(quat_to_R([qx, qy, qz, qw]), [tx, ty, tz])
    return ts, poses


def write_trajectory(path: str | Path, timestamps: np.ndarray, poses: np.ndarray) -> Path:
    """Write camera-to-world poses in TUM format (readable by ``evo``)."""
    from .geometry import R_to_quat

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for t, T in zip(timestamps, poses, strict=True):
        q = R_to_quat(T[:3, :3])
        x, y, z = T[:3, 3]
        lines.append(f"{t:.6f} {x:.6f} {y:.6f} {z:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}")
    path.write_text("\n".join(lines) + "\n")
    return path


@dataclass
class TumSequence:
    """Associated RGB / depth / ground-truth frames of one TUM sequence."""

    root: Path
    timestamps: np.ndarray  # RGB timestamps
    rgb_paths: list[Path]
    depth_paths: list[Path]
    gt_poses: np.ndarray | None  # Nx4x4 camera-to-world, or None if GT is missing
    camera: PinholeCamera
    dist: np.ndarray
    extra: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.timestamps)

    def rgb(self, i: int) -> np.ndarray:
        from .io import read_image

        return read_image(self.rgb_paths[i])

    def depth(self, i: int) -> np.ndarray:
        return read_depth(self.depth_paths[i])


def load_sequence(root: str | Path, max_dt: float = 0.02, require_gt: bool = True) -> TumSequence:
    """Associate RGB, depth and ground truth of a TUM sequence directory.

    Frames without a depth match (or without a GT pose when ``require_gt``) are dropped.
    """
    root = Path(root)
    rgb = read_file_list(root / "rgb.txt")
    depth = read_file_list(root / "depth.txt")
    pairs = associate(rgb, depth, max_dt=max_dt)
    gt_file = root / "groundtruth.txt"
    gt_poses = None
    if gt_file.exists():
        gt_ts, gt_T = read_trajectory(gt_file)
        idx = {t: i for i, t in enumerate(gt_ts)}
        gt_pairs = dict(associate([p[0] for p in pairs], list(gt_ts), max_dt=max_dt))
        keep = [p for p in pairs if p[0] in gt_pairs] if require_gt else pairs
        gt_poses = np.stack([gt_T[idx[gt_pairs[t]]] for t, _ in keep]) if require_gt else None
        pairs = keep
    cam, dist = camera_for(root)
    return TumSequence(
        root=root,
        timestamps=np.array([p[0] for p in pairs]),
        rgb_paths=[root / rgb[p[0]][0] for p in pairs],
        depth_paths=[root / depth[p[1]][0] for p in pairs],
        gt_poses=gt_poses,
        camera=cam,
        dist=dist,
    )
