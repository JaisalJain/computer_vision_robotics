"""COCO-17 human keypoint helpers: names, skeleton, OKS, YOLO-pose labels, drawing."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

KPT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]  # fmt: skip
# Per-keypoint standard deviations from the COCO evaluation (cocoeval.py): eyes are easy to
# localise precisely (small sigma), hips are ambiguous under clothing (large sigma).
COCO_SIGMAS = np.array(
    [0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72,
     0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89]
) / 10.0  # fmt: skip
# Limbs as 0-based keypoint index pairs (COCO skeleton).
SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7),
    (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
]  # fmt: skip
# Keypoint permutation for a horizontal flip (left <-> right).
FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


def side(i: int) -> int:
    """0 = centre (nose), 1 = left body side, 2 = right body side."""
    return 0 if i == 0 else 1 if "left" in KPT_NAMES[i] else 2


def oks(gt: np.ndarray, pred: np.ndarray, area: float, sigmas: np.ndarray = COCO_SIGMAS) -> float:
    """Object Keypoint Similarity between one GT pose (17x3, v-flag) and one prediction (17x2+).

    ``OKS = mean_i exp(-d_i^2 / (2 s^2 k_i^2))`` over labelled GT keypoints (``v > 0``), with
    ``s^2`` = object area in pixels^2 and ``k_i = 2 sigma_i``. Returns NaN without labels.
    """
    gt = np.asarray(gt, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    vis = gt[:, 2] > 0
    if not vis.any():
        return float("nan")
    d2 = np.sum((gt[:, :2] - pred[:, :2]) ** 2, axis=1)
    e = d2 / (2.0 * area * (2.0 * sigmas) ** 2 + np.finfo(float).eps)
    return float(np.exp(-e)[vis].mean())


def oks_matrix(gts: np.ndarray, areas: np.ndarray, preds: np.ndarray) -> np.ndarray:
    """OKS for every (GT, prediction) pair -> ``(n_gt, n_pred)``."""
    out = np.zeros((len(gts), len(preds)))
    for i, (g, a) in enumerate(zip(gts, areas)):
        for j, p in enumerate(preds):
            out[i, j] = oks(g, p, a)
    return out


def match_by_oks(gts: np.ndarray, areas: np.ndarray, preds: np.ndarray) -> np.ndarray:
    """Best one-to-one assignment (Hungarian on OKS). Returns per-GT OKS (0 if unmatched)."""
    best = np.zeros(len(gts))
    if len(gts) == 0 or len(preds) == 0:
        return best
    m = oks_matrix(gts, areas, preds)
    rows, cols = linear_sum_assignment(-m)
    best[rows] = m[rows, cols]
    return best


def load_yolo_pose_labels(label_file: Path, width: int, height: int):
    """Read a YOLO-pose label file -> ``(boxes_xyxy Nx4, keypoints Nx17x3, areas N)`` in pixels.

    Each line is ``cls cx cy w h x1 y1 v1 ... x17 y17 v17`` (normalised to [0, 1]); the area
    used for OKS is ``0.53 * w * h``, the box-to-segment ratio used by Ultralytics.
    """
    rows = np.loadtxt(label_file, ndmin=2)
    cx, cy, w, h = (rows[:, 1] * width, rows[:, 2] * height, rows[:, 3] * width,
                    rows[:, 4] * height)  # fmt: skip
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    kpts = rows[:, 5:].reshape(-1, 17, 3).copy()
    kpts[..., 0] *= width
    kpts[..., 1] *= height
    return boxes, kpts, 0.53 * w * h


def joint_angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at ``b`` formed by points ``a-b-c`` (e.g. shoulder-elbow-wrist), in degrees."""
    u, v = np.asarray(a, float) - b, np.asarray(c, float) - b
    cos = u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def draw_pose(
    img: np.ndarray,
    kpts: np.ndarray,
    conf: np.ndarray | None = None,
    thr: float = 0.5,
    colors: tuple = ((240, 228, 66), (0, 158, 115), (230, 159, 0)),
    radius: int | None = None,
) -> np.ndarray:
    """Draw skeletons (N x 17 x 2 keypoints) on a copy of an RGB image.

    Keypoints whose confidence is below ``thr`` are skipped. Colours: centre / left / right.
    """
    out = np.ascontiguousarray(img.copy())
    kpts = np.asarray(kpts).reshape(-1, 17, kpts.shape[-1])
    conf = np.ones(kpts.shape[:2]) if conf is None else np.asarray(conf).reshape(-1, 17)
    scale = max(out.shape[:2]) / 640
    r = radius or max(2, round(3 * scale))
    lw = max(1, round(2 * scale))
    for person, c in zip(kpts, conf):
        ok = c >= thr
        for i, j in SKELETON:
            if ok[i] and ok[j]:
                p, q = person[i, :2].round().astype(int), person[j, :2].round().astype(int)
                col = colors[side(j)] if side(i) == side(j) else colors[0]
                cv2.line(out, tuple(p), tuple(q), col, lw, cv2.LINE_AA)
        for i in np.flatnonzero(ok):
            cv2.circle(out, tuple(person[i, :2].round().astype(int)), r, colors[side(i)], -1,
                       cv2.LINE_AA)  # fmt: skip
    return out
