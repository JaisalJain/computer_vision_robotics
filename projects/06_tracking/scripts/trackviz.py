"""Static matplotlib / OpenCV figures for project 06 (MOT and point tracking)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from motlib import DISTRACTOR_CLASSES, PEDESTRIAN, MotSequence, draw_tracks, id_color


def draw_gt(img: np.ndarray, gt_frame: np.ndarray) -> np.ndarray:
    """Draw one frame of MOT17 GT: scored pedestrians coloured by id, ignored rows in grey."""
    import cv2

    out = np.ascontiguousarray(img.copy())
    th = max(1, round(2 * max(out.shape[:2]) / 640))
    scored = (gt_frame[:, 7] == PEDESTRIAN) & (gt_frame[:, 6] != 0)
    for r in gt_frame[~scored]:
        x, y, w, h = r[2:6].astype(int)
        color = (160, 160, 160) if r[7] in DISTRACTOR_CLASSES else (90, 90, 90)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, max(1, th // 2))
    return draw_tracks(out, gt_frame[scored][:, :7], thickness=2)


def track_trails(rows: np.ndarray, t: int, length: int = 20) -> dict[int, list]:
    """Foot points ``(x + w/2, y + h)`` of every id over the last ``length`` frames up to ``t``."""
    sel = rows[(rows[:, 0] <= t) & (rows[:, 0] > t - length)]
    trails: dict[int, list[tuple[float, float]]] = {}
    for r in sel[np.argsort(sel[:, 0], kind="stable")]:
        trails.setdefault(int(r[1]), []).append((r[2] + r[4] / 2, r[3] + r[5]))
    return trails


def render_tracks(seq: MotSequence, rows: np.ndarray, t: int, trail: int = 20,
                  max_side: int | None = None) -> np.ndarray:  # fmt: skip
    """RGB frame ``t`` with tracker boxes, ids and short motion trails."""
    from cvlab.io import resize_max

    img = draw_tracks(seq.frame(t), rows[rows[:, 0] == t], trails=track_trails(rows, t, trail))
    return resize_max(img, max_side) if max_side else img


def plot_identity_timeline(ax, matches: np.ndarray, gt_rows: np.ndarray, gt_ids: Sequence[int],
                           idsw: np.ndarray | None = None, title: str = "") -> None:  # fmt: skip
    """One row per GT id: grey = GT present, colour = matched tracker id, x = identity switch.

    Args:
        matches: ``(frame_index, gt_id, tracker_id)`` rows from ``clear_metrics(return_matches)``.
        gt_rows: Scored GT rows (frame, id, ...), frames 1-based.
        gt_ids: GT ids to show (top to bottom).
        idsw: ``(frame_index, gt_id, old_id, new_id)`` identity-switch events.
    """
    for row, gid in enumerate(gt_ids):
        present = np.sort(gt_rows[gt_rows[:, 1] == gid, 0]) - 1  # 0-based frame index
        ax.broken_barh([(f, 1) for f in present], (row - 0.4, 0.8), color="0.85", lw=0)
        m = matches[matches[:, 1] == gid]
        for tid in np.unique(m[:, 2]):
            fr = m[m[:, 2] == tid, 0]
            c = tuple(v / 255 for v in id_color(int(tid)))
            ax.broken_barh([(f, 1) for f in fr], (row - 0.4, 0.8), color=c, lw=0)
        if idsw is not None and len(idsw):
            ev = idsw[idsw[:, 1] == gid]
            ax.plot(ev[:, 0], np.full(len(ev), row), "kx", ms=7, mew=2)
    ax.set_yticks(range(len(gt_ids)), [str(g) for g in gt_ids])
    ax.set_ylim(len(gt_ids) - 0.5, -0.5)
    ax.set_xlabel("frame index")
    ax.set_ylabel("GT pedestrian id")
    if title:
        ax.set_title(title)


def draw_points(img: np.ndarray, pts: np.ndarray, visible: np.ndarray,
                colors: np.ndarray | None = None, radius: int = 3,
                history: np.ndarray | None = None) -> np.ndarray:  # fmt: skip
    """Draw tracked 2D points (filled = visible, ring = occluded) and optional trails.

    Args:
        pts: (N, 2) pixel coordinates in this frame.
        visible: (N,) bool.
        colors: (N, 3) uint8 RGB per point (default: hue by initial x position).
        history: (K, N, 2) previous positions for trails (oldest first).
    """
    import cv2

    out = np.ascontiguousarray(img.copy())
    n = len(pts)
    if colors is None:
        colors = point_colors(pts)
    if history is not None and len(history) > 1:
        for i in range(n):
            p = np.asarray(history[:, i], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [p], False, tuple(int(c) for c in colors[i]), 1, cv2.LINE_AA)
    for i in range(n):
        c = tuple(int(v) for v in colors[i])
        center = (int(round(pts[i, 0])), int(round(pts[i, 1])))
        cv2.circle(out, center, radius, c, -1 if visible[i] else 1, cv2.LINE_AA)
    return out


def point_colors(pts0: np.ndarray) -> np.ndarray:
    """RGB colours from the x position of each point in the query frame (rainbow left-right)."""
    import matplotlib as mpl

    x = np.asarray(pts0)[:, 0]
    u = (x - x.min()) / max(np.ptp(x), 1e-6)
    return (mpl.colormaps["turbo"](u)[:, :3] * 255).astype(np.uint8)
