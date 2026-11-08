#!/usr/bin/env python
"""Solution, notebook 02, exercise 1: CoTracker3 visibility of points on an occluded person.

Picks the MOT17-05 pedestrian (first 64 frames) whose GT visibility drops the most, places 5
query points on the torso in the first frame where it is >= 80 % visible, tracks them offline
and compares the share of points predicted visible with the GT visibility ratio over time.
GPU: run under the lock.

    flock /tmp/cv-gpu.lock python projects/06_tracking/solutions/nb02_ex1_occlusion_visibility.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cvlab.io import data_dir, outputs_dir, repo_root  # noqa: E402
from cvlab.viz import set_style  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import motlib as ml  # noqa: E402
import pointtrack as pt  # noqa: E402

T = 64


def main() -> None:
    seq = ml.load_sequence(data_dir("mot17", "MOT17-05-FRCNN"))
    gt = seq.gt()
    gt = gt[(gt[:, 0] <= T) & (gt[:, 7] == ml.PEDESTRIAN) & (gt[:, 6] != 0)]
    # Pedestrian with the largest visibility drop after a well-visible start.
    best, best_drop = None, 0.0
    for pid in np.unique(gt[:, 1]):
        g = gt[gt[:, 1] == pid]
        g = g[np.argsort(g[:, 0])]
        ok = np.flatnonzero(g[:, 8] >= 0.8)
        if len(ok) == 0 or len(g) < 30:
            continue
        drop = g[ok[0] :, 8].max() - g[ok[0] :, 8].min()
        if drop > best_drop:
            best, best_drop = g[ok[0] :], drop
    pid, t0 = int(best[0, 1]), int(best[0, 0])
    x, y, w, h = best[0, 2:6]
    xs = x + w * np.array([0.4, 0.6, 0.5, 0.4, 0.6])
    ys = y + h * np.array([0.3, 0.3, 0.4, 0.5, 0.5])
    clip = [seq.frame(t) for t in range(1, T + 1)]
    queries = np.column_stack([np.full(5, t0 - 1), xs, ys])  # (t, x, y), t 0-based
    model = pt.load_cotracker("offline")
    res = pt.track_offline(model, clip, queries=queries)
    frames = best[:, 0].astype(int)
    pred_vis = res["visible"][frames - 1].mean(axis=1)
    corr = np.corrcoef(pred_vis, best[:, 8])[0, 1]
    print(
        f"pedestrian {pid}: frames {frames[0]}-{frames[-1]}, GT visibility "
        f"{best[:, 8].min():.2f}-{best[:, 8].max():.2f}"
    )
    print(
        f"correlation(predicted visible share, GT visibility) = {corr:.2f}; "
        f"peak VRAM {res['peak_vram_mb']:.0f} MiB"
    )
    set_style()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(frames, best[:, 8], label="GT visibility ratio")
    ax.plot(frames, pred_vis, label="share of 5 torso points predicted visible")
    ax.set_xlabel("frame")
    ax.set_ylabel("fraction")
    ax.set_title(f"MOT17-05 pedestrian {pid}: CoTracker3 visibility through an occlusion")
    ax.legend()
    path = outputs_dir("06_tracking", create=True) / "nb02_ex1_visibility.png"
    fig.savefig(path, dpi=110)
    print("figure:", path.relative_to(repo_root()))


if __name__ == "__main__":
    main()
