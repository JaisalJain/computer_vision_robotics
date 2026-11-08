#!/usr/bin/env python
"""Solution, notebook 02, exercise 3: BoT-SORT's GMC vs camera motion from CoTracker3 tracks.

GMC (sparse optical flow + RANSAC affine, as used by BoT-SORT) gives a frame-to-frame warp
W_t (frame t-1 -> t). Chaining them gives frame 1 -> t, compared here with the homography
fitted to CoTracker3 point tracks over the same 64 MOT17-05 frames. GPU: run under the lock.

    flock /tmp/cv-gpu.lock python projects/06_tracking/solutions/nb02_ex3_gmc_vs_cotracker.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from ultralytics.trackers.utils.gmc import GMC  # noqa: E402

from cvlab.io import data_dir, outputs_dir, repo_root  # noqa: E402
from cvlab.viz import set_style  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import motlib as ml  # noqa: E402
import pointtrack as pt  # noqa: E402

T = 64


def main() -> None:
    seq = ml.load_sequence(data_dir("mot17", "MOT17-05-FRCNN"))
    clip = [seq.frame(t) for t in range(1, T + 1)]
    centre = np.array([seq.width / 2, seq.height / 2, 1.0])
    # Chained GMC warps (3x3 homogeneous), frame 1 -> t.
    gmc, M, gmc_disp = GMC(method="sparseOptFlow"), np.eye(3), []
    for t in range(1, T + 1):
        W = gmc.apply(cv2.imread(str(seq.frame_path(t))))  # 2x3, frame t-1 -> t
        M = np.vstack([W, [0, 0, 1]]) @ M
        gmc_disp.append((M @ centre)[:2] - centre[:2])
    gmc_disp = np.asarray(gmc_disp)
    # CoTracker3 homographies, frame 1 -> t.
    model = pt.load_cotracker("offline")
    res = pt.track_offline(model, clip, grid_size=30)
    _, Hs = pt.ego_motion_residuals(res["tracks"], res["visible"])
    ct_disp = []
    for H in Hs:
        p = H @ centre if H is not None else np.full(3, np.nan)
        ct_disp.append(p[:2] / p[2] - centre[:2])
    ct_disp = np.asarray(ct_disp)
    diff = np.linalg.norm(gmc_disp - ct_disp, axis=1)
    print(
        f"image-centre displacement after {T} frames: GMC {gmc_disp[-1].round(1)} px, "
        f"CoTracker {ct_disp[-1].round(1)} px"
    )
    print(f"mean |GMC - CoTracker| = {np.nanmean(diff):.1f} px, max = {np.nanmax(diff):.1f} px")
    set_style()
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    tt = np.arange(1, T + 1)
    ax.plot(tt, gmc_disp[:, 0], label="GMC chained, horizontal")
    ax.plot(tt, ct_disp[:, 0], "--", label="CoTracker homography, horizontal")
    ax.plot(tt, gmc_disp[:, 1], label="GMC chained, vertical")
    ax.plot(tt, ct_disp[:, 1], "--", label="CoTracker homography, vertical")
    ax.set_xlabel("frame")
    ax.set_ylabel("image-centre displacement [px]")
    ax.set_title("MOT17-05: camera motion from GMC vs CoTracker3 tracks")
    ax.legend(fontsize=8)
    path = outputs_dir("06_tracking", create=True) / "nb02_ex3_gmc_vs_cotracker.png"
    fig.savefig(path, dpi=110)
    print("figure:", path.relative_to(repo_root()))


if __name__ == "__main__":
    main()
