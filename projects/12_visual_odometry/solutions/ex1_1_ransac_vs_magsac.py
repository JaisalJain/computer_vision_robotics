"""Notebook 01, exercise 1: RANSAC vs MAGSAC++ for the essential matrix (fr1/xyz, gap 10)."""

import cv2
import numpy as np
import pandas as pd
from _common import gray, load, vo

from cvlab.geometry import angle_between_deg, invert_T, rotation_error_deg

seq = load("fr1_xyz")
K, dist = seq.camera.K, seq.dist
fe = vo.FeatureFrontend("orb", 2000)
rows = []
for i in np.linspace(0, len(seq) - 11, 50).astype(int):
    j = i + 10
    uv1_raw, d1 = fe.detect(gray(seq, i))
    uv2_raw, d2 = fe.detect(gray(seq, j))
    uv1, uv2 = vo.undistort_pixels(uv1_raw, K, dist), vo.undistort_pixels(uv2_raw, K, dist)
    m = fe.match(d1, d2)
    T_gt = invert_T(seq.gt_poses[j]) @ seq.gt_poses[i]
    for method in ("ransac", "magsac"):
        cv2.setRNGSeed(0)
        T, inl = vo.relative_pose_essential(uv1[m[:, 0]], uv2[m[:, 1]], K, method=method)
        if T is None:
            rows.append({"method": method, "failed": 1})
            continue
        rows.append(
            {
                "method": method,
                "failed": 0,
                "rot err [deg]": rotation_error_deg(T[:3, :3], T_gt[:3, :3]),
                "dir err [deg]": angle_between_deg(T[:3, 3], T_gt[:3, 3]),
                "inlier ratio": inl.mean(),
            }
        )
df = pd.DataFrame(rows)
summary = df.groupby("method").agg(
    pairs=("failed", "size"),
    failed=("failed", "sum"),
    median_rot_err_deg=("rot err [deg]", "median"),
    median_dir_err_deg=("dir err [deg]", "median"),
    mean_inlier_ratio=("inlier ratio", "mean"),
)
print(summary.round(3).to_string())
print(
    "\nUSAC_MAGSAC scores hypotheses with MAGSAC++ (no hard threshold) and polishes the best"
    "\none with local optimisation; the classic RANSAC flag ranks them by a hard inlier count."
)
