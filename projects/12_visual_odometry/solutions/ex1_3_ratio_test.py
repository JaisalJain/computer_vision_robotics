"""Notebook 01, exercise 3: sweep Lowe's ratio for ORB matching + PnP (fr1/xyz)."""

import cv2
import numpy as np
import pandas as pd
from _common import gray, load, vo

from cvlab.geometry import invert_T, unproject

seq = load("fr1_xyz")
K, dist = seq.camera.K, seq.dist
pairs = [(i, i + 10) for i in np.linspace(0, len(seq) - 11, 50).astype(int)]
rows = []
for ratio in (0.6, 0.7, 0.8, 0.9, 1.0):
    fe = vo.FeatureFrontend("orb", 2000, ratio=ratio)
    for i, j in pairs:
        uv1_raw, d1 = fe.detect(gray(seq, i))
        uv2_raw, d2 = fe.detect(gray(seq, j))
        uv1, uv2 = vo.undistort_pixels(uv1_raw, K, dist), vo.undistort_pixels(uv2_raw, K, dist)
        dep = seq.depth(i)
        z = vo.lookup_depth(dep, uv1_raw, smooth_mask=vo.depth_edge_mask(dep))
        X = unproject(uv1, np.nan_to_num(z), K)
        m = fe.match(d1, d2)
        m = m[np.isfinite(z[m[:, 0]])]
        cv2.setRNGSeed(0)
        T, inl = vo.pose_pnp(X[m[:, 0]], uv2[m[:, 1]], K)
        T_gt = invert_T(seq.gt_poses[j]) @ seq.gt_poses[i]
        rows.append(
            {
                "ratio": ratio,
                "matches": len(m),
                "inlier ratio": inl.mean() if len(m) else np.nan,
                "t err [cm]": 100 * np.linalg.norm(T[:3, 3] - T_gt[:3, 3])
                if T is not None
                else np.nan,
            }
        )
df = pd.DataFrame(rows)
print(df.groupby("ratio").median().round(3).to_string())
failed = df["t err [cm]"].isna().groupby(df["ratio"]).sum()
print("\nfailed PnP per ratio:", failed.astype(int).to_dict())
