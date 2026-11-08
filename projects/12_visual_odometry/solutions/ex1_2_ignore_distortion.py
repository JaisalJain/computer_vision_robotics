"""Notebook 01, exercise 2: what happens if the fr1 lens distortion is ignored?"""

import cv2
import numpy as np
import pandas as pd
from _common import gray, load, vo

from cvlab.geometry import angle_between_deg, invert_T, rotation_error_deg

seq = load("fr1_xyz")
fe = vo.FeatureFrontend("orb", 2000)

# 1) full RGB-D frame-to-frame VO with and without undistortion
rows = []
for undistort in (True, False):
    res = vo.run_pnp_vo(seq, fe, keyframes=False, undistort=undistort)
    s = vo.evaluate(res, seq, align="se3")
    rows.append(
        {
            "undistort keypoints": undistort,
            "ATE [cm]": 100 * s["ate_rmse_m"],
            "RPE 1s [cm]": 100 * s["rpe_trans_rmse_m"],
            "RPE 1s [deg]": s["rpe_rot_rmse_deg"],
        }
    )
print("RGB-D frame-to-frame VO on fr1/xyz")
print(pd.DataFrame(rows).round(3).to_string(index=False))

# 2) essential matrix on 40 pairs (gap 10) with raw vs undistorted keypoints
K, dist = seq.camera.K, seq.dist
rows = []
for i in np.linspace(0, len(seq) - 11, 40).astype(int):
    j = i + 10
    uv1_raw, d1 = fe.detect(gray(seq, i))
    uv2_raw, d2 = fe.detect(gray(seq, j))
    m = fe.match(d1, d2)
    T_gt = invert_T(seq.gt_poses[j]) @ seq.gt_poses[i]
    for undistort in (True, False):
        p1, p2 = uv1_raw[m[:, 0]], uv2_raw[m[:, 1]]
        if undistort:
            p1, p2 = vo.undistort_pixels(p1, K, dist), vo.undistort_pixels(p2, K, dist)
        cv2.setRNGSeed(0)
        T, inl = vo.relative_pose_essential(p1, p2, K)
        if T is not None:
            rows.append(
                {
                    "undistort keypoints": undistort,
                    "rot err [deg]": rotation_error_deg(T[:3, :3], T_gt[:3, :3]),
                    "dir err [deg]": angle_between_deg(T[:3, 3], T_gt[:3, 3]),
                    "inlier ratio": inl.mean(),
                }
            )
print("\nEssential matrix, 40 pairs with a 10-frame gap (medians)")
print(pd.DataFrame(rows).groupby("undistort keypoints").median().round(3).to_string())
