#!/usr/bin/env python
"""Solutions to the exercises of notebook 01 (epipolar geometry, E, triangulation).

Run (CPU only, about 10 s)::

    python projects/09_multiview_geometry/solutions/solutions_01.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import mvgeo as M  # noqa: E402

from cvlab import geometry as G  # noqa: E402
from cvlab.io import data_dir  # noqa: E402
from cvlab.tum import load_sequence  # noqa: E402

# Same synthetic scene as the notebook.
rng = np.random.default_rng(0)
W, H = 640, 480
cam = G.PinholeCamera.from_fov(W, H, 60.0)
K = cam.K
X_world = rng.uniform([-1.0, -0.5, 0.0], [1.0, 1.0, 1.5], size=(400, 3))
T_wc1 = G.look_at(eye=[-0.8, -3.0, 1.2], target=[0.0, 0.2, 0.7])
T_wc2 = G.look_at(eye=[0.6, -2.8, 1.4], target=[0.0, 0.2, 0.7])
R_gt, t_gt = M.relative_pose_from_c2w(T_wc1, T_wc2)
x1_all, _ = G.project_points(X_world, K, G.invert_T(T_wc1))
x2_all, _ = G.project_points(X_world, K, G.invert_T(T_wc2))


def pose_from_F(F, a, b):
    E = M.essential_from_fundamental(F, K, K)
    _, R, t, _ = cv2.recoverPose(E, a, b, K)
    return R, t.ravel()


# ----------------------------------------------------------------------------- exercise 1
def ransac_eight_point(a, b, thr_px=1.5, iters=2000, seed=0):
    """RANSAC with the normalised 8-point solver and the Sampson distance."""
    r = np.random.default_rng(seed)
    best = np.zeros(len(a), bool)
    for _ in range(iters):
        idx = r.choice(len(a), 8, replace=False)
        F = M.eight_point(a[idx], b[idx])
        inl = M.sampson_distance(F, a, b) < thr_px**2
        if inl.sum() > best.sum():
            best = inl
    F = M.eight_point(a[best], b[best])  # least-squares refit on all inliers
    return F, M.sampson_distance(F, a, b) < thr_px**2


def exercise_1() -> None:
    r = np.random.default_rng(0)
    idx = r.choice(len(x1_all), 100, replace=False)
    a = x1_all[idx] + r.normal(0, 0.5, (100, 2))
    b = x2_all[idx] + r.normal(0, 0.5, (100, 2))
    b[:25] = r.uniform([0, 0], [W, H], (25, 2))  # 25 % outliers (the first 25)
    F_own, inl_own = ransac_eight_point(a, b)
    F_cv, mask = cv2.findFundamentalMat(a, b, cv2.FM_RANSAC, 1.5, 0.999)
    inl_cv = mask.ravel().astype(bool)
    print("Exercise 1 - RANSAC + 8-point (25 % outliers, sigma 0.5 px)")
    for name, F, inl in [("own RANSAC + refit", F_own, inl_own), ("cv2 FM_RANSAC", F_cv, inl_cv)]:
        R, t = pose_from_F(F, a[inl], b[inl])
        er, et = M.pose_errors(R, t, R_gt, t_gt)
        print(f"  {name:20s} inliers {inl.sum():3d} (true outliers kept: {inl[:25].sum()}) "
              f"rot err {er:.3f} deg, t err {et:.3f} deg")


# ----------------------------------------------------------------------------- exercise 2
def exercise_2() -> None:
    T_rot = G.make_T(T_wc1[:3, :3] @ G.so3_exp(np.deg2rad([2.0, 5.0, 1.0])), T_wc1[:3, 3])
    uv_a, za = G.project_points(X_world, K, G.invert_T(T_wc1))
    uv_b, zb = G.project_points(X_world, K, G.invert_T(T_rot))
    ok = (za > 0) & (zb > 0) & cam.in_image(uv_a) & cam.in_image(uv_b)
    r = np.random.default_rng(1)
    a = uv_a[ok] + r.normal(0, 0.5, (ok.sum(), 2))
    b = uv_b[ok] + r.normal(0, 0.5, (ok.sum(), 2))
    E, mask = cv2.findEssentialMat(a, b, K, cv2.USAC_MAGSAC, 0.999, 1.0)
    _, R, t, _ = cv2.recoverPose(E[:3], a, b, K, mask=mask.copy())
    # Transfer error combines the noise of both points (~sqrt(2) * 0.5 px): use a 2 px threshold.
    Hm, hmask = cv2.findHomography(a, b, cv2.USAC_MAGSAC, 2.0)
    proj = cv2.perspectiveTransform(a.reshape(-1, 1, 2), Hm).reshape(-1, 2)
    H_true = K @ M.relative_pose_from_c2w(T_wc1, T_rot)[0] @ np.linalg.inv(K)
    Hn, Htn = Hm / Hm[2, 2], H_true / H_true[2, 2]
    H_rel = np.linalg.norm(Hn - Htn) / np.linalg.norm(Htn)
    print("Exercise 2 - pure rotation (camera centres coincide)")
    print(f"  E-based pose returns |t| = {np.linalg.norm(t):.1f} (always unit length), "
          f"direction {np.round(t.ravel(), 3)}: arbitrary, there is no baseline")
    print(f"  homography: {hmask.sum()}/{len(a)} inliers, mean transfer error "
          f"{np.linalg.norm(proj - b, axis=1).mean():.3f} px; relative difference to K R K^-1: "
          f"{H_rel:.2e}")
    print("  -> monocular SLAM/SfM initialisers test H vs F/E and refuse to triangulate when the")
    print("     homography explains the motion (no parallax = no depth).")


# ----------------------------------------------------------------------------- exercise 3
def exercise_3() -> None:
    seq = load_sequence(data_dir("tum", "fr1_desk", "rgbd_dataset_freiburg1_desk"))
    Kt, dist = seq.camera.K, seq.dist

    def und(x):
        return cv2.undistortPoints(x.reshape(-1, 1, 2), Kt, dist, P=Kt).reshape(-1, 2)

    # Same selection as the notebook: gap-20 pair (every 30 frames) with the most SIFT matches.
    best = None
    for i in range(0, len(seq) - 20, 30):
        R_rel, t_rel = M.relative_pose_from_c2w(seq.gt_poses[i], seq.gt_poses[i + 20])
        if np.linalg.norm(t_rel) < 0.02:
            continue
        m = M.sift_match(seq.rgb(i), seq.rgb(i + 20))
        if best is None or len(m) > len(best[2]):
            best = (i, i + 20, m)
    i, j, m = best
    a, b = und(m.x1), und(m.x2)
    E, mask = cv2.findEssentialMat(a, b, Kt, cv2.USAC_MAGSAC, 0.999, 1.0)
    F = np.linalg.inv(Kt).T @ E[:3] @ np.linalg.inv(Kt)
    e1 = np.linalg.svd(F)[2][-1]
    e2 = np.linalg.svd(F.T)[2][-1]
    e1, e2 = e1[:2] / e1[2], e2[:2] / e2[2]
    c_i_in_j, _ = G.project_points(seq.gt_poses[i][:3, 3][None], Kt, G.invert_T(seq.gt_poses[j]))
    c_j_in_i, _ = G.project_points(seq.gt_poses[j][:3, 3][None], Kt, G.invert_T(seq.gt_poses[i]))
    _, R, t, _ = cv2.recoverPose(E[:3], a, b, Kt, mask=mask.copy())
    R_gt_, t_gt_ = M.relative_pose_from_c2w(seq.gt_poses[i], seq.gt_poses[j])
    er, et = M.pose_errors(R, t.ravel(), R_gt_, t_gt_)
    print(f"Exercise 3 - epipoles of TUM frames {i} and {j}")
    print(f"  e1 from F = {e1.round(1)} px | camera {j} centre projected into frame {i} = "
          f"{c_j_in_i[0].round(1)} px")
    print(f"  e2 from F = {e2.round(1)} px | camera {i} centre projected into frame {j} = "
          f"{c_i_in_j[0].round(1)} px")
    print(f"  pose error: rotation {er:.2f} deg, translation direction {et:.2f} deg")
    print("  The epipole is the image of the translation direction: e ~ K t. A few degrees of")
    print("  direction error move it by roughly f * tan(err) pixels, and much more when the")
    print("  motion is mostly sideways (epipole far outside the image, nearly at infinity).")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
