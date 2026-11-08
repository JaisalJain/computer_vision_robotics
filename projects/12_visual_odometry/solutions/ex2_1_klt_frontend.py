"""Notebook 02, exercise 1: KLT optical-flow front end + PnP (RGB-D, frame-to-frame)."""

import time

import cv2
import numpy as np
import pandas as pd
from _common import gray, load, vo

from cvlab.geometry import invert_T, unproject

LK = {
    "winSize": (21, 21),
    "maxLevel": 3,
    "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
}


def run_klt(seq, max_corners=1500, fb_px=1.0):
    """Track Shi-Tomasi corners frame to frame with pyramidal Lucas-Kanade, pose by PnP."""
    cv2.setRNGSeed(0)
    K, dist = seq.camera.K, seq.dist
    T_wc = seq.gt_poses[0].copy()
    poses, ids, ms, status = [T_wc.copy()], [0], [], ["kf"]
    prev = gray(seq, 0)
    for i in range(1, len(seq)):
        cur = gray(seq, i)
        t0 = time.perf_counter()
        p0 = cv2.goodFeaturesToTrack(prev, max_corners, qualityLevel=0.01, minDistance=7)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None, **LK)
        p0b, st_b, _ = cv2.calcOpticalFlowPyrLK(cur, prev, p1, None, **LK)  # backward check
        fb = np.linalg.norm((p0 - p0b).reshape(-1, 2), axis=1)
        ok = (st.ravel() == 1) & (st_b.ravel() == 1) & (fb < fb_px)
        p0, p1 = p0.reshape(-1, 2)[ok].astype(float), p1.reshape(-1, 2)[ok].astype(float)
        dep = seq.depth(i - 1)
        z = vo.lookup_depth(dep, p0, smooth_mask=vo.depth_edge_mask(dep))
        good = np.isfinite(z)
        X = unproject(vo.undistort_pixels(p0[good], K, dist), z[good], K)
        T_cp, inl = vo.pose_pnp(X, vo.undistort_pixels(p1[good], K, dist), K)
        if T_cp is not None and inl.sum() >= 20:
            T_wc = T_wc @ invert_T(T_cp)
            status.append("kf")
        else:
            status.append("lost")  # keep the previous pose
        ms.append(1000 * (time.perf_counter() - t0))
        poses.append(T_wc.copy())
        ids.append(i)
        prev = cur
    ids = np.array(ids)
    log = {"frame": ids, "ms": np.array([0.0, *ms]), "status": np.array(status)}
    return vo.VOResult("RGB-D frame-to-frame, KLT", ids, seq.timestamps[ids], np.stack(poses), log)


rows = []
for name in ("fr1_xyz", "fr1_desk"):
    seq = load(name)
    for label, res in [
        ("KLT + PnP", run_klt(seq)),
        ("ORB + PnP", vo.run_pnp_vo(seq, vo.FeatureFrontend("orb", 2000), keyframes=False)),
    ]:
        s = vo.evaluate(res, seq, align="se3")
        rows.append(
            {
                "sequence": name,
                "front end": label,
                "ATE [cm]": 100 * s["ate_rmse_m"],
                "RPE 1s [cm]": 100 * s["rpe_trans_rmse_m"],
                "RPE 1s [deg]": s["rpe_rot_rmse_deg"],
                "lost": s["lost"],
                "ms/frame": s["ms_per_frame"],
            }
        )
print(pd.DataFrame(rows).round(2).to_string(index=False))
