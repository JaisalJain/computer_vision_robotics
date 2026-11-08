#!/usr/bin/env python
"""Solutions to the exercises of notebook 02 (fiducial markers + PnP). CPU only::

python projects/07_pose_estimation/solutions/02_fiducial_pose_solutions.py
"""

from __future__ import annotations

import sys

import cv2
import numpy as np

from cvlab import io as cvio
from cvlab import tum

PROJECT = "07_pose_estimation"
sys.path.insert(0, str(cvio.project_dir(PROJECT) / "scripts"))
import fiducial_sim as fs  # noqa: E402

SEQ = cvio.data_dir("tum", "fr1_desk", "rgbd_dataset_freiburg1_desk")
CAM, DIST = tum.camera_for(SEQ.name)
K, W, H = CAM.K, CAM.width, CAM.height
RGB = sorted((SEQ / "rgb").glob("*.png"))
BGS = [cv2.imread(str(p)) for p in RGB[::40]]


# --------------------------------------------- Exercise 1: dictionary vs range vs false positives
def exercise1(rng):
    print("Exercise 1 - 10 cm markers: 90 % detection range and false positives on 120 TUM frames")
    dicts = {"DICT_4X4_50": cv2.aruco.DICT_4X4_50, "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
             "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11}  # fmt: skip
    dists = [1, 2, 3, 4, 5, 6, 7, 8]
    negatives = [cv2.imread(str(p)) for p in RGB[::5][:120]]  # desk frames without markers
    for name, did in dicts.items():
        d = cv2.aruco.getPredefinedDictionary(did)
        tex = fs.marker_texture(d, 3, 0.10)
        det = fs.make_detector(d, "subpix")
        rates = []
        for z in dists:
            hits = 0
            for _ in range(30):
                R, t = fs.sample_pose(tex, K, (W, H), z, rng.uniform(0, 45), rng)
                img = fs.render(tex, K, R, t, BGS[rng.integers(len(BGS))], rng=rng)
                hits += 3 in fs.detect(det, img)
            rates.append(hits / 30)
        ok = [z for z, r in zip(dists, rates) if r >= 0.9]
        false_pos = sum(len(fs.detect(det, img)) for img in negatives)
        print(f"  {name:20s} detection rate per distance {dict(zip(dists, rates))}")
        print(f"  {'':20s} 90 % range: {max(ok) if ok else 0} m | false detections: {false_pos}")


# --------------------------------------------- Exercise 2: temporal disambiguation of IPPE
def exercise2(rng, ratio_thr: float = 1.5):
    print("Exercise 2 - 60-frame sequence at 2.5 m: oblique start (40 deg), then 5-15 deg views")
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    tex = fs.marker_texture(d, 7, 0.10)
    det = fs.make_detector(d, "subpix")
    errs = {"lowest reprojection": [], "temporal prior": []}
    prev = None
    for k in range(60):
        # Smooth motion: start oblique (unambiguous), settle around 10 deg (ambiguous).
        view = 40 - 3 * k if k < 10 else 10 + 5 * np.sin(k / 6)
        R, t = fs.pose_facing(tex, K, (W / 2, H / 2), 2.5, tilt_deg=view, tilt_dir_deg=2 * k,
                              roll_deg=20)  # fmt: skip
        img = fs.render(tex, K, R, t, BGS[0], noise_sigma=3.0, rng=rng)
        found = fs.detect(det, img)
        if 7 not in found:
            continue
        res = fs.solve_marker_pose(found[7], 0.10, K)
        naive = res.candidates[0][0]
        (_, _, e1), (_, _, e2) = res.candidates[0], res.candidates[1]
        if prev is None or e2 / max(e1, 1e-9) > ratio_thr:
            chosen = naive  # unambiguous frame (or no history): trust the reprojection error
        else:  # ambiguous: keep the branch that is consistent with the previous frame
            chosen = min(
                (c[0] for c in res.candidates), key=lambda Rc: fs.rotation_error_deg(prev, Rc)
            )
        prev = chosen
        errs["lowest reprojection"].append(fs.rotation_error_deg(R, naive))
        errs["temporal prior"].append(fs.rotation_error_deg(R, chosen))
    for name, e in errs.items():
        e = np.array(e)
        print(
            f"  {name:20s}: frames {len(e)}, wrong branch (> 5 deg) {int((e > 5).sum())}, "
            f"median {np.median(e):.2f} deg, max {e.max():.1f} deg"
        )
    print(f"  (temporal prior: reprojection error decides when e2/e1 > {ratio_thr}, otherwise the")
    print("   candidate closest to the previous rotation is kept)")


# --------------------------------------------- Exercise 3: lens distortion
def distort_image(ideal: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Re-image an ideal pinhole render through the lens model (cv2.remap with undistortPoints)."""
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    pts = np.stack([u.ravel(), v.ravel()], 1).reshape(-1, 1, 2)
    und = cv2.undistortPoints(pts, K, dist, P=K).reshape(H, W, 2)  # where each pixel comes from
    return cv2.remap(
        ideal, und[..., 0], und[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )


def exercise3(rng):
    print("Exercise 3 - TUM fr1 lens distortion, 10 cm tag at 0.6 m, 30 deg view (20 trials each)")
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    tex = fs.marker_texture(d, 7, 0.10)
    det = fs.make_detector(d, "subpix")
    obj = fs.marker_object_points(0.10)
    for source in ("exact corners + 0.1 px noise", "rendered + cv2.aruco SUBPIX"):
        print(f"  corners from: {source}")
        for where, px in [("centre", (W / 2, H / 2)), ("near corner", (95, 85))]:
            res = {"(a) ignore distortion": [], "(b) distCoeffs in PnP": [],
                   "(c) undistort corners": []}  # fmt: skip
            for _ in range(20):
                R, t = fs.pose_facing(tex, K, px, 0.6, 30, rng.uniform(0, 360), rng.uniform(0, 360))
                if source.startswith("exact"):
                    rvec = cv2.Rodrigues(R)[0]
                    c = cv2.projectPoints(obj, rvec, t, K, DIST)[0].reshape(4, 2)
                    c = c + rng.normal(0, 0.1, c.shape)
                else:
                    ideal = fs.render(tex, K, R, t, BGS[1], blur_sigma=0, noise_sigma=0)
                    found = fs.detect(det, fs.degrade(distort_image(ideal, K, DIST), 0.6, 2.0, rng))
                    if 7 not in found:
                        continue
                    c = found[7]
                pa = fs.solve_marker_pose(c, 0.10, K)
                pb = fs.solve_marker_pose(c, 0.10, K, DIST)
                cu = cv2.undistortPoints(c.reshape(-1, 1, 2), K, DIST, P=K).reshape(4, 2)
                pc = fs.solve_marker_pose(cu, 0.10, K)
                for key, p in zip(res, (pa, pb, pc)):
                    res[key].append((fs.rotation_error_deg(R, p.R), 1000 * np.linalg.norm(p.t - t)))
            for key, v in res.items():
                v = np.array(v)
                rot, trans = np.median(v[:, 0]), np.median(v[:, 1])
                print(f"    {where:11s} {key:24s}: n={len(v)}, median rot {rot:.2f} deg, "
                      f"median trans {trans:.1f} mm")  # fmt: skip
    print(
        "  With rendered images the SUBPIX corner bias (about -0.26 px radially, notebook 02) adds"
    )
    print("  a few mm of range error on top, which can partly cancel the distortion error at the")
    print("  image centre; the exact-corner rows isolate the lens model.")


def main() -> None:
    rng = np.random.default_rng(0)
    exercise1(rng)
    exercise2(rng)
    exercise3(rng)


if __name__ == "__main__":
    main()
