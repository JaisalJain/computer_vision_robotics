#!/usr/bin/env python
"""Solutions to the exercises of notebook 03 (frames, transform chains, localisation). CPU only::

python projects/07_pose_estimation/solutions/03_frames_solutions.py
"""

from __future__ import annotations

import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from cvlab import geometry as G
from cvlab import io as cvio
from cvlab import tum
from cvlab.metrics import ate

PROJECT = "07_pose_estimation"
sys.path.insert(0, str(cvio.project_dir(PROJECT) / "scripts"))
import fiducial_sim as fs  # noqa: E402
import table_scene as ts  # noqa: E402

SEQ = cvio.data_dir("tum", "fr1_desk", "rgbd_dataset_freiburg1_desk")
CAM, _ = tum.camera_for(SEQ.name)
K, W, H = CAM.K, CAM.width, CAM.height
BG = np.full((H, W, 3), 60, np.uint8)
SCENE = ts.make_scene(cv2.imread(str(sorted((SEQ / "rgb").glob("*.png"))[300])))
try:
    import pupil_apriltags

    PUPIL = pupil_apriltags.Detector(families="tag36h11", nthreads=1, quad_decimate=1.0)
except ImportError:
    PUPIL = None
ARUCO = fs.make_detector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), "subpix")


def detect_tags(img):
    if PUPIL is not None:
        return {i: r["corners"] for i, r in fs.pupil_detect(PUPIL, img, K, 1.0).items()}
    return fs.detect(ARUCO, img)


# --------------------------------------------- Exercise 1: chordal L2 rotation averaging
def chordal_mean(rotations: list[np.ndarray]) -> np.ndarray:
    """Rotation closest (Frobenius norm) to the arithmetic mean of rotation matrices."""
    U, _, Vt = np.linalg.svd(np.sum(rotations, axis=0))
    return U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt


def localize_average(dets: dict[int, np.ndarray]):
    poses = []
    for i, c in dets.items():
        if i not in SCENE.T_world_tag:
            continue
        r = fs.solve_marker_pose(c, SCENE.tag_size[i], K)
        poses.append(SCENE.T_world_tag[i] @ G.invert_T(G.make_T(r.R, r.t)))
    if not poses:
        return None
    return G.make_T(chordal_mean([T[:3, :3] for T in poses]), np.mean([T[:3, 3] for T in poses], 0))


def exercise1(rng):
    print("Exercise 1 - per-tag poses averaged (chordal L2) vs single tag vs joint PnP, 60 frames")
    traj = ts.orbit_trajectory(60)
    est = {"single tag": [], "chordal average": [], "joint PnP": []}
    for T_wc in traj:
        d = detect_tags(fs.render_scene(SCENE.items, K, G.invert_T(T_wc), BG, rng=rng))
        est["single tag"].append(ts.localize_single(SCENE, d, K))
        est["chordal average"].append(localize_average(d))
        est["joint PnP"].append(ts.localize_joint(SCENE, d, K))
    for name, poses in est.items():
        ok = [p is not None for p in poses]
        E, Gt = np.stack([p for p in poses if p is not None]), traj[ok]
        rot = [fs.rotation_error_deg(a[:3, :3], b[:3, :3]) for a, b in zip(Gt, E)]
        ate_mm = 1000 * ate(E[:, :3, 3], Gt[:, :3, 3], align="none")["rmse"]
        print(f"  {name:16s}: ATE {ate_mm:7.1f} mm, max rotation error {max(rot):6.2f} deg")
    print("  A single flipped tag drags the mean (it is not robust); a median / RANSAC-style")
    print("  consensus or the joint PnP is unaffected by one bad tag.")


# --------------------------------------------- Exercise 2: ROS frames
def exercise2(rng):
    print("Exercise 2 - object pose in camera_link (REP-103) vs camera_optical (OpenCV)")
    img = fs.render_scene(SCENE.items, K, G.invert_T(SCENE.T_world_optical), BG, rng=rng)
    r = fs.solve_marker_pose(detect_tags(img)[ts.OBJ_TAG_ID], ts.OBJ_TAG_SIZE, K)
    T_opt_object = G.make_T(r.R, r.t) @ G.invert_T(SCENE.T_object_objtag)
    T_base_object = SCENE.T_base_camlink @ SCENE.T_camlink_optical @ T_opt_object
    T_link_object = G.invert_T(SCENE.T_base_camlink) @ T_base_object
    for name, T in [("camera_optical", T_opt_object), ("camera_link", T_link_object)]:
        print(
            f"  in {name:15s}: xyz {T[:3, 3].round(4)}, quat xyzw {G.R_to_quat(T[:3, :3]).round(4)}"
        )
    R = G.R_ROS_FROM_OPENCV
    print(
        "  p_link == R_ROS_FROM_OPENCV @ p_optical:",
        np.allclose(T_link_object[:3, 3], R @ T_opt_object[:3, 3]),
    )
    print(
        "  R_link == R_ROS_FROM_OPENCV @ R_optical:",
        np.allclose(T_link_object[:3, :3], R @ T_opt_object[:3, :3]),
    )
    rpy = Rotation.from_matrix(R).as_euler("xyz", degrees=True)
    print(f"  camera_link -> camera_optical_frame rpy [deg]: {rpy.round(1)} (roll, pitch, yaw)")


# --------------------------------------------- Exercise 3: hand-eye calibration (Park-Martin)
def park_martin(As: list[np.ndarray], Bs: list[np.ndarray]) -> np.ndarray:
    """Solve AX = XB for X (4x4) from motion pairs (Park & Martin, 1994)."""
    M = np.zeros((3, 3))
    for A, B in zip(As, Bs):
        M += np.outer(G.so3_log(B[:3, :3]), G.so3_log(A[:3, :3]))  # beta alpha^T
    w, V = np.linalg.eigh(M.T @ M)
    R_X = V @ np.diag(1.0 / np.sqrt(w)) @ V.T @ M.T
    C = np.concatenate([A[:3, :3] - np.eye(3) for A in As])
    d = np.concatenate([R_X @ B[:3, 3] - A[:3, 3] for A, B in zip(As, Bs)])
    t_X = np.linalg.lstsq(C, d, rcond=None)[0]
    return G.make_T(R_X, t_X)


def exercise3(rng):
    print("Exercise 3 - eye-in-hand calibration, 10 robot poses, 10 cm tag, PnP from noisy corners")
    X_gt = G.make_T(ts.Rz(90) @ ts.Rx(4), [0.03, -0.05, 0.08])  # gripper <- camera
    T_base_tag = G.make_T(np.eye(3), [0.6, 0.1, 0.0])  # tag on the table, z up
    L = 0.10
    obj = fs.marker_object_points(L)
    for sigma in (0.0, 0.2, 0.5):
        T_g, T_ct = [], []
        for _ in range(10):
            # A camera pose looking at the tag from 0.3-0.5 m, with a random roll.
            eye = T_base_tag[:3, 3] + rng.uniform([-0.2, -0.2, 0.3], [0.2, 0.2, 0.45])
            T_base_cam = G.look_at(eye, T_base_tag[:3, 3] + rng.normal(0, 0.02, 3))
            T_base_cam = T_base_cam @ G.make_T(ts.Rz(rng.uniform(-60, 60)), np.zeros(3))
            T_g.append(T_base_cam @ G.invert_T(X_gt))  # what the robot's kinematics report
            T_cam_tag = G.invert_T(T_base_cam) @ T_base_tag
            uv = fs.project(obj, K, T_cam_tag[:3, :3], T_cam_tag[:3, 3])[0]
            r = fs.solve_marker_pose(uv + rng.normal(0, sigma, uv.shape), L, K, refine_lm=True)
            T_ct.append(G.make_T(r.R, r.t))
        As = [G.invert_T(T_g[i]) @ T_g[j] for i in range(10) for j in range(i + 1, 10)]
        Bs = [T_ct[i] @ G.invert_T(T_ct[j]) for i in range(10) for j in range(i + 1, 10)]
        X = park_martin(As, Bs)
        r_err = fs.rotation_error_deg(X_gt[:3, :3], X[:3, :3])
        t_err = 1000 * np.linalg.norm(X[:3, 3] - X_gt[:3, 3])
        print(
            f"  corner noise {sigma:.1f} px: rotation error {r_err:.3f} deg, "
            f"translation error {t_err:.2f} mm"
        )


def main() -> None:
    rng = np.random.default_rng(0)
    exercise1(rng)
    exercise2(rng)
    exercise3(rng)


if __name__ == "__main__":
    main()
