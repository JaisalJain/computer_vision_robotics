"""Project 07: synthetic fiducial renderer, frame conventions and PnP helpers (CPU only)."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "projects/07_pose_estimation/scripts"))

import fiducial_sim as fs  # noqa: E402

K = np.array([[517.3, 0.0, 318.6], [0.0, 516.5, 255.3], [0.0, 0.0, 1.0]])
SIZE = (640, 480)


def _background(seed: int = 0) -> np.ndarray:
    """Smooth random texture (no dataset needed)."""
    rng = np.random.default_rng(seed)
    small = rng.uniform(60, 200, (12, 16, 3)).astype(np.float32)
    return cv2.resize(small, SIZE, interpolation=cv2.INTER_CUBIC).clip(0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def tag():
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    return d, fs.marker_texture(d, 7, 0.10)


def test_marker_texture_maps_black_border_to_object_corners(tag):
    _, tex = tag
    pad, side = tex.meta["pad_px"], tex.meta["side_px"]
    # Outer black-border corners (pixel-centre convention) in TL, TR, BR, BL order.
    a, b = pad - 0.5, pad + side - 0.5
    px = np.array([[a, a, 1], [b, a, 1], [b, b, 1], [a, b, 1]])
    xy = (tex.A @ px.T).T[:, :2]
    np.testing.assert_allclose(xy, fs.marker_object_points(0.10)[:, :2], atol=1e-12)


def test_pose_facing_viewing_angle_and_distance(tag):
    _, tex = tag
    R, t = fs.pose_facing(tex, K, (100.0, 400.0), 1.7, tilt_deg=35.0, tilt_dir_deg=20, roll_deg=50)
    np.testing.assert_allclose(np.linalg.norm(t), 1.7)
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
    # Angle between the marker normal (+Z, towards the viewer) and the direction to the camera.
    view = np.degrees(np.arccos(R[:, 2] @ (-t / np.linalg.norm(t))))
    assert view == pytest.approx(35.0, abs=1e-9)


@pytest.mark.parametrize("method", ["IPPE_SQUARE", "IPPE", "SQPNP", "ITERATIVE"])
def test_exact_corners_give_exact_pose(tag, method):
    _, tex = tag
    rng = np.random.default_rng(1)
    obj = fs.marker_object_points(0.10)
    for _ in range(20):
        R, t = fs.sample_pose(tex, K, SIZE, rng.uniform(0.3, 2.0), rng.uniform(0, 70), rng)
        uv, _ = fs.project(obj, K, R, t)
        res = fs.solve_marker_pose(uv, 0.10, K, method=method)
        assert fs.rotation_error_deg(R, res.R) < 0.01
        assert np.linalg.norm(res.t - t) < 1e-5


def test_render_detect_and_pose(tag):
    d, tex = tag
    det = fs.make_detector(d, "subpix")
    rng = np.random.default_rng(2)
    bg = _background()
    for _ in range(5):
        R, t = fs.sample_pose(tex, K, SIZE, 0.5, 30.0, rng)
        img = fs.render(tex, K, R, t, bg, rng=rng)
        found = fs.detect(det, img)
        assert set(found) == {7}
        gt, _ = fs.project(fs.marker_object_points(0.10), K, R, t)
        assert np.linalg.norm(found[7] - gt, axis=1).max() < 0.6  # same corner order as IPPE
        res = fs.solve_marker_pose(found[7], 0.10, K)
        assert fs.rotation_error_deg(R, res.R) < 2.0
        assert fs.translation_errors(t, res.t)["t_err"] < 0.01  # < 1 cm at 0.5 m


def test_charuco_texture_frame():
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    board = cv2.aruco.CharucoBoard((5, 7), 0.03, 0.022, d)
    tex = fs.charuco_texture(board)
    R, t = fs.pose_facing(tex, K, (320, 240), 0.6, tilt_deg=20, tilt_dir_deg=30, roll_deg=10)
    img = fs.render(tex, K, R, t, _background(1), rng=np.random.default_rng(3))
    corners, ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(img)
    assert ids is not None and len(ids) >= 20
    obj, imgp = board.matchImagePoints(corners, ids)
    ok, rvec, tvec = cv2.solvePnP(obj, imgp, K, None, flags=cv2.SOLVEPNP_SQPNP)
    assert ok
    assert fs.rotation_error_deg(R, cv2.Rodrigues(rvec)[0]) < 1.0
    assert np.linalg.norm(tvec.ravel() - t) < 0.005


def test_image_texture_and_scene_render():
    photo = _background(2)
    tex = fs.image_texture(photo, 1.2)
    assert tex.size == pytest.approx((1.2, 0.9))
    T_world_cam = np.eye(4)
    T_world_cam[:3, :3] = fs.R_FRONT_MARKER  # camera looking down -Z of the plane
    T_world_cam[:3, 3] = [0.0, 0.0, 1.0]
    T_cam_world = np.linalg.inv(T_world_cam)
    img = fs.render_scene([(tex, np.eye(4))], K, T_cam_world, np.zeros((480, 640, 3), np.uint8),
                          noise_sigma=0.0)  # fmt: skip
    assert img.mean() > 50  # the plane fills most of the view


def test_pupil_apriltags_conventions(tag):
    pupil = pytest.importorskip("pupil_apriltags")
    d, tex = tag
    detector = pupil.Detector(families="tag36h11", nthreads=1)
    rng = np.random.default_rng(4)
    errs = []
    for _ in range(5):
        R, t = fs.sample_pose(tex, K, SIZE, 0.6, 30.0, rng)
        img = fs.render(tex, K, R, t, _background(), blur_sigma=0.0, noise_sigma=0.0)
        out = fs.pupil_detect(detector, img, K, 0.10, estimate_pose=True)[7]
        gt, _ = fs.project(fs.marker_object_points(0.10), K, R, t)
        errs.append(out["corners"] - gt)
        assert fs.rotation_error_deg(R, out["R"]) < 2.0
        assert np.linalg.norm(out["t"] - t) < 0.005
    bias = np.concatenate(errs).mean(0)
    assert np.abs(bias).max() < 0.15  # half-pixel convention removed


def test_match_corner_order():
    ref = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    shuffled = np.roll(ref[::-1], 1, axis=0)
    np.testing.assert_allclose(fs.match_corner_order(shuffled, ref), ref)
