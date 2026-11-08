"""CPU tests for the project-01 calibration CLI on synthetic checkerboard views."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from cvlab.geometry import so3_exp

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "projects" / "01_classical_cv" / "scripts" / "calibrate_camera.py"
PATTERN = (9, 6)
SQ_PX = 40  # texture pixels per board square
MARGIN = 40  # white margin around the 10x7 squares [texture px]
K_TRUE = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])


@pytest.fixture(scope="module")
def calib():
    spec = importlib.util.spec_from_file_location("p01_calibrate_camera", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def board_texture() -> np.ndarray:
    """10x7 squares with a white margin; inner corner (0, 0) at texture x = MARGIN + SQ_PX - 0.5."""
    cols, rows = PATTERN[0] + 1, PATTERN[1] + 1
    tex = np.full((rows * SQ_PX + 2 * MARGIN, cols * SQ_PX + 2 * MARGIN), 255, np.uint8)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                y, x = MARGIN + r * SQ_PX, MARGIN + c * SQ_PX
                tex[y : y + SQ_PX, x : x + SQ_PX] = 0
    return tex


def render_view(tex: np.ndarray, rotvec: np.ndarray, distance: float) -> np.ndarray:
    """Image of the board (units: squares) seen by K_TRUE; board centre on the optical axis."""
    R = so3_exp(np.asarray(rotvec, float))
    center = np.array([(PATTERN[0] - 1) / 2, (PATTERN[1] - 1) / 2, 0.0])
    t = np.array([0.0, 0.0, distance]) - R @ center
    # texture px -> board units (squares), then the plane-induced homography K [r1 r2 t]
    s = 1.0 / SQ_PX
    off = (MARGIN + SQ_PX - 0.5) / SQ_PX  # pixel-centre convention: square edges lie at i - 0.5
    S = np.array([[s, 0, -off], [0, s, -off], [0, 0, 1.0]])
    H = K_TRUE @ np.column_stack([R[:, 0], R[:, 1], t]) @ S
    return cv2.warpPerspective(tex, H, (640, 480), flags=cv2.INTER_LINEAR, borderValue=128)


@pytest.fixture(scope="module")
def views() -> list[np.ndarray]:
    tex = board_texture()
    poses = [
        ((0.0, 0.0, 0.0), 14.0),
        ((0.35, 0.0, 0.0), 13.0),
        ((-0.35, 0.0, 0.1), 13.5),
        ((0.0, 0.4, 0.0), 14.0),
        ((0.0, -0.4, -0.1), 13.0),
        ((0.3, 0.3, 0.2), 15.0),
        ((-0.3, -0.3, 0.0), 15.0),
        ((0.25, -0.35, 0.3), 14.5),
    ]
    return [render_view(tex, rv, d) for rv, d in poses]


def test_board_points_layout(calib):
    obj = calib.board_points(PATTERN, 0.025)
    assert obj.shape == (54, 3) and obj.dtype == np.float32
    np.testing.assert_allclose(obj[1], [0.025, 0.0, 0.0])  # x varies fastest
    np.testing.assert_allclose(obj[9], [0.0, 0.025, 0.0])
    assert np.all(obj[:, 2] == 0)


def test_detect_corners_shape_and_accuracy(calib, views):
    c = calib.detect_corners(views[0], PATTERN)
    assert c is not None and c.shape == (54, 2)
    obj = calib.board_points(PATTERN)
    center = np.array([(PATTERN[0] - 1) / 2, (PATTERN[1] - 1) / 2, 0.0])
    proj, _ = cv2.projectPoints(obj, np.zeros(3), np.array([0.0, 0.0, 14.0]) - center, K_TRUE, None)
    err = np.linalg.norm(c - proj.reshape(-1, 2), axis=1)
    # the detector may start from either end of the symmetric grid
    err_rev = np.linalg.norm(c[::-1] - proj.reshape(-1, 2), axis=1)
    assert min(err.mean(), err_rev.mean()) < 0.15


def test_detect_corners_returns_none_without_board(calib):
    assert calib.detect_corners(np.full((480, 640), 128, np.uint8), PATTERN) is None


def test_calibration_recovers_intrinsics(calib, views):
    res = calib.calibrate_images(views, PATTERN, 1.0, half_window=5, loo=True)
    K = res["K"]
    assert len(res["used"]) == len(views) and not res["skipped"]
    assert res["rms"] < 0.2
    assert abs(K[0, 0] - 500) / 500 < 0.01 and abs(K[1, 1] - 500) / 500 < 0.01
    assert abs(K[0, 2] - 320) < 3 and abs(K[1, 2] - 240) < 3
    assert np.all(np.abs(res["dist"][:2]) < 0.05)  # no distortion was simulated
    assert res["loo_rms"].shape == (len(views),) and np.all(res["loo_rms"] < 0.5)


def test_camera_info_yaml_roundtrip(calib, views):
    res = calib.calibrate_images(views, PATTERN, 1.0, half_window=5)
    info = yaml.safe_load(yaml.safe_dump(calib.camera_info(res, "synthetic")))
    assert info["image_width"] == 640 and info["image_height"] == 480
    assert info["distortion_model"] == "plumb_bob"
    assert len(info["camera_matrix"]["data"]) == 9 and len(info["projection_matrix"]["data"]) == 12
    assert len(info["distortion_coefficients"]["data"]) == 5
    np.testing.assert_allclose(np.reshape(info["camera_matrix"]["data"], (3, 3)), res["K"])


def test_too_few_views_raises(calib):
    blank = [np.full((480, 640), 128, np.uint8)] * 3
    with pytest.raises(RuntimeError):
        calib.calibrate_images(blank, PATTERN)
