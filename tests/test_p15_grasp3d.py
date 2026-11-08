"""CPU tests for the capstone geometry (project 15): rays, planes, boxes and grasps."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "projects/15_robotics_capstone/scripts")
)

import grasp3d as g3  # noqa: E402

from cvlab.geometry import make_T, so3_exp  # noqa: E402

K = np.array([[517.3, 0.0, 318.6], [0.0, 516.5, 255.3], [0.0, 0.0, 1.0]])
DIST = np.array([0.2624, -0.9531, -0.0054, 0.0026, 1.1633])


def box_surface(extents, n=4000, seed=0, faces=("top", "sides")) -> np.ndarray:
    """Points sampled on the top and the four side faces of an axis-aligned box resting on z=0."""
    rng = np.random.default_rng(seed)
    lx, ly, lz = extents
    pts = []
    if "top" in faces:
        pts.append(
            np.c_[rng.uniform(-lx / 2, lx / 2, n), rng.uniform(-ly / 2, ly / 2, n), np.full(n, lz)]
        )
    if "sides" in faces:
        for sx in (-1, 1):
            pts.append(
                np.c_[
                    np.full(n // 2, sx * lx / 2),
                    rng.uniform(-ly / 2, ly / 2, n // 2),
                    rng.uniform(0, lz, n // 2),
                ]
            )
        for sy in (-1, 1):
            pts.append(
                np.c_[
                    rng.uniform(-lx / 2, lx / 2, n // 2),
                    np.full(n // 2, sy * ly / 2),
                    rng.uniform(0, lz, n // 2),
                ]
            )
    return np.vstack(pts)


def table_points(n=3000, size=0.6, seed=1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.c_[rng.uniform(-size, size, (n, 2)), np.zeros(n)]


# A camera 0.8 m above the table looking down at it (table frame: z up).
T_TABLE_CAM = make_T(so3_exp([np.pi * 0.8, 0.0, 0.0]), [0.0, 0.3, 0.75])  # camera -> table
T_CAM_TABLE = np.linalg.inv(T_TABLE_CAM)


def to_cam(P: np.ndarray) -> np.ndarray:
    return P @ T_CAM_TABLE[:3, :3].T + T_CAM_TABLE[:3, 3]


def test_pixel_rays_roundtrip_with_distortion():
    rays = g3.pixel_rays(K, DIST, 640, 480)
    assert rays.shape == (480, 640, 2)
    uv = np.array([[0, 0], [639, 0], [320, 240], [100, 400], [639, 479]])
    P = np.c_[rays[uv[:, 1], uv[:, 0]], np.ones(len(uv))] * 1.3  # any depth
    back = g3.project_points(P, K, DIST)
    assert np.abs(back - uv).max() < 1e-3  # px


def test_pixel_rays_pinhole_matches_closed_form():
    rays = g3.pixel_rays(K, None, 640, 480)
    assert rays[255, 318, 0] == pytest.approx((318 - 318.6) / 517.3)
    assert rays[100, 50, 1] == pytest.approx((100 - 255.3) / 516.5)


def test_backproject_and_project_are_inverse():
    rays = g3.pixel_rays(K, DIST, 640, 480)
    depth = np.full((480, 640), 2.0)
    depth[:10] = 0.0  # invalid rows are dropped
    pts, pix = g3.backproject(depth, rays, stride=7)
    assert np.all(pix[:, 0] >= 10)
    uv = g3.project_points(pts, K, DIST)
    assert np.abs(uv - pix[:, ::-1]).max() < 1e-3
    assert np.allclose(pts[:, 2], 2.0)


def test_ransac_recovers_plane_with_outliers():
    rng = np.random.default_rng(0)
    P = to_cam(table_points(2000))
    P += rng.normal(0, 0.002, P.shape)
    clutter = rng.uniform(-0.5, 0.5, (800, 3)) + [0, 0, 1.0]
    plane, inl = g3.fit_plane_ransac(np.vstack([P, clutter]), threshold=0.01)
    n_true = T_CAM_TABLE[:3, :3] @ np.array([0, 0, 1.0])  # table z axis in the camera frame
    assert abs(plane.normal @ n_true) > 0.999
    assert plane.normal @ n_true > 0  # oriented towards the camera ("up")
    assert inl[:2000].mean() > 0.95
    assert plane.d > 0


def test_min_area_rect_of_rotated_rectangle():
    rng = np.random.default_rng(0)
    xy = rng.uniform([-0.10, -0.03], [0.10, 0.03], (500, 2))
    th = np.deg2rad(30)
    Rz = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    center, axes, ext = g3.min_area_rect(xy @ Rz.T + [0.5, -0.2])
    assert np.allclose(center, [0.5, -0.2], atol=0.005)
    assert ext[0] == pytest.approx(0.2, abs=0.01) and ext[1] == pytest.approx(0.06, abs=0.01)
    assert abs(axes[:, 0] @ Rz[:, 0]) > 0.999  # long side along the rotated x axis


def test_table_box_recovers_extruded_box():
    ext = np.array([0.18, 0.06, 0.10])
    obj = to_cam(box_surface(ext))
    plane, _ = g3.fit_plane_ransac(to_cam(table_points()), threshold=0.005)
    box = g3.table_box(obj, plane)
    assert box.kind == "table"
    assert np.allclose(box.extents, ext, atol=0.006)
    c_true = to_cam(np.array([[0, 0, ext[2] / 2]]))[0]
    assert np.linalg.norm(box.center - c_true) < 0.005
    assert np.isclose(np.linalg.det(box.R), 1.0)
    assert abs(box.R[:, 2] @ plane.normal) > 0.999


def test_table_box_extrudes_when_only_the_top_is_visible():
    ext = np.array([0.12, 0.05, 0.07])
    top_only = to_cam(box_surface(ext, faces=("top",)))
    plane, _ = g3.fit_plane_ransac(to_cam(table_points()), threshold=0.005)
    box = g3.table_box(top_only, plane)
    assert box.extents[2] == pytest.approx(0.07, abs=0.003)  # height from the plane
    pca = g3.pca_box(top_only)
    assert pca.extents[2] < 0.01  # the visible-surface box is flat


def test_grasp_on_narrow_box_closes_along_minor_axis():
    ext = np.array([0.18, 0.05, 0.08])
    obj = to_cam(box_surface(ext))
    plane, _ = g3.fit_plane_ransac(to_cam(table_points()), threshold=0.005)
    box = g3.table_box(obj, plane)
    grasps = g3.top_down_grasps(obj, box, plane)
    best = grasps[0]
    assert best.feasible and best.axis == "minor"
    assert best.width == pytest.approx(0.05, abs=0.006)
    R = best.T[:3, :3]
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-9) and np.linalg.det(R) > 0
    assert R[:, 2] @ plane.normal < -0.999  # approach points down into the table
    assert abs(R[:, 0] @ box.R[:, 1]) > 0.999  # closing direction = minor axis
    h = plane.height(best.position[None])[0]
    assert 0.0 < h < ext[2]


def test_grasp_rejects_wide_and_flat_objects():
    plane, _ = g3.fit_plane_ransac(to_cam(table_points()), threshold=0.005)
    wide = to_cam(box_surface([0.30, 0.20, 0.05]))
    assert not any(gr.feasible for gr in g3.top_down_grasps(wide, g3.table_box(wide, plane), plane))
    flat = to_cam(box_surface([0.06, 0.04, 0.004]))
    res = g3.top_down_grasps(flat, g3.table_box(flat, plane), plane)
    assert not any(gr.feasible for gr in res)
    assert any("flat" in gr.reason for gr in res)


def test_grasp_detects_collision_with_neighbour():
    ext = np.array([0.15, 0.05, 0.08])
    obj = to_cam(box_surface(ext))
    plane, _ = g3.fit_plane_ransac(to_cam(table_points()), threshold=0.005)
    box = g3.table_box(obj, plane)
    # a tall neighbour touching the object's long side blocks every minor-axis grasp
    wall = box_surface([0.3, 0.01, 0.12], seed=3) + [0.0, 0.025 + 0.01 + 0.006, 0.0]
    grasps = g3.top_down_grasps(obj, box, plane, scene_points=to_cam(wall))
    minor = [gr for gr in grasps if gr.axis == "minor" and gr.width > 0]
    assert minor and all(not gr.feasible for gr in minor)
    assert any("collision" in gr.reason for gr in minor)


def test_interpolate_pose_midpoint():
    T0 = make_T(np.eye(3), [0, 0, 0])
    T1 = make_T(so3_exp([0, 0, np.pi / 2]), [1.0, 2.0, 0.0])
    T = g3.interpolate_pose(np.array([0.0, 1.0]), np.stack([T0, T1]), 0.5)
    assert np.allclose(T[:3, 3], [0.5, 1.0, 0.0])
    assert np.allclose(T[:3, :3], so3_exp([0, 0, np.pi / 4]), atol=1e-9)


def test_object_cloud_gates_background_leak():
    rays = g3.pixel_rays(K, None, 640, 480)
    depth = np.full((480, 640), 2.0)  # background at 2 m
    mask = np.zeros((480, 640), bool)
    mask[200:280, 280:360] = True
    depth[205:275, 285:355] = 0.8  # the object; the mask border leaks onto the background
    cloud = g3.object_cloud(depth, mask, rays, erode_px=0, voxel=0.002)
    assert cloud is not None
    assert np.allclose(cloud.points[:, 2], 0.8, atol=1e-6)
    assert cloud.n_gated < cloud.n_valid
