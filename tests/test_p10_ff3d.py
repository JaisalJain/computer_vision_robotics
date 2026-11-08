"""CPU tests for projects/10_feedforward_3d/scripts/ff3d.py (pose metrics, Sim(3), caching)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_SCRIPTS = Path(__file__).resolve().parents[1] / "projects/10_feedforward_3d/scripts"
sys.path.insert(0, str(PROJECT_SCRIPTS))
import ff3d  # noqa: E402

from cvlab import geometry as G  # noqa: E402


@pytest.fixture(scope="module")
def ring_poses():
    """12 camera-from-world poses on a ring around the origin, all looking at it."""
    poses = []
    for a in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        eye = np.array([3 * np.cos(a), 3 * np.sin(a), 1.0 + 0.2 * np.sin(3 * a)])
        T_wc = G.look_at(eye, [0.0, 0.0, 0.3])
        poses.append(G.invert_T(T_wc))
    return np.stack(poses)


def random_sim3(seed: int = 0) -> tuple[float, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return 0.37, G.so3_exp(rng.normal(size=3)), rng.normal(size=3)


def test_camera_centers(ring_poses):
    C = ff3d.camera_centers(ring_poses)
    for T, c in zip(ring_poses, C):
        np.testing.assert_allclose(G.invert_T(T)[:3, 3], c, atol=1e-12)


def test_sim3_apply_poses_is_consistent(ring_poses):
    """Mapping the world by X' = sRX + t moves the centres the same way and scales camera z."""
    s, R, t = random_sim3()
    T2 = ff3d.sim3_apply_poses(ring_poses, s, R, t)
    np.testing.assert_allclose(ff3d.camera_centers(T2),
                               ff3d.sim3_apply(ff3d.camera_centers(ring_poses), s, R, t),
                               atol=1e-10)  # fmt: skip
    X = np.array([0.1, -0.2, 0.4])
    Xc = ring_poses[3, :3, :3] @ X + ring_poses[3, :3, 3]
    Xc2 = T2[3, :3, :3] @ ff3d.sim3_apply(X[None], s, R, t)[0] + T2[3, :3, 3]
    np.testing.assert_allclose(Xc2, s * Xc, atol=1e-10)


def test_relative_errors_are_gauge_free(ring_poses):
    s, R, t = random_sim3(1)
    pe = ff3d.relative_pose_errors(ff3d.sim3_apply_poses(ring_poses, s, R, t), ring_poses)
    assert len(pe["rot_deg"]) == 12 * 11 // 2
    # arccos near 1 limits the precision to ~1e-4 deg in float64
    assert pe["rot_deg"].max() < 1e-3
    assert pe["trans_deg"].max() < 1e-3


def test_pose_auc_matches_hand_computation():
    assert ff3d.pose_auc(np.zeros(10), np.zeros(10), 5) == pytest.approx(1.0)
    # Every pair at 2.5 deg falls in the [2, 3) bin -> acc = [0, 0, 1, 1, 1] -> mean 0.6.
    assert ff3d.pose_auc(np.full(4, 2.5), np.zeros(4), 5) == pytest.approx(0.6)
    # The pair error is the max of the rotation and translation errors.
    assert ff3d.pose_auc(np.zeros(4), np.full(4, 2.5), 5) == pytest.approx(0.6)
    assert ff3d.pose_auc(np.full(4, 40.0), np.zeros(4), 30) == pytest.approx(0.0)


def test_sim3_align_recovers_gauge(ring_poses):
    s, R, t = random_sim3(2)
    pred = ff3d.sim3_apply_poses(ring_poses, s, R, t)  # same cameras, other gauge
    al = ff3d.sim3_align(pred, ring_poses)
    assert al["s"] == pytest.approx(1 / s, rel=1e-9)
    assert al["rot_err_deg"].max() < 1e-3
    assert al["centre_err"].max() < 1e-9


def test_evaluate_poses_handles_missing_and_noise(ring_poses):
    rng = np.random.default_rng(3)
    pred = ring_poses.copy()
    for k in range(len(pred)):  # ~1 deg rotation noise per camera
        noise = G.so3_exp(np.radians(1.0) * rng.normal(size=3) / np.sqrt(3))
        pred[k, :3, :3] = noise @ pred[k, :3, :3]
    pred[4] = np.nan  # e.g. an image COLMAP did not register
    m = ff3d.evaluate_poses(pred, ring_poses, metric_units=True)
    assert m["n_eval"] == 11 and m["n_total"] == 12
    assert 0.0 < m["auc@5"] < m["auc@15"] < m["auc@30"] < 1.0
    assert 0.0 < m["abs_rot_med_deg"] < 3.0
    assert m["rra@5"] == pytest.approx(1.0)
    assert {"ate_pct", "ate_m", "sim3_scale"} <= set(m)
    few = ff3d.evaluate_poses(pred[3:6], ring_poses[3:6])  # 2 valid images: too few for metrics
    assert few["n_eval"] == 2 and "auc@5" not in few


def test_evenly_spaced():
    np.testing.assert_array_equal(ff3d.evenly_spaced(128, 2), [0, 127])
    idx = ff3d.evenly_spaced(128, 24)
    assert len(idx) == 24 and idx[0] == 0 and idx[-1] == 127 and np.all(np.diff(idx) > 0)
    np.testing.assert_array_equal(ff3d.evenly_spaced(5, 10), np.arange(5))


def test_upright_rotation(ring_poses):
    # look_at() cameras have their 'up' (-y in OpenCV) roughly along +z of the world.
    R_up = ff3d.upright_rotation(ring_poses)
    np.testing.assert_allclose(R_up @ R_up.T, np.eye(3), atol=1e-12)
    up = -np.mean([G.invert_T(T)[:3, 1] for T in ring_poses], axis=0)
    up_rot = R_up @ (up / np.linalg.norm(up))
    np.testing.assert_allclose(up_rot, [0.0, 0.0, 1.0], atol=1e-9)


def test_fov_and_chamfer():
    K = np.array([[320.0, 0, 320], [0, 320.0, 240], [0, 0, 1]])
    assert ff3d.fov_deg(K, 640) == pytest.approx(90.0)
    pts = np.random.default_rng(4).normal(size=(500, 3))
    c = ff3d.chamfer(pts, pts)
    assert c["chamfer_mean"] == pytest.approx(0.0)
    c2 = ff3d.chamfer(pts + [0.01, 0, 0], pts)
    assert c2["acc_median"] <= 0.01 + 1e-12


def test_recon_round_trip(tmp_path, ring_poses):
    rng = np.random.default_rng(5)
    n, h, w = 3, 8, 10
    rec = ff3d.Recon(method="vggt", T_cw=ring_poses[:n], K=np.tile(np.eye(3), (n, 1, 1)),
                     proc_hw=(h, w), depth=rng.random((n, h, w), dtype=np.float32),
                     world_points=rng.random((n, h, w, 3), dtype=np.float32),
                     runtime_s=1.5, peak_vram_mb=1234.0, extra={"precision": "mixed"})  # fmt: skip
    back = ff3d.Recon.load(rec.save(tmp_path / "r.npz"))
    assert back.method == "vggt" and back.proc_hw == (h, w) and back.extra["precision"] == "mixed"
    np.testing.assert_allclose(back.T_cw, rec.T_cw)
    np.testing.assert_allclose(back.world_points, rec.world_points, atol=1e-3)  # stored as fp16
    assert back.conf is None and back.runtime_s == 1.5
