"""CPU unit tests for projects/12_visual_odometry (synthetic data, no downloads)."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from cvlab.geometry import (
    PinholeCamera,
    angle_between_deg,
    invert_T,
    make_T,
    project,
    rotation_error_deg,
    so3_exp,
    unproject,
)
from cvlab.metrics import rpe as cvlab_rpe

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "projects/12_visual_odometry/scripts"))
import synthetic  # noqa: E402
import vo  # noqa: E402

CAM = PinholeCamera(400.0, 400.0, 320.0, 240.0, 640, 480)


def _scene(rng, n=300):
    """Random 3D points 2-6 m in front of camera 1."""
    return np.column_stack([rng.uniform(-2, 2, n), rng.uniform(-1.5, 1.5, n), rng.uniform(2, 6, n)])


def _two_view(rng, n=300, noise=0.3):
    X1 = _scene(rng, n)
    T21 = make_T(so3_exp([0.02, -0.05, 0.01]), [0.3, -0.05, 0.1])
    X2 = X1 @ T21[:3, :3].T + T21[:3, 3]
    uv1, _ = project(X1, CAM.K)
    uv2, _ = project(X2, CAM.K)
    uv1 += rng.normal(0, noise, uv1.shape)
    uv2 += rng.normal(0, noise, uv2.shape)
    return X1, T21, uv1, uv2


@pytest.mark.parametrize(("method", "tol_deg"), [("magsac", 0.2), ("ransac", 1.0)])
def test_relative_pose_essential_recovers_rotation_and_direction(rng, method, tol_deg):
    _, T21, uv1, uv2 = _two_view(rng)
    uv2[:30] = rng.uniform([0, 0], [640, 480], (30, 2))  # 10 % gross outliers
    T, inl = vo.relative_pose_essential(uv1, uv2, CAM.K, method=method)
    assert T is not None  # plain RANSAC has no local optimisation -> looser tolerance
    assert rotation_error_deg(T[:3, :3], T21[:3, :3]) < tol_deg
    assert angle_between_deg(T[:3, 3], T21[:3, 3]) < 5 * tol_deg
    assert np.linalg.norm(T[:3, 3]) == pytest.approx(1.0)  # scale is unobservable
    assert inl[:30].mean() < 0.2 and inl[30:].mean() > 0.9


def test_pose_pnp_metric_pose_and_outliers(rng):
    X1, T21, _, uv2 = _two_view(rng)
    uv2[:60] += rng.uniform(20, 60, (60, 2))  # 20 % outliers
    T, inl = vo.pose_pnp(X1, uv2, CAM.K, reproj_px=2.0)
    assert T is not None
    assert np.linalg.norm(T[:3, 3] - T21[:3, 3]) < 0.01  # metric translation (m)
    assert rotation_error_deg(T[:3, :3], T21[:3, :3]) < 0.1
    assert not inl[:60].any() and inl[60:].mean() > 0.95


def test_triangulate_noise_free_and_rejects_low_parallax(rng):
    X1, T21, uv1, uv2 = _two_view(rng, noise=0.0)
    Xc, ok = vo.triangulate(uv1, uv2, T21, CAM.K, min_angle_deg=0.5)
    X2 = X1 @ T21[:3, :3].T + T21[:3, 3]
    np.testing.assert_allclose(Xc[ok], X2[ok], atol=1e-6)
    assert ok.mean() > 0.9
    # a tiny baseline gives tiny ray angles -> every point is rejected
    T_small = make_T(np.eye(3), [0.001, 0, 0])
    uv2s, _ = project(X1 + T_small[:3, 3], CAM.K)
    _, ok_small = vo.triangulate(uv1, uv2s, T_small, CAM.K, min_angle_deg=0.5)
    assert not ok_small.any()


def test_undistort_pixels_inverts_opencv_distortion(rng):
    dist = np.array([0.2624, -0.9531, -0.0054, 0.0026, 1.1633])  # TUM fr1 RGB
    ideal = rng.uniform([60, 60], [580, 420], (100, 2))
    rays = np.column_stack([(ideal - CAM.K[:2, 2]) / [CAM.fx, CAM.fy], np.ones(100)])
    distorted, _ = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), CAM.K, dist)
    back = vo.undistort_pixels(distorted.reshape(-1, 2), CAM.K, dist)
    np.testing.assert_allclose(back, ideal, atol=1e-6)
    np.testing.assert_array_equal(vo.undistort_pixels(ideal, CAM.K, None), ideal)


def test_lookup_depth_and_edge_mask():
    depth = np.full((40, 60), 1.0, np.float32)
    depth[:, 30:] = 2.0  # occlusion boundary at u = 30
    depth[5, 5] = 0.0  # hole
    smooth = vo.depth_edge_mask(depth, ksize=5)
    uv = np.array([[10.0, 20.0], [29.6, 20.0], [45.0, 20.0], [5.0, 5.0], [10.0, 20.0]])
    z = vo.lookup_depth(depth, uv, z_range=(0.2, 5.0), smooth_mask=smooth)
    assert z[0] == pytest.approx(1.0) and z[2] == pytest.approx(2.0)
    assert np.isnan(z[1]) and np.isnan(z[3])  # near the edge / in a hole
    z_far = vo.lookup_depth(depth, uv, z_range=(0.2, 1.5))
    assert np.isnan(z_far[2]) and z_far[0] == pytest.approx(1.0)


def test_feature_frontend_matches_shifted_image(rng):
    img = cv2.GaussianBlur(rng.random((240, 320)).astype(np.float32), (0, 0), 2) * 255
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    shifted = np.roll(img, (3, 7), axis=(0, 1))
    for kind in ("orb", "sift"):
        fe = vo.FeatureFrontend(kind, 500)
        uv1, d1 = fe.detect(img)
        uv2, d2 = fe.detect(shifted)
        m = fe.match(d1, d2)
        assert len(m) > 50
        assert len(np.unique(m[:, 1])) == len(m)  # one-to-one
        flow = np.median(uv2[m[:, 1]] - uv1[m[:, 0]], axis=0)
        np.testing.assert_allclose(flow, [7, 3], atol=0.5)
    assert len(vo.FeatureFrontend("orb").match(None, d2)) == 0


def _random_walk(rng, n=40):
    poses = [np.eye(4)]
    for _ in range(n - 1):
        step = make_T(so3_exp(rng.normal(0, 0.02, 3)), rng.normal(0, 0.05, 3))
        poses.append(poses[-1] @ step)
    return np.stack(poses)


def test_rpe_time_matches_frame_rpe_and_alignment(rng):
    gt = _random_walk(rng)
    noisy = gt.copy()
    noisy[:, :3, 3] += rng.normal(0, 0.01, (len(gt), 3))
    ts = np.arange(len(gt)) * 0.1  # 10 Hz -> 1 s = 10 frames
    r = vo.rpe_time(ts, noisy, gt, delta_s=1.0, tol_s=0.05)
    ref = cvlab_rpe(noisy, gt, delta=10)
    assert len(r["trans"]) == len(gt) - 10
    assert np.sqrt(np.mean(r["trans"] ** 2)) == pytest.approx(ref["trans_rmse"])
    assert np.sqrt(np.mean(r["rot"] ** 2)) == pytest.approx(ref["rot_rmse_deg"], abs=1e-6)
    # a Sim(3)-transformed copy aligns back exactly
    S = make_T(so3_exp([0.3, -0.2, 1.0]), [1.0, 2.0, -0.5])
    est = np.einsum("ij,njk->nik", S, gt)
    est[:, :3, 3] = 0.25 * est[:, :3, 3]
    aligned, s = vo.align_poses(est, gt, "sim3")
    assert s == pytest.approx(4.0)
    np.testing.assert_allclose(aligned, gt, atol=1e-8)


def test_evo_metrics_agree_with_cvlab(rng):
    pytest.importorskip("evo")
    from cvlab.metrics import ate

    gt = _random_walk(rng, 60)
    noise = [make_T(so3_exp(rng.normal(0, 0.01, 3)), rng.normal(0, 0.02, 3)) for _ in gt]
    est = np.einsum("nij,njk->nik", gt, np.stack(noise))
    ts = np.arange(60) / 30.0
    ev = vo.evo_metrics(ts, est, gt, align="se3", delta_frames=5)
    assert ev["ate_rmse_m"] == pytest.approx(ate(est[:, :3, 3], gt[:, :3, 3], "se3")["rmse"])
    ref = cvlab_rpe(est, gt, delta=5)  # RPE is invariant to the rigid alignment
    assert ev["rpe_trans_rmse_m"] == pytest.approx(ref["trans_rmse"])
    assert ev["rpe_rot_rmse_deg"] == pytest.approx(ref["rot_rmse_deg"], abs=1e-6)


@pytest.fixture(scope="module")
def synthetic_seq(tmp_path_factory):
    return synthetic.make_sequence(tmp_path_factory.mktemp("p12_syn"), n_frames=45)


def test_synthetic_sequence_is_consistent(synthetic_seq):
    seq = synthetic_seq
    # a pixel's back-projected depth lands on the same world point seen from another frame
    K, d0 = seq.camera.K, seq.depth(0)
    uv = np.array([[100.0, 80.0], [200.0, 150.0], [300.0, 200.0]])
    Xc = unproject(uv, d0[uv[:, 1].astype(int), uv[:, 0].astype(int)], K)
    Xw = Xc @ seq.gt_poses[0][:3, :3].T + seq.gt_poses[0][:3, 3]
    T_cw = invert_T(seq.gt_poses[10])
    uv10, z10 = project(Xw @ T_cw[:3, :3].T + T_cw[:3, 3], K)
    d10 = seq.depth(10)
    for (u, v), z in zip(uv10, z10):
        assert d10[int(round(v)), int(round(u))] == pytest.approx(z, rel=0.02)


@pytest.mark.parametrize("keyframes", [False, True])
def test_rgbd_pnp_vo_on_synthetic(synthetic_seq, keyframes):
    seq = synthetic_seq
    res = vo.run_pnp_vo(seq, vo.FeatureFrontend("sift", 1000), keyframes=keyframes)
    assert len(res.frame_ids) == len(seq)  # every frame tracked
    m = vo.evaluate(res, seq, align="none")  # anchored at the first GT pose: no alignment
    assert m["ate_rmse_m"] < 0.02
    assert m["rpe_rot_rmse_deg"] < 0.5
    assert (m["keyframes"] < len(seq)) == keyframes


def test_mono_vo_on_synthetic(synthetic_seq, tmp_path):
    seq = synthetic_seq
    fe = vo.FeatureFrontend("sift", 1000)
    res = vo.run_mono_essential(seq, fe, scale="gt", min_parallax_px=8)
    assert len(res.frame_ids) >= 4
    assert vo.evaluate(res, seq, align="none")["ate_rmse_m"] < 0.02
    unit = vo.run_mono_essential(seq, fe, scale="unit", min_parallax_px=8)
    assert np.allclose(np.linalg.norm(np.diff(unit.poses[:, :3, 3], axis=0), axis=1), 1.0)
    mono_map = vo.run_pnp_vo(seq, fe, depth="triangulated")
    m = vo.evaluate(mono_map, seq, align="sim3")
    assert m["ate_rmse_m"] < 0.02 and 0 < m["scale"]
    path = mono_map.save_tum(tmp_path / "est.txt")
    assert len(path.read_text().splitlines()) == len(mono_map.frame_ids)


def test_translational_parallax_ignores_pure_rotation(rng):
    X1 = _scene(rng)
    uv1, _ = project(X1, CAM.K)
    R = so3_exp([0.03, -0.04, 0.02])  # ~3 deg: tens of pixels of image motion
    uv_rot, _ = project(X1 @ R.T, CAM.K)
    assert np.median(np.linalg.norm(uv_rot - uv1, axis=1)) > 10
    assert vo._translational_parallax(uv1, uv_rot, R, CAM.K) < 1e-6
    uv_tr, _ = project(X1 @ R.T + [0.2, 0.0, 0.0], CAM.K)
    assert vo._translational_parallax(uv1, uv_tr, R, CAM.K) > 5


class _IdentityMatcher:
    """Stand-in front end: keypoint k of one frame matches keypoint k of the other."""

    def match(self, d1, d2):
        n = min(len(d1), len(d2))
        return np.column_stack([np.arange(n), np.arange(n)])


def test_mono_keyframe_rescales_new_points_to_map_scale(rng):
    # map units are 2x metres; PnP returned a baseline half as long as it should be in map
    # units, which without the relative-scale step would plant new points at half the scale
    X1 = _scene(rng, 400)
    R, t = so3_exp([0.01, 0.03, 0.0]), np.array([0.3, 0.0, 0.05])
    uv1, _ = project(X1, CAM.K)
    uv2, _ = project(X1 @ R.T + t, CAM.K)
    X_map = np.full_like(X1, np.nan)
    X_map[:200] = 2.0 * X1[:200]  # only the first half is already in the map
    desc = np.zeros((400, 32), np.uint8)
    kf = vo._Keyframe(0, np.eye(4), uv1, desc, X_map)
    T_ck = make_T(R, t)  # metric baseline = half of the 0.6 map units it should be
    new_kf = vo._mono_keyframe(kf, 1, np.eye(4), T_ck, uv2, desc, _IdentityMatcher(), CAM.K)
    z_true_map = 2.0 * (X1 @ R.T + t)[:, 2]
    new = np.arange(200, 400)
    ok = np.isfinite(new_kf.X[new, 2])
    assert ok.mean() > 0.8
    ratio = new_kf.X[new[ok], 2] / z_true_map[new[ok]]
    assert np.median(ratio) == pytest.approx(1.0, abs=0.05)
