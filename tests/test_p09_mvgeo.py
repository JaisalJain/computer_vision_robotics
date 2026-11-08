"""CPU tests for projects/09_multiview_geometry/scripts/mvgeo.py (no datasets needed)."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

PROJECT_SCRIPTS = Path(__file__).resolve().parents[1] / "projects/09_multiview_geometry/scripts"
sys.path.insert(0, str(PROJECT_SCRIPTS))
import mvgeo as M  # noqa: E402

from cvlab import geometry as G  # noqa: E402


@pytest.fixture(scope="module")
def two_view():
    """Random 3D points seen by two cameras with a known relative pose."""
    rng = np.random.default_rng(0)
    cam = G.PinholeCamera.from_fov(640, 480, 60.0)
    X = rng.uniform([-1.0, -0.5, 0.0], [1.0, 1.0, 1.5], size=(200, 3))
    T1 = G.look_at([-0.8, -3.0, 1.2], [0.0, 0.2, 0.7])
    T2 = G.look_at([0.6, -2.8, 1.4], [0.0, 0.2, 0.7])
    R, t = M.relative_pose_from_c2w(T1, T2)
    x1, _ = G.project_points(X, cam.K, G.invert_T(T1))
    x2, _ = G.project_points(X, cam.K, G.invert_T(T2))
    return {"K": cam.K, "R": R, "t": t, "x1": x1, "x2": x2, "X": X, "T1": T1, "T2": T2}


def test_relative_pose_maps_camera1_to_camera2(two_view):
    X1 = G.transform_points(G.invert_T(two_view["T1"]), two_view["X"])
    X2 = G.transform_points(G.invert_T(two_view["T2"]), two_view["X"])
    np.testing.assert_allclose(X1 @ two_view["R"].T + two_view["t"], X2, atol=1e-12)


@pytest.mark.parametrize("normalize", [True, False])
def test_eight_point_exact_data(two_view, normalize):
    F = M.eight_point(two_view["x1"], two_view["x2"], normalize=normalize)
    assert np.linalg.matrix_rank(F, tol=1e-9) == 2
    assert np.abs(M.algebraic_residual(F, two_view["x1"], two_view["x2"])).max() < 1e-6
    F_gt = G.fundamental_from_essential(
        G.essential_from_Rt(two_view["R"], two_view["t"]), two_view["K"], two_view["K"]
    )
    assert min(np.linalg.norm(F - F_gt), np.linalg.norm(F + F_gt)) < 1e-6


def test_normalisation_improves_noisy_estimate(two_view):
    rng = np.random.default_rng(1)
    errs = {True: [], False: []}
    for _ in range(20):
        a = two_view["x1"] + rng.normal(0, 1.0, two_view["x1"].shape)
        b = two_view["x2"] + rng.normal(0, 1.0, two_view["x2"].shape)
        for norm in (True, False):
            F = M.eight_point(a, b, normalize=norm)
            # error on the noise-free correspondences
            errs[norm].append(np.sqrt(M.sampson_distance(F, two_view["x1"], two_view["x2"])).mean())
    assert np.median(errs[True]) < np.median(errs[False])


def test_normalize_points_statistics():
    x = np.random.default_rng(2).uniform(0, 640, (50, 2))
    xn, T = M.normalize_points(x)
    np.testing.assert_allclose(xn.mean(0), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(xn, axis=1).mean(), np.sqrt(2.0))
    np.testing.assert_allclose((M.to_h(x) @ T.T)[:, :2], xn)


def test_sampson_and_epipolar_distance_zero_on_exact_data(two_view):
    F = G.fundamental_from_essential(
        G.essential_from_Rt(two_view["R"], two_view["t"]), two_view["K"], two_view["K"]
    )
    assert M.sampson_distance(F, two_view["x1"], two_view["x2"]).max() < 1e-12
    lines = M.to_h(two_view["x1"]) @ F.T
    assert M.point_line_distance(lines, two_view["x2"]).max() < 1e-6


def test_decompose_essential_and_cheirality(two_view):
    E = G.essential_from_Rt(two_view["R"], two_view["t"])
    K = two_view["K"]
    x1n = cv2.undistortPoints(two_view["x1"].reshape(-1, 1, 2), K, None).reshape(-1, 2)
    x2n = cv2.undistortPoints(two_view["x2"].reshape(-1, 1, 2), K, None).reshape(-1, 2)
    cands = M.decompose_essential(E)
    counts = [M.cheirality_count(R, t, x1n, x2n)[0] for R, t in cands]
    best = int(np.argmax(counts))
    assert counts[best] == len(x1n)
    assert sorted(counts)[:3] == [0, 0, 0]
    R, t = cands[best]
    rot_err, t_err = M.pose_errors(R, t, two_view["R"], two_view["t"])
    assert rot_err < 1e-4 and t_err < 1e-4  # arccos limits precision near 0


def test_essential_from_fundamental_and_check(two_view):
    E = G.essential_from_Rt(two_view["R"], two_view["t"])
    F = G.fundamental_from_essential(E, two_view["K"], two_view["K"])
    E2 = M.essential_from_fundamental(F, two_view["K"], two_view["K"])
    chk = M.essential_check(E2)
    np.testing.assert_allclose(chk["singular_values"], [1.0, 1.0, 0.0], atol=1e-9)
    assert chk["cubic_residual"] < 1e-9
    En = E / np.linalg.norm(E) * np.sqrt(2)
    assert min(np.linalg.norm(E2 - En), np.linalg.norm(E2 + En)) < 1e-6


def test_ransac_essential_recovers_pose_with_outliers(two_view):
    rng = np.random.default_rng(3)
    x1 = two_view["x1"] + rng.normal(0, 0.3, two_view["x1"].shape)
    x2 = two_view["x2"] + rng.normal(0, 0.3, two_view["x2"].shape)
    x2[:40] = rng.uniform([0, 0], [640, 480], (40, 2))
    m = M.Matches(x1, x2, len(x1), len(x2))
    est = M.ransac_essential(m, two_view["K"], thresh_px=1.0)
    rot_err, t_err = M.pose_errors(est["R"], est["t"], two_view["R"], two_view["t"])
    assert rot_err < 1.0 and t_err < 3.0
    assert est["n_inliers"] >= 150


def test_pose_auc_bounds():
    assert M.pose_auc(np.zeros(10))["auc@5"] == pytest.approx(1.0)
    assert M.pose_auc(np.full(10, np.inf))["auc@10"] == pytest.approx(0.0)
    half = M.pose_auc(np.array([0.0, np.inf]))
    assert half["auc@20"] == pytest.approx(0.5)


def test_triangulation_angle():
    X = np.array([[0.0, 0.0, 1.0]])
    ang = M.triangulation_angle_deg(X, np.array([-1.0, 0, 0]), np.array([1.0, 0, 0]))
    assert ang[0] == pytest.approx(90.0)


def test_pfm_roundtrip(tmp_path):
    img = np.arange(12, dtype=np.float32).reshape(3, 4)
    img[1, 2] = np.inf
    path = M.write_pfm(tmp_path / "d.pfm", img)
    out = M.read_pfm(path)
    assert out.shape == (3, 4) and out.dtype == np.float32
    np.testing.assert_array_equal(out, img)


def test_read_middlebury_calib(tmp_path):
    p = tmp_path / "calib.txt"
    p.write_text(
        "cam0=[999.421 0 294.182; 0 999.421 252.932; 0 0 1]\n"
        "cam1=[999.421 0 326.96; 0 999.421 252.932; 0 0 1]\n"
        "doffs=32.778\nbaseline=193.001\nwidth=741\nheight=497\nndisp=70\nisint=0\n"
    )
    c = M.read_middlebury_calib(p)
    assert c["cam0"].shape == (3, 3) and c["cam1"][0, 2] == pytest.approx(326.96)
    assert c["ndisp"] == 70 and isinstance(c["ndisp"], int)
    assert c["doffs"] == pytest.approx(c["cam1"][0, 2] - c["cam0"][0, 2])


def test_disparity_to_depth():
    d = np.array([[10.0, 0.0, -5.0]])
    z = M.disparity_to_depth(d, f=1000.0, baseline=100.0, doffs=0.0)
    assert z[0, 0] == pytest.approx(10000.0)
    assert np.isnan(z[0, 1]) and np.isnan(z[0, 2])
    z2 = M.disparity_to_depth(np.array([[10.0]]), 1000.0, 100.0, doffs=10.0)
    assert z2[0, 0] == pytest.approx(5000.0)


def test_disparity_metrics():
    gt = np.full((2, 5), 10.0)
    pred = np.array([[10.0, 10.5, 13.0, np.nan, 10.0], [10.0, 10.0, 10.0, 10.0, 10.0]])
    mask = np.ones_like(gt, bool)
    m = M.disparity_metrics(pred, gt, mask, thresholds=(1.0, 2.0))
    assert m["density"] == pytest.approx(0.9)
    assert m["bad-2"] == pytest.approx(100 / 9)
    assert m["bad-2 (all)"] == pytest.approx(20.0)
    assert m["epe"] == pytest.approx(3.5 / 9)


def test_lr_consistency_and_fill_holes():
    h, w = 4, 20
    dl = np.full((h, w), 3.0)
    dr = np.full((h, w), 3.0)
    dl[1, 10] = 7.0  # inconsistent with the right view
    ok = M.lr_consistency(dl, dr, 1.0)
    assert not ok[1, 10]
    assert not ok[:, :3].any()  # x - d < 0: no partner in the right image
    assert ok[:, 3:10].all() and ok[0, 10:].all()
    row = np.array([[5.0, np.nan, np.nan, 2.0, np.nan]])
    np.testing.assert_allclose(M.fill_holes(row), [[5.0, 2.0, 2.0, 2.0, 2.0]])


def test_stereo_sgbm_recovers_constant_shift():
    rng = np.random.default_rng(0)
    tex = cv2.GaussianBlur((rng.uniform(0, 255, (120, 200)).astype(np.uint8)), (3, 3), 0)
    left = np.dstack([tex] * 3)
    right = np.roll(left, -8, axis=1)  # every pixel shifted by 8 px -> disparity 8
    d = M.stereo_sgbm(left, right, ndisp=32)
    centre = d[20:-20, 60:-20]
    assert np.nanmedian(centre) == pytest.approx(8.0, abs=0.1)
    x1, x2 = np.array([[10.0, 0], [50.0, 0]]), np.array([[20.0, 0], [5.0, 0]])
    lo, n = M.disparity_search_range(x1, x2)
    assert lo <= -10 and lo + n >= 45 and n % 16 == 0


def test_rectify_calibrated_identity_for_rectified_rig():
    K = np.array([[500.0, 0, 160], [0, 500.0, 120], [0, 0, 1]])
    img = np.zeros((240, 320, 3), np.uint8)
    out = M.rectify_calibrated(img, img, K, K, np.eye(3), np.array([-100.0, 0, 0]))
    np.testing.assert_allclose(out["R1"], np.eye(3), atol=1e-9)
    np.testing.assert_allclose(out["R2"], np.eye(3), atol=1e-9)


def test_rectified_disparity_to_left_depth():
    K = np.array([[500.0, 0, 160], [0, 500.0, 120], [0, 0, 1]])
    img = np.zeros((240, 320, 3), np.uint8)
    out = M.rectify_calibrated(img, img, K, K, np.eye(3), np.array([-100.0, 0, 0]), alpha=-1)
    f_new = out["P1"][0, 0]
    disp = np.full((240, 320), 25.0)
    depth = M.rectified_disparity_to_left_depth(disp, out["Q"], out["R1"], K, (240, 320))
    assert np.nanmedian(depth) == pytest.approx(f_new * 100.0 / 25.0, rel=1e-6)


def test_warp_rotation_identity():
    img = np.random.default_rng(0).integers(0, 255, (60, 80, 3), dtype=np.uint8)
    K = np.array([[100.0, 0, 40], [0, 100.0, 30], [0, 0, 1]])
    np.testing.assert_array_equal(M.warp_rotation(img, K, np.eye(3)), img)


def test_render_point_cloud_zbuffer():
    K = np.array([[100.0, 0, 10], [0, 100.0, 10], [0, 0, 1]])
    pts = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]])  # same pixel, different depth
    cols = np.array([[255, 0, 0], [0, 0, 255]], np.uint8)
    img = M.render_point_cloud(pts, cols, K, np.eye(4), 20, 20, splat=1)
    np.testing.assert_array_equal(img[10, 10], [255, 0, 0])  # the nearer point wins
    assert (img[0, 0] == 255).all()
    T = M.orbit_pose(90.0, np.array([0.0, 0.0, 1.0]))
    pivot = np.array([[0.0, 0.0, 1.0]])
    np.testing.assert_allclose(G.transform_points(T, pivot), pivot, atol=1e-12)


def test_gravity_align():
    R_tilt = G.so3_exp([0.3, -0.2, 0.1])
    cams = {}
    for k in range(5):
        T = np.eye(4)
        T[:3, :3] = R_tilt @ G.look_at([k, 0.0, 0.0], [k, 1.0, 0.0])[:3, :3]
        cams[str(k)] = {"T_wc": T}
    R = M.gravity_align(cams)
    up = -cams["0"]["T_wc"][:3, 1]
    np.testing.assert_allclose(R @ up, [0, 0, 1], atol=1e-9)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)


def test_compare_to_reference_recovers_similarity():
    rng = np.random.default_rng(4)
    ref, est = {}, {}
    s, R = 2.5, G.so3_exp([0.1, 0.4, -0.3])
    t = np.array([1.0, -2.0, 0.5])
    for k in range(8):
        T_ref = G.make_T(G.so3_exp(rng.normal(0, 0.5, 3)), rng.normal(0, 3, 3))
        ref[f"im{k}"] = {"T_wc": T_ref}
        # est = ref expressed in another similarity frame: C_ref = s R C_est + t
        C_est = R.T @ (T_ref[:3, 3] - t) / s
        est[f"im{k}"] = {"T_wc": G.make_T(R.T @ T_ref[:3, :3], C_est)}
    out = M.compare_to_reference(est, ref)
    assert out["scale"] == pytest.approx(s)
    assert out["rot_err_deg"].max() < 1e-5 and out["center_err"].max() < 1e-9


def test_colmap_K_to_opencv():
    K = np.array([[800.0, 0, 512.0], [0, 800.0, 384.0], [0, 0, 1]])
    Kc = M.colmap_K_to_opencv(K)
    assert Kc[0, 2] == 511.5 and Kc[1, 2] == 383.5 and K[0, 2] == 512.0


def test_sequential_pairs():
    names = ["a", "b", "c", "d", "e"]
    assert len(M.sequential_pairs(names)) == 10  # exhaustive: N (N - 1) / 2
    assert M.sequential_pairs(names, 0) == M.sequential_pairs(names)
    assert M.sequential_pairs(names, 1) == [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e")]
    pairs = M.sequential_pairs(names, 2)
    assert len(pairs) == 7 and ("a", "c") in pairs and ("a", "d") not in pairs
