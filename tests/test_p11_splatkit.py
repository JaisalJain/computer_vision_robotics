"""CPU tests for projects/11_gaussian_splatting/scripts/splatkit.py (no dataset, no GPU)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_SCRIPTS = Path(__file__).resolve().parents[1] / "projects/11_gaussian_splatting/scripts"
sys.path.insert(0, str(PROJECT_SCRIPTS))
import splatkit as S  # noqa: E402

from cvlab import geometry as G  # noqa: E402


def test_sh_dc_roundtrip():
    rgb = np.random.default_rng(0).random((100, 3))
    np.testing.assert_allclose(S.sh_to_rgb(S.rgb_to_sh(rgb)), rgb, atol=1e-12)
    assert S.rgb_to_sh(np.array(0.5)) == 0.0  # mid-grey has a zero DC coefficient


def test_quat_wxyz_matches_cvlab_xyzw():
    q = np.random.default_rng(1).normal(size=(20, 4))
    R = S.quat_to_rotmat_wxyz(q)
    for qi, Ri in zip(q, R, strict=True):
        qn = qi / np.linalg.norm(qi)
        np.testing.assert_allclose(Ri, G.quat_to_R(np.r_[qn[1:], qn[0]]), atol=1e-12)
        np.testing.assert_allclose(Ri @ Ri.T, np.eye(3), atol=1e-12)


def test_covariance_eigenvalues_are_squared_scales():
    scales = np.array([0.3, 0.1, 0.02])
    cov = S.covariance_from_scale_rot(scales, np.array([0.9, 0.1, -0.3, 0.2]))
    np.testing.assert_allclose(cov, cov.T, atol=1e-15)
    np.testing.assert_allclose(np.sort(np.linalg.eigvalsh(cov)), np.sort(scales**2), rtol=1e-10)


def test_ewa_projection_matches_monte_carlo():
    """The linearised (EWA) 2D covariance matches sampling + exact perspective projection."""
    rng = np.random.default_rng(2)
    K = np.array([[500.0, 0, 320], [0, 520.0, 240], [0, 0, 1]])
    w2c = G.invert_T(G.look_at([0.3, -2.0, 0.4], [0.0, 0.0, 0.0]))
    mean = np.array([0.1, 0.05, -0.05])
    cov = S.covariance_from_scale_rot(np.array([0.02, 0.008, 0.004]), np.array([1, 0.4, 0.2, 0]))
    m2d, c2d = S.project_covariance(mean, cov, w2c, K)
    X = rng.multivariate_normal(mean, cov, size=200_000)
    uv, _ = G.project_points(X, K, w2c)
    np.testing.assert_allclose(uv.mean(0), m2d, atol=0.05)
    np.testing.assert_allclose(np.cov(uv.T), c2d, rtol=0.03, atol=0.02)


def _ssim_reference(x: np.ndarray, y: np.ndarray) -> float:
    """SciPy Gaussian-filter SSIM restricted to pixels whose 11x11 window is inside the image."""
    from scipy.ndimage import gaussian_filter

    c1, c2 = 0.01**2, 0.03**2
    vals = []
    for ch in range(x.shape[0]):
        a, b = x[ch].astype(np.float64), y[ch].astype(np.float64)

        def blur(t):
            return gaussian_filter(t, 1.5, truncate=3.5)[5:-5, 5:-5]

        mu_a, mu_b = blur(a), blur(b)
        s_aa, s_bb, s_ab = blur(a * a) - mu_a**2, blur(b * b) - mu_b**2, blur(a * b) - mu_a * mu_b
        num = (2 * mu_a * mu_b + c1) * (2 * s_ab + c2)
        vals.append(np.mean(num / ((mu_a**2 + mu_b**2 + c1) * (s_aa + s_bb + c2))))
    return float(np.mean(vals))


def test_ssim_loss_matches_reference_and_is_differentiable():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(3)
    x = rng.random((3, 40, 50))
    y = np.clip(x + 0.1 * rng.normal(size=x.shape), 0, 1)
    tx = torch.tensor(x[None], dtype=torch.float64, requires_grad=True)
    ty = torch.tensor(y[None], dtype=torch.float64)
    val = S.ssim(tx, ty)
    assert abs(val.item() - _ssim_reference(x, y)) < 1e-10
    assert abs(float(S.ssim(ty, ty)) - 1.0) < 1e-12
    val.backward()
    assert tx.grad is not None and torch.isfinite(tx.grad).all()


def test_knn_mean_dist_on_grid():
    g = np.stack(np.meshgrid(np.arange(5.0), np.arange(5.0), np.arange(5.0)), -1).reshape(-1, 3)
    d = S.knn_mean_dist(g, k=3)
    interior = np.all((g > 0) & (g < 4), axis=1)
    np.testing.assert_allclose(d[interior], 1.0)


def test_closest_point_to_rays_recovers_target():
    rng = np.random.default_rng(4)
    target = np.array([0.5, -1.0, 2.0])
    origins = rng.normal(size=(30, 3)) * 3
    dirs = target - origins
    np.testing.assert_allclose(S.closest_point_to_rays(origins, dirs), target, atol=1e-10)


def _ring_scene(n: int = 24, radius: float = 2.0, height: float = 0.5) -> S.ColmapScene:
    """Cameras on a horizontal circle (world z up) looking at the origin."""
    K = np.array([[300.0, 0, 160], [0, 300.0, 120], [0, 0, 1]])
    w2c = []
    for a in np.linspace(0, 2 * np.pi, n, endpoint=False):
        eye = [radius * np.cos(a), radius * np.sin(a), height]
        w2c.append(G.invert_T(G.look_at(eye, [0.0, 0.0, 0.0])))
    return S.ColmapScene(
        scene_dir=Path("."), names=[f"{i:03d}.png" for i in range(n)],
        image_paths=[Path(f"{i:03d}.png") for i in range(n)], w2c=np.stack(w2c),
        K=np.repeat(K[None], n, 0), widths=np.full(n, 320), heights=np.full(n, 240),
        points=np.zeros((10, 3)), points_rgb=np.zeros((10, 3), np.uint8),
        points_track_len=np.full(10, 3), points_error=np.zeros(10),
        train_idx=np.arange(1, n), test_idx=np.array([0]), camera_model="PINHOLE",
        colmap_size=(320, 240),
    )  # fmt: skip


def test_ring_scene_properties_and_orbit_path():
    scene = _ring_scene()
    np.testing.assert_allclose(S.fit_up_vector(scene.c2w), [0, 0, 1], atol=0.25)
    poses = S.orbit_path(scene, n_frames=12)
    assert poses.shape == (12, 4, 4)
    c2w = np.linalg.inv(poses)
    np.testing.assert_allclose(np.linalg.det(c2w[:, :3, :3]), 1.0, atol=1e-9)
    # every novel camera looks at the common target (the origin, up to the fitted geometry)
    target = S.closest_point_to_rays(scene.centers, scene.c2w[:, :3, 2])
    for T in c2w:
        to_t = target - T[:3, 3]
        assert G.angle_between_deg(T[:3, 2], to_t) < 1e-3  # arccos round-off ~1e-5 deg
    radii = np.linalg.norm(c2w[:, :2, 3] - target[:2], axis=1)
    np.testing.assert_allclose(radii, 2.0, rtol=0.05)


def test_init_points_random_and_colmap():
    scene = _ring_scene()
    pts, rgb = S.init_points(scene, "random", n_random=5000, random_extent=1.0)
    c = scene.centers.mean(0)
    assert pts.shape == (5000, 3) and rgb.shape == (5000, 3)
    assert np.all(np.abs(pts - c) <= scene.scene_scale + 1e-5)
    assert rgb.min() >= 0 and rgb.max() <= 1
    p2, c2 = S.init_points(scene, "colmap")
    assert p2.shape == (10, 3) and c2.max() <= 1.0
    with pytest.raises(ValueError):
        S.init_points(scene, "da3")


def test_create_splats_shapes_and_init_values():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(5)
    pts = rng.random((200, 3)).astype(np.float32)
    rgb = rng.random((200, 3)).astype(np.float32)
    cfg = S.TrainConfig(sh_degree=2)
    splats, opts = S.create_splats(pts, rgb, scene_scale=2.0, cfg=cfg, device="cpu")
    assert splats["shN"].shape == (200, 8, 3) and splats["sh0"].shape == (200, 1, 3)
    np.testing.assert_allclose(torch.sigmoid(splats["opacities"]).detach().numpy(), 0.1,
                               rtol=1e-5)  # fmt: skip
    np.testing.assert_allclose(S.sh_to_rgb(splats["sh0"][:, 0].detach().numpy()), rgb, atol=1e-6)
    assert opts["means"].param_groups[0]["lr"] == pytest.approx(cfg.lr_means * 2.0)
    scales = torch.exp(splats["scales"]).detach().numpy()
    np.testing.assert_allclose(scales[:, 0], S.knn_mean_dist(pts.astype(np.float64)), rtol=1e-5)


def test_load_colmap_scene_rescales_intrinsics(tmp_path):
    pycolmap = pytest.importorskip("pycolmap")
    from PIL import Image

    sparse = tmp_path / "sparse" / "0"
    sparse.mkdir(parents=True)
    rec = pycolmap.Reconstruction()
    cam = pycolmap.Camera.create_from_model_name(1, "PINHOLE", 400.0, 800, 600)
    cam.params = [400.0, 410.0, 400.0, 300.0]
    rec.add_camera_with_trivial_rig(cam)
    img_dir = tmp_path / "images_2"
    img_dir.mkdir()
    for i in range(10):
        pose = pycolmap.Rigid3d(pycolmap.Rotation3d(np.eye(3)), np.array([0.1 * i, 0.0, 0.0]))
        im = pycolmap.Image(name=f"img_{i:02d}.jpg", camera_id=1, image_id=i + 1)
        rec.add_image_with_trivial_frame(im, pose)
        Image.new("RGB", (400, 300)).save(img_dir / f"img_{i:02d}.png")
    rec.add_point3D(np.array([0.0, 0.0, 5.0]), pycolmap.Track(), np.array([255, 0, 0], np.uint8))
    rec.write(str(sparse))
    scene = S.load_colmap_scene(tmp_path, images="images_2", test_every=4)
    assert scene.n_images == 10 and scene.colmap_size == (800, 600)
    np.testing.assert_array_equal(scene.test_idx, [0, 4, 8])
    np.testing.assert_allclose(scene.K[0], [[200, 0, 200], [0, 205, 150], [0, 0, 1]])
    np.testing.assert_allclose(scene.centers[3], [-0.3, 0.0, 0.0], atol=1e-12)
    assert scene.points_rgb[0].tolist() == [255, 0, 0]
