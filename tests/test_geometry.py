import numpy as np
import pytest

from cvlab import geometry as G


def random_rotation(rng):
    return G.so3_exp(rng.normal(size=3))


def test_project_unproject_roundtrip(rng):
    cam = G.PinholeCamera.from_fov(640, 480, 60.0)
    pts = rng.uniform([-1, -1, 1], [1, 1, 5], size=(100, 3))
    uv, z = cam.project(pts)
    back = cam.unproject(uv, z)
    np.testing.assert_allclose(back, pts, atol=1e-9)


def test_principal_point_projects_to_center():
    cam = G.PinholeCamera(500.0, 500.0, 319.5, 239.5, 640, 480)
    uv, _ = cam.project(np.array([[0.0, 0.0, 2.0]]))
    np.testing.assert_allclose(uv[0], [319.5, 239.5])
    assert cam.in_image(uv).all()


def test_scaled_intrinsics_match_resized_projection(rng):
    cam = G.PinholeCamera(600.0, 610.0, 320.0, 240.0, 640, 480)
    half = cam.scaled(0.5)
    assert (half.width, half.height) == (320, 240)
    pts = rng.uniform([-1, -1, 2], [1, 1, 4], size=(10, 3))
    uv_full, _ = cam.project(pts)
    uv_half, _ = half.project(pts)
    # pixel-centre convention: u_half = (u_full + 0.5) * 0.5 - 0.5
    np.testing.assert_allclose(uv_half, (uv_full + 0.5) * 0.5 - 0.5, atol=1e-9)


def test_fov_roundtrip():
    cam = G.PinholeCamera.from_fov(800, 600, 90.0)
    assert cam.fov_deg[0] == pytest.approx(90.0, abs=0.2)


def test_so3_exp_log_roundtrip(rng):
    for _ in range(50):
        w = rng.normal(size=3)
        w *= rng.uniform(0, np.pi - 1e-3) / np.linalg.norm(w)
        R = G.so3_exp(w)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0)
        np.testing.assert_allclose(G.so3_log(R), w, atol=1e-6)


def test_so3_log_near_pi():
    w = np.array([0.0, 0.0, np.pi])
    R = G.so3_exp(w)
    w2 = G.so3_log(R)
    np.testing.assert_allclose(G.so3_exp(w2), R, atol=1e-9)


def test_quaternion_roundtrip(rng):
    for _ in range(50):
        R = random_rotation(rng)
        q = G.R_to_quat(R)
        assert q[3] >= 0
        np.testing.assert_allclose(G.quat_to_R(q), R, atol=1e-9)


def test_quaternion_known_value():
    # 90 degrees about z: (x, y, z, w) = (0, 0, sin45, cos45)
    q = np.array([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)])
    R = G.quat_to_R(q)
    np.testing.assert_allclose(R @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-12)


def test_invert_T_and_transform(rng):
    T = G.make_T(random_rotation(rng), rng.normal(size=3))
    np.testing.assert_allclose(G.invert_T(T) @ T, np.eye(4), atol=1e-12)
    pts = rng.normal(size=(20, 3))
    back = G.transform_points(G.invert_T(T), G.transform_points(T, pts))
    np.testing.assert_allclose(back, pts, atol=1e-12)
    stack = np.stack([T, T])
    np.testing.assert_allclose(G.invert_T(stack)[1], G.invert_T(T))


def test_project_points_with_extrinsics(rng):
    cam = G.PinholeCamera.from_fov(640, 480, 70.0)
    T_wc = G.look_at(eye=[0.0, -3.0, 1.0], target=[0.0, 0.0, 0.0], up=[0.0, 0.0, 1.0])
    uv, z = G.project_points(np.zeros((1, 3)), cam.K, G.invert_T(T_wc))
    np.testing.assert_allclose(uv[0], [cam.cx, cam.cy], atol=1e-9)
    assert z[0] == pytest.approx(np.sqrt(10.0))


def test_look_at_is_rotation():
    T = G.look_at([1.0, 2.0, 3.0], [0.0, 0.0, 0.0])
    R = T[:3, :3]
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0)


def test_epipolar_constraint_and_triangulation(rng):
    K = G.PinholeCamera.from_fov(640, 480, 60.0).K
    R = G.so3_exp([0.02, -0.1, 0.03])
    t = np.array([-0.5, 0.05, 0.02])
    X = rng.uniform([-1, -1, 4], [1, 1, 8], size=(30, 3))
    x1, _ = G.project(X, K)
    x2, _ = G.project(X @ R.T + t, K)
    F = G.fundamental_from_essential(G.essential_from_Rt(R, t), K, K)
    h1 = np.hstack([x1, np.ones((30, 1))])
    h2 = np.hstack([x2, np.ones((30, 1))])
    residual = np.abs(np.einsum("ni,ij,nj->n", h2, F, h1))
    assert residual.max() < 1e-8
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t[:, None]])
    np.testing.assert_allclose(G.triangulate_dlt(P1, P2, x1, x2), X, atol=1e-6)


def test_umeyama_recovers_similarity(rng):
    src = rng.normal(size=(100, 3))
    R = random_rotation(rng)
    s, t = 2.5, np.array([1.0, -2.0, 0.5])
    dst = s * src @ R.T + t
    s2, R2, t2 = G.umeyama_alignment(src, dst)
    assert s2 == pytest.approx(s)
    np.testing.assert_allclose(R2, R, atol=1e-9)
    np.testing.assert_allclose(t2, t, atol=1e-9)
    s3, _, _ = G.umeyama_alignment(src, dst, with_scale=False)
    assert s3 == 1.0


def test_rotation_errors():
    assert G.rotation_error_deg(np.eye(3), G.so3_exp([0, 0, np.deg2rad(10)])) == pytest.approx(10)
    assert G.angle_between_deg([1, 0, 0], [0, 1, 0]) == pytest.approx(90)


def test_ros_frame_rotation():
    # camera optical axis (z forward) is ROS x forward; image right (x) is ROS -y
    R = G.R_ROS_FROM_OPENCV
    np.testing.assert_allclose(R @ [0, 0, 1], [1, 0, 0])
    np.testing.assert_allclose(R @ [1, 0, 0], [0, -1, 0])
    np.testing.assert_allclose(R @ [0, 1, 0], [0, 0, -1])
    assert np.linalg.det(R) == pytest.approx(1.0)
