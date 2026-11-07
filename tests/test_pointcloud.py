import numpy as np
import pytest

from cvlab import pointcloud as PC
from cvlab.geometry import PinholeCamera


def test_depth_to_points_plane():
    cam = PinholeCamera(100.0, 100.0, 31.5, 23.5, 64, 48)
    depth = np.full((48, 64), 2.0)
    depth[0, 0] = 0.0  # invalid
    pts, cols = PC.depth_to_points(depth, cam, rgb=np.full((48, 64, 3), 255, np.uint8))
    assert pts.shape == (48 * 64 - 1, 3)
    np.testing.assert_allclose(pts[:, 2], 2.0)
    np.testing.assert_allclose(cols, 1.0)
    # pixel (u=63, v=47) -> x = (63 - 31.5) / 100 * 2
    assert pts[:, 0].max() == pytest.approx((63 - 31.5) / 100 * 2)


def test_depth_points_depth_roundtrip(rng):
    cam = PinholeCamera(80.0, 80.0, 31.5, 23.5, 64, 48)
    depth = rng.uniform(1.0, 3.0, size=(48, 64))
    pts, _ = PC.depth_to_points(depth, cam.K)
    rendered = PC.points_to_depth(pts, cam, 64, 48)
    np.testing.assert_allclose(rendered, depth, atol=1e-9)


def test_mask_and_stride():
    depth = np.ones((10, 10))
    mask = np.zeros((10, 10), bool)
    mask[:5] = True
    pts, _ = PC.depth_to_points(depth, np.eye(3), mask=mask, stride=2)
    assert len(pts) == 3 * 5  # rows 0,2,4 x cols 0,2,4,6,8


def test_voxel_downsample():
    pts = np.array([[0.01, 0.01, 0.01], [0.02, 0.02, 0.02], [1.0, 1.0, 1.0]])
    cols = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    out, oc = PC.voxel_downsample(pts, 0.1, cols)
    assert len(out) == 2
    i = int(np.argmin(out[:, 0]))
    np.testing.assert_allclose(out[i], [0.015] * 3)
    np.testing.assert_allclose(oc[i], [0.5, 0.5, 0.0])


def test_outlier_removal(rng):
    pts = rng.normal(scale=0.01, size=(500, 3))
    pts = np.vstack([pts, [[5.0, 5.0, 5.0]]])
    keep = PC.remove_statistical_outliers(pts, k=8)
    assert not keep[-1]
    assert keep[:-1].mean() > 0.9


def test_normals_on_plane(rng):
    pts = np.column_stack([rng.uniform(-1, 1, 400), rng.uniform(-1, 1, 400), np.full(400, 2.0)])
    n = PC.estimate_normals(pts, k=10)
    # plane z=2 seen from the origin: normals point towards -z
    np.testing.assert_allclose(n, np.tile([0, 0, -1.0], (400, 1)), atol=1e-6)


def test_pca_obb_box(rng):
    local = rng.uniform([-2, -1, -0.5], [2, 1, 0.5], size=(20000, 3))
    from cvlab.geometry import so3_exp

    R = so3_exp([0.3, 0.2, -0.4])
    pts = local @ R.T + [1.0, 2.0, 3.0]
    center, axes, extents = PC.pca_obb(pts)
    np.testing.assert_allclose(center, [1, 2, 3], atol=0.05)
    np.testing.assert_allclose(extents, [4, 2, 1], atol=0.15)  # PCA axes ~1 deg off
    assert np.linalg.det(axes) == pytest.approx(1.0)
    corners = PC.obb_corners(center, axes, extents)
    assert corners.shape == (8, 3)


@pytest.mark.parametrize("binary", [True, False])
def test_ply_roundtrip(tmp_path, rng, binary):
    pts = rng.normal(size=(50, 3)).astype(np.float32)
    cols = rng.integers(0, 255, size=(50, 3)).astype(np.uint8)
    normals = rng.normal(size=(50, 3)).astype(np.float32)
    path = PC.write_ply(tmp_path / "cloud.ply", pts, colors=cols, normals=normals, binary=binary)
    data = PC.read_ply(path)
    np.testing.assert_allclose(data["points"], pts, atol=1e-6)
    np.testing.assert_array_equal(data["colors"], cols)
    np.testing.assert_allclose(data["normals"], normals, atol=1e-6)
