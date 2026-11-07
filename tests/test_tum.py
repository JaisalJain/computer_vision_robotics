import numpy as np
import pytest

from cvlab import tum
from cvlab.geometry import R_to_quat, so3_exp


@pytest.fixture
def fake_sequence(tmp_path):
    import cv2

    root = tmp_path / "rgbd_dataset_freiburg1_xyz"
    (root / "rgb").mkdir(parents=True)
    (root / "depth").mkdir()
    rgb_lines, depth_lines, gt_lines = ["# rgb"], ["# depth"], ["# gt"]
    for i in range(5):
        t = 100.0 + 0.1 * i
        cv2.imwrite(str(root / "rgb" / f"{t:.6f}.png"), np.full((48, 64, 3), i * 10, np.uint8))
        cv2.imwrite(
            str(root / "depth" / f"{t + 0.005:.6f}.png"),
            np.full((48, 64), 5000 * (i + 1), np.uint16),
        )
        rgb_lines.append(f"{t:.6f} rgb/{t:.6f}.png")
        depth_lines.append(f"{t + 0.005:.6f} depth/{t + 0.005:.6f}.png")
        q = R_to_quat(so3_exp([0.0, 0.0, 0.1 * i]))
        gt_lines.append(f"{t + 0.001:.4f} {i} 0 0 {q[0]} {q[1]} {q[2]} {q[3]}")
    (root / "rgb.txt").write_text("\n".join(rgb_lines))
    (root / "depth.txt").write_text("\n".join(depth_lines))
    (root / "groundtruth.txt").write_text("\n".join(gt_lines[:-1]))  # last frame has no GT
    return root


def test_associate_one_to_one():
    m = tum.associate([1.0, 2.0, 3.0], [1.01, 1.015, 2.5, 3.0], max_dt=0.02)
    assert m == [(1.0, 1.01), (3.0, 3.0)]


def test_load_sequence(fake_sequence):
    seq = tum.load_sequence(fake_sequence)
    assert len(seq) == 4  # frame without GT is dropped
    assert seq.camera.fx == pytest.approx(517.3)
    assert seq.depth(2).max() == pytest.approx(3.0)
    assert seq.rgb(1).shape == (48, 64, 3)
    np.testing.assert_allclose(seq.gt_poses[3][:3, 3], [3, 0, 0])
    np.testing.assert_allclose(seq.gt_poses[2][:3, :3], so3_exp([0, 0, 0.2]), atol=1e-6)


def test_trajectory_roundtrip(tmp_path):
    ts = np.array([1.0, 2.0])
    poses = np.stack([np.eye(4), np.eye(4)])
    poses[1][:3, :3] = so3_exp([0.1, 0.2, 0.3])
    poses[1][:3, 3] = [1, 2, 3]
    path = tum.write_trajectory(tmp_path / "traj.txt", ts, poses)
    ts2, poses2 = tum.read_trajectory(path)
    np.testing.assert_allclose(ts2, ts)
    np.testing.assert_allclose(poses2, poses, atol=1e-5)


def test_camera_for_names():
    cam, dist = tum.camera_for("rgbd_dataset_freiburg3_long_office_household")
    assert cam.fx == pytest.approx(535.4) and not dist.any()
