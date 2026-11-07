import numpy as np
import pytest

from cvlab import io as cio
from cvlab import viz


def test_repo_root_contains_package():
    root = cio.repo_root()
    assert (root / "pyproject.toml").is_file()
    assert (root / "src" / "cvlab").is_dir()
    assert cio.data_dir("x").parts[-2:] == ("data", "x")


def test_image_roundtrip(tmp_path, rng):
    img = rng.integers(0, 255, size=(40, 60, 3), dtype=np.uint8)
    path = cio.write_image(tmp_path / "a" / "img.png", img)
    back = cio.read_image(path)
    np.testing.assert_array_equal(back, img)
    small = cio.read_image(path, max_side=30)
    assert small.shape == (20, 30, 3)
    with pytest.raises(FileNotFoundError):
        cio.read_image(tmp_path / "missing.png")


def test_json_roundtrip(tmp_path):
    obj = {"a": np.float32(1.5), "b": np.arange(3), "c": np.int64(7)}
    path = cio.save_json(obj, tmp_path / "x.json")
    assert cio.load_json(path) == {"a": 1.5, "b": [0, 1, 2], "c": 7}


def test_video_frames(tmp_path):
    import cv2

    path = str(tmp_path / "v.avi")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 24))
    for i in range(10):
        writer.write(np.full((24, 32, 3), i * 20, np.uint8))
    writer.release()
    frames = cio.read_video_frames(path, stride=2, max_frames=3)
    assert len(frames) == 3 and frames[0].shape == (24, 32, 3)


def test_drawing_helpers(rng):
    img = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    out = viz.draw_boxes(img, [[5, 5, 30, 30]], labels=["cup"], scores=[0.9])
    assert out.shape == img.shape and not np.array_equal(out, img)
    mask = np.zeros((64, 64), bool)
    mask[10:20, 10:20] = True
    ov = viz.overlay_masks(img, mask)
    assert ov.shape == img.shape
    depth = rng.uniform(0.5, 3.0, size=(64, 64))
    depth[0, 0] = 0
    rgb = viz.colorize_depth(depth)
    assert rgb.dtype == np.uint8 and rgb.shape == (64, 64, 3)
    assert tuple(rgb[0, 0]) == (0, 0, 0)


def test_plots_and_savefig(tmp_path, monkeypatch, rng):
    import matplotlib.pyplot as plt

    monkeypatch.setenv("CVLAB_ROOT", str(tmp_path))
    viz.set_style()
    fig = viz.show_images([np.zeros((10, 10, 3), np.uint8)] * 3, titles=["a", "b", "c"])
    path = viz.savefig(fig, "00_test", "grid")
    assert path.exists() and path.stat().st_size < 1_000_000
    plt.close(fig)
    ax = viz.plot_point_cloud(rng.normal(size=(200, 3)))
    poses = np.stack([np.eye(4)] * 2)
    viz.plot_camera_frustums(ax, poses, np.diag([100.0, 100.0, 1.0]), 64, 48)
    plt.close("all")
    viz.plot_trajectories({"gt": rng.normal(size=(20, 3)), "est": rng.normal(size=(20, 3))})
    plt.close("all")
    gif = viz.save_gif([np.zeros((50, 50, 3), np.uint8)] * 4, tmp_path / "a.gif")
    assert gif.stat().st_size > 0


def test_add_third_party_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("CVLAB_ROOT", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="setup_third_party"):
        cio.add_third_party("does-not-exist")
    (tmp_path / "third_party" / "pkg" / "src").mkdir(parents=True)
    path = cio.add_third_party("pkg", "src")
    import sys

    assert str(path) in sys.path
    sys.path.remove(str(path))
