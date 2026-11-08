"""Project 07: COCO keypoint helpers (OKS, matching, label parsing, joint angles)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "projects/07_pose_estimation/scripts"))

import keypoints as kp  # noqa: E402


def _pose(rng: np.random.Generator) -> np.ndarray:
    g = np.zeros((17, 3))
    g[:, :2] = rng.uniform(100, 300, (17, 2))
    g[:, 2] = 2
    return g


def test_oks_perfect_and_decay(rng):
    g = _pose(rng)
    assert kp.oks(g, g[:, :2], area=5000.0) == pytest.approx(1.0)
    shifted = g[:, :2] + 5.0
    o1 = kp.oks(g, shifted, area=5000.0)
    o2 = kp.oks(g, shifted, area=20000.0)
    assert 0 < o1 < o2 < 1  # same pixel error is more forgivable on a bigger person


def test_oks_ignores_unlabelled_keypoints(rng):
    g = _pose(rng)
    g[10:, 2] = 0
    p = g[:, :2].copy()
    p[10:] += 500.0  # wild errors only on unlabelled keypoints
    assert kp.oks(g, p, area=5000.0) == pytest.approx(1.0)
    g[:, 2] = 0
    assert np.isnan(kp.oks(g, p, area=5000.0))


def test_oks_matches_ultralytics(rng):
    torch = pytest.importorskip("torch")
    metrics = pytest.importorskip("ultralytics.utils.metrics")
    gts = np.stack([_pose(rng) for _ in range(3)])
    preds = gts[:, :, :2] + rng.normal(0, 4, (3, 17, 2))
    preds = np.concatenate([preds, np.ones((3, 17, 1))], axis=-1)
    areas = np.array([4000.0, 9000.0, 16000.0])
    ref = metrics.kpt_iou(torch.tensor(gts), torch.tensor(preds), torch.tensor(areas),
                          sigma=metrics.OKS_SIGMA).numpy()  # fmt: skip
    np.testing.assert_allclose(kp.oks_matrix(gts, areas, preds), ref, rtol=1e-6, atol=1e-9)


def test_match_by_oks_is_one_to_one(rng):
    gts = np.stack([_pose(rng) for _ in range(2)])
    preds = gts[::-1, :, :2]  # same persons, reversed order
    best = kp.match_by_oks(gts, np.array([5000.0, 5000.0]), preds)
    np.testing.assert_allclose(best, [1.0, 1.0])
    np.testing.assert_allclose(kp.match_by_oks(gts, np.ones(2), preds[:0]), [0.0, 0.0])


def test_load_yolo_pose_labels(tmp_path):
    row = [0, 0.5, 0.5, 0.2, 0.4] + [0.5, 0.25, 2] * 17
    f = tmp_path / "x.txt"
    f.write_text(" ".join(map(str, row)) + "\n")
    boxes, kpts, areas = kp.load_yolo_pose_labels(f, 640, 480)
    np.testing.assert_allclose(boxes, [[256, 144, 384, 336]])
    np.testing.assert_allclose(kpts[0, 0], [320, 120, 2])
    assert areas[0] == pytest.approx(0.53 * 128 * 192)


def test_joint_angle_and_flip_idx():
    assert kp.joint_angle_deg([0, 0], [1, 0], [1, 1]) == pytest.approx(90.0)
    assert kp.joint_angle_deg([0, 0], [1, 0], [2, 0]) == pytest.approx(180.0)
    flipped = [kp.KPT_NAMES[i] for i in kp.FLIP_IDX]
    for a, b in zip(kp.KPT_NAMES, flipped):
        assert a.replace("left", "X").replace("right", "left").replace("X", "right") == b


def test_draw_pose_draws_something(rng):
    img = np.zeros((200, 200, 3), np.uint8)
    pts = rng.uniform(20, 180, (1, 17, 2))
    out = kp.draw_pose(img, pts)
    assert out.sum() > 0 and img.sum() == 0
