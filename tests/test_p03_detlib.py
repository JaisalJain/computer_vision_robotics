"""CPU tests for projects/03_object_detection/scripts/detlib.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "projects" / "03_object_detection" / "scripts"
sys.path.insert(0, str(SCRIPTS))
import detlib  # noqa: E402


def _sample(boxes, classes, w=100, h=100):
    return detlib.Sample(
        Path("x.jpg"), w, h, np.asarray(boxes, float).reshape(-1, 4), np.asarray(classes, int)
    )


def test_yolo_to_xyxy():
    out = detlib.yolo_to_xyxy(np.array([[0.5, 0.5, 0.2, 0.4]]), 200, 100)
    np.testing.assert_allclose(out, [[80, 30, 120, 70]])


def test_read_yolo_labels(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("1 0.5 0.5 0.5 0.5\n0 0.25 0.25 0.1 0.1\n")
    boxes, cls = detlib.read_yolo_labels(p, 100, 100)
    np.testing.assert_array_equal(cls, [1, 0])
    np.testing.assert_allclose(boxes[0], [25, 25, 75, 75])
    empty_b, empty_c = detlib.read_yolo_labels(tmp_path / "missing.txt", 100, 100)
    assert empty_b.shape == (0, 4) and empty_c.size == 0


def test_load_split(tmp_path):
    from PIL import Image

    (tmp_path / "images/val").mkdir(parents=True)
    (tmp_path / "labels/val").mkdir(parents=True)
    Image.new("RGB", (64, 32)).save(tmp_path / "images/val/a.jpg")
    (tmp_path / "labels/val/a.txt").write_text("0 0.5 0.5 1 1\n")
    (s,) = detlib.load_split(tmp_path, "val")
    assert (s.width, s.height) == (64, 32)
    np.testing.assert_allclose(s.boxes, [[0, 0, 64, 32]])


def test_evaluate_perfect_and_empty():
    samples = [_sample([[10, 10, 50, 50]], [0]), _sample([[20, 20, 60, 60], [0, 0, 5, 5]], [0, 1])]
    perfect = [detlib.Detections(s.boxes, np.ones(len(s.boxes)), s.classes) for s in samples]
    res = detlib.evaluate(samples, perfect, ["a", "b"])
    assert res.map50 == pytest.approx(1.0)
    assert res.map50_95 == pytest.approx(1.0)
    none = [detlib.Detections.empty() for _ in samples]
    assert detlib.evaluate(samples, none, ["a", "b"]).map50 == 0.0


def test_evaluate_ranking_matters():
    # One GT; a high-scoring false positive ranked above the true positive halves precision.
    samples = [_sample([[10, 10, 50, 50]], [0])]
    good_first = detlib.Detections(
        np.array([[10, 10, 50, 50], [70, 70, 90, 90]], float),
        np.array([0.9, 0.1]),
        np.array([0, 0]),
    )
    bad_first = detlib.Detections(good_first.boxes, np.array([0.1, 0.9]), np.array([0, 0]))
    assert detlib.evaluate(samples, [good_first], ["a"]).map50 == pytest.approx(1.0)
    assert detlib.evaluate(samples, [bad_first], ["a"]).map50 == pytest.approx(0.5)


def test_evaluate_iou_threshold_sweep():
    # Prediction shifted so IoU = 0.6: TP at 0.5/0.55/0.6, FP above -> AP50-95 = 3/10.
    samples = [_sample([[0, 0, 40, 40]], [0])]
    iou_target = 0.6
    dx = 40 * (1 - iou_target) / (1 + iou_target)  # shift that gives the target IoU
    pred = detlib.Detections(np.array([[dx, 0, 40 + dx, 40]]), np.array([0.9]), np.array([0]))
    res = detlib.evaluate(samples, [pred], ["a"])
    assert res.ap[0, 0] == pytest.approx(1.0)
    assert res.map50_95 == pytest.approx(0.3)


def test_nms_matches_torchvision():
    torch = pytest.importorskip("torch")
    from torchvision.ops import nms

    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 200, (200, 2))
    wh = rng.uniform(10, 60, (200, 2))
    boxes = np.concatenate([xy, xy + wh], axis=1)
    scores = rng.uniform(0, 1, 200)
    ours = detlib.nms_numpy(boxes, scores, 0.5)
    ref = nms(torch.tensor(boxes), torch.tensor(scores), 0.5).numpy()
    np.testing.assert_array_equal(ours, ref)


def test_detection_flags_threshold_consistency():
    samples = [_sample([[0, 0, 10, 10], [20, 20, 30, 30]], [0, 1])]
    pred = detlib.Detections(
        np.array([[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30]], float),
        np.array([0.9, 0.8, 0.3]),
        np.array([0, 0, 0]),
    )
    scores, img_idx, tp = detlib.detection_flags(samples, [pred], 0.5)
    np.testing.assert_array_equal(tp, [True, False, False])  # duplicate + wrong class are FPs
    np.testing.assert_array_equal(img_idx, [0, 0, 0])
    assert scores.tolist() == [0.9, 0.8, 0.3]
