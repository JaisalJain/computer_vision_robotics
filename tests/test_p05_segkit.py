"""CPU tests for the segmentation metrics and prompt helpers of project 05 (no models)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "projects/05_segmentation/scripts"))

import segkit as sk  # noqa: E402


def disk(h: int, w: int, cx: float, cy: float, r: float) -> np.ndarray:
    yy, xx = np.mgrid[:h, :w]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2


def test_j_measure_basic_and_empty():
    a = np.zeros((10, 10), bool)
    a[:5] = True
    b = np.zeros((10, 10), bool)
    b[:, :5] = True
    assert sk.j_measure(a, b) == pytest.approx(25 / 75)
    assert sk.j_measure(a, a) == 1.0
    assert sk.j_measure(np.zeros_like(a), np.zeros_like(a)) == 1.0
    assert sk.j_measure(a, np.zeros_like(a)) == 0.0


def test_seg2bmap_is_inner_one_pixel_boundary():
    m = np.zeros((9, 9), bool)
    m[2:7, 2:7] = True
    b = sk.seg2bmap(m)
    assert b.sum() == 16  # perimeter of a 5x5 square
    assert not b[3:6, 3:6].any()  # interior is not boundary
    assert (b <= m).all()
    full = np.ones((4, 4), bool)  # mask touching the image border: border pixels are boundary
    assert sk.seg2bmap(full).sum() == 12


def test_f_measure_tolerance_and_conventions():
    h, w = 120, 160
    gt = disk(h, w, 80, 60, 30)
    assert sk.boundary_radius((h, w)) == int(np.ceil(0.008 * np.hypot(h, w)))  # 2 px
    assert sk.f_measure(gt, gt) == 1.0
    # a 1-px shift is inside the 2-px tolerance -> perfect boundary score, J < 1
    shifted = np.roll(gt, 1, axis=1)
    assert sk.f_measure(shifted, gt) == 1.0
    assert sk.j_measure(shifted, gt) < 1.0
    # a 10-px shift is far outside it
    assert sk.f_measure(np.roll(gt, 10, axis=1), gt) < 0.5
    empty = np.zeros_like(gt)
    assert sk.f_measure(empty, empty) == 1.0
    assert sk.f_measure(empty, gt) == 0.0
    assert sk.f_measure(gt, empty) == 0.0


def test_jf_statistics_decay_and_recall():
    x = np.r_[np.ones(8), np.zeros(8)]
    st = sk.jf_statistics(x)
    assert st["mean"] == pytest.approx(0.5)
    assert st["recall"] == pytest.approx(0.5)
    assert st["decay"] == pytest.approx(1.0)  # first quarter all 1, last quarter all 0
    assert sk.jf_statistics(np.full(10, 0.7))["decay"] == pytest.approx(0.0)


def test_evaluate_vos_skips_first_and_last_frame():
    gt = np.zeros((5, 20, 20), np.uint8)
    gt[:, 5:15, 5:15] = 1
    pred = gt.copy()
    pred[0] = 0  # wrong on the prompt frame and the last frame: both are excluded
    pred[-1] = 0
    per = sk.evaluate_vos(pred, gt)
    assert per[1]["J"].shape == (3,)
    assert sk.summarize_vos(per)["JF_mean"] == 1.0


def test_interior_point_lies_inside_nonconvex_mask():
    ring = disk(101, 101, 50, 50, 40) & ~disk(101, 101, 50, 50, 25)
    x, y = sk.interior_point(ring)
    assert ring[int(y), int(x)]
    cy, cx = np.argwhere(ring).mean(0)
    assert not ring[int(round(cy)), int(round(cx))]  # the centroid would miss the object


def test_mask_to_box_uses_pixel_edges():
    m = np.zeros((10, 10), bool)
    m[2:5, 3:8] = True
    assert sk.mask_to_box(m) == [3.0, 2.0, 8.0, 5.0]


def test_next_click_targets_largest_error():
    gt = np.zeros((50, 50), bool)
    gt[10:40, 10:40] = True
    pred = gt.copy()
    pred[10:40, 25:40] = False  # large false-negative region on the right
    pred[0:3, 0:3] = True  # tiny false positive
    (x, y), label = sk.next_click(pred, gt)
    assert label == 1 and gt[int(y), int(x)] and not pred[int(y), int(x)]
    assert sk.next_click(gt, gt) is None
    (x, y), label = sk.next_click(gt | disk(50, 50, 45, 45, 4), gt)
    assert label == 0  # only a false-positive blob is left -> negative click


def test_polygons_to_masks_rasterises_normalised_polygons():
    text = "3 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5\n0 0.6 0.6 0.9 0.6 0.9 0.9\n"
    masks, classes = sk.polygons_to_masks(text, 100, 200)
    assert masks.shape == (2, 100, 200) and classes.tolist() == [3, 0]
    assert masks[0, 30, 60] and not masks[0, 80, 160]
    assert abs(masks[0].sum() - 40 * 80) < 200  # 0.4 x 0.4 of a 100 x 200 image


def test_logits_to_labels_argmax_and_background():
    logits = np.full((1, 2, 4, 4), -1.0, np.float32)
    logits[0, 0, :2] = 2.0
    logits[0, 1, 1:3] = 3.0
    lab = sk.logits_to_labels(logits, [5, 7])[0]
    assert lab[0, 0] == 5 and lab[1, 0] == 7 and lab[2, 0] == 7 and lab[3, 0] == 0


def _sample(masks, classes):
    m = np.asarray(masks)
    return sk.InstanceSample("x", np.zeros((*m.shape[1:], 3), np.uint8), m, np.asarray(classes))


def test_mask_ap_perfect_and_ranking():
    a, b = disk(64, 64, 20, 20, 10), disk(64, 64, 45, 45, 10)
    gt = [_sample([a, b], [0, 1])]
    perfect = [sk.Detections(np.stack([a, b]), np.array([0.9, 0.8]), np.array([0, 1]))]
    res = sk.mask_ap(perfect, gt)
    assert res["AP"] == pytest.approx(1.0) and res["n_gt"] == 2 and res["n_classes"] == 2
    # a confident false positive ranked above the true positive halves precision at recall 1
    fp = disk(64, 64, 45, 20, 8)
    noisy = [sk.Detections(np.stack([fp, a, b]), np.array([0.95, 0.9, 0.8]), np.array([0, 0, 1]))]
    res = sk.mask_ap(noisy, gt)
    assert res["AP50"] == pytest.approx((0.5 + 1.0) / 2)
    # wrong class never matches
    wrong = [sk.Detections(np.stack([a]), np.array([0.9]), np.array([1]))]
    assert sk.mask_ap(wrong, gt)["AP"] == 0.0


def test_mask_iou_matrix_matches_pairwise_j():
    rng = np.random.default_rng(0)
    p = rng.random((3, 16, 16)) > 0.5
    g = rng.random((2, 16, 16)) > 0.5
    M = sk.mask_iou_matrix(p, g)
    for i in range(3):
        for j in range(2):
            assert M[i, j] == pytest.approx(sk.j_measure(p[i], g[j]))
