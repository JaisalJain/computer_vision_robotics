import numpy as np
import pytest

from cvlab import metrics as M
from cvlab.geometry import invert_T, make_T, so3_exp


def test_box_iou():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
    np.testing.assert_allclose(M.box_iou(a, b), [[1.0, 50 / 150, 0.0]])


def test_mask_iou_and_dice():
    a = np.zeros((4, 4), bool)
    b = np.zeros((4, 4), bool)
    a[:2] = True
    b[1:3] = True
    assert M.mask_iou(a, b) == pytest.approx(4 / 12)
    assert M.dice(a, b) == pytest.approx(2 * 4 / 16)
    assert M.mask_iou(np.zeros((2, 2)), np.zeros((2, 2))) == 1.0


def test_average_precision_perfect_and_half():
    p, r = M.precision_recall([0.9, 0.8], [True, True], n_gt=2)
    assert M.average_precision(r, p, "continuous") == pytest.approx(1.0)
    assert M.average_precision(r, p, "coco") == pytest.approx(1.0)
    p, r = M.precision_recall([0.9, 0.8], [True, False], n_gt=2)
    assert M.average_precision(r, p, "continuous") == pytest.approx(0.5)
    assert M.average_precision(r, p, "coco") == pytest.approx(51 / 101)  # thresholds 0..0.5


def test_match_detections_greedy():
    gt = np.array([[0, 0, 10, 10]])
    pred = np.array([[0, 0, 10, 10], [1, 1, 10, 10]])
    tp = M.match_detections(pred, [0.5, 0.9], gt, iou_thr=0.5)
    # the higher-scoring (second) box claims the GT; the first becomes a duplicate FP
    assert tp.tolist() == [False, True]


def test_classification_metrics():
    logits = np.array([[0.1, 0.9, 0.0], [0.8, 0.1, 0.1], [0.3, 0.3, 0.4]])
    labels = np.array([1, 1, 2])
    assert M.topk_accuracy(logits, labels, 1) == pytest.approx(2 / 3)
    assert M.topk_accuracy(logits, labels, 2) == pytest.approx(1.0)
    cm = M.confusion_matrix(labels, logits.argmax(1), 3)
    assert cm.sum() == 3 and cm[1, 0] == 1


def test_psnr_ssim(rng):
    img = rng.uniform(size=(32, 32, 3))
    assert M.psnr(img, img) == float("inf")
    noisy = np.clip(img + rng.normal(scale=0.1, size=img.shape), 0, 1)
    assert 15 < M.psnr(noisy, img) < 25
    assert M.ssim(img, img) == pytest.approx(1.0)
    assert M.ssim(noisy, img) < 0.95


def test_depth_alignment_and_metrics(rng):
    gt = rng.uniform(1, 5, size=(20, 20))
    pred = 0.5 * gt
    s = M.align_depth_scale(pred, gt)
    assert s == pytest.approx(2.0)
    m = M.depth_metrics(s * pred, gt)
    assert m["abs_rel"] == pytest.approx(0.0, abs=1e-12)
    assert m["delta1"] == 1.0
    s2, t2 = M.align_depth_scale_shift(3 * gt + 1, gt)
    assert s2 == pytest.approx(1 / 3) and t2 == pytest.approx(-1 / 3)
    gt[0, 0] = 0  # invalid pixels are ignored
    assert M.depth_metrics(gt * 1.1, gt)["abs_rel"] == pytest.approx(0.1)


def test_ate_with_sim3(rng):
    gt = np.cumsum(rng.normal(size=(100, 3)), axis=0)
    R = so3_exp([0.1, 0.2, 0.3])
    est = (gt @ R.T) * 0.3 + [1, 2, 3]
    res = M.ate(est, gt, align="sim3")
    assert res["rmse"] == pytest.approx(0.0, abs=1e-9)
    assert res["scale"] == pytest.approx(1 / 0.3)
    assert M.ate(est, gt, align="none")["rmse"] > 1.0


def test_rpe_constant_bias():
    gt = np.stack([make_T(np.eye(3), [i, 0, 0]) for i in range(10)])
    est = np.stack([make_T(np.eye(3), [1.1 * i, 0, 0]) for i in range(10)])
    res = M.rpe(est, gt, delta=1)
    assert res["trans_rmse"] == pytest.approx(0.1)
    assert res["rot_rmse_deg"] == pytest.approx(0.0, abs=1e-6)
    same = M.rpe(gt, gt)
    assert same["trans_rmse"] == pytest.approx(0.0)
    assert np.allclose(invert_T(gt[1]) @ gt[1], np.eye(4))
