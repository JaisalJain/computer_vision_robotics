"""CPU tests for the alignment / evaluation helpers of project 08 (no models are loaded)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "projects/08_monocular_depth/scripts"))

import mono_depth as md  # noqa: E402


@pytest.fixture
def scene() -> tuple[np.ndarray, np.ndarray]:
    """Synthetic GT depth (metres) with a hole, like a Kinect frame."""
    rng = np.random.default_rng(0)
    gt = rng.uniform(0.5, 4.0, size=(48, 64))
    gt[:5, :5] = 0.0  # missing measurements
    return gt, rng.uniform(0.0, 1.0, size=gt.shape)


def test_median_scale_recovers_depth_up_to_scale(scene):
    gt, _ = scene
    pred = 0.37 * gt  # depth up to scale (what DA3 any-view predicts)
    res = md.evaluate_frame(pred, "depth", gt, ["median"])["median"]
    assert res["scale"] == pytest.approx(1 / 0.37)
    assert res["abs_rel"] < 1e-9 and res["delta1"] == 1.0


def test_lsq_disparity_alignment_recovers_affine_inverse_depth(scene):
    gt, _ = scene
    valid = gt > 0
    disp = np.where(valid, 2.5 / np.where(valid, gt, 1.0) + 0.3, 0.3)  # affine-invariant
    depth, s, t = md.align_prediction(disp, "disparity", gt, md.valid_mask(gt), "lsq")
    assert s == pytest.approx(1 / 2.5)
    assert t == pytest.approx(-0.3 / 2.5)
    np.testing.assert_allclose(depth[valid], gt[valid], rtol=1e-9)


def test_median_scaling_cannot_fix_a_disparity_shift(scene):
    gt, _ = scene
    valid = gt > 0
    disp = np.where(valid, 1.0 / np.where(valid, gt, 1.0) + 0.5, 0.5)
    with pytest.raises(ValueError):
        md.align_prediction(disp, "disparity", gt, md.valid_mask(gt), "median")
    # Inverting the shifted disparity and median-scaling it leaves a large error.
    naive = md.evaluate_frame(1.0 / disp, "depth", gt, ["median"])["median"]
    assert naive["abs_rel"] > 0.1


def test_raw_metric_and_masking(scene):
    gt, noise = scene
    pred = gt * 1.1 + 0.0 * noise
    res = md.evaluate_frame(pred, "metric", gt)
    assert res["none"]["abs_rel"] == pytest.approx(0.1)
    assert res["median"]["abs_rel"] < 1e-9  # a pure scale error disappears after median scaling
    with pytest.raises(ValueError):
        md.align_prediction(pred, "depth", gt, md.valid_mask(gt), "none")


def test_binned_errors_and_maps(scene):
    gt, _ = scene
    pred = gt + 0.1
    abs_rel, sq, count = md.binned_errors(pred, gt)
    assert count.sum() == md.valid_mask(gt).sum()
    np.testing.assert_allclose(sq[count > 0] / count[count > 0], 0.01)
    err = md.abs_rel_map(pred, gt)
    assert np.isnan(err[0, 0]) and np.nanmax(err) <= 0.1 / 0.5 + 1e-12


def test_chamfer_and_focal_helpers():
    pts = np.random.default_rng(1).normal(size=(500, 3))
    stats = md.chamfer_stats(pts, pts + [0.01, 0.0, 0.0])
    assert stats["chamfer_m"] == pytest.approx(0.01, abs=1e-9)
    assert stats["fscore@5cm"] == 1.0
    K = np.array([[517.3, 0, 318.6], [0, 516.5, 255.3], [0, 0, 1]])
    assert md.da3_metric_focal(K, (480, 640), (378, 504)) == pytest.approx(
        0.5 * (517.3 * 504 / 640 + 516.5 * 378 / 480)
    )
    assert list(md.select_frames(573, 4)) == [0, 191, 381, 572]
