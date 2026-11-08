"""CPU tests for the MOT metrics, MOT17 preprocessing and trackers of project 06."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "projects/06_tracking/scripts"))

import motlib as ml  # noqa: E402


def make_gt(n_obj: int = 3, n_frames: int = 20, seed: int = 0) -> np.ndarray:
    """Pedestrian GT rows (frame, id, x, y, w, h, flag=1, class=1, vis=1) moving linearly."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_obj):
        x0, y0 = rng.uniform(0, 400), rng.uniform(0, 300)
        vx, vy = rng.uniform(-3, 3), rng.uniform(-1, 1)
        w, h = rng.uniform(30, 50), rng.uniform(80, 120)
        for t in range(1, n_frames + 1):
            rows.append([t, i + 1, x0 + vx * t + 150 * i, y0 + vy * t, w, h, 1, 1, 1])
    return np.asarray(rows, dtype=float)


def to_pred(gt: np.ndarray) -> np.ndarray:
    return gt[:, :7].copy()


def test_perfect_tracking_scores_one():
    gt = make_gt()
    res = ml.evaluate(gt, to_pred(gt), length=20)
    assert res["MOTA"] == pytest.approx(1.0)
    assert res["IDF1"] == pytest.approx(1.0)
    assert res["HOTA"] == pytest.approx(1.0)
    assert res["MOTP"] == pytest.approx(1.0)
    assert res["IDSW"] == 0 and res["CLR_FP"] == 0 and res["CLR_FN"] == 0
    assert res["MT"] == 3 and res["ML"] == 0


def test_identity_switch_is_counted_once():
    gt = make_gt(n_obj=1, n_frames=10)
    pred = to_pred(gt)
    pred[pred[:, 0] > 6, 1] = 99  # the tracker changes the id of the only object at frame 7
    res = ml.evaluate(gt, pred, length=10)
    assert res["IDSW"] == 1
    assert res["MOTA"] == pytest.approx(1 - 1 / 10)
    # IDF1: the best one-to-one assignment keeps 6 of the 10 frames.
    assert res["IDTP"] == 6
    assert res["IDF1"] == pytest.approx(6 / 10)
    clr = ml.clear_metrics(ml.prepare_mot17(gt, pred, 10), return_matches=True)
    assert clr["idsw_events"].tolist() == [[6, 1, 1, 99]]  # frame index 6 = frame 7


def test_misses_and_false_positives():
    gt = make_gt(n_obj=2, n_frames=10)
    pred = to_pred(gt)
    pred = pred[~((pred[:, 1] == 2) & (pred[:, 0] <= 4))]  # object 2 missed in frames 1-4
    fp = np.array([[5, 50, 900, 900, 20, 40, 0.9]])  # a box far from everything
    res = ml.evaluate(gt, np.vstack([pred, fp]), length=10)
    assert res["CLR_FN"] == 4 and res["CLR_FP"] == 1 and res["IDSW"] == 0
    assert res["MOTA"] == pytest.approx(1 - 5 / 20)


def test_fragmentation_counts_interruptions():
    gt = make_gt(n_obj=2, n_frames=10)
    pred = to_pred(gt)
    gap = (pred[:, 1] == 1) & (pred[:, 0] >= 4) & (pred[:, 0] <= 5)
    res = ml.evaluate(gt, pred[~gap], length=10)  # object 1 lost in frames 4-5, same id after
    assert res["Frag"] == 1 and res["IDSW"] == 0
    # TrackEval quirk (reproduced on purpose): a frame without *any* tracker box does not
    # reset the "tracked in the previous frame" state, so this gap is not a fragmentation.
    only_one = pred[(pred[:, 1] == 1) & ~gap]
    assert ml.evaluate(gt[gt[:, 1] == 1], only_one, length=10)["Frag"] == 0


def test_distractor_matches_are_removed():
    gt = make_gt(n_obj=2, n_frames=5)
    gt[gt[:, 1] == 2, 7] = 7  # object 2 is a "static person" (distractor class)
    pred = to_pred(gt)  # the tracker also tracks the static person
    res = ml.evaluate(gt, pred, length=5)
    assert res["CLR_FP"] == 0 and res["MOTA"] == pytest.approx(1.0)
    assert res["removed_distractor_matches"] == 5
    raw = ml.evaluate(gt, pred, length=5, preprocess=False)  # no protocol: they count
    assert raw["num_gt_dets"] == 10


def test_zero_marked_gt_is_ignored():
    gt = make_gt(n_obj=2, n_frames=5)
    gt[gt[:, 1] == 2, 6] = 0  # consider-flag 0
    pred = to_pred(gt)[to_pred(gt)[:, 1] == 1]
    res = ml.evaluate(gt, pred, length=5)
    assert res["num_gt_dets"] == 5 and res["MOTA"] == pytest.approx(1.0)


def test_empty_tracker_output():
    gt = make_gt(n_obj=2, n_frames=5)
    res = ml.evaluate(gt, np.zeros((0, 7)), length=5)
    assert res["CLR_FN"] == 10 and res["MOTA"] == 0.0 and res["HOTA"] == 0.0
    assert res["ML"] == 2


def random_case(seed: int, n_frames: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Noisy tracker output with misses, swaps and false positives."""
    rng = np.random.default_rng(seed)
    gt = make_gt(n_obj=6, n_frames=n_frames, seed=seed)
    pred = to_pred(gt)
    pred[:, 2:4] += rng.normal(0, 6, size=(len(pred), 2))  # localisation noise
    pred = pred[rng.random(len(pred)) > 0.15]  # 15 % misses
    swap = (pred[:, 1] <= 2) & (pred[:, 0] > n_frames // 2)
    pred[swap, 1] = 3 - pred[swap, 1]  # objects 1 and 2 swap ids halfway
    fps = np.column_stack([rng.integers(1, n_frames + 1, 15), 100 + np.arange(15),
                           rng.uniform(0, 600, (15, 2)), rng.uniform(20, 60, (15, 2)),
                           np.ones(15)])  # fmt: skip
    return gt, np.vstack([pred, fps])


def to_trackeval(data: ml.EvalData) -> dict:
    return {
        "num_timesteps": len(data.gt_ids), "num_gt_ids": data.num_gt_ids,
        "num_tracker_ids": data.num_tr_ids, "num_gt_dets": data.num_gt_dets,
        "num_tracker_dets": data.num_tr_dets, "gt_ids": data.gt_ids,
        "tracker_ids": data.tr_ids, "similarity_scores": data.similarity, "seq": "synthetic",
    }  # fmt: skip


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_reference_trackeval(seed):
    te = pytest.importorskip("trackeval")
    gt, pred = random_case(seed)
    data = ml.prepare_mot17(gt, pred, 30)
    ref_data = to_trackeval(data)
    cfg = {"PRINT_CONFIG": False}
    ref = {
        **te.metrics.CLEAR(cfg).eval_sequence(ref_data),
        **te.metrics.Identity(cfg).eval_sequence(ref_data),
    }
    hota_ref = te.metrics.HOTA().eval_sequence(ref_data)
    ours = {**ml.clear_metrics(data), **ml.identity_metrics(data), **ml.hota_metrics(data)}
    for k in ("MOTA", "MOTP", "IDSW", "CLR_FP", "CLR_FN", "MT", "PT", "ML", "Frag", "IDF1",
              "IDP", "IDR"):  # fmt: skip
        assert ours[k] == pytest.approx(float(ref[k]), abs=1e-9), k
    for k in ("HOTA", "DetA", "AssA", "LocA", "DetRe", "AssRe"):
        assert ours[k] == pytest.approx(float(np.mean(hota_ref[k])), abs=1e-9), k


def test_combine_matches_reference_trackeval():
    te = pytest.importorskip("trackeval")
    cases = [random_case(s) for s in (3, 4)]
    ours = ml.combine([ml.evaluate(g, p, 30) for g, p in cases])
    hota, clear = te.metrics.HOTA(), te.metrics.CLEAR({"PRINT_CONFIG": False})
    datas = [to_trackeval(ml.prepare_mot17(g, p, 30)) for g, p in cases]
    ref_h = hota.combine_sequences({str(i): hota.eval_sequence(d) for i, d in enumerate(datas)})
    ref_c = clear.combine_sequences({str(i): clear.eval_sequence(d) for i, d in enumerate(datas)})
    assert ours["MOTA"] == pytest.approx(float(ref_c["MOTA"]))
    assert ours["HOTA"] == pytest.approx(float(np.mean(ref_h["HOTA"])))
    assert ours["AssA"] == pytest.approx(float(np.mean(ref_h["AssA"])))


def test_mot_txt_roundtrip(tmp_path):
    rows = np.array([[1, 3, 10.5, 20.25, 30, 40, 0.9], [2, 3, 11, 21, 30, 40, 0.8]])
    path = ml.write_mot_txt(tmp_path / "seq.txt", rows)
    back = ml.load_mot_txt(path)
    np.testing.assert_allclose(back[:, :7], rows, atol=1e-2)


def test_kalman_filter_learns_constant_velocity():
    z0 = np.array([100.0, 50.0, 40.0, 100.0])
    v = np.array([4.0, -2.0, 0.0, 0.0])
    kf = ml.KalmanBoxFilter(z0)
    for k in range(1, 30):
        kf.predict()
        kf.update(z0 + k * v)
    pred = kf.predict()
    np.testing.assert_allclose(pred, z0 + 30 * v, atol=0.5)
    np.testing.assert_allclose(kf.x[4:], v, atol=0.1)


def synthetic_detections(gt: np.ndarray, n_frames: int, drop: tuple[int, ...] = ()):
    cache_rows = gt[~np.isin(gt[:, 0], drop)]
    xyxy = ml.xywh_to_xyxy(cache_rows[:, 2:6]).astype(np.float32)
    conf = np.full(len(cache_rows), 0.9, np.float32)
    return ml.DetectionCache(cache_rows[:, 0].astype(int), xyxy, conf)


def test_simple_sort_keeps_identities_through_a_gap():
    gt = make_gt(n_obj=2, n_frames=25)
    dets = synthetic_detections(gt, 25, drop=(12, 13))
    rows, timing = ml.run_tracker(ml.SimpleSORT(min_hits=1), dets, 25)
    assert len(np.unique(rows[:, 1])) == 2
    res = ml.evaluate(gt, rows, 25)
    assert res["IDSW"] == 0 and res["MOTA"] > 0.85
    assert timing["tracker_fps"] > 0


def test_ultralytics_bytetrack_runs_on_cached_detections():
    pytest.importorskip("ultralytics")
    gt = make_gt(n_obj=2, n_frames=25)
    dets = synthetic_detections(gt, 25)
    tracker = ml.make_tracker("bytetrack.yaml")
    rows, _ = ml.run_tracker(tracker, dets, 25)
    assert len(np.unique(rows[:, 1])) == 2
    assert ml.evaluate(gt, rows, 25)["IDSW"] == 0


def test_detections_adapter_indexing():
    d = ml.Detections(np.array([[0, 0, 10, 20], [5, 5, 9, 9]]), np.array([0.9, 0.2]))
    np.testing.assert_allclose(d.xywh[0], [5, 10, 10, 20])
    sub = d[d.conf > 0.5]
    assert len(sub) == 1 and sub.xyxy.shape == (1, 4)
