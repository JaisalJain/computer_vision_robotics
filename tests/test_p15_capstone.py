"""CPU tests for the capstone pipeline helpers (project 15): phrases, caches, object map."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "projects/15_robotics_capstone/scripts")
)

import capstone as cap  # noqa: E402


def test_match_query_maps_partial_phrases():
    queries = ["game controller", "cereal box", "cup"]
    assert cap.match_query("cup", queries) == "cup"
    assert cap.match_query("Cereal", queries) == "cereal box"
    assert cap.match_query("controller", queries) == "game controller"
    assert cap.match_query("monitor", queries) is None


def test_postprocess_nms_and_large_box_filter():
    dets = cap.Detections(
        np.array([[0, 0, 100, 100], [2, 2, 102, 102], [0, 0, 640, 480], [200, 200, 260, 260.0]]),
        np.array([0.9, 0.8, 0.7, 0.6]),
        ["cup", "cup", "book", "cup"],
    )
    out = cap.postprocess(dets, (480, 640))
    assert out.labels == ["cup", "cup"]  # duplicate suppressed, full-image box removed
    np.testing.assert_allclose(out.scores, [0.9, 0.6])


def test_mask_cache_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    cache = {
        8: {
            "boxes": np.array([[1, 2, 30, 40.0]]),
            "scores": np.array([0.5]),
            "labels": ["cup"],
            "sam3": rng.random((1, 480, 640)) > 0.5,
        },
        0: {
            "boxes": np.zeros((0, 4)),
            "scores": np.zeros(0),
            "labels": [],
            "sam3": np.zeros((0, 480, 640), bool),
        },
    }
    path = cap.save_mask_cache(tmp_path / "c.npz", cache)
    back = cap.load_mask_cache(path)
    assert list(back) == [0, 8]
    assert back[8]["labels"] == ["cup"]
    np.testing.assert_array_equal(back[8]["sam3"], cache[8]["sam3"])
    assert back[0]["sam3"].shape == (0, 480, 640)


def _obs(frame, label, p):
    return {"frame": frame, "label": label, "box_center_w": np.asarray(p, dtype=float)}


def test_object_map_association_and_consistency():
    rng = np.random.default_rng(1)
    obs = []
    for f in range(10):
        obs.append(_obs(f, "cup", [0.0, 0.0, 0.8] + rng.normal(0, 0.01, 3)))
        obs.append(_obs(f, "book", [0.5, 0.0, 0.8] + rng.normal(0, 0.01, 3)))
        obs.append(_obs(f, "cup", [0.05, 0.0, 0.8] + rng.normal(0, 0.01, 3)))  # 2nd cup, 5 cm away
    obs.append(_obs(3, "cup", [2.0, 2.0, 0.8]))  # one-off false positive
    lms = cap.build_object_map(obs, gate=0.15)
    sizes = sorted(len(lm["obs"]) for lm in lms)
    assert sizes == [1, 10, 10, 10]  # at most one observation per landmark per frame
    c = cap.consistency(obs, gate=0.15, min_obs=5)
    assert c["stable landmarks"] == 3 and c["landmarks"] == 4
    assert abs(c["obs in stable landmarks [%]"] - 100 * 30 / 31) < 1e-9
    assert c["median dev [cm]"] < 3.0
