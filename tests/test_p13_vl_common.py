"""CPU tests for the pure helpers of project 13 (no models are loaded, no network)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "projects/13_vision_language/scripts"))

import vl_common as vl  # noqa: E402


def test_build_prompts_is_class_major():
    p = vl.build_prompts(["cat", "dog"], ["a {}.", "the {}!"])
    assert p == ["a cat.", "the cat!", "a dog.", "the dog!"]


def test_ensemble_is_mean_of_unit_vectors(rng):
    emb = rng.normal(size=(3 * 4, 8)) * rng.uniform(0.5, 5.0, size=(12, 1))  # mixed norms
    out = vl.ensemble_text_embeddings(emb, n_classes=3)
    assert out.shape == (3, 8)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-6)
    unit = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    ref = unit[4:8].mean(0)
    np.testing.assert_allclose(out[1], ref / np.linalg.norm(ref), atol=1e-6)


def test_single_template_ensemble_is_identity(rng):
    emb = rng.normal(size=(5, 16))
    out = vl.ensemble_text_embeddings(emb, n_classes=5)
    np.testing.assert_allclose(out, emb / np.linalg.norm(emb, axis=1, keepdims=True), atol=1e-6)


def test_softmax_rows_sum_to_one_and_sigmoid_is_independent(rng):
    img = rng.normal(size=(6, 4))
    img /= np.linalg.norm(img, axis=1, keepdims=True)
    cls = rng.normal(size=(3, 4))
    cls /= np.linalg.norm(cls, axis=1, keepdims=True)
    p_soft = vl.zero_shot_scores(img, cls, scale=100.0, kind="softmax")
    np.testing.assert_allclose(p_soft.sum(1), 1.0)
    p_sig = vl.zero_shot_scores(img, cls, scale=10.0, bias=-5.0, kind="sigmoid")
    assert np.all((p_sig > 0) & (p_sig < 1))
    # Adding a class never changes the sigmoid score of the existing ones ...
    extra = np.vstack([cls, -cls[:1]])
    np.testing.assert_allclose(
        vl.zero_shot_scores(img, extra, 10.0, -5.0, "sigmoid")[:, :3], p_sig
    )  # fmt: skip
    # ... but it does change the softmax scores.
    assert not np.allclose(vl.zero_shot_scores(img, extra, 100.0)[:, :3], p_soft)
    # Both rank classes the same way (monotonic in the cosine similarity).
    np.testing.assert_array_equal(p_soft.argmax(1), p_sig.argmax(1))
    with pytest.raises(ValueError):
        vl.zero_shot_scores(img, cls, 1.0, kind="relu")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Television", "tv"), ("a man", "person"), ("cars", "car"), ("Buses", "bus"),
        ("motorbike", "motorcycle"), ("dining table", "dining table"), ("mugs", "cup"),
        ("computer monitor", "tv"), ("lamp", "lamp"), ("the wine glasses", "wine glass"),
    ],
)  # fmt: skip
def test_normalize_label(raw, expected):
    assert vl.normalize_label(raw) == expected


def test_match_by_label_counts():
    gt = np.array([[0, 0, 10, 10], [20, 20, 30, 30], [50, 50, 60, 60]], float)
    gl = ["person", "car", "dog"]
    pred = np.array(
        [[0, 0, 10, 10], [1, 1, 10, 10], [20, 20, 30, 30], [50, 50, 60, 60], [80, 80, 90, 90]],
        float,
    )
    pl = ["man", "woman", "cat", "dog", "tv"]  # duplicate person, wrong label, hit, extra
    r = vl.match_by_label(pred, pl, gt, gl)
    assert r == {"tp": 2, "fp": 3, "fn": 1}
    assert vl.match_by_label(np.zeros((0, 4)), [], gt, gl) == {"tp": 0, "fp": 0, "fn": 3}


@pytest.mark.parametrize(
    ("answer", "yn", "count", "lr"),
    [
        ("Yes.", True, None, None), ("No, there is not.", False, 0, None),
        ("3", None, 3, None), ("There are two people.", None, 2, None),
        ("Left", None, None, "left"), ("It is to the right of the cup.", None, None, "right"),
        ("I see a cat", None, None, None), ("none", None, 0, None),
    ],
)  # fmt: skip
def test_answer_parsers(answer, yn, count, lr):
    assert vl.parse_yes_no(answer) == yn
    assert vl.parse_count(answer) == count
    assert vl.parse_left_right(answer) == lr


def _item(boxes, classes, w=100, h=100):
    return vl.CocoItem(Path("x.jpg"), w, h, np.asarray(boxes, float), np.asarray(classes))


def test_presence_and_count_probes():
    it = _item([[0, 0, 10, 10], [20, 0, 30, 10], [50, 0, 60, 10]], [0, 0, 2])  # 2 person, 1 car
    pres = vl.presence_probes([it], n_pos=2, n_neg=3, seed=1)
    assert sum(p["truth"] for p in pres) == 2 and len(pres) == 5
    assert {p["object"] for p in pres if p["truth"]} == {"person", "car"}
    assert all(p["object"] not in {"person", "car"} for p in pres if not p["truth"])
    cnt = {p["object"]: p["truth"] for p in vl.count_probes([it])}
    assert cnt == {"car": 1, "person": 2}


def test_spatial_probe_truth_matches_geometry():
    # car on the left (x<20), dog on the right (x>60); two persons are ignored (not single).
    it = _item([[0, 0, 20, 10], [60, 0, 90, 10], [30, 0, 40, 10], [42, 0, 50, 10]],
               [2, 16, 0, 0])  # fmt: skip
    for seed in range(6):
        (p,) = vl.spatial_probes([it], seed=seed)
        first = p["question"].split("Is the ")[1].split(" to the")[0]
        assert p["truth"] == ("left" if first == "car" else "right")
        assert vl.score_probe("spatial", p["truth"], f"{p['truth'].title()}.") == (p["truth"], True)
    overlapping = _item([[0, 0, 50, 10], [45, 0, 90, 10]], [2, 16])
    assert vl.spatial_probes([overlapping]) == []


def test_load_yolo_split(tmp_path):
    from PIL import Image

    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    Image.new("RGB", (200, 100)).save(tmp_path / "images" / "a.jpg")
    (tmp_path / "labels" / "a.txt").write_text("41 0.5 0.5 0.2 0.4\n")
    Image.new("RGB", (50, 50)).save(tmp_path / "images" / "b.jpg")  # no label file
    a, b = vl.load_yolo_split(tmp_path / "images", tmp_path / "labels")
    np.testing.assert_allclose(a.boxes, [[80, 30, 120, 70]])
    assert a.names() == ["cup"] and (a.width, a.height) == (200, 100)
    assert b.boxes.shape == (0, 4) and b.classes.shape == (0,)


def test_plural_and_count_question():
    assert vl.plural("person") == "people" and vl.plural("cup") == "cups"
    assert vl.plural("wine glass") == "wine glasses"
    it = _item([[0, 0, 10, 10]], [41])  # one cup
    (p,) = vl.count_probes([it])
    assert p["question"].startswith("How many cups are in the image?") and p["truth"] == 1


def test_vertical_spatial_probe():
    # clock at the top, car at the bottom, overlapping in x -> only an above/below probe.
    it = _item([[10, 0, 60, 20], [20, 70, 80, 95]], [74, 2])
    for seed in range(6):
        (p,) = vl.spatial_probes([it], seed=seed)
        first = p["question"].split("Is the ")[1].split(" above")[0]
        assert p["truth"] == ("above" if first == "clock" else "below")
        assert vl.score_probe("spatial", p["truth"], f"It is {p['truth']} it.")[1]


@pytest.mark.parametrize(
    ("kind", "truth", "answer", "ok"),
    [
        ("state", "on", "The screen is on.", True), ("state", "on", "Off", False),
        ("keyword", ["phone"], "The telephone.", True), ("keyword", "geometry", "A book", False),
        ("spatial", "below", "Below the sink.", True), ("presence", False, "No.", True),
    ],
)  # fmt: skip
def test_score_probe_kinds(kind, truth, answer, ok):
    assert vl.score_probe(kind, truth, answer)[1] is ok
    with pytest.raises(ValueError):
        vl.score_probe("colour", "red", "red")


def test_wilson_ci():
    lo, hi = vl.wilson_ci(8, 10)
    assert 0.0 < lo < 0.8 < hi < 1.0
    assert vl.wilson_ci(0, 0) == (0.0, 1.0)
    lo, hi = vl.wilson_ci(0, 20)
    assert lo == pytest.approx(0.0, abs=1e-12) and 0.1 < hi < 0.2  # never collapses to [0, 0]
