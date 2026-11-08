"""CPU tests for the project-04 detector utilities (no model weights needed)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "projects" / "04_open_vocab_detection" / "scripts"))

ovd = pytest.importorskip("ovd")
pytest.importorskip("pycocotools")


def _dets(boxes, scores, labels, names=("a", "b")):
    return ovd.Detections(np.asarray(boxes, np.float32), np.asarray(scores, np.float32),
                          np.asarray(labels, np.int64), list(names))  # fmt: skip


def _gt():
    return {
        "images": [{"id": 1, "width": 100, "height": 100, "file_name": "x.jpg"},
                   {"id": 2, "width": 100, "height": 100, "file_name": "y.jpg"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 30, 30], "area": 900,
             "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [50, 50, 40, 40], "area": 1600,
             "iscrowd": 0},
            {"id": 3, "image_id": 2, "category_id": 1, "bbox": [0, 0, 50, 50], "area": 2500,
             "iscrowd": 0},
        ],
        "categories": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}],
    }  # fmt: skip


def test_detections_select_topk_and_above():
    d = _dets([[0, 0, 1, 1], [0, 0, 2, 2], [0, 0, 3, 3]], [0.2, 0.9, 0.5], [0, 1, 0])
    assert d.topk(2).scores.tolist() == pytest.approx([0.9, 0.5])
    assert len(d.above(0.5)) == 2
    assert d.topk(1).label_names == ["b"]
    assert len(ovd.Detections.empty(["a"])) == 0


def test_topk_flat_allows_multiple_labels_per_box():
    import torch

    scores = torch.tensor([[0.9, 0.8], [0.1, 0.3]])
    boxes = torch.tensor([[0.0, 0, 1, 1], [1.0, 1, 2, 2]])
    b, s, lab = ovd._topk_flat(scores, boxes, k=3, min_score=0.2)
    assert s.tolist() == pytest.approx([0.9, 0.8, 0.3])
    assert lab.tolist() == [0, 1, 1]
    assert b[0].tolist() == b[1].tolist() == [0.0, 0, 1, 1]  # same box, two classes


def test_nms_per_class_keeps_other_classes():
    d = _dets([[0, 0, 10, 10], [1, 1, 10, 10], [0, 0, 10, 10]], [0.9, 0.8, 0.7], [0, 0, 1])
    out = ovd.nms_per_class(d, iou=0.5)
    assert sorted(out.scores.tolist()) == pytest.approx([0.7, 0.9])


def test_detections_to_coco_uses_xywh_and_category_ids():
    d = _dets([[10, 20, 40, 60]], [0.5], [1])
    (r,) = ovd.detections_to_coco(d, image_id=7, cat_ids=[3, 9])
    assert r["image_id"] == 7 and r["category_id"] == 9
    assert r["bbox"] == [10.0, 20.0, 30.0, 40.0]


def test_coco_evaluate_perfect_and_empty():
    gt = ovd.coco_from_dict(_gt())
    perfect = [{"image_id": a["image_id"], "category_id": a["category_id"], "bbox": a["bbox"],
                "score": 0.9} for a in _gt()["annotations"]]  # fmt: skip
    m = ovd.coco_evaluate(gt, perfect)
    assert m["AP"] == pytest.approx(1.0)
    assert m["per_class_AP"]["a"] == pytest.approx(1.0)
    assert len(m["pr_curve_iou50"]) == 101
    assert ovd.coco_evaluate(gt, [])["AP"] == 0.0


def test_coco_evaluate_missing_class_gives_zero_ap_for_it():
    gt = ovd.coco_from_dict(_gt())
    only_a = [{"image_id": a["image_id"], "category_id": 1, "bbox": a["bbox"], "score": 0.9}
              for a in _gt()["annotations"] if a["category_id"] == 1]  # fmt: skip
    m = ovd.coco_evaluate(gt, only_a)
    assert m["per_class_AP"]["a"] == pytest.approx(1.0)
    assert m["per_class_AP"]["b"] == pytest.approx(0.0)
    assert m["AP"] == pytest.approx(0.5)


def test_f1_vs_threshold_picks_operating_point():
    gt = ovd.coco_from_dict(_gt())
    res = [{"image_id": a["image_id"], "category_id": a["category_id"], "bbox": a["bbox"],
            "score": 0.8} for a in _gt()["annotations"]]  # fmt: skip
    res.append({"image_id": 2, "category_id": 2, "bbox": [60, 60, 10, 10], "score": 0.3})  # FP
    f = ovd.f1_vs_threshold(gt, res, np.array([0.1, 0.5, 0.9]))
    assert f["precision"][0] == pytest.approx(0.75) and f["recall"][0] == pytest.approx(1.0)
    assert f["f1"][1] == pytest.approx(1.0) and f["best_threshold"] == pytest.approx(0.5)
    assert f["recall"][2] == pytest.approx(0.0)


def test_yolo_labels_to_coco(tmp_path):
    from PIL import Image

    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    Image.new("RGB", (200, 100)).save(tmp_path / "images" / "a.jpg")
    (tmp_path / "labels" / "a.txt").write_text("1 0.5 0.5 0.2 0.4\n")
    gt = ovd.yolo_labels_to_coco(tmp_path / "images", tmp_path / "labels", ["x", "y"])
    (ann,) = gt["annotations"]
    assert ann["category_id"] == 2
    assert ann["bbox"] == pytest.approx([80.0, 30.0, 40.0, 40.0])
    assert gt["images"][0]["width"] == 200


class _WordTokenizer:
    """Stand-in for the BERT tokenizer: one token per word or '.', plus [CLS]/[SEP]."""

    def __call__(self, text, return_offsets_mapping=True):
        import re

        offsets = [(0, 0)] + [m.span() for m in re.finditer(r"[a-z]+|\.", text)] + [(0, 0)]
        return {"offset_mapping": offsets}


def test_grounding_dino_positive_map_and_caption_chunking():
    from types import SimpleNamespace

    det = object.__new__(ovd.GroundingDinoDetector)  # no weights: only the text logic is tested
    det.processor = SimpleNamespace(tokenizer=_WordTokenizer())
    det.max_tokens = 8
    caption, posmap, n_tok = det._caption_and_posmap(["Cat", "remote control"])
    assert caption == "cat. remote control."
    # tokens: [CLS] cat . remote control . [SEP]
    assert n_tok == 7
    assert posmap.tolist() == [[0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 0, 0]]
    # 4 classes need 11 tokens > 8, so the vocabulary is split into two captions
    det._set_vocab(["a", "b c", "d", "e"])
    assert [c["offset"] for c in det.chunks] == [0, 2]
    assert [c["caption"] for c in det.chunks] == ["a. b c.", "d. e."]
    assert all(c["n_tokens"] <= 8 for c in det.chunks)
