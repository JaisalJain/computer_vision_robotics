"""Small, framework-agnostic helpers for the object-detection notebooks.

* reading Ultralytics/YOLO-format datasets (``images/<split>`` + ``labels/<split>``),
* a transparent COCO-style mAP evaluator built on :mod:`cvlab.metrics`,
* greedy NMS written out in NumPy for teaching (compare with ``torchvision.ops.nms``).

Import from a notebook with::

    sys.path.insert(0, str(project_dir("03_object_detection") / "scripts"))
    import detlib
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cvlab.metrics import average_precision, box_iou, match_detections, precision_recall

IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COCO_IOU_THRESHOLDS = np.round(np.linspace(0.5, 0.95, 10), 2)


@dataclass
class Sample:
    """One image of a YOLO-format split with its ground truth in pixel ``xyxy``."""

    image: Path
    width: int
    height: int
    boxes: np.ndarray  # (N, 4) float, x1 y1 x2 y2 in pixels
    classes: np.ndarray  # (N,) int


def yolo_to_xyxy(xywhn: np.ndarray, width: int, height: int) -> np.ndarray:
    """Normalised YOLO ``(cx, cy, w, h)`` -> pixel ``(x1, y1, x2, y2)``."""
    b = np.asarray(xywhn, dtype=np.float64).reshape(-1, 4)
    cx, cy, w, h = b[:, 0] * width, b[:, 1] * height, b[:, 2] * width, b[:, 3] * height
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def read_yolo_labels(path: Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Read one YOLO label file (``cls cx cy w h`` per line, normalised)."""
    path = Path(path)
    if not path.exists() or not path.read_text().strip():
        return np.zeros((0, 4)), np.zeros(0, dtype=int)
    arr = np.loadtxt(path, ndmin=2)
    return yolo_to_xyxy(arr[:, 1:5], width, height), arr[:, 0].astype(int)


def load_split(root: Path, split: str = "val") -> list[Sample]:
    """All images of ``root/images/<split>`` with labels from ``root/labels/<split>``."""
    from PIL import Image

    root = Path(root)
    samples = []
    for img in sorted((root / "images" / split).iterdir()):
        if img.suffix.lower() not in IMG_SUFFIXES:
            continue
        with Image.open(img) as im:
            w, h = im.size
        boxes, cls = read_yolo_labels(root / "labels" / split / f"{img.stem}.txt", w, h)
        samples.append(Sample(img, w, h, boxes, cls))
    return samples


@dataclass
class Detections:
    """Predictions for one image (pixel ``xyxy`` boxes, confidences, class ids)."""

    boxes: np.ndarray
    scores: np.ndarray
    classes: np.ndarray

    @classmethod
    def empty(cls) -> Detections:
        return cls(np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int))


@dataclass
class EvalResult:
    """Output of :func:`evaluate`; ``ap[c, t]`` is the AP of class ``c`` at IoU ``iou_thrs[t]``."""

    ap: np.ndarray
    iou_thrs: np.ndarray
    n_gt: np.ndarray
    class_names: list[str]
    pr_curves: dict[int, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    @property
    def valid(self) -> np.ndarray:
        """Classes with at least one ground-truth box (others are excluded from the mean)."""
        return self.n_gt > 0

    @property
    def map50(self) -> float:
        """Mean AP at IoU 0.5 over classes with ground truth (NaN if there are none)."""
        return float(self.ap[self.valid, 0].mean()) if self.valid.any() else float("nan")

    @property
    def map50_95(self) -> float:
        """Mean AP over classes and IoU thresholds 0.50:0.95 (NaN without ground truth)."""
        return float(self.ap[self.valid].mean()) if self.valid.any() else float("nan")

    def per_class(self) -> dict[str, dict[str, float]]:
        return {
            name: {
                "AP50": float(self.ap[c, 0]),
                "AP50-95": float(self.ap[c].mean()),
                "n_gt": int(self.n_gt[c]),
            }
            for c, name in enumerate(self.class_names)
        }


def evaluate(
    samples: list[Sample],
    predictions: list[Detections],
    class_names: list[str],
    iou_thrs: np.ndarray = COCO_IOU_THRESHOLDS,
) -> EvalResult:
    """COCO-style AP per class and IoU threshold (101-point interpolation).

    For every class and threshold, predictions of that class are matched greedily (in
    descending score order) to unmatched ground-truth boxes of the same class in the
    same image; a prediction is a true positive if its best free match has
    ``IoU >= thr``. Precision/recall are accumulated over the whole dataset and the
    area under the interpolated PR curve is the AP. ``pr_curves[c]`` holds
    ``(recall, precision)`` at the first threshold (IoU 0.5).
    """
    if len(samples) != len(predictions):
        raise ValueError("one Detections object per sample is required")
    nc = len(class_names)
    ap = np.zeros((nc, len(iou_thrs)))
    n_gt = np.zeros(nc, dtype=int)
    curves: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for c in range(nc):
        n_gt[c] = int(sum((s.classes == c).sum() for s in samples))
        for t, thr in enumerate(iou_thrs):
            scores, tps = [], []
            for s, p in zip(samples, predictions, strict=True):
                keep = p.classes == c
                if not keep.any():
                    continue
                gt = s.boxes[s.classes == c]
                scores.append(p.scores[keep])
                tps.append(match_detections(p.boxes[keep], p.scores[keep], gt, float(thr)))
            if not scores:
                continue
            prec, rec = precision_recall(np.concatenate(scores), np.concatenate(tps), n_gt[c])
            ap[c, t] = average_precision(rec, prec, method="coco")
            if t == 0:
                curves[c] = (rec, prec)
    return EvalResult(ap, np.asarray(iou_thrs), n_gt, list(class_names), curves)


def detection_flags(
    samples: list[Sample], predictions: list[Detections], iou_thr: float = 0.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-prediction ``(scores, image index, is_tp)`` at one IoU threshold (class-aware).

    Because matching is greedy in score order, thresholding the scores afterwards gives
    the same TP flags as re-matching at that threshold, so precision/recall/F1 at any
    confidence ``t`` follow from ``is_tp[scores >= t]``.
    """
    scores, image_idx, flags = [], [], []
    for i, (s, p) in enumerate(zip(samples, predictions, strict=True)):
        tp = np.zeros(len(p.scores), dtype=bool)
        for c in np.unique(p.classes):
            keep = p.classes == c
            tp[keep] = match_detections(
                p.boxes[keep], p.scores[keep], s.boxes[s.classes == c], iou_thr
            )
        scores.append(p.scores)
        image_idx.append(np.full(len(p.scores), i))
        flags.append(tp)
    if not scores:
        return np.zeros(0), np.zeros(0, dtype=int), np.zeros(0, dtype=bool)
    return np.concatenate(scores), np.concatenate(image_idx).astype(int), np.concatenate(flags)


def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.7) -> np.ndarray:
    """Greedy non-maximum suppression; returns kept indices sorted by descending score.

    Repeatedly keep the highest-scoring remaining box and discard every remaining box
    whose IoU with it exceeds ``iou_thr``.
    """
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        ious = box_iou(boxes[i : i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_thr]
    return np.asarray(keep, dtype=int)


def ultralytics_to_detections(result, class_map: dict[int, int] | None = None) -> Detections:
    """Convert one ``ultralytics`` ``Results`` object to :class:`Detections`.

    ``class_map`` optionally re-labels model classes (e.g. COCO ids -> dataset ids);
    predictions of unmapped classes are dropped.
    """
    b = result.boxes
    if b is None or len(b) == 0:
        return Detections.empty()
    boxes = b.xyxy.cpu().numpy().astype(np.float64)
    scores = b.conf.cpu().numpy().astype(np.float64)
    classes = b.cls.cpu().numpy().astype(int)
    if class_map is not None:
        keep = np.array([c in class_map for c in classes], dtype=bool)
        boxes, scores = boxes[keep], scores[keep]
        classes = np.array([class_map[c] for c in classes[keep]], dtype=int)
    return Detections(boxes, scores, classes)
