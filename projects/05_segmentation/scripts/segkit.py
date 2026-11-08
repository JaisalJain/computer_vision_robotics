"""Segmentation toolkit for project 05: data loaders, metrics, prompts and model adapters.

The notebooks, the evaluation scripts and the exercise solutions all import this module, so
every number in the README comes from the same code path.

Contents
--------
* **Data**: coco8-seg / coco128-seg (YOLO polygon labels -> instance masks), DAVIS 2017
  480p sequences (palette PNG annotations -> per-object masks).
* **Metrics**: DAVIS region similarity ``J`` (IoU) and boundary F-measure ``F``, J&F
  statistics (mean / recall / decay), COCO-style mask AP for instance segmentation.
* **Prompts**: interior click (distance-transform peak), tight box, error-driven clicks for
  the interactive "clicks-to-IoU" protocol.
* **Models** (Hugging Face ``transformers``): promptable image segmentation with SAM 3
  (``Sam3TrackerModel``) or SAM 2.1 (``Sam2Model``), concept (text) segmentation with SAM 3
  (``Sam3Model``), Grounding DINO + SAM 2.1 ("Grounded-SAM"), and video propagation with
  ``Sam3TrackerVideoModel`` / ``Sam2VideoModel`` / ``Sam3VideoModel`` (text prompts).

Usage::

    import sys; sys.path.insert(0, "projects/05_segmentation/scripts")
    import segkit as sk
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cvlab.io import data_dir, project_dir, read_image
from cvlab.metrics import average_precision, precision_recall

PROJECT = "05_segmentation"

SAM3_ID = "facebook/sam3"
SAM2_IDS = {
    "tiny": "facebook/sam2.1-hiera-tiny",
    "small": "facebook/sam2.1-hiera-small",
    "base-plus": "facebook/sam2.1-hiera-base-plus",
}
GDINO_ID = "IDEA-Research/grounding-dino-tiny"

# Short labels used in tables and figures.
MODEL_LABELS = {
    SAM3_ID: "SAM 3",
    SAM2_IDS["tiny"]: "SAM 2.1-T",
    SAM2_IDS["small"]: "SAM 2.1-S",
    SAM2_IDS["base-plus"]: "SAM 2.1-B+",
}


def results_dir() -> Path:
    """``projects/05_segmentation/results`` (small committed JSON/CSV metrics)."""
    path = project_dir(PROJECT) / "results"
    path.mkdir(parents=True, exist_ok=True)
    return path


def show_jpeg(fig, quality: int = 85, dpi: int = 100) -> None:
    """Display a photo-heavy matplotlib figure inline as JPEG and close it.

    Photographs compress 5-10x better as JPEG than as PNG, which keeps executed notebooks
    small enough for GitHub. Use it for image grids; keep charts as PNG (``plt.show()``).
    """
    import io

    import matplotlib.pyplot as plt
    from IPython.display import Image, display

    buf = io.BytesIO()
    fig.savefig(buf, format="jpeg", dpi=dpi, bbox_inches="tight", pil_kwargs={"quality": quality})
    plt.close(fig)
    display(Image(data=buf.getvalue(), format="jpeg"))


# ======================================================================== data: COCO-seg
@dataclass
class InstanceSample:
    """One image with its instance masks (``N x H x W`` bool) and COCO class ids."""

    name: str
    image: np.ndarray
    masks: np.ndarray
    classes: np.ndarray

    @property
    def boxes(self) -> np.ndarray:
        return np.array([mask_to_box(m) for m in self.masks]).reshape(-1, 4)


def coco_names() -> dict[int, str]:
    """The 80 COCO class names in Ultralytics' contiguous indexing (0 = person)."""
    import importlib.util

    import yaml

    spec = importlib.util.find_spec("ultralytics")
    if spec is None or spec.origin is None:  # pragma: no cover
        raise ImportError("ultralytics is needed for the COCO class list")
    path = Path(spec.origin).parent / "cfg" / "datasets" / "coco8-seg.yaml"
    return {int(k): v for k, v in yaml.safe_load(path.read_text())["names"].items()}


def polygons_to_masks(label_text: str, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Rasterise YOLO-seg label lines ``cls x1 y1 x2 y2 ...`` (normalised) into masks.

    Returns ``(masks N x H x W bool, classes N int)``.
    """
    import cv2

    masks, classes = [], []
    for line in label_text.strip().splitlines():
        vals = line.split()
        if len(vals) < 7:  # a polygon needs >= 3 vertices
            continue
        cls = int(vals[0])
        xy = np.asarray(vals[1:], dtype=np.float64).reshape(-1, 2) * [w, h]
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [np.round(xy).astype(np.int32)], 1)
        masks.append(m.astype(bool))
        classes.append(cls)
    if not masks:
        return np.zeros((0, h, w), bool), np.zeros(0, int)
    return np.stack(masks), np.asarray(classes, dtype=int)


def load_coco_seg(
    name: str = "coco8-seg", splits: tuple[str, ...] = ("train", "val"), min_area: int = 0
) -> list[InstanceSample]:
    """Load an Ultralytics COCO segmentation subset from ``data/ultralytics/<name>``.

    Args:
        name: ``coco8-seg`` or ``coco128-seg``.
        splits: Image folders to read.
        min_area: Drop instances smaller than this many pixels (tiny masks are mostly
            annotation noise for a click-based study).
    """
    root = data_dir("ultralytics", name)
    if not root.is_dir():
        raise FileNotFoundError(f"data/ultralytics/{name} missing - run make data-{PROJECT}")
    out = []
    for split in splits:
        for img_path in sorted((root / "images" / split).glob("*.jpg")):
            img = read_image(img_path)
            lbl = root / "labels" / split / (img_path.stem + ".txt")
            text = lbl.read_text() if lbl.exists() else ""
            masks, classes = polygons_to_masks(text, *img.shape[:2])
            keep = masks.sum(axis=(1, 2)) >= min_area
            out.append(InstanceSample(img_path.stem, img, masks[keep], classes[keep]))
    return out


# =========================================================================== data: DAVIS
def davis_root() -> Path:
    return data_dir("davis", "DAVIS")


def load_davis(seq: str, max_frames: int | None = None) -> tuple[list[np.ndarray], np.ndarray]:
    """Load a DAVIS 2017 480p sequence.

    Returns:
        ``frames`` (list of ``H x W x 3`` uint8 RGB) and ``labels`` (``T x H x W`` uint8,
        0 = background, 1..K = object id; the palette PNG stores the ids directly).
    """
    from PIL import Image

    root = davis_root()
    jpgs = sorted((root / "JPEGImages" / "480p" / seq).glob("*.jpg"))[:max_frames]
    if not jpgs:
        raise FileNotFoundError(f"DAVIS sequence {seq!r} missing - run make data-{PROJECT}")
    frames = [read_image(p) for p in jpgs]
    pngs = [root / "Annotations" / "480p" / seq / (p.stem + ".png") for p in jpgs]
    labels = np.stack([np.array(Image.open(p)) for p in pngs]).astype(np.uint8)
    labels[labels == 255] = 0  # 255 = "void" in some DAVIS releases
    return frames, labels


def object_ids(labels: np.ndarray, frame: int = 0) -> list[int]:
    """Object ids present in a label map (frame ``frame`` of a ``T x H x W`` stack)."""
    lab = labels[frame] if labels.ndim == 3 else labels
    return [int(i) for i in np.unique(lab) if i != 0]


# =============================================================================== metrics
def j_measure(pred: np.ndarray, gt: np.ndarray) -> float:
    """Region similarity J = |P n G| / |P u G| (1.0 when both masks are empty)."""
    pred, gt = np.asarray(pred, bool), np.asarray(gt, bool)
    union = np.logical_or(pred, gt).sum()
    return 1.0 if union == 0 else float(np.logical_and(pred, gt).sum() / union)


def seg2bmap(mask: np.ndarray) -> np.ndarray:
    """One-pixel-wide inner boundary of a binary mask (mask minus its 3x3 erosion)."""
    import cv2

    m = np.asarray(mask, np.uint8)
    if not m.any():
        return np.zeros(m.shape, bool)
    # BORDER_CONSTANT 0: pixels on the image edge that belong to the mask count as boundary
    er = cv2.erode(m, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return (m > 0) & (er == 0)


def boundary_radius(shape: tuple[int, int], bound_th: float = 0.008) -> int:
    """DAVIS boundary tolerance in pixels: ``ceil(bound_th * image diagonal)`` (or absolute)."""
    if bound_th >= 1:
        return int(bound_th)
    return int(np.ceil(bound_th * np.hypot(*shape[:2])))


def f_measure(pred: np.ndarray, gt: np.ndarray, bound_th: float = 0.008) -> float:
    """DAVIS boundary F-measure.

    Boundary pixels of each mask are matched to the other mask's boundary dilated by a disk of
    radius ``ceil(0.008 * diagonal)`` (8 px at 854x480); ``F = 2PR / (P + R)``. Conventions
    for empty masks follow the official davis2017-evaluation code.
    """
    import cv2

    r = boundary_radius(np.shape(gt), bound_th)
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    pb, gb = seg2bmap(pred), seg2bmap(gt)
    n_p, n_g = int(pb.sum()), int(gb.sum())
    if n_p == 0 and n_g == 0:
        return 1.0
    if n_p == 0 or n_g == 0:
        return 0.0  # (P, R) = (1, 0) or (0, 1) -> F = 0
    p_dil = cv2.dilate(pb.astype(np.uint8), disk) > 0
    g_dil = cv2.dilate(gb.astype(np.uint8), disk) > 0
    precision = float((pb & g_dil).sum() / n_p)
    recall = float((gb & p_dil).sum() / n_g)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def jf_statistics(values: np.ndarray, n_bins: int = 4) -> dict[str, float]:
    """DAVIS summary of a per-frame metric: mean, recall (fraction > 0.5) and decay.

    Decay = mean of the first quarter of the frames minus mean of the last quarter (positive
    values mean the tracker degrades over time). Same binning as davis2017-evaluation.
    """
    x = np.asarray(values, dtype=np.float64)
    ids = np.round(np.linspace(1, len(x), n_bins + 1) + 1e-10) - 1
    bins = [x[int(ids[i]) : int(ids[i + 1]) + 1] for i in range(n_bins)]
    decay = float(np.nanmean(bins[0]) - np.nanmean(bins[-1])) if len(x) >= n_bins else 0.0
    return {"mean": float(np.nanmean(x)), "recall": float(np.mean(x > 0.5)), "decay": decay}


def evaluate_vos(
    pred_labels: np.ndarray, gt_labels: np.ndarray, skip_first_last: bool = True
) -> dict[int, dict[str, np.ndarray]]:
    """Per-object, per-frame J and F of predicted vs GT label maps (``T x H x W``).

    As in the official DAVIS protocol the first frame (which carries the prompt) and the last
    frame are excluded when ``skip_first_last``.
    """
    sl = slice(1, -1) if skip_first_last else slice(None)
    out = {}
    for oid in object_ids(gt_labels, 0):
        g, p = gt_labels[sl] == oid, pred_labels[sl] == oid
        out[oid] = {
            "J": np.array([j_measure(a, b) for a, b in zip(p, g)]),
            "F": np.array([f_measure(a, b) for a, b in zip(p, g)]),
        }
    return out


def summarize_vos(per_obj: dict[int, dict[str, np.ndarray]]) -> dict[str, float]:
    """Average J/F statistics over objects (DAVIS reports the object mean)."""
    stats = {}
    for key in ("J", "F"):
        s = [jf_statistics(v[key]) for v in per_obj.values()]
        for k in ("mean", "recall", "decay"):
            stats[f"{key}_{k}"] = float(np.mean([d[k] for d in s]))
    stats["JF_mean"] = 0.5 * (stats["J_mean"] + stats["F_mean"])
    return stats


def mask_iou_matrix(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Pairwise IoU between ``N x H x W`` and ``M x H x W`` boolean masks -> ``N x M``."""
    if len(pred) == 0 or len(gt) == 0:
        return np.zeros((len(pred), len(gt)))
    p = np.asarray(pred, bool).reshape(len(pred), -1).astype(np.float32)
    g = np.asarray(gt, bool).reshape(len(gt), -1).astype(np.float32)
    inter = p @ g.T
    union = p.sum(1)[:, None] + g.sum(1)[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


@dataclass
class Detections:
    """Instance predictions for one image (masks ``N x H x W`` bool, scores, class ids)."""

    masks: np.ndarray
    scores: np.ndarray
    classes: np.ndarray
    boxes: np.ndarray | None = None
    extra: dict = field(default_factory=dict)

    @staticmethod
    def empty(h: int, w: int) -> Detections:
        return Detections(np.zeros((0, h, w), bool), np.zeros(0), np.zeros(0, int))

    @staticmethod
    def concat(items: list[Detections], h: int, w: int) -> Detections:
        items = [d for d in items if len(d.scores)]
        if not items:
            return Detections.empty(h, w)
        return Detections(
            np.concatenate([d.masks for d in items]),
            np.concatenate([d.scores for d in items]),
            np.concatenate([d.classes for d in items]),
        )


def mask_ap(
    preds: list[Detections],
    gts: list[InstanceSample],
    iou_thrs: np.ndarray | None = None,
    max_dets: int = 100,
) -> dict[str, float]:
    """COCO-style mask AP (101-point interpolation, mean over classes present in the GT).

    For every class and IoU threshold, predictions are matched greedily in score order to
    unmatched GT masks of the same class in the same image.

    Returns ``AP`` (mean over IoU 0.50:0.05:0.95), ``AP50``, ``AP75`` and the number of
    GT instances / classes evaluated.
    """
    if iou_thrs is None:
        iou_thrs = np.round(np.arange(0.5, 0.96, 0.05), 2)
    classes = sorted({int(c) for g in gts for c in g.classes})
    ap = np.zeros((len(iou_thrs), len(classes)))
    for ci, c in enumerate(classes):
        n_gt = 0
        per_thr_scores: list[list[float]] = [[] for _ in iou_thrs]
        per_thr_tp: list[list[bool]] = [[] for _ in iou_thrs]
        for det, gt in zip(preds, gts):
            g = gt.masks[gt.classes == c]
            n_gt += len(g)
            sel = np.where(det.classes == c)[0]
            sel = sel[np.argsort(-det.scores[sel], kind="stable")][:max_dets]
            ious = mask_iou_matrix(det.masks[sel], g)
            for ti, thr in enumerate(iou_thrs):
                taken = np.zeros(len(g), bool)
                for k in range(len(sel)):
                    tp = False
                    if len(g):
                        cand = np.where(~taken, ious[k], -1.0)
                        j = int(np.argmax(cand))
                        if cand[j] >= thr:
                            taken[j], tp = True, True
                    per_thr_scores[ti].append(float(det.scores[sel[k]]))
                    per_thr_tp[ti].append(tp)
        for ti in range(len(iou_thrs)):
            prec, rec = precision_recall(
                np.array(per_thr_scores[ti]), np.array(per_thr_tp[ti]), n_gt
            )
            ap[ti, ci] = average_precision(rec, prec, method="coco")
    i50 = int(np.argmin(np.abs(iou_thrs - 0.5)))
    i75 = int(np.argmin(np.abs(iou_thrs - 0.75)))
    return {
        "AP": float(ap.mean()) if ap.size else 0.0,
        "AP50": float(ap[i50].mean()) if ap.size else 0.0,
        "AP75": float(ap[i75].mean()) if ap.size else 0.0,
        "n_gt": int(sum(len(g.classes) for g in gts)),
        "n_classes": len(classes),
    }


# =============================================================================== prompts
def mask_to_box(mask: np.ndarray) -> list[float]:
    """Tight ``[x1, y1, x2, y2]`` box (pixel edges, so a 1-pixel mask is 1 px wide)."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def interior_point(mask: np.ndarray) -> tuple[float, float]:
    """The pixel deepest inside the mask (peak of the distance transform), as ``(x, y)``.

    Unlike the centroid it always lies on the object, even for C- or ring-shaped masks.
    """
    import cv2

    m = np.pad(np.asarray(mask, np.uint8), 1)  # pad so the image border counts as outside
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)[1:-1, 1:-1]
    y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
    return float(x), float(y)


def next_click(pred: np.ndarray, gt: np.ndarray) -> tuple[tuple[float, float], int] | None:
    """Error-driven click of the standard interactive-segmentation protocol.

    Click at the interior point of the larger error region: a positive click (label 1) inside
    a false-negative region or a negative click (label 0) inside a false-positive region.
    Returns ``None`` when the prediction is already perfect.
    """
    import cv2

    pred, gt = np.asarray(pred, bool), np.asarray(gt, bool)
    fn, fp = gt & ~pred, pred & ~gt
    best = None
    for err, label in ((fn, 1), (fp, 0)):
        if not err.any():
            continue
        dist = cv2.distanceTransform(np.pad(err.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
        d = float(dist.max())
        if best is None or d > best[0]:
            y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
            best = (d, (float(x), float(y)), label)
    return None if best is None else (best[1], best[2])


# ================================================================================ models
def _torch():
    import torch

    return torch


def model_label(model_id: str) -> str:
    return MODEL_LABELS.get(model_id, model_id.split("/")[-1])


def count_params(model) -> float:
    """Parameters in millions."""
    return sum(p.numel() for p in model.parameters()) / 1e6


class PromptableSegmenter:
    """Point / box prompted image segmentation with SAM 3 (tracker head) or SAM 2.1.

    Both expose the same interface: ``set_image`` runs the heavy image encoder once,
    ``predict`` runs the light prompt encoder + mask decoder for any number of prompts.
    """

    def __init__(self, model_id: str = SAM3_ID, device: str = "cuda", dtype=None) -> None:
        torch = _torch()
        import transformers as tf

        self.model_id, self.device = model_id, device
        self.dtype = dtype if dtype is not None else torch.bfloat16
        if "sam3" in model_id:
            model_cls, proc_cls = tf.Sam3TrackerModel, tf.Sam3TrackerProcessor
        else:
            model_cls, proc_cls = tf.Sam2Model, tf.Sam2Processor
        self.processor = proc_cls.from_pretrained(model_id)
        self.model = model_cls.from_pretrained(model_id, dtype=self.dtype).to(device).eval()
        self.label = model_label(model_id)
        self._emb = None
        self._sizes = None

    @property
    def n_params_m(self) -> float:
        return count_params(self.model)

    def set_image(self, image: np.ndarray) -> None:
        """Encode an RGB uint8 image (cached for the following ``predict`` calls)."""
        torch = _torch()
        inputs = self.processor(images=image, return_tensors="pt")
        self._sizes = inputs["original_sizes"]
        with torch.inference_mode():
            pix = inputs["pixel_values"].to(self.device, self.dtype)
            self._emb = self.model.get_image_embeddings(pix)

    def predict(
        self,
        points: list | None = None,
        labels: list | None = None,
        boxes: list | None = None,
        mask_logits=None,
        multimask: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, object]:
        """Segment objects in the current image.

        Args:
            points: Per object a list of ``(x, y)`` pixel clicks -> ``[[(x, y), ...], ...]``.
            labels: Per object a list of click labels (1 positive, 0 negative).
            boxes: Per object one ``[x1, y1, x2, y2]`` box.
            mask_logits: Optional low-res mask logits from a previous call (``O x 1 x h x w``),
                fed back as a dense prompt during interactive refinement (256 x 256 for SAM 2.1,
                288 x 288 for SAM 3).
            multimask: Return 3 candidate masks per object (resolves click ambiguity).

        Returns:
            ``masks`` (``O x M x H x W`` bool), ``scores`` (``O x M`` predicted IoU) and the
            low-res logits (``O x M x h x w`` torch tensor) for further refinement.
        """
        torch = _torch()
        kw = {}
        if points is not None:
            kw["input_points"] = [[[list(map(float, p)) for p in obj] for obj in points]]
            kw["input_labels"] = [[list(map(int, lab)) for lab in labels]]
        if boxes is not None:
            kw["input_boxes"] = [[list(map(float, b)) for b in boxes]]
        prompt = self.processor(original_sizes=self._sizes, return_tensors="pt", **kw)
        args = {
            k: v.to(self.device)
            for k, v in prompt.items()
            if k in ("input_points", "input_labels", "input_boxes")
        }
        if "input_points" in args:
            args["input_points"] = args["input_points"].to(self.dtype)
        if "input_boxes" in args:
            args["input_boxes"] = args["input_boxes"].to(self.dtype)
        if mask_logits is not None:
            args["input_masks"] = mask_logits.to(self.device, self.dtype)
        with torch.inference_mode():
            out = self.model(image_embeddings=self._emb, multimask_output=multimask, **args)
        logits = out.pred_masks[0]  # O x M x h x w (low-res)
        masks = self.processor.post_process_masks(out.pred_masks.float().cpu(), self._sizes)[0]
        return masks.numpy().astype(bool), out.iou_scores[0].float().cpu().numpy(), logits

    def interactive(self, gt: np.ndarray, n_clicks: int = 5, return_steps: bool = False):
        """Simulated interactive session: IoU after each error-driven click (one object).

        Click 1 is the interior point of the GT mask; later clicks go to the centre of the
        largest error region. The previous low-res logits are fed back as a mask prompt and
        only the first click uses multimask output (best mask by predicted IoU).

        Returns the IoU after each click, plus (with ``return_steps``) a list of
        ``(mask, points, labels)`` per click for visualisation.
        """
        pts, labs = [interior_point(gt)], [1]
        ious, steps, logits = [], [], None
        pred = None
        for k in range(n_clicks):
            if k > 0:
                click = next_click(pred, gt)
                if click is None:  # already perfect: IoU stays 1
                    ious.append(1.0)
                    steps.append((pred, list(pts), list(labs)))
                    continue
                pts.append(click[0])
                labs.append(click[1])
            masks, scores, lg = self.predict([pts], [labs], mask_logits=logits, multimask=(k == 0))
            best = int(np.argmax(scores[0]))
            pred = masks[0, best]
            logits = lg[:, best : best + 1]
            ious.append(j_measure(pred, gt))
            steps.append((pred, list(pts), list(labs)))
        return (ious, steps) if return_steps else ious


class ConceptSegmenter:
    """SAM 3 promptable concept segmentation: text (and/or exemplar boxes) -> all instances."""

    def __init__(self, device: str = "cuda", dtype=None) -> None:
        torch = _torch()
        from transformers import Sam3Model, Sam3Processor

        self.device = device
        self.dtype = dtype if dtype is not None else torch.bfloat16
        self.processor = Sam3Processor.from_pretrained(SAM3_ID)
        self.model = Sam3Model.from_pretrained(SAM3_ID, dtype=self.dtype).to(device).eval()
        self._vision = None
        self._sizes = None

    @property
    def n_params_m(self) -> float:
        return count_params(self.model)

    def set_image(self, image: np.ndarray) -> None:
        """Run the vision backbone once; text prompts then only run the DETR-style heads."""
        torch = _torch()
        inputs = self.processor(images=image, return_tensors="pt")
        self._sizes = inputs["original_sizes"].tolist()
        with torch.inference_mode():
            pix = inputs["pixel_values"].to(self.device, self.dtype)
            self._vision = self.model.get_vision_features(pixel_values=pix)

    def segment(
        self,
        text: str | None,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        boxes: list | None = None,
        box_labels: list | None = None,
    ) -> dict[str, np.ndarray]:
        """Masks/boxes/scores of every instance of ``text`` in the current image.

        ``boxes`` / ``box_labels`` add visual exemplars (1 = "more like this", 0 = "not this").
        """
        torch = _torch()
        kw = {}
        if text is not None:
            kw["text"] = text
        if boxes is not None:
            kw["input_boxes"] = [boxes]
            kw["input_boxes_labels"] = [box_labels or [1] * len(boxes)]
            kw["original_sizes"] = self._sizes
        prompt = self.processor(return_tensors="pt", **kw)
        args = {k: v.to(self.device) for k, v in prompt.items() if k != "original_sizes"}
        if "input_boxes" in args:
            args["input_boxes"] = args["input_boxes"].to(self.dtype)
        with torch.inference_mode():
            out = self.model(vision_embeds=self._vision, **args)
        res = self.processor.post_process_instance_segmentation(
            out, threshold=threshold, mask_threshold=mask_threshold, target_sizes=self._sizes
        )[0]
        return {
            "masks": res["masks"].cpu().numpy().astype(bool),
            "boxes": res["boxes"].float().cpu().numpy(),
            "scores": res["scores"].float().cpu().numpy(),
            "semantic": out.semantic_seg[0, 0].float().cpu().numpy()
            if getattr(out, "semantic_seg", None) is not None
            else None,
        }


class GroundingDino:
    """Grounding DINO (open-vocabulary detector): text -> boxes."""

    def __init__(self, model_id: str = GDINO_ID, device: str = "cuda") -> None:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        # fp32: the tiny model is small and its deformable attention is precision-sensitive
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()

    @property
    def n_params_m(self) -> float:
        return count_params(self.model)

    def detect(
        self,
        image: np.ndarray,
        text: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> dict[str, np.ndarray]:
        """Boxes (``x1 y1 x2 y2`` pixels) and scores for one noun phrase."""
        torch = _torch()
        phrase = text.lower().strip().rstrip(".") + "."  # GDINO expects '.'-terminated phrases
        inputs = self.processor(images=image, text=phrase, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self.model(**inputs)
        res = self.processor.post_process_grounded_object_detection(
            out,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.shape[:2]],
        )[0]
        return {
            "boxes": res["boxes"].float().cpu().numpy(),
            "scores": res["scores"].float().cpu().numpy(),
        }


class GroundedSam:
    """The "Grounded-SAM" pattern: Grounding DINO boxes become SAM 2.1 box prompts."""

    def __init__(self, sam_id: str = SAM2_IDS["small"], device: str = "cuda") -> None:
        self.detector = GroundingDino(device=device)
        self.sam = PromptableSegmenter(sam_id, device=device)

    def set_image(self, image: np.ndarray) -> None:
        self.image = image
        self.sam.set_image(image)

    def segment(
        self, text: str, box_threshold: float = 0.35, text_threshold: float = 0.25
    ) -> dict[str, np.ndarray]:
        det = self.detector.detect(self.image, text, box_threshold, text_threshold)
        h, w = self.image.shape[:2]
        if len(det["scores"]) == 0:
            return {"masks": np.zeros((0, h, w), bool), **det}
        masks, _, _ = self.sam.predict(boxes=det["boxes"].tolist(), multimask=False)
        return {"masks": masks[:, 0], **det}


# ------------------------------------------------------------------------------ video
def _video_classes(model_id: str):
    import transformers as tf

    if "sam3" in model_id:
        return tf.Sam3TrackerVideoModel, tf.Sam3TrackerVideoProcessor
    return tf.Sam2VideoModel, tf.Sam2VideoProcessor


class VideoTracker:
    """Semi-supervised video object segmentation with SAM 3 (tracker) or SAM 2.1.

    Prompts (one mask, box or click per object) are given on frame 0 and propagated through
    the memory bank to the remaining frames.
    """

    def __init__(self, model_id: str = SAM3_ID, device: str = "cuda", dtype=None) -> None:
        torch = _torch()
        model_cls, proc_cls = _video_classes(model_id)
        self.model_id, self.device = model_id, device
        self.dtype = dtype if dtype is not None else torch.bfloat16
        self.processor = proc_cls.from_pretrained(model_id)
        self.model = model_cls.from_pretrained(model_id, dtype=self.dtype).to(device).eval()
        self.label = model_label(model_id)

    @property
    def n_params_m(self) -> float:
        return count_params(self.model)

    def track(
        self, frames: list[np.ndarray], first_labels: np.ndarray, prompt: str = "mask"
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Propagate frame-0 prompts derived from the GT label map ``first_labels``.

        Args:
            frames: RGB uint8 frames.
            first_labels: ``H x W`` object-id map of frame 0 (the prompt source).
            prompt: ``"mask"`` (full GT mask, the DAVIS semi-supervised protocol), ``"box"``
                (tight GT box) or ``"click"`` (one positive click at the mask's interior point).

        Returns:
            Predicted ``T x H x W`` uint8 label maps and timing info (``ms_per_frame``, ``fps``).
        """
        torch = _torch()
        h, w = first_labels.shape
        ids = object_ids(first_labels)
        gt_masks = [first_labels == i for i in ids]
        session = self.processor.init_video_session(
            video=frames,
            inference_device=self.device,
            video_storage_device="cpu",
            processing_device="cpu",
            dtype=self.dtype,
        )
        if prompt == "mask":
            kw = {"input_masks": [m.astype(np.float32) for m in gt_masks]}
        elif prompt == "box":
            kw = {"input_boxes": [[mask_to_box(m) for m in gt_masks]]}
        elif prompt == "click":
            kw = {
                "input_points": [[[list(interior_point(m))] for m in gt_masks]],
                "input_labels": [[[1] for _ in gt_masks]],
            }
        else:
            raise ValueError(f"unknown prompt type: {prompt}")
        self.processor.add_inputs_to_inference_session(
            inference_session=session, frame_idx=0, obj_ids=list(ids), **kw
        )
        logits = np.full((len(frames), len(ids), h, w), -np.inf, dtype=np.float32)
        sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
        sync()
        t0 = time.perf_counter()
        with torch.inference_mode():
            self.model(inference_session=session, frame_idx=0)
            for out in self.model.propagate_in_video_iterator(session):
                m = self.processor.post_process_masks(
                    [out.pred_masks.float()], original_sizes=[[h, w]], binarize=False
                )[0]  # n_obj x 1 x H x W logits
                order = out.object_ids if out.object_ids is not None else ids
                for k, oid in enumerate(order):  # map output rows to our object order
                    logits[out.frame_idx, ids.index(int(oid))] = m[k, 0].cpu().numpy()
        sync()
        dt = time.perf_counter() - t0
        labels = logits_to_labels(logits, ids)
        return labels, {"ms_per_frame": 1000 * dt / len(frames), "fps": len(frames) / dt}


def logits_to_labels(logits: np.ndarray, ids: list[int]) -> np.ndarray:
    """``T x O x H x W`` mask logits -> ``T x H x W`` label map (argmax where logit > 0)."""
    if len(ids) == 0:
        return np.zeros((logits.shape[0], *logits.shape[2:]), np.uint8)
    best = np.argmax(logits, axis=1)
    fg = np.max(logits, axis=1) > 0
    lut = np.asarray(ids, dtype=np.uint8)
    return np.where(fg, lut[best], 0).astype(np.uint8)


class TextVideoTracker:
    """SAM 3 concept tracking in video: a noun phrase -> masklets of *every* matching object."""

    def __init__(self, device: str = "cuda", dtype=None) -> None:
        torch = _torch()
        from transformers import Sam3VideoModel, Sam3VideoProcessor

        self.device = device
        self.dtype = dtype if dtype is not None else torch.bfloat16
        self.processor = Sam3VideoProcessor.from_pretrained(SAM3_ID)
        self.model = Sam3VideoModel.from_pretrained(SAM3_ID, dtype=self.dtype).to(device).eval()

    @property
    def n_params_m(self) -> float:
        return count_params(self.model)

    def track(self, frames: list[np.ndarray], text: str) -> tuple[dict[int, np.ndarray], dict]:
        """Returns ``{track_id: T x H x W bool}`` and timing/score info."""
        torch = _torch()
        h, w = frames[0].shape[:2]
        session = self.processor.init_video_session(
            video=frames,
            inference_device=self.device,
            video_storage_device="cpu",
            processing_device="cpu",
            dtype=self.dtype,
        )
        session = self.processor.add_text_prompt(session, text)
        tracks: dict[int, np.ndarray] = {}
        scores: dict[int, list[float]] = {}
        sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
        sync()
        t0 = time.perf_counter()
        with torch.inference_mode():
            for out in self.model.propagate_in_video_iterator(
                inference_session=session, max_frame_num_to_track=len(frames)
            ):
                res = self.processor.postprocess_outputs(session, out)
                for oid, m, s in zip(
                    res["object_ids"].tolist(), res["masks"].cpu().numpy(), res["scores"].tolist()
                ):
                    if oid not in tracks:
                        tracks[oid] = np.zeros((len(frames), h, w), bool)
                    tracks[oid][out.frame_idx] = m
                    scores.setdefault(oid, []).append(float(s))
        sync()
        dt = time.perf_counter() - t0
        info = {
            "ms_per_frame": 1000 * dt / len(frames),
            "fps": len(frames) / dt,
            "mean_score": {k: float(np.mean(v)) for k, v in scores.items()},
        }
        return tracks, info


def match_tracks_to_gt(
    tracks: dict[int, np.ndarray], gt_labels: np.ndarray
) -> dict[int, int | None]:
    """Hungarian assignment of predicted masklets to GT objects by mean per-frame IoU."""
    from scipy.optimize import linear_sum_assignment

    gt_ids = object_ids(gt_labels)
    tids = list(tracks)
    if not tids:
        return {g: None for g in gt_ids}
    cost = np.zeros((len(gt_ids), len(tids)))
    for i, g in enumerate(gt_ids):
        gm = gt_labels == g
        for j, t in enumerate(tids):
            cost[i, j] = -np.mean([j_measure(a, b) for a, b in zip(tracks[t], gm)])
    rows, cols = linear_sum_assignment(cost)
    out: dict[int, int | None] = {g: None for g in gt_ids}
    for r, c in zip(rows, cols):
        if -cost[r, c] > 0:
            out[gt_ids[r]] = tids[c]
    return out
