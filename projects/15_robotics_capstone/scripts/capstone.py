"""Open-vocabulary RGB-D perception for grasping: the robotics capstone pipeline (project 15).

``text query -> 2D boxes -> instance masks -> object point cloud -> 3D box -> grasp pose``

* **Detection** (open vocabulary): Grounding DINO-T or OWLv2-B/16 (Hugging Face transformers).
* **Segmentation** (box prompts): SAM 3 (``Sam3TrackerModel``) or SAM 2.1-S (``Sam2Model``);
  SAM 3 can also segment straight from text (``Sam3Model``), skipping the detector.
* **Depth**: the RGB-D sensor (TUM fr1/desk, Kinect v1) or monocular metric depth
  (Depth Anything V2 Metric-Indoor-Small).
* **3D + grasps**: :mod:`grasp3d` (NumPy only, shared with the ROS 2 node).

The notebooks, the exercise solutions and ``ros2/perception_worker.py`` all import this module.

Usage::

    import sys; sys.path.insert(0, "projects/15_robotics_capstone/scripts")
    import capstone as cap
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import grasp3d as g3
import numpy as np

from cvlab.geometry import invert_T
from cvlab.io import data_dir, outputs_dir, project_dir, runs_dir
from cvlab.tum import TUM_INTRINSICS, TumSequence, load_sequence, read_trajectory

PROJECT = "15_robotics_capstone"
SEQUENCE = "rgbd_dataset_freiburg1_desk"

# Hugging Face model ids pinned to the revisions used for every number in the README.
MODELS = {
    "gdino": ("IDEA-Research/grounding-dino-tiny", "a2bb814dd30d776dcf7e30523b00659f4f141c71"),
    "owlv2": ("google/owlv2-base-patch16-ensemble", "cfd3195ba4ea9592eec887ded089f4c08eff231d"),
    "sam3": ("facebook/sam3", "3c879f39826c281e95690f02c7821c4de09afae7"),
    "sam2.1-s": ("facebook/sam2.1-hiera-small", "ee5bba1d82bb8749febdf90f45e84b687142ba03"),
    "da2-metric": (
        "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
        "8078d68a9c75a972131914f6afd0c1723be0da7f",
    ),
}
MODEL_LABELS = {
    "gdino": "Grounding DINO-T",
    "owlv2": "OWLv2-B/16",
    "sam3": "SAM 3",
    "sam2.1-s": "SAM 2.1-S",
    "sam3-text": "SAM 3 (text)",
    "da2-metric": "DA-V2 Metric-Indoor-S",
}

# Objects on the TUM desks that a robot could plausibly be asked to fetch.
QUERIES = [
    "keyboard",
    "book",
    "game controller",
    "telephone",
    "computer mouse",
    "cup",
    "cereal box",
]

# Camera models compared in notebook 02 (TUM file-format page: fr1 RGB calibration with
# distortion, or the ROS default pinhole recommended for the pre-registered depth).
_FR1, _ROS = TUM_INTRINSICS["fr1"], TUM_INTRINSICS["ros_default"]


def _K(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:  # noqa: N802
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


CAMERAS = {
    "fr1 + distortion": (_K(*_FR1[0]), np.array(_FR1[1])),
    "fr1, no distortion": (_K(*_FR1[0]), None),
    "ROS default": (_K(*_ROS[0]), None),
}
DEFAULT_CAMERA = "fr1 + distortion"


# ================================================================================= data
def results_dir() -> Path:
    """``projects/15_robotics_capstone/results`` (small committed JSON/CSV metrics)."""
    path = project_dir(PROJECT) / "results"
    path.mkdir(parents=True, exist_ok=True)
    return path


def capstone_outputs() -> Path:
    """``outputs/15_robotics_capstone`` (gitignored: .rrd recordings, .ply clouds)."""
    return outputs_dir(PROJECT, create=True)


def sequence_dir() -> Path:
    path = data_dir("tum", "fr1_desk", SEQUENCE)
    if not (path / "rgb.txt").exists():
        raise FileNotFoundError(f"{SEQUENCE} missing - run `make data-{PROJECT}`")
    return path


@dataclass
class Frames:
    """TUM fr1/desk with full-rate ground truth for pose interpolation."""

    seq: TumSequence
    depth_ts: np.ndarray  # timestamp of the associated depth image of every frame
    gt_ts: np.ndarray  # 100 Hz motion-capture timestamps
    gt_poses: np.ndarray  # matching camera-to-world poses

    def __len__(self) -> int:
        return len(self.seq)

    def rgb(self, i: int) -> np.ndarray:
        return self.seq.rgb(i)

    def depth(self, i: int) -> np.ndarray:
        return self.seq.depth(i)

    def pose(self, i: int, at: str = "depth") -> np.ndarray:
        """Camera-to-world pose ``T_wc``, interpolated at the RGB or the depth timestamp."""
        t = self.depth_ts[i] if at == "depth" else self.seq.timestamps[i]
        return g3.interpolate_pose(self.gt_ts, self.gt_poses, float(t))


def load_frames() -> Frames:
    root = sequence_dir()
    seq = load_sequence(root)
    depth_ts = np.array([float(Path(p).stem) for p in seq.depth_paths])
    gt_ts, gt_poses = read_trajectory(root / "groundtruth.txt")
    return Frames(seq, depth_ts, gt_ts, gt_poses)


# ================================================================================ models
def _torch():
    import torch

    return torch


def _sync() -> None:
    torch = _torch()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def count_params_m(model) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


@dataclass
class Detections:
    """2D detections: ``boxes`` (N x 4, x1 y1 x2 y2 px), ``scores`` (N,), ``labels`` (N,)."""

    boxes: np.ndarray
    scores: np.ndarray
    labels: list[str]
    source: str = ""

    def __len__(self) -> int:
        return len(self.scores)

    def select(self, keep) -> Detections:
        keep = np.asarray(keep)
        if keep.dtype == bool:
            keep = np.nonzero(keep)[0]
        return Detections(
            self.boxes[keep], self.scores[keep], [self.labels[i] for i in keep], self.source
        )

    @staticmethod
    def empty(source: str = "") -> Detections:
        return Detections(np.zeros((0, 4)), np.zeros(0), [], source)


def match_query(text: str, queries: list[str]) -> str | None:
    """Map a (possibly partial) phrase returned by Grounding DINO back to one query."""
    text = text.strip().lower()
    if text in queries:
        return text
    words = set(text.split())
    best, overlap = None, 0
    for q in queries:
        o = len(words & set(q.split()))
        if o > overlap:
            best, overlap = q, o
    return best


def postprocess(
    dets: Detections, image_shape: tuple[int, int], iou: float = 0.5, max_area_frac: float = 0.5
) -> Detections:
    """Per-label NMS and removal of boxes covering more than ``max_area_frac`` of the image."""
    if len(dets) == 0:
        return dets
    torch = _torch()
    from torchvision.ops import batched_nms

    h, w = image_shape
    area = (dets.boxes[:, 2] - dets.boxes[:, 0]) * (dets.boxes[:, 3] - dets.boxes[:, 1])
    dets = dets.select(area <= max_area_frac * h * w)
    if len(dets) == 0:
        return dets
    ids = {q: i for i, q in enumerate(sorted(set(dets.labels)))}
    keep = batched_nms(
        torch.as_tensor(dets.boxes, dtype=torch.float32),
        torch.as_tensor(dets.scores, dtype=torch.float32),
        torch.as_tensor([ids[x] for x in dets.labels]),
        iou,
    )
    return dets.select(np.sort(keep.numpy()))


class GroundingDinoDetector:
    """Grounding DINO-T: all queries in one caption ``"keyboard. book. cup."``."""

    key = "gdino"

    def __init__(
        self, device: str = "cuda", box_threshold: float = 0.35, text_threshold: float = 0.25
    ) -> None:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        repo, rev = MODELS[self.key]
        self.device, self.box_threshold, self.text_threshold = device, box_threshold, text_threshold
        self.processor = AutoProcessor.from_pretrained(repo, revision=rev)
        # fp32: the tiny model is small and its deformable attention is precision-sensitive
        self.model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(repo, revision=rev)
            .to(device)
            .eval()
        )

    def detect(self, rgb: np.ndarray, queries: list[str]) -> Detections:
        torch = _torch()
        caption = ". ".join(q.lower() for q in queries) + "."
        inputs = self.processor(images=rgb, text=caption, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self.model(**inputs)
        res = self.processor.post_process_grounded_object_detection(
            out,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[rgb.shape[:2]],
        )[0]
        labels = [match_query(t, queries) for t in res["text_labels"]]
        keep = [i for i, lab in enumerate(labels) if lab is not None]
        dets = Detections(
            res["boxes"].float().cpu().numpy()[keep],
            res["scores"].float().cpu().numpy()[keep],
            [labels[i] for i in keep],
            self.key,
        )
        return postprocess(dets, rgb.shape[:2])


class Owlv2Detector:
    """OWLv2-B/16: image patches scored against each text embedding ("a photo of a {q}")."""

    key = "owlv2"

    def __init__(
        self, device: str = "cuda", threshold: float = 0.2, template: str = "a photo of a {}"
    ) -> None:
        torch = _torch()
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        repo, rev = MODELS[self.key]
        self.device, self.threshold, self.template = device, threshold, template
        self.processor = Owlv2Processor.from_pretrained(repo, revision=rev)
        self.model = (
            Owlv2ForObjectDetection.from_pretrained(repo, revision=rev, dtype=torch.float16)
            .to(device)
            .eval()
        )

    def detect(self, rgb: np.ndarray, queries: list[str]) -> Detections:
        torch = _torch()
        texts = [[self.template.format(q) for q in queries]]
        inputs = self.processor(text=texts, images=rgb, return_tensors="pt").to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].half()
        with torch.inference_mode():
            out = self.model(**inputs)
        # OWLv2 pads the image to a square at the bottom/right: boxes live in that square
        side = max(rgb.shape[:2])
        res = self.processor.post_process_grounded_object_detection(
            out, threshold=self.threshold, target_sizes=[(side, side)]
        )[0]
        boxes = res["boxes"].float().cpu().numpy()
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, rgb.shape[1])
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, rgb.shape[0])
        dets = Detections(
            boxes,
            res["scores"].float().cpu().numpy(),
            [queries[i] for i in res["labels"].tolist()],
            self.key,
        )
        return postprocess(dets, rgb.shape[:2])


class BoxSegmenter:
    """Box-prompted masks with SAM 3 (tracker / SAM-2-style head) or SAM 2.1-S.

    ``set_image`` runs the heavy image encoder once; ``segment`` decodes one mask per box.
    """

    def __init__(self, key: str = "sam3", device: str = "cuda") -> None:
        torch = _torch()
        import transformers as tf

        self.key, self.device, self.dtype = key, device, torch.bfloat16
        repo, rev = MODELS[key]
        if key == "sam3":
            model_cls, proc_cls = tf.Sam3TrackerModel, tf.Sam3TrackerProcessor
        else:
            model_cls, proc_cls = tf.Sam2Model, tf.Sam2Processor
        self.processor = proc_cls.from_pretrained(repo, revision=rev)
        self.model = (
            model_cls.from_pretrained(repo, revision=rev, dtype=self.dtype).to(device).eval()
        )
        self._emb = self._sizes = None

    def set_image(self, rgb: np.ndarray) -> None:
        torch = _torch()
        inputs = self.processor(images=rgb, return_tensors="pt")
        self._sizes = inputs["original_sizes"]
        with torch.inference_mode():
            self._emb = self.model.get_image_embeddings(
                inputs["pixel_values"].to(self.device, self.dtype)
            )

    def segment(self, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Masks (N x H x W bool) and predicted IoU (N,) for N boxes in the current image."""
        torch = _torch()
        h, w = (int(x) for x in self._sizes[0])
        if len(boxes) == 0:
            return np.zeros((0, h, w), bool), np.zeros(0)
        prompt = self.processor(
            input_boxes=[[list(map(float, b)) for b in boxes]],
            original_sizes=self._sizes,
            return_tensors="pt",
        )
        with torch.inference_mode():
            out = self.model(
                image_embeddings=self._emb,
                multimask_output=False,
                input_boxes=prompt["input_boxes"].to(self.device, self.dtype),
            )
        masks = self.processor.post_process_masks(out.pred_masks.float().cpu(), self._sizes)[0]
        return masks[:, 0].numpy().astype(bool), out.iou_scores[0, :, 0].float().cpu().numpy()


class Sam3TextSegmenter:
    """SAM 3 promptable concept segmentation: a noun phrase -> every matching instance."""

    key = "sam3-text"

    def __init__(self, device: str = "cuda", threshold: float = 0.5) -> None:
        torch = _torch()
        from transformers import Sam3Model, Sam3Processor

        repo, rev = MODELS["sam3"]
        self.device, self.dtype, self.threshold = device, torch.bfloat16, threshold
        self.processor = Sam3Processor.from_pretrained(repo, revision=rev)
        self.model = (
            Sam3Model.from_pretrained(repo, revision=rev, dtype=self.dtype).to(device).eval()
        )
        self._vision = self._sizes = None

    def set_image(self, rgb: np.ndarray) -> None:
        torch = _torch()
        inputs = self.processor(images=rgb, return_tensors="pt")
        self._sizes = inputs["original_sizes"].tolist()
        with torch.inference_mode():
            self._vision = self.model.get_vision_features(
                pixel_values=inputs["pixel_values"].to(self.device, self.dtype)
            )

    def segment(self, queries: list[str]) -> tuple[Detections, np.ndarray]:
        torch = _torch()
        h, w = self._sizes[0]
        boxes, scores, labels = [np.zeros((0, 4))], [np.zeros(0)], []
        masks = [np.zeros((0, h, w), bool)]
        for q in queries:
            t = self.processor(text=q, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                out = self.model(vision_embeds=self._vision, **t)
            r = self.processor.post_process_instance_segmentation(
                out, threshold=self.threshold, mask_threshold=0.5, target_sizes=self._sizes
            )[0]
            if len(r["scores"]) == 0:  # no instance: the empty mask tensor is not resized
                continue
            boxes.append(r["boxes"].float().cpu().numpy().reshape(-1, 4))
            scores.append(r["scores"].float().cpu().numpy())
            masks.append(r["masks"].cpu().numpy().astype(bool).reshape(-1, h, w))
            labels += [q] * len(r["scores"])
        dets = Detections(np.concatenate(boxes), np.concatenate(scores), labels, self.key)
        return dets, np.concatenate(masks)


class MetricDepth:
    """Depth Anything V2 Metric-Indoor-Small: one RGB image -> depth in metres."""

    key = "da2-metric"

    def __init__(self, device: str = "cuda") -> None:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        repo, rev = MODELS[self.key]
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(repo, revision=rev)
        self.model = (
            AutoModelForDepthEstimation.from_pretrained(repo, revision=rev).to(device).eval()
        )

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        torch = _torch()
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self.model(**inputs)
        d = self.processor.post_process_depth_estimation(out, target_sizes=[rgb.shape[:2]])[0]
        return d["predicted_depth"].float().cpu().numpy()


# ============================================================================ 3D objects
@dataclass
class ObjectResult:
    """Everything the pipeline knows about one detected object (camera frame)."""

    label: str
    score: float
    box2d: np.ndarray
    mask: np.ndarray
    cloud: g3.ObjectCloud | None = None
    plane: g3.Plane | None = None
    box: g3.Box3D | None = None
    grasps: list[g3.Grasp] = field(default_factory=list)
    status: str = "ok"

    @property
    def best_grasp(self) -> g3.Grasp | None:
        return self.grasps[0] if self.grasps and self.grasps[0].feasible else None

    @property
    def centroid(self) -> np.ndarray | None:
        return None if self.cloud is None else self.cloud.centroid


def scene_cloud(
    depth: np.ndarray, rays: np.ndarray, stride: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """Whole-frame point cloud (for collision checks) and its pixel indices."""
    return g3.backproject(depth, rays, stride=stride)


def estimate_objects(
    rgb: np.ndarray,
    depth: np.ndarray,
    dets: Detections,
    masks: np.ndarray,
    rays: np.ndarray,
    gripper: g3.Gripper | None = None,
    grasps: bool = True,
    timings: dict | None = None,
    min_mask_px: int = 150,
) -> list[ObjectResult]:
    """CPU half of the pipeline: masks + depth -> clouds -> support planes -> boxes -> grasps."""
    t = timings if timings is not None else {}
    scene, scene_pix = scene_cloud(depth, rays) if grasps else (None, None)
    out = []
    for box2d, score, label, mask in zip(dets.boxes, dets.scores, dets.labels, masks):
        res = ObjectResult(label, float(score), np.asarray(box2d), mask)
        out.append(res)
        if mask.sum() < min_mask_px:
            res.status = "mask too small"
            continue
        t0 = time.perf_counter()
        res.cloud = g3.object_cloud(depth, mask, rays, rgb)
        t1 = time.perf_counter()
        t["cloud_ms"] = t.get("cloud_ms", 0.0) + (t1 - t0) * 1e3
        if res.cloud is None:
            res.status = "too few depth points"
            continue
        res.plane = g3.support_plane(depth, rays, box2d, mask)
        if res.plane is not None and np.median(res.plane.height(res.cloud.points)) > 0.0:
            res.box = g3.table_box(res.cloud.points, res.plane)
        else:
            res.plane = None
            res.box = g3.pca_box(res.cloud.points)
            res.status = "no support plane (PCA box)"
        t2 = time.perf_counter()
        t["box_ms"] = t.get("box_ms", 0.0) + (t2 - t1) * 1e3
        if grasps and res.plane is not None:
            near = np.linalg.norm(scene - res.box.center, axis=1) < 0.3
            keep = near & ~g3.dilate_mask(mask, 5)[scene_pix[:, 0], scene_pix[:, 1]]
            res.grasps = g3.top_down_grasps(
                res.cloud.points, res.box, res.plane, scene[keep], gripper
            )
            t["grasp_ms"] = t.get("grasp_ms", 0.0) + (time.perf_counter() - t2) * 1e3
    return out


class Pipeline:
    """End-to-end pipeline with per-stage timing (GPU stages are CUDA-synchronised).

    ``detector`` / ``segmenter`` are model instances (see :meth:`create` to build them).
    """

    def __init__(
        self,
        detector,
        segmenter: BoxSegmenter,
        camera: str = DEFAULT_CAMERA,
        gripper: g3.Gripper | None = None,
        size: tuple[int, int] = (640, 480),
    ) -> None:
        self.detector, self.segmenter = detector, segmenter
        self.K, self.dist = CAMERAS[camera]
        self.rays = g3.pixel_rays(self.K, self.dist, *size)
        self.gripper = gripper or g3.Gripper()

    @classmethod
    def create(
        cls, detector: str = "gdino", segmenter: str = "sam3", device: str = "cuda", **kw
    ) -> Pipeline:
        det = (GroundingDinoDetector if detector == "gdino" else Owlv2Detector)(device)
        return cls(det, BoxSegmenter(segmenter, device), **kw)

    def __call__(
        self, rgb: np.ndarray, depth: np.ndarray, queries: list[str]
    ) -> tuple[list[ObjectResult], dict]:
        t: dict = {}
        t0 = time.perf_counter()
        dets = self.detector.detect(rgb, queries)
        _sync()
        t1 = time.perf_counter()
        self.segmenter.set_image(rgb)
        masks, _ = self.segmenter.segment(dets.boxes)
        _sync()
        t2 = time.perf_counter()
        t.update(detect_ms=(t1 - t0) * 1e3, segment_ms=(t2 - t1) * 1e3)
        results = estimate_objects(rgb, depth, dets, masks, self.rays, self.gripper, timings=t)
        t["total_ms"] = (time.perf_counter() - t0) * 1e3
        return results, t


class VramMeter:
    """Peak CUDA memory of a block *above* what was already allocated when it started.

    ``with VramMeter() as m: ...`` -> ``m.delta_mb`` (this block alone) and ``m.peak_mb``
    (absolute peak, including tensors that were already resident).
    """

    def __enter__(self) -> VramMeter:
        torch = _torch()
        self.base = 0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.base = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(self, *exc: object) -> None:
        torch = _torch()
        self.delta_mb = self.peak_mb = 0.0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            self.peak_mb, self.delta_mb = peak / 2**20, (peak - self.base) / 2**20


def free(*objs) -> None:
    """Drop model references and return their CUDA memory to the allocator."""
    import gc

    for o in objs:
        del o
    gc.collect()
    torch = _torch()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =========================================================================== world frame
def to_world(T_wc: np.ndarray, results: list[ObjectResult]) -> list[dict]:
    """World-frame summary (centroid, box, best grasp) of every object with a 3D estimate."""
    out = []
    for r in results:
        if r.cloud is None or r.box is None:
            continue
        c_w = T_wc[:3, :3] @ r.centroid + T_wc[:3, 3]
        b_w = r.box.transformed(T_wc)
        g = r.best_grasp
        out.append(
            {
                "label": r.label,
                "score": r.score,
                "centroid_w": c_w,
                "box_center_w": b_w.center,
                "box_R_w": b_w.R,
                "extents": b_w.extents,
                "box_kind": r.box.kind,
                "grasp_T_w": None if g is None else T_wc @ g.T,
                "grasp_width": None if g is None else g.width,
                "n_points": len(r.cloud.points),
            }
        )
    return out


def build_object_map(
    observations: list[dict], key: str = "box_center_w", gate: float = 0.15
) -> list[dict]:
    """Greedy data association of per-frame observations into object landmarks.

    ``observations``: dicts with ``frame``, ``label`` and a world position under ``key``.
    An observation joins the closest landmark *of the same label* whose running median is
    within ``gate`` metres (each landmark takes at most one observation per frame);
    otherwise it starts a new landmark.
    """
    landmarks: list[dict] = []
    by_frame: dict[int, list[dict]] = {}
    for o in observations:
        by_frame.setdefault(o["frame"], []).append(o)
    for f in sorted(by_frame):
        pairs = []
        for oi, o in enumerate(by_frame[f]):
            for li, lm in enumerate(landmarks):
                if lm["label"] == o["label"]:
                    d = float(np.linalg.norm(o[key] - lm["median"]))
                    if d < gate:
                        pairs.append((d, oi, li))
        used_o, used_l = set(), set()
        for _, oi, li in sorted(pairs):
            if oi in used_o or li in used_l:
                continue
            used_o.add(oi)
            used_l.add(li)
            lm = landmarks[li]
            lm["obs"].append(by_frame[f][oi])
            lm["median"] = np.median([x[key] for x in lm["obs"]], axis=0)
        for oi, o in enumerate(by_frame[f]):
            if oi not in used_o:
                landmarks.append({"label": o["label"], "obs": [o], "median": np.asarray(o[key])})
    return landmarks


def landmark_stats(
    landmarks: list[dict], key: str = "box_center_w", min_obs: int = 5
) -> list[dict]:
    """Spread of each landmark's observations around their median position (metres)."""
    rows = []
    for i, lm in enumerate(landmarks):
        if len(lm["obs"]) < min_obs:
            continue
        P = np.array([o[key] for o in lm["obs"]])
        med = np.median(P, axis=0)
        dist = np.linalg.norm(P - med, axis=1)
        rows.append(
            {
                "landmark": i,
                "label": lm["label"],
                "n_obs": len(P),
                "median_dev_cm": 100 * float(np.median(dist)),
                "rms_dev_cm": 100 * float(np.sqrt(np.mean(np.sum((P - P.mean(0)) ** 2, axis=1)))),
                "p90_dev_cm": 100 * float(np.percentile(dist, 90)),
                "frac_within_5cm": float(np.mean(dist < 0.05)),
                "position": med,
            }
        )
    return rows


# ============================================================================ sequences
CACHE_NAME = "fr1_desk_stride8_masks.npz"


def cache_path() -> Path:
    """``runs/15_robotics_capstone/fr1_desk_stride8_masks.npz`` (written by notebook 03)."""
    return runs_dir(PROJECT, create=True) / CACHE_NAME


def save_mask_cache(path: str | Path, cache: dict[int, dict]) -> Path:
    """Store per-frame detections and masks compactly (masks bit-packed along the width).

    ``cache[frame] = {"boxes": Nx4, "scores": N, "labels": [str], <mask source>: NxHxW bool}``.
    """
    arrays: dict[str, np.ndarray] = {}
    for f, c in cache.items():
        arrays[f"f{f}__boxes"] = np.asarray(c["boxes"], dtype=np.float32).reshape(-1, 4)
        arrays[f"f{f}__scores"] = np.asarray(c["scores"], dtype=np.float32)
        arrays[f"f{f}__labels"] = np.asarray(c["labels"], dtype=str)
        for k, v in c.items():
            if k not in ("boxes", "scores", "labels"):
                arrays[f"f{f}__mask_{k}"] = np.packbits(np.asarray(v, dtype=bool), axis=-1)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def load_mask_cache(path: str | Path | None = None, width: int = 640) -> dict[int, dict]:
    """Inverse of :func:`save_mask_cache`."""
    path = Path(path) if path is not None else cache_path()
    if not path.exists():
        raise FileNotFoundError(f"{path.name} missing - run notebook 03 first")
    cache: dict[int, dict] = {}
    with np.load(path) as z:
        for key in z.files:
            frame, name = key.split("__", 1)
            c = cache.setdefault(int(frame[1:]), {})
            v = z[key]
            if name.startswith("mask_"):
                c[name[5:]] = np.unpackbits(v, axis=-1, count=width).astype(bool)
            elif name == "labels":
                c[name] = [str(x) for x in v]
            else:
                c[name] = v
    return dict(sorted(cache.items()))


def sequence_observations(
    frames: Frames,
    cache: dict[int, dict],
    mask_key: str = "sam3",
    camera: str = DEFAULT_CAMERA,
    pose_at: str = "depth",
    grasps: bool = False,
    gripper: g3.Gripper | None = None,
    min_score: float = 0.0,
) -> tuple[list[dict], list[dict]]:
    """Re-run the CPU 3D stage over cached masks and return world-frame observations.

    ``mask_key``: ``"sam3"``, ``"sam2.1-s"`` (cached masks) or ``"box"`` (the filled 2D box,
    i.e. no segmentation). ``pose_at``: interpolate the ground-truth pose at the ``"depth"``
    or the ``"rgb"`` timestamp. Returns ``(observations, per-frame timings)``.
    """
    rays = g3.pixel_rays(*CAMERAS[camera], 640, 480)
    obs, times = [], []
    for i, c in cache.items():
        rgb, depth = frames.rgb(i), frames.depth(i)
        dets = Detections(np.asarray(c["boxes"]), np.asarray(c["scores"]), list(c["labels"]))
        if mask_key == "box":
            masks = np.array([g3.box_to_mask(b, depth.shape) for b in dets.boxes], dtype=bool)
            masks = masks.reshape(-1, *depth.shape)
        else:
            masks = c[mask_key]
        keep = dets.scores >= min_score
        dets, masks = dets.select(keep), masks[keep]
        t: dict = {}
        res = estimate_objects(rgb, depth, dets, masks, rays, gripper, grasps=grasps, timings=t)
        for o in to_world(frames.pose(i, at=pose_at), res):
            o["frame"] = i
            obs.append(o)
        times.append(t)
    return obs, times


def consistency(
    observations: list[dict], key: str = "box_center_w", gate: float = 0.15, min_obs: int = 5
) -> dict:
    """Multi-frame consistency of world-frame object positions (the capstone accuracy metric).

    Observations are associated into landmarks (:func:`build_object_map`); for landmarks seen in
    at least ``min_obs`` frames, every observation's distance to its landmark's median position
    is pooled. A static object observed with a perfect pipeline would give zero spread.
    """
    lms = build_object_map(observations, key=key, gate=gate)
    stable = [lm for lm in lms if len(lm["obs"]) >= min_obs]
    dev = []
    for lm in stable:
        P = np.array([o[key] for o in lm["obs"]])
        dev.append(np.linalg.norm(P - np.median(P, axis=0), axis=1))
    d = np.concatenate(dev) if dev else np.zeros(0)
    n = len(observations)
    return {
        "observations": n,
        "landmarks": len(lms),
        "stable landmarks": len(stable),
        "obs in stable landmarks [%]": 100.0 * len(d) / max(n, 1),
        "median dev [cm]": 100.0 * float(np.median(d)) if len(d) else float("nan"),
        "p90 dev [cm]": 100.0 * float(np.percentile(d, 90)) if len(d) else float("nan"),
        "within 5 cm [%]": 100.0 * float(np.mean(d < 0.05)) if len(d) else float("nan"),
    }


# ============================================================================ drawing
def draw_results(
    rgb: np.ndarray,
    results: list[ObjectResult],
    K: np.ndarray,
    dist: np.ndarray | None = None,
    show_box2d: bool = False,
) -> np.ndarray:
    """Masks, projected 3D boxes and the best grasp of every object, drawn on the image."""
    import cv2

    from cvlab.viz import color, overlay_masks

    masks = [r.mask for r in results if r.box is not None]
    cols = [color(i) for i, r in enumerate(results) if r.box is not None]
    out = overlay_masks(rgb, np.array(masks), alpha=0.35, colors=cols) if masks else rgb.copy()
    k = 0
    for r in results:
        if r.box is None:
            continue
        c = tuple(int(x) for x in color(k))
        k += 1
        uv = g3.project_points(r.box.corners(), K, dist)
        for a, b in g3.BOX_EDGES:
            cv2.line(
                out,
                tuple(np.round(uv[a]).astype(int)),
                tuple(np.round(uv[b]).astype(int)),
                c,
                2,
                cv2.LINE_AA,
            )
        if show_box2d:
            x1, y1, x2, y2 = np.round(r.box2d).astype(int)
            cv2.rectangle(out, (x1, y1), (x2, y2), c, 1)
        g = r.best_grasp
        if g is not None:
            R, p = g.T[:3, :3], g.T[:3, 3]
            tips = np.array(
                [p - R[:, 0] * g.opening / 2, p + R[:, 0] * g.opening / 2, p - R[:, 2] * 0.06]
            )
            t = np.round(g3.project_points(tips, K, dist)).astype(int)
            cv2.line(out, tuple(t[0]), tuple(t[1]), (255, 255, 255), 3, cv2.LINE_AA)
            cv2.line(out, tuple(t[0]), tuple(t[1]), (220, 20, 60), 2, cv2.LINE_AA)
            cv2.line(out, tuple(t[2]), tuple((t[0] + t[1]) // 2), (220, 20, 60), 2, cv2.LINE_AA)
            for q in t[:2]:
                cv2.circle(out, tuple(q), 4, (220, 20, 60), -1, cv2.LINE_AA)
        x1, y1 = np.round(r.box2d[:2]).astype(int)
        cv2.putText(
            out,
            r.label,
            (x1 + 2, max(y1 - 4, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            r.label,
            (x1 + 2, max(y1 - 4, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            c,
            1,
            cv2.LINE_AA,
        )
    return out


def plot_box3d(ax, box: g3.Box3D, color_rgb, lw: float = 1.2, label: str | None = None) -> None:
    """Draw the 12 edges of a box on a matplotlib 3D axis."""
    C = box.corners()
    for k, (a, b) in enumerate(g3.BOX_EDGES):
        ax.plot(*np.stack([C[a], C[b]]).T, color=color_rgb, lw=lw, label=label if k == 0 else None)


def plot_grasp3d(
    ax,
    T: np.ndarray,
    opening: float,
    color_rgb=(0.86, 0.08, 0.24),
    depth: float = 0.035,
    lw: float = 2.0,
) -> None:
    """Draw a parallel-jaw gripper (two fingers + palm + approach stem) at pose ``T``."""
    R, p = T[:3, :3], T[:3, 3]
    x, z = R[:, 0], R[:, 2]
    half = opening / 2
    tips = [p - x * half + z * depth / 2, p + x * half + z * depth / 2]
    bases = [p - x * half - z * depth / 2, p + x * half - z * depth / 2]
    for a, b in zip(tips, bases):
        ax.plot(*np.stack([a, b]).T, color=color_rgb, lw=lw)
    ax.plot(*np.stack(bases).T, color=color_rgb, lw=lw)
    mid = (bases[0] + bases[1]) / 2
    ax.plot(*np.stack([mid, mid - z * 0.05]).T, color=color_rgb, lw=lw)


# ============================================================================== rerun
class RerunRecorder:
    """Write an ``.rrd`` recording (open it later with ``rerun <file>.rrd``).

    Entity layout: ``world/camera`` (pose + pinhole + image/depth/segmentation),
    ``world/objects/<label>`` (points, oriented boxes), ``world/grasps`` (gripper axes),
    ``world/map`` (landmarks of the multi-frame object map).
    """

    def __init__(
        self,
        path: str | Path,
        K: np.ndarray,
        width: int = 640,
        height: int = 480,
        app_id: str = "capstone_fr1_desk",
    ) -> None:
        import rerun as rr

        self.rr, self.path = rr, Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rr.init(app_id, spawn=False)
        rr.save(str(self.path))  # stream everything that follows into the file
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        rr.log(
            "world/camera/image",
            rr.Pinhole(
                image_from_camera=K, width=width, height=height, camera_xyz=rr.ViewCoordinates.RDF
            ),
            static=True,
        )

    def log_frame(
        self,
        frame: int,
        T_wc: np.ndarray,
        rgb: np.ndarray,
        depth: np.ndarray,
        results: list[ObjectResult],
    ) -> None:
        rr = self.rr
        from cvlab.viz import color

        rr.set_time("frame", sequence=int(frame))
        rr.log("world/camera", rr.Transform3D(translation=T_wc[:3, 3], mat3x3=T_wc[:3, :3]))
        rr.log("world/camera/image/rgb", rr.Image(rgb).compress(jpeg_quality=80))
        rr.log(
            "world/camera/image/depth",
            rr.DepthImage((depth * 5000).astype(np.uint16), meter=5000.0),
        )
        seg = np.zeros(depth.shape, np.uint8)
        for k, r in enumerate(results):
            seg[r.mask] = k + 1
        rr.log("world/camera/image/segmentation", rr.SegmentationImage(seg))
        rr.log("world/objects", rr.Clear(recursive=True))
        rr.log("world/grasps", rr.Clear(recursive=True))
        for k, r in enumerate(results):
            if r.box is None:
                continue
            c = color(k)
            P = r.cloud.points @ T_wc[:3, :3].T + T_wc[:3, 3]
            name = f"world/objects/{k}_{r.label.replace(' ', '_')}"
            rr.log(f"{name}/points", rr.Points3D(P, colors=r.cloud.colors, radii=0.003))
            b = r.box.transformed(T_wc)
            rr.log(
                f"{name}/box",
                rr.Boxes3D(
                    centers=[b.center],
                    half_sizes=[b.extents / 2],
                    quaternions=[rr.Quaternion(xyzw=b.quat_xyzw)],
                    colors=[c],
                    labels=[r.label],
                ),
            )
            g = r.best_grasp
            if g is not None:
                Tg = T_wc @ g.T
                o = Tg[:3, 3]
                rr.log(
                    f"world/grasps/{k}",
                    rr.Arrows3D(
                        origins=[o - Tg[:3, 2] * 0.08, o - Tg[:3, 0] * g.opening / 2],
                        vectors=[Tg[:3, 2] * 0.08, Tg[:3, 0] * g.opening],
                        colors=[(220, 20, 60), (255, 255, 255)],
                    ),
                )

    def log_map(self, stats: list[dict], frame: int) -> None:
        rr = self.rr
        if not stats:
            return
        rr.set_time("frame", sequence=int(frame))
        rr.log(
            "world/map",
            rr.Points3D(
                [s["position"] for s in stats], radii=0.02, labels=[s["label"] for s in stats]
            ),
        )

    def close(self) -> Path:
        self.rr.disconnect()
        return self.path


def camera_to_world_cloud(T_wc: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ T_wc[:3, :3].T + T_wc[:3, 3]


def world_to_camera(T_wc: np.ndarray) -> np.ndarray:
    return invert_T(T_wc)
