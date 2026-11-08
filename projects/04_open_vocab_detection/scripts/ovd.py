"""Open-vocabulary detectors behind one small interface (project 04).

Every detector takes an RGB ``uint8`` image plus a vocabulary (a list of class names)
and returns :class:`Detections` with boxes in pixel ``(x1, y1, x2, y2)``, scores in
``[0, 1]`` and integer labels that index the vocabulary. This makes it possible to
evaluate a closed-set COCO detector and several open-vocabulary detectors with the
same COCO-style code.

Implemented backends:

* ``yolo26s``  - Ultralytics YOLO26-s, closed-set (80 COCO classes, no text input).
* ``yoloe26s`` - Ultralytics YOLOE-26-s-seg, text prompts re-parameterised into the head.
* ``gdino_t``  - Grounding DINO-T (transformers), one text prompt "a. b. c." per pass.
* ``llmdet_t`` - LLMDet-T (MM-Grounding-DINO architecture, same API as Grounding DINO).
* ``owlv2_b``  - OWLv2 B/16 ensemble (transformers), per-patch boxes x text embeddings.
* ``omdet_t``  - OmDet-Turbo Swin-T (transformers), real-time oriented, built-in NMS.
* ``sam3``     - SAM 3 (transformers), one noun phrase per pass, boxes + instance masks.

Also: COCO-style evaluation helpers (pycocotools) and a JPEG figure display helper.
"""

from __future__ import annotations

import contextlib
import io
import os
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("YOLO_AUTOINSTALL", "false")  # never pip-install at runtime
os.environ.setdefault("YOLO_VERBOSE", "false")

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]  # fmt: skip

# Pretty labels for plots/tables (key -> display name, kind).
MODEL_INFO = {
    "yolo26s": ("YOLO26-s (closed-set)", "closed"),
    "yoloe26s": ("YOLOE-26-s", "open"),
    "gdino_t": ("Grounding DINO-T", "open"),
    "llmdet_t": ("LLMDet-T", "open"),
    "owlv2_b": ("OWLv2-B/16", "open"),
    "omdet_t": ("OmDet-Turbo-T", "open"),
    "sam3": ("SAM 3", "open"),
}
HF_REPOS = {
    "gdino_t": ("IDEA-Research/grounding-dino-tiny", "a2bb814dd30d776dcf7e30523b00659f4f141c71"),
    "llmdet_t": ("iSEE-Laboratory/llmdet_tiny", "d05199165a19320a9236396e20fca0a5e065189c"),
    "owlv2_b": ("google/owlv2-base-patch16-ensemble", "cfd3195ba4ea9592eec887ded089f4c08eff231d"),
    "omdet_t": ("omlab/omdet-turbo-swin-tiny-hf", "7fe93cecfb770c4d76cf71163956221249cab566"),
    "sam3": ("facebook/sam3", "3c879f39826c281e95690f02c7821c4de09afae7"),
}
# Prompt template used by default per model ("{}" = bare class name).
DEFAULT_TEMPLATES = {"owlv2_b": "a photo of a {}"}


# ----------------------------------------------------------------------------- results
@dataclass
class Detections:
    """Detections on one image; ``labels`` index into ``names``."""

    boxes: np.ndarray  # (N, 4) float32, x1 y1 x2 y2 in pixels
    scores: np.ndarray  # (N,) float32 in [0, 1]
    labels: np.ndarray  # (N,) int64
    names: list[str]
    masks: np.ndarray | None = None  # (N, H, W) bool, if the model predicts masks
    extra: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.scores)

    @classmethod
    def empty(cls, names: list[str]) -> Detections:
        return cls(np.zeros((0, 4), np.float32), np.zeros(0, np.float32),
                   np.zeros(0, np.int64), list(names))  # fmt: skip

    def select(self, keep: np.ndarray) -> Detections:
        keep = np.asarray(keep)
        masks = None if self.masks is None else self.masks[keep]
        return Detections(self.boxes[keep], self.scores[keep], self.labels[keep], self.names,
                          masks, dict(self.extra))  # fmt: skip

    def above(self, threshold: float) -> Detections:
        """Keep detections with ``score >= threshold``."""
        return self.select(self.scores >= threshold)

    def topk(self, k: int) -> Detections:
        order = np.argsort(-self.scores, kind="stable")[:k]
        return self.select(order)

    @property
    def label_names(self) -> list[str]:
        return [self.names[i] for i in self.labels]


def nms_per_class(dets: Detections, iou: float = 0.5) -> Detections:
    """Class-aware non-maximum suppression (torchvision ``batched_nms``)."""
    import torch
    from torchvision.ops import batched_nms

    if len(dets) == 0:
        return dets
    keep = batched_nms(torch.as_tensor(dets.boxes, dtype=torch.float32),
                       torch.as_tensor(dets.scores, dtype=torch.float32),
                       torch.as_tensor(dets.labels), iou)  # fmt: skip
    return dets.select(keep.numpy())


def _topk_flat(scores, boxes, k: int, min_score: float):
    """Top-k over a (num_boxes x num_classes) score matrix: one box may get several labels.

    This is the standard DETR/GLIP evaluation protocol and gives slightly higher AP than
    keeping only the arg-max class per box.
    """
    import torch

    n_box, n_cls = scores.shape
    flat = scores.flatten()
    k = min(k, flat.numel())
    vals, idx = torch.topk(flat, k)
    keep = vals >= min_score
    vals, idx = vals[keep], idx[keep]
    box_idx, labels = idx // n_cls, idx % n_cls
    return boxes[box_idx], vals, labels


def _to_np(boxes, scores, labels) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (boxes.detach().float().cpu().numpy().astype(np.float32),
            scores.detach().float().cpu().numpy().astype(np.float32),
            labels.detach().cpu().numpy().astype(np.int64))  # fmt: skip


def _clip_boxes(boxes: np.ndarray, h: int, w: int) -> np.ndarray:
    boxes = boxes.copy()
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
    return boxes


# ---------------------------------------------------------------------------- detectors
class Detector:
    """Common interface. Subclasses implement :meth:`_set_vocab` and :meth:`_predict`."""

    key = "base"
    kind = "open"

    def __init__(self, device: str = "cuda", template: str = "{}") -> None:
        self.device = device
        self.template = template
        self.names: list[str] = []

    @property
    def label(self) -> str:
        return MODEL_INFO.get(self.key, (self.key, ""))[0]

    def prompts(self, names: list[str]) -> list[str]:
        return [self.template.format(n) for n in names]

    def set_vocabulary(self, names: list[str], prompts: list[str] | None = None) -> Detector:
        """Set the class names to detect (text encoders run once here where possible).

        ``prompts`` optionally replaces the text sent to the model for each name (same
        order); by default each name is formatted with the detector's ``template``.
        """
        self.names = list(names)
        prompts = self.prompts(self.names) if prompts is None else list(prompts)
        if len(prompts) != len(self.names):
            raise ValueError("need exactly one prompt per class name")
        self._set_vocab(prompts)
        return self

    def predict(self, image: np.ndarray, min_score: float = 0.05, max_det: int = 100,
                masks: bool = False) -> Detections:  # fmt: skip
        """Detect the current vocabulary in an RGB uint8 image."""
        import torch

        if not self.names:
            raise RuntimeError("call set_vocabulary(names) first")
        with torch.inference_mode():
            dets = self._predict(image, min_score, max_det, masks)
        dets.boxes = _clip_boxes(dets.boxes, *image.shape[:2])
        return dets

    def _set_vocab(self, prompts: list[str]) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def _predict(self, image, min_score, max_det, masks) -> Detections:  # pragma: no cover
        raise NotImplementedError


class UltralyticsDetector(Detector):
    """YOLO26 (closed-set) and YOLOE-26 (open-vocabulary) through Ultralytics."""

    def __init__(self, weights: str, device: str = "cuda", imgsz: int = 640, half: bool = True,
                 **kw) -> None:  # fmt: skip
        super().__init__(device, **kw)
        from ultralytics import YOLO, YOLOE
        from ultralytics.utils import LOGGER, SETTINGS

        LOGGER.setLevel("ERROR")  # Ultralytics logs absolute paths at INFO level
        path = Path(SETTINGS["weights_dir"]) / weights  # weights/ultralytics/ (see download script)
        self.is_yoloe = weights.startswith("yoloe")
        self.model = (YOLOE if self.is_yoloe else YOLO)(str(path))
        self.imgsz, self.half = imgsz, half and device.startswith("cuda")
        self.native_names = [self.model.names[i] for i in range(len(self.model.names))]
        self._map: np.ndarray | None = None

    def _set_vocab(self, prompts: list[str]) -> None:
        if self.is_yoloe:
            # Text -> MobileCLIP2 embedding (once), fused into the classification head.
            with contextlib.redirect_stdout(io.StringIO()):
                self.model.set_classes(prompts)
            self._map = np.arange(len(prompts))
        else:
            # Closed-set: a requested name can only be detected if it IS a training class.
            lookup = {n: i for i, n in enumerate(self.native_names)}
            self._map = np.array([lookup.get(p, -1) for p in prompts])

    def _predict(self, image, min_score, max_det, masks) -> Detections:
        bgr = np.ascontiguousarray(image[..., ::-1])  # Ultralytics expects OpenCV BGR arrays
        # retina_masks=True returns masks at the original image resolution (no letterbox pad)
        res = self.model.predict(bgr, imgsz=self.imgsz, conf=min_score, max_det=300,
                                 half=self.half, device=self.device, retina_masks=masks,
                                 verbose=False)[0]  # fmt: skip
        b = res.boxes
        boxes = b.xyxy.float().cpu().numpy()
        scores = b.conf.float().cpu().numpy()
        native = b.cls.cpu().numpy().astype(np.int64)
        if self.is_yoloe:
            labels, keep = native, np.ones(len(native), bool)
        else:
            inv = {int(c): i for i, c in enumerate(self._map) if c >= 0}
            labels = np.array([inv.get(int(c), -1) for c in native], dtype=np.int64)
            keep = labels >= 0
        m = None
        if masks and res.masks is not None:
            m = res.masks.data.cpu().numpy().astype(bool)
            if m.shape[1:] != image.shape[:2]:  # masks come at network resolution
                import cv2

                size = image.shape[1::-1]
                m = np.stack([cv2.resize(x.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
                              for x in m]).astype(bool)  # fmt: skip
            m = m[keep]
        dets = Detections(boxes[keep].astype(np.float32), scores[keep].astype(np.float32),
                          labels[keep], self.names, m)  # fmt: skip
        return dets.topk(max_det)


class GroundingDinoDetector(Detector):
    """Grounding DINO / MM-Grounding-DINO / LLMDet via ``AutoModelForZeroShotObjectDetection``.

    The vocabulary is written into one caption ``"cat. remote control. dog."``. The model
    predicts, for each of its 900 queries, a probability for every *text token*. A class
    score is the mean probability over that class's tokens (the GLIP / mmdetection
    "positive map" protocol). Captions are limited to 256 tokens, so long vocabularies are
    split into chunks that are run as separate forward passes.
    """

    def __init__(self, repo: str, revision: str | None = None, device: str = "cuda",
                 dtype: str = "float16", max_tokens: int = 256, **kw) -> None:  # fmt: skip
        super().__init__(device, **kw)
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.dtype = getattr(torch, dtype)
        self.processor = AutoProcessor.from_pretrained(repo, revision=revision)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            repo, revision=revision, dtype=self.dtype).to(device).eval()  # fmt: skip
        self.max_tokens = max_tokens
        self.chunks: list[dict] = []

    def _caption_and_posmap(self, prompts: list[str]):
        """Caption text and a (num_classes, num_tokens) 0/1 map from classes to tokens."""
        import torch

        spans, caption = [], ""
        for p in prompts:
            p = p.strip().lower()
            spans.append((len(caption), len(caption) + len(p)))
            caption += p + ". "
        caption = caption.strip()
        enc = self.processor.tokenizer(caption, return_offsets_mapping=True)
        offsets = enc["offset_mapping"]
        posmap = torch.zeros(len(prompts), len(offsets))
        for c, (s, e) in enumerate(spans):
            for t, (a, b) in enumerate(offsets):
                if b > a and a >= s and b <= e:  # special tokens have (0, 0) offsets
                    posmap[c, t] = 1.0
        return caption, posmap, len(offsets)

    def _set_vocab(self, prompts: list[str]) -> None:
        self.chunks, start = [], 0
        while start < len(prompts):
            end = len(prompts)
            while True:  # greedily grow the chunk while the caption fits in max_tokens
                caption, posmap, n_tok = self._caption_and_posmap(prompts[start:end])
                if n_tok <= self.max_tokens or end - start == 1:
                    break
                end -= 1
            self.chunks.append({"offset": start, "caption": caption, "posmap": posmap,
                                "n_tokens": n_tok})  # fmt: skip
            start = end

    def raw_outputs(self, image: np.ndarray, caption: str):
        """One forward pass; returns the transformers output and the processor inputs."""
        from PIL import Image

        inputs = self.processor(images=Image.fromarray(image), text=caption, return_tensors="pt")
        inputs = inputs.to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        return self.model(**inputs), inputs

    def _predict(self, image, min_score, max_det, masks) -> Detections:
        import torch

        h, w = image.shape[:2]
        scale = torch.tensor([w, h, w, h], device=self.device, dtype=torch.float32)
        all_b, all_s, all_l = [], [], []
        for ch in self.chunks:
            out, _ = self.raw_outputs(image, ch["caption"])
            prob = out.logits[0].float().sigmoid()  # (900 queries, 256 tokens), -inf -> 0
            pm = ch["posmap"].to(prob.device)
            prob = prob[:, : pm.shape[1]]
            cls_scores = (prob @ pm.T) / pm.sum(1).clamp(min=1)  # mean over class tokens
            cx, cy, bw, bh = out.pred_boxes[0].float().unbind(-1)  # normalised cxcywh
            boxes = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], -1) * scale
            b, s, lab = _topk_flat(cls_scores, boxes, max_det, min_score)
            all_b.append(b), all_s.append(s), all_l.append(lab + ch["offset"])
        dets = Detections(*_to_np(torch.cat(all_b), torch.cat(all_s), torch.cat(all_l)),
                          self.names)  # fmt: skip
        return dets.topk(max_det)


class Owlv2Detector(Detector):
    """OWLv2: every ViT patch predicts a box and an embedding; class score = sigmoid of the
    (scaled, shifted) cosine similarity between that embedding and each text query."""

    def __init__(self, repo: str, revision: str | None = None, device: str = "cuda",
                 dtype: str = "float16", nms_iou: float = 0.5, **kw) -> None:  # fmt: skip
        super().__init__(device, **kw)
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.dtype = getattr(torch, dtype)
        self.processor = Owlv2Processor.from_pretrained(repo, revision=revision)
        self.model = Owlv2ForObjectDetection.from_pretrained(
            repo, revision=revision, dtype=self.dtype).to(device).eval()  # fmt: skip
        self.nms_iou = nms_iou
        self.text_inputs = None

    def _set_vocab(self, prompts: list[str]) -> None:
        self.text_inputs = self.processor(text=[prompts], return_tensors="pt").to(self.device)

    def _predict(self, image, min_score, max_det, masks) -> Detections:
        import torch
        from PIL import Image

        h, w = image.shape[:2]
        pix = self.processor(images=Image.fromarray(image), return_tensors="pt")["pixel_values"]
        out = self.model(pixel_values=pix.to(self.device, self.dtype), **self.text_inputs)
        scores = out.logits[0].float().sigmoid()  # (3600 patches, num_classes)
        cx, cy, bw, bh = out.pred_boxes[0].float().unbind(-1)
        # Images are zero-padded at the bottom/right to a square before resizing to 960x960,
        # so normalised coordinates refer to the padded square of side max(h, w).
        s = float(max(h, w))
        boxes = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], -1) * s
        b, sc, lab = _topk_flat(scores, boxes, 4 * max_det, min_score)
        dets = Detections(*_to_np(b, sc, lab), self.names)
        if self.nms_iou:
            dets = nms_per_class(dets, self.nms_iou)  # neighbouring patches give duplicates
        return dets.topk(max_det)


class OmDetTurboDetector(Detector):
    """OmDet-Turbo: class and task prompts are encoded separately (and cached by the model),
    a fast RT-DETR-style decoder scores every query against every class; NMS removes
    duplicates."""

    def __init__(self, repo: str, revision: str | None = None, device: str = "cuda",
                 dtype: str = "float16", nms_iou: float = 0.5, **kw) -> None:  # fmt: skip
        super().__init__(device, **kw)
        import torch
        from transformers import AutoProcessor, OmDetTurboForObjectDetection

        self.dtype = getattr(torch, dtype)
        self.processor = AutoProcessor.from_pretrained(repo, revision=revision)
        self.model = OmDetTurboForObjectDetection.from_pretrained(
            repo, revision=revision, dtype=self.dtype).to(device).eval()  # fmt: skip
        self.nms_iou = nms_iou
        self.prompts_ = []

    def _set_vocab(self, prompts: list[str]) -> None:
        self.prompts_ = list(prompts)

    def _predict(self, image, min_score, max_det, masks) -> Detections:
        import torch
        from PIL import Image

        h, w = image.shape[:2]
        inputs = self.processor(images=Image.fromarray(image), text=self.prompts_,
                                return_tensors="pt").to(self.device)  # fmt: skip
        inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        out = self.model(**inputs)
        res = self.processor.post_process_grounded_object_detection(
            out, threshold=min_score, nms_threshold=self.nms_iou, target_sizes=[(h, w)],
            max_num_det=max(max_det, 300))[0]  # fmt: skip
        dets = Detections(*_to_np(res["boxes"], res["scores"], torch.as_tensor(res["labels"])),
                          self.names)  # fmt: skip
        return dets.topk(max_det)


def _no_mask_decoder():
    """Stand-in module for SAM 3's mask decoder when only boxes are needed (saves time and
    memory: the pixel decoder upsamples features for all 200 queries of every prompt)."""
    import torch

    class _NoMaskDecoder(torch.nn.Module):
        def forward(self, **kwargs):
            return SimpleNamespace(pred_masks=None, semantic_seg=None, attentions=None)

    return _NoMaskDecoder()


class Sam3Detector(Detector):
    """SAM 3 promptable concept segmentation used as an open-vocabulary detector.

    SAM 3 takes ONE noun phrase per forward pass and returns every instance of it
    (boxes, masks and a per-query score multiplied by an image-level "presence" score).
    For a vocabulary we encode the image once, encode all phrases once, and run the
    DETR-style fusion encoder/decoder for batches of phrases against the cached image
    features.
    """

    def __init__(self, repo: str = "facebook/sam3", revision: str | None = None,
                 device: str = "cuda", dtype: str = "bfloat16", prompt_batch: int = 8,
                 **kw) -> None:  # fmt: skip
        super().__init__(device, **kw)
        import torch
        from transformers import Sam3Model, Sam3Processor

        self.dtype = getattr(torch, dtype)
        self.processor = Sam3Processor.from_pretrained(repo, revision=revision)
        self.model = Sam3Model.from_pretrained(repo, revision=revision,
                                               dtype=self.dtype).to(device).eval()  # fmt: skip
        self._mask_decoder = self.model.mask_decoder
        self._no_masks = _no_mask_decoder()
        self.prompt_batch = prompt_batch
        self.text = None

    def _set_vocab(self, prompts: list[str]) -> None:
        import torch

        enc = self.processor(text=prompts, return_tensors="pt", padding="max_length",
                             max_length=32, truncation=True).to(self.device)  # fmt: skip
        with torch.inference_mode():
            feats = self.model.get_text_features(input_ids=enc["input_ids"],
                                                 attention_mask=enc["attention_mask"])  # fmt: skip
        self.text = (feats.pooler_output, enc["attention_mask"])

    def _predict(self, image, min_score, max_det, masks) -> Detections:
        import torch
        from PIL import Image
        from transformers.modeling_outputs import BaseModelOutputWithPooling

        h, w = image.shape[:2]
        inputs = self.processor(images=Image.fromarray(image), return_tensors="pt")
        pix = inputs["pixel_values"].to(self.device, self.dtype)
        vis = self.model.get_vision_features(pixel_values=pix)
        text_feats, text_mask = self.text
        if not masks:
            self.model.mask_decoder = self._no_masks
        try:
            all_b, all_s, all_l, all_m = [], [], [], []
            for s in range(0, len(self.names), self.prompt_batch):
                e = min(s + self.prompt_batch, len(self.names))
                n = e - s
                rep = lambda ts, n=n: tuple(t.expand(n, *t.shape[1:]) for t in ts)  # noqa: E731
                vis_b = type(vis)(fpn_hidden_states=rep(vis.fpn_hidden_states),
                                  fpn_position_encoding=rep(vis.fpn_position_encoding))  # fmt: skip
                out = self.model(vision_embeds=vis_b,
                                 text_embeds=BaseModelOutputWithPooling(pooler_output=text_feats[s:e]),
                                 attention_mask=text_mask[s:e])  # fmt: skip
                score = out.pred_logits.float().sigmoid()  # (n, 200 queries)
                if out.presence_logits is not None:
                    score = score * out.presence_logits.float().sigmoid()  # (n, 1) broadcast
                boxes = out.pred_boxes.float() * torch.tensor([w, h, w, h], device=score.device)
                for i in range(n):
                    keep = torch.topk(score[i], min(max_det, score.shape[1])).indices
                    keep = keep[score[i, keep] >= min_score]
                    all_b.append(boxes[i, keep]), all_s.append(score[i, keep])
                    all_l.append(torch.full_like(keep, s + i))
                    if masks and len(keep):
                        # 288x288 mask logits cover the whole (square-resized) image
                        m = out.pred_masks[i, keep].float()[None]
                        m = torch.nn.functional.interpolate(m, size=(h, w), mode="bilinear")[0]
                        all_m.append((m > 0).cpu().numpy())
                    elif masks:
                        all_m.append(np.zeros((0, h, w), bool))
        finally:
            self.model.mask_decoder = self._mask_decoder
        dets = Detections(*_to_np(torch.cat(all_b), torch.cat(all_s), torch.cat(all_l)),
                          self.names, np.concatenate(all_m) if masks else None)  # fmt: skip
        return dets.topk(max_det)


def load_detector(key: str, device: str = "cuda", template: str | None = None,
                  **kw) -> Detector:  # fmt: skip
    """Build one of the detectors listed in :data:`MODEL_INFO`."""
    template = DEFAULT_TEMPLATES.get(key, "{}") if template is None else template
    if key == "yolo26s":
        det: Detector = UltralyticsDetector("yolo26s.pt", device, template="{}", **kw)
    elif key == "yoloe26s":
        det = UltralyticsDetector("yoloe-26s-seg.pt", device, template=template, **kw)
    elif key in ("gdino_t", "llmdet_t"):
        repo, rev = HF_REPOS[key]
        det = GroundingDinoDetector(repo, rev, device, template=template, **kw)
    elif key == "owlv2_b":
        repo, rev = HF_REPOS[key]
        det = Owlv2Detector(repo, rev, device, template=template, **kw)
    elif key == "omdet_t":
        repo, rev = HF_REPOS[key]
        det = OmDetTurboDetector(repo, rev, device, template=template, **kw)
    elif key == "sam3":
        repo, rev = HF_REPOS[key]
        det = Sam3Detector(repo, rev, device, template=template, **kw)
    else:
        raise KeyError(f"unknown detector {key!r}; choose from {list(MODEL_INFO)}")
    det.key = key
    det.kind = MODEL_INFO[key][1]
    return det


def free_detector(det: Detector | None) -> None:
    """Release a detector's GPU memory (notebooks load models one after another)."""
    import gc

    import torch

    if det is not None:
        # every attribute that may hold GPU tensors or modules (SAM 3 keeps its mask decoder)
        for attr in ("model", "processor", "_mask_decoder", "_no_masks", "text", "text_inputs",
                     "chunks"):  # fmt: skip
            if hasattr(det, attr):
                setattr(det, attr, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def show_jpeg(fig, quality: int = 82, dpi: int = 90) -> None:
    """Display a matplotlib figure in a notebook as JPEG and close it.

    Photo grids saved as PNG cost about 1 MB each in a notebook; JPEG keeps them at roughly a
    tenth of that, so notebooks with many qualitative figures stay small. Use plain
    ``plt.show()`` (PNG) for line plots and charts, where JPEG artefacts would be visible.
    """
    import matplotlib.pyplot as plt
    from IPython.display import Image, display

    buf = io.BytesIO()
    fig.savefig(buf, format="jpeg", dpi=dpi, pil_kwargs={"quality": quality})
    plt.close(fig)
    display(Image(data=buf.getvalue(), format="jpeg"))


# ------------------------------------------------------------------------ COCO-style eval
def detections_to_coco(dets: Detections, image_id: int, cat_ids: list[int]) -> list[dict]:
    """Convert to the COCO results format (``bbox`` = x, y, w, h in pixels)."""
    out = []
    for (x1, y1, x2, y2), s, lab in zip(dets.boxes, dets.scores, dets.labels):
        out.append({"image_id": int(image_id), "category_id": int(cat_ids[lab]),
                    "bbox": [round(float(x1), 2), round(float(y1), 2),
                             round(float(x2 - x1), 2), round(float(y2 - y1), 2)],
                    "score": round(float(s), 5)})  # fmt: skip
    return out


def yolo_labels_to_coco(images_dir, labels_dir, names: list[str]) -> dict:
    """Ground truth of a YOLO-format split (``cls cx cy w h`` normalised) as a COCO dict."""
    from pathlib import Path

    from PIL import Image

    images, anns = [], []
    paths = sorted(p for p in Path(images_dir).iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"})  # fmt: skip
    for img_id, p in enumerate(paths, start=1):
        with Image.open(p) as im:
            w, h = im.size
        images.append({"id": img_id, "file_name": p.name, "width": w, "height": h})
        lab = Path(labels_dir) / f"{p.stem}.txt"
        rows = np.loadtxt(lab, ndmin=2) if lab.exists() and lab.stat().st_size else np.zeros((0, 5))
        for c, cx, cy, bw, bh in rows:
            bw_px, bh_px = bw * w, bh * h
            anns.append({"id": len(anns) + 1, "image_id": img_id, "category_id": int(c) + 1,
                         "bbox": [(cx - bw / 2) * w, (cy - bh / 2) * h, bw_px, bh_px],
                         "area": bw_px * bh_px, "iscrowd": 0})  # fmt: skip
    cats = [{"id": i + 1, "name": n} for i, n in enumerate(names)]
    return {"images": images, "annotations": anns, "categories": cats}


def coco_from_dict(gt: dict):
    """pycocotools ``COCO`` object from an in-memory dict (silences its prints)."""
    from pycocotools.coco import COCO

    coco = COCO()
    coco.dataset = gt
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
    return coco


def coco_evaluate(coco_gt, results: list[dict], img_ids: list[int] | None = None) -> dict:
    """Run COCOeval (bbox) and return summary metrics, per-class AP and PR curves."""
    from pycocotools.cocoeval import COCOeval

    cat_ids = sorted(coco_gt.getCatIds())
    names = [coco_gt.cats[c]["name"] for c in cat_ids]
    if not results:
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0, "APs": 0.0, "APm": 0.0, "APl": 0.0,
                "AR100": 0.0, "per_class_AP": dict.fromkeys(names, 0.0),
                "per_class_AP50": dict.fromkeys(names, 0.0)}  # fmt: skip
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(results)
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        if img_ids is not None:
            ev.params.imgIds = list(img_ids)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    s = ev.stats
    prec = ev.eval["precision"]  # (T=10 IoUs, R=101 recalls, K classes, A=4 areas, M=3 maxDets)

    def class_ap(t_slice) -> dict:
        out = {}
        for k, n in enumerate(names):
            p = prec[t_slice, :, k, 0, 2]
            p = p[p > -1]
            out[n] = float(p.mean()) if p.size else float("nan")  # nan = class absent from GT
        return out

    pr50 = prec[0, :, :, 0, 2]  # IoU=0.5, all areas, maxDets=100 -> (101, K)
    valid = (pr50 > -1).all(0)
    return {
        "AP": float(s[0]), "AP50": float(s[1]), "AP75": float(s[2]), "APs": float(s[3]),
        "APm": float(s[4]), "APl": float(s[5]), "AR100": float(s[8]),
        "per_class_AP": class_ap(slice(None)), "per_class_AP50": class_ap(slice(0, 1)),
        "pr_curve_iou50": pr50[:, valid].mean(1).round(4).tolist() if valid.any() else [],
    }  # fmt: skip


def f1_vs_threshold(coco_gt, results: list[dict], thresholds: np.ndarray,
                    iou_thr: float = 0.5) -> dict:  # fmt: skip
    """Micro-averaged precision / recall / F1 over all classes as a function of the score
    threshold (greedy matching at ``iou_thr``; crowd boxes ignored). Useful to choose an
    operating point: unlike AP, a robot has to commit to one threshold."""
    from collections import defaultdict

    from cvlab.metrics import match_detections

    gts = defaultdict(list)
    n_gt = 0
    for a in coco_gt.dataset["annotations"]:
        if a.get("iscrowd", 0):
            continue
        x, y, w, h = a["bbox"]
        gts[(a["image_id"], a["category_id"])].append([x, y, x + w, y + h])
        n_gt += 1
    preds = defaultdict(list)
    for r in results:
        x, y, w, h = r["bbox"]
        preds[(r["image_id"], r["category_id"])].append([x, y, x + w, y + h, r["score"]])
    scores, tps = [], []
    for key, p in preds.items():
        p = np.asarray(p)
        tp = match_detections(p[:, :4], p[:, 4], np.asarray(gts.get(key, [])), iou_thr)
        scores.append(p[:, 4]), tps.append(tp)
    scores = np.concatenate(scores) if scores else np.zeros(0)
    tps = np.concatenate(tps) if tps else np.zeros(0, bool)
    prec, rec, f1 = [], [], []
    for t in thresholds:
        sel = scores >= t
        tp = float(tps[sel].sum())
        p = tp / max(sel.sum(), 1)
        r = tp / max(n_gt, 1)
        prec.append(p), rec.append(r), f1.append(2 * p * r / max(p + r, 1e-12))
    best = int(np.argmax(f1))
    return {"thresholds": np.round(thresholds, 4).tolist(), "precision": np.round(prec, 4).tolist(),
            "recall": np.round(rec, 4).tolist(), "f1": np.round(f1, 4).tolist(),
            "best_threshold": float(thresholds[best]), "best_f1": float(f1[best])}  # fmt: skip
