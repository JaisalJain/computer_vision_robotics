"""Shared helpers for ``projects/13_vision_language``.

Three model families are wrapped behind small, readable classes so the notebooks can
focus on the ideas instead of boilerplate:

* :class:`SigLIP2Encoder` / :class:`OpenCLIPEncoder` - dual encoders for zero-shot
  classification (image and text embeddings in one space),
* :class:`Florence2` - one seq2seq model for captioning, detection and phrase grounding,
* :class:`SmallVLM` - a chat VLM (Qwen3.5-0.8B) for visual question answering, with
  latency (time-to-first-token, decode tokens/s) and peak-VRAM bookkeeping.

The pure-Python/NumPy parts (prompt ensembling, score functions, label normalisation,
detection matching, answer parsing, probe generation) are unit-tested on the CPU in
``tests/test_p13_vl_common.py``.

Checkpoints and their pinned revisions live in ``configs/models.yaml``.
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cvlab.io import project_dir, repo_root

PROJECT = "13_vision_language"

# ----------------------------------------------------------------------------- config
_CFG_CACHE: dict[str, Any] | None = None


def model_cfg(key: str | None = None) -> dict[str, Any]:
    """Entries of ``configs/models.yaml`` (all of them, or the one named ``key``)."""
    global _CFG_CACHE
    if _CFG_CACHE is None:
        import yaml

        _CFG_CACHE = yaml.safe_load((project_dir(PROJECT) / "configs" / "models.yaml").read_text())
    return _CFG_CACHE if key is None else _CFG_CACHE[key]


def rel(path: str | Path) -> str:
    """Repository-relative path string (keeps absolute paths out of notebook outputs)."""
    p = Path(path).resolve()
    try:
        return str(p.relative_to(repo_root()))
    except ValueError:
        return p.name


# ----------------------------------------------------------------------- bookkeeping
class vram_meter:  # noqa: N801 - lowercase context manager, like cvlab.device.track_vram
    """Peak CUDA memory of a block: absolute and on top of what was allocated before.

    ``cvlab.device.track_vram`` reports the absolute peak; when several models stay
    loaded, the increment ``delta_mb`` isolates the cost of the code inside the block.
    """

    def __enter__(self) -> vram_meter:
        import torch

        self.base_mb = self.peak_mb = self.delta_mb = 0.0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.base_mb = torch.cuda.memory_allocated() / 2**20
            torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(self, *exc: object) -> None:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.peak_mb = torch.cuda.max_memory_allocated() / 2**20
            self.delta_mb = self.peak_mb - self.base_mb


def n_params(module) -> float:
    """Number of parameters of a ``torch.nn.Module`` in millions."""
    return sum(p.numel() for p in module.parameters()) / 1e6


# ------------------------------------------------------------------------------- data
def load_rgb(path: str | Path):
    """Open an image file as an RGB ``PIL.Image`` (file handle closed)."""
    from PIL import Image

    with Image.open(path) as im:
        return im.convert("RGB")


def desk_questions() -> tuple[dict[str, Path], list[dict[str, Any]]]:
    """Hand-labelled TUM fr1_desk frames and questions from ``configs/desk_questions.yaml``."""
    import yaml

    from cvlab.io import data_dir

    cfg = yaml.safe_load((project_dir(PROJECT) / "configs" / "desk_questions.yaml").read_text())
    rgb = data_dir("tum", "fr1_desk", "rgbd_dataset_freiburg1_desk", "rgb")
    frames = {k: rgb / v for k, v in cfg["frames"].items()}
    return frames, cfg["questions"]


# Imagenette folder (WordNet id) -> class name. The names are the cleaned ImageNet names
# that OpenAI published with CLIP (open_clip.zero_shot_metadata.IMAGENET_CLASSNAMES).
IMAGENETTE_CLASSES: dict[str, str] = {
    "n01440764": "tench",
    "n02102040": "English Springer Spaniel",
    "n02979186": "cassette player",
    "n03000684": "chainsaw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute",
}

COCO_NAMES: list[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]  # fmt: skip


def imagenette_split(root: str | Path, split: str = "val") -> tuple[list[Path], np.ndarray]:
    """Image paths and integer labels (index into ``IMAGENETTE_CLASSES``) of one split."""
    root = Path(root) / split
    paths, labels = [], []
    for k, wnid in enumerate(IMAGENETTE_CLASSES):
        files = sorted((root / wnid).glob("*.JPEG"))
        paths += files
        labels += [k] * len(files)
    return paths, np.asarray(labels, dtype=np.int64)


@dataclass
class CocoItem:
    """One YOLO-format image with pixel ``(x1, y1, x2, y2)`` boxes and COCO class ids."""

    path: Path
    width: int
    height: int
    boxes: np.ndarray  # (N, 4) float, pixels
    classes: np.ndarray  # (N,) int, index into COCO_NAMES

    def names(self) -> list[str]:
        return [COCO_NAMES[c] for c in self.classes]


def load_yolo_split(images_dir: str | Path, labels_dir: str | Path) -> list[CocoItem]:
    """Read a YOLO-format split (``cls cx cy w h`` normalised) into :class:`CocoItem` s."""
    from PIL import Image

    items = []
    for img_path in sorted(Path(images_dir).glob("*.jpg")):
        with Image.open(img_path) as im:
            w, h = im.size
        lbl = Path(labels_dir) / (img_path.stem + ".txt")
        rows = np.loadtxt(lbl, ndmin=2) if lbl.exists() and lbl.stat().st_size else np.zeros((0, 5))
        cls = rows[:, 0].astype(np.int64)
        cx, cy, bw, bh = rows[:, 1] * w, rows[:, 2] * h, rows[:, 3] * w, rows[:, 4] * h
        boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
        items.append(CocoItem(img_path, w, h, boxes.reshape(-1, 4), cls))
    return items


# ------------------------------------------------------------------ zero-shot helpers
def build_prompts(classnames: Sequence[str], templates: Sequence[str]) -> list[str]:
    """Class-major list of prompts: ``[t0(c0), t1(c0), ..., t0(c1), ...]``.

    Templates use ``{}`` as the class-name placeholder (``"a photo of a {}."``).
    """
    return [t.format(c) for c in classnames for t in templates]


def ensemble_text_embeddings(text_emb: np.ndarray, n_classes: int) -> np.ndarray:
    """Prompt ensembling: average the L2-normalised embeddings of each class's prompts.

    Args:
        text_emb: ``(n_classes * n_templates, D)`` embeddings in :func:`build_prompts` order.
        n_classes: Number of classes.

    Returns:
        ``(n_classes, D)`` unit-norm class embeddings (the zero-shot classifier weights).
    """
    e = np.asarray(text_emb, dtype=np.float64)
    e = e / np.linalg.norm(e, axis=-1, keepdims=True)
    e = e.reshape(n_classes, -1, e.shape[-1]).mean(axis=1)
    return (e / np.linalg.norm(e, axis=-1, keepdims=True)).astype(np.float32)


def zero_shot_scores(
    img_emb: np.ndarray, cls_emb: np.ndarray, scale: float, bias: float = 0.0, kind: str = "softmax"
) -> np.ndarray:
    """Class probabilities from unit-norm embeddings.

    ``logits = scale * cos(img, cls) + bias``. CLIP was trained with a softmax over the
    batch, so its scores are a softmax over classes (rows sum to 1). SigLIP was trained
    with an independent sigmoid per image-text pair, so each class gets its own
    probability in ``[0, 1]`` (rows need not sum to 1, "none of these" is possible).
    """
    logits = scale * (np.asarray(img_emb, np.float64) @ np.asarray(cls_emb, np.float64).T) + bias
    if kind == "softmax":
        z = logits - logits.max(axis=1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(axis=1, keepdims=True)
    if kind == "sigmoid":
        return 1.0 / (1.0 + np.exp(-logits))
    raise ValueError(f"unknown score kind: {kind}")


def _batches(seq: Sequence, n: int):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


class _DualEncoder:
    """Common interface: unit-norm float32 embeddings returned as NumPy arrays."""

    name = "dual-encoder"
    kind = "softmax"
    scale = 100.0
    bias = 0.0

    def _img(self, pixel_values):  # pragma: no cover - implemented by subclasses
        raise NotImplementedError

    def preprocess(self, images: Sequence) -> Any:  # pragma: no cover
        raise NotImplementedError

    def encode_texts(
        self, texts: Sequence[str], batch_size: int = 256
    ) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def encode_images(self, images: Sequence, batch_size: int = 64) -> np.ndarray:
        """Embed PIL images or image paths (decoded on the fly)."""
        import torch

        out = []
        for chunk in _batches(list(images), batch_size):
            pil = [load_rgb(x) if isinstance(x, (str, Path)) else x for x in chunk]
            pv = self.preprocess(pil)
            with torch.inference_mode():
                f = self._img(pv)
            out.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy())
        return np.concatenate(out)


class SigLIP2Encoder(_DualEncoder):
    """SigLIP 2 (transformers ``AutoModel``), fp16 by default. Scores use a sigmoid."""

    kind = "sigmoid"

    def __init__(self, device: str = "cuda", dtype=None) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        cfg = model_cfg("siglip2")
        self.name = "SigLIP 2 B/16-224"
        self.device, self.dtype = str(device), dtype or torch.float16
        self.processor = AutoProcessor.from_pretrained(cfg["repo"], revision=cfg["revision"])
        self.model = AutoModel.from_pretrained(
            cfg["repo"], revision=cfg["revision"], dtype=self.dtype
        ).to(device).eval()  # fmt: skip
        # Learned temperature t = exp(t') and bias b (b starts at -10 in training).
        self.scale = float(self.model.logit_scale.detach().float().exp())
        self.bias = float(self.model.logit_bias.detach().float())

    def preprocess(self, images: Sequence):
        px = self.processor(images=list(images), return_tensors="pt")["pixel_values"]
        return px.to(self.device, self.dtype)

    def _img(self, pixel_values):
        return self.model.get_image_features(pixel_values=pixel_values).pooler_output

    def encode_texts(
        self, texts: Sequence[str], batch_size: int = 256, padding: str = "max_length"
    ) -> np.ndarray:
        import torch

        out = []
        for chunk in _batches(list(texts), batch_size):
            # SigLIP 2 was trained on text padded to exactly 64 tokens (and it has no
            # attention mask in the text tower), so the processor must use
            # padding="max_length", max_length=64; other padding silently changes the
            # embeddings. ``padding`` is exposed only to demonstrate that pitfall.
            kw = {"max_length": 64} if padding == "max_length" else {}
            tok = self.processor(
                text=list(chunk), padding=padding, truncation=True, return_tensors="pt", **kw
            ).to(self.device)  # fmt: skip
            with torch.inference_mode():
                f = self.model.get_text_features(**tok).pooler_output
            out.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy())
        return np.concatenate(out)


class OpenCLIPEncoder(_DualEncoder):
    """open_clip CLIP (ViT-B/32, LAION-2B), fp16 by default. Scores use a softmax."""

    kind = "softmax"

    def __init__(self, device: str = "cuda", dtype=None) -> None:
        import open_clip
        import torch

        from cvlab.download import hf_download

        cfg = model_cfg("clip")
        arch, tag = cfg["open_clip_arch"], cfg["open_clip_tag"]
        self.name = "CLIP B/32 (LAION-2B)"
        self.device, self.dtype = str(device), dtype or torch.float16
        # Load the pinned checkpoint file and pass the matching preprocessing explicitly
        # (identical to pretrained=<tag>, but reproducible: no implicit "main" revision).
        pre = open_clip.get_pretrained_cfg(arch, tag)
        ckpt = hf_download(cfg["repo"], "open_clip_model.safetensors", revision=cfg["revision"])
        model, _, self.transform = open_clip.create_model_and_transforms(
            arch, pretrained=str(ckpt), image_mean=pre["mean"], image_std=pre["std"],
            image_interpolation=pre["interpolation"], image_resize_mode=pre["resize_mode"],
        )  # fmt: skip
        self.model = model.to(device, self.dtype).eval()
        self.tokenizer = open_clip.get_tokenizer(arch)
        self.scale = float(self.model.logit_scale.detach().float().exp())  # clamped at 100
        self.bias = 0.0

    def preprocess(self, images: Sequence):
        import torch

        return torch.stack([self.transform(im) for im in images]).to(self.device, self.dtype)

    def _img(self, pixel_values):
        return self.model.encode_image(pixel_values)

    def encode_texts(self, texts: Sequence[str], batch_size: int = 256) -> np.ndarray:
        import torch

        out = []
        for chunk in _batches(list(texts), batch_size):
            tok = self.tokenizer(list(chunk)).to(self.device)
            with torch.inference_mode():
                f = self.model.encode_text(tok)
            out.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy())
        return np.concatenate(out)


# ------------------------------------------------------------------------- Florence-2
@dataclass
class FlorenceResult:
    """Parsed Florence-2 output for one image and task."""

    task: str
    text: str = ""  # caption / OCR text
    boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    labels: list[str] = field(default_factory=list)
    quads: np.ndarray = field(default_factory=lambda: np.zeros((0, 8)))  # OCR regions
    latency_ms: float = 0.0
    n_tokens: int = 0
    raw: str = ""


class Florence2:
    """Florence-2 (native transformers port, no remote code).

    Every task is a text prompt (``"<OD>"``, ``"<CAPTION>"``, ...). The decoder emits
    plain words plus location tokens ``<loc_0>`` ... ``<loc_999>``: coordinates are
    quantised into 1000 bins per axis, and the processor maps bin ``k`` back to pixels
    as ``(k + 0.5) * W / 1000``.
    """

    def __init__(self, device: str = "cuda", dtype=None) -> None:
        import torch
        from transformers import AutoProcessor, Florence2ForConditionalGeneration

        cfg = model_cfg("florence2")
        self.device, self.dtype = str(device), dtype or torch.float16
        self.processor = AutoProcessor.from_pretrained(cfg["repo"], revision=cfg["revision"])
        self.model = Florence2ForConditionalGeneration.from_pretrained(
            cfg["repo"], revision=cfg["revision"], dtype=self.dtype
        ).to(device).eval()  # fmt: skip

    def run(
        self,
        image,
        task: str,
        text: str | None = None,
        num_beams: int = 3,
        max_new_tokens: int = 256,
    ) -> FlorenceResult:
        """Run one task prompt on a PIL image and parse the answer."""
        import torch

        prompt = task + (text or "")
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        inputs = inputs.to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, num_beams=num_beams, do_sample=False
            )  # fmt: skip
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        raw = self.processor.batch_decode(ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(
            raw, task=task, image_size=(image.width, image.height)
        )[task]  # fmt: skip
        res = FlorenceResult(task=task, latency_ms=ms, n_tokens=int(ids.shape[1]), raw=raw)
        if isinstance(parsed, str):
            res.text = parsed.strip()
        elif "quad_boxes" in parsed:  # OCR with regions: 4 corner points per text line
            res.quads = np.asarray(parsed["quad_boxes"], dtype=np.float64).reshape(-1, 8)
            res.labels = [str(s).replace("</s>", "").strip() for s in parsed.get("labels", [])]
        else:
            res.boxes = np.asarray(parsed.get("bboxes", []), dtype=np.float64).reshape(-1, 4)
            labels = parsed.get("labels", parsed.get("bboxes_labels", []))
            res.labels = [str(s).strip() for s in labels]
        return res


# -------------------------------------------------- detection matching (free-text labels)
# Florence-2 names objects in free text. Map common synonyms onto the COCO vocabulary so
# that "man", "television" or "motorbike" can match COCO "person", "tv", "motorcycle".
LABEL_SYNONYMS: dict[str, str] = {
    "man": "person", "woman": "person", "boy": "person", "girl": "person", "child": "person",
    "people": "person", "human face": "person", "human": "person", "lady": "person",
    "player": "person", "baseball player": "person", "tennis player": "person",
    "television": "tv", "monitor": "tv", "computer monitor": "tv", "screen": "tv",
    "motorbike": "motorcycle", "aeroplane": "airplane", "plane": "airplane",
    "sofa": "couch", "table": "dining table", "mobile phone": "cell phone",
    "cellphone": "cell phone", "phone": "cell phone", "computer mouse": "mouse",
    "computer keyboard": "keyboard", "doughnut": "donut", "houseplant": "potted plant",
    "plant": "potted plant", "flowerpot": "potted plant", "coffee cup": "cup", "mug": "cup",
    "ball": "sports ball", "tennis ball": "sports ball", "football": "sports ball",
    "soccer ball": "sports ball", "glass": "wine glass", "hydrant": "fire hydrant",
    "teddy": "teddy bear", "racket": "tennis racket", "bag": "handbag", "puppy": "dog",
    "kitten": "cat", "taxi": "car", "van": "car", "suv": "car", "lorry": "truck",
    "ship": "boat", "sailboat": "boat", "fridge": "refrigerator", "bike": "bicycle",
}  # fmt: skip


def normalize_label(label: str) -> str:
    """Lower-case, strip articles/plurals and map synonyms to the COCO vocabulary."""
    s = re.sub(r"[^a-z ]", " ", label.lower())
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    if s in LABEL_SYNONYMS:
        return LABEL_SYNONYMS[s]
    if s in COCO_NAMES:
        return s
    for cand in (s[:-1] if s.endswith("s") else None, s[:-2] if s.endswith("es") else None):
        if cand and (cand in COCO_NAMES or cand in LABEL_SYNONYMS):
            return LABEL_SYNONYMS.get(cand, cand)
    return s


def match_by_label(
    pred_boxes: np.ndarray,
    pred_labels: Sequence[str],
    gt_boxes: np.ndarray,
    gt_labels: Sequence[str],
    iou_thr: float = 0.5,
) -> dict[str, int]:
    """Greedy one-to-one matching of label-free-score detections to ground truth.

    A prediction is a true positive when its normalised label equals the GT label and
    IoU >= ``iou_thr`` with a still-unmatched GT box (highest IoU first). Returns counts
    ``tp``, ``fp``, ``fn`` (Florence-2 gives no confidences, so no PR curve / AP).
    """
    from cvlab.metrics import box_iou

    pb = np.asarray(pred_boxes, np.float64).reshape(-1, 4)
    gb = np.asarray(gt_boxes, np.float64).reshape(-1, 4)
    pl = [normalize_label(s) for s in pred_labels]
    gl = [normalize_label(s) for s in gt_labels]
    if len(pb) == 0 or len(gb) == 0:
        return {"tp": 0, "fp": len(pb), "fn": len(gb)}
    iou = box_iou(pb, gb)
    same = np.array([[a == b for b in gl] for a in pl])
    iou = np.where(same, iou, 0.0)
    pairs = sorted(
        ((iou[i, j], i, j) for i in range(len(pb)) for j in range(len(gb)) if iou[i, j] >= iou_thr),
        reverse=True,
    )
    used_p, used_g = set(), set()
    for _, i, j in pairs:
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
    tp = len(used_p)
    return {"tp": tp, "fp": len(pb) - tp, "fn": len(gb) - tp}


# ------------------------------------------------------------------------------- VLM
@dataclass
class VLMAnswer:
    """One VLM generation with its cost."""

    text: str
    prompt_tokens: int
    image_tokens: int
    new_tokens: int
    latency_s: float

    @property
    def tokens_per_s(self) -> float:
        return self.new_tokens / max(self.latency_s, 1e-9)


class SmallVLM:
    """Chat VLM (Qwen3.5-0.8B by default) loaded natively in transformers, bf16.

    ``max_pixels`` caps the image resolution the processor feeds to the vision
    encoder; each visual token covers a 32x32 pixel block (16-px patches merged 2x2),
    so a 640x480 frame becomes 20 * 15 = 300 visual tokens.
    """

    def __init__(self, device: str = "cuda", dtype=None, max_pixels: int | None = None) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        cfg = model_cfg("vlm")
        self.name = cfg["repo"].split("/")[-1]
        self.device, self.dtype = str(device), dtype or torch.bfloat16
        self.processor = AutoProcessor.from_pretrained(cfg["repo"], revision=cfg["revision"])
        if max_pixels is not None:
            self.set_pixel_budget(max_pixels)
        self.model = AutoModelForImageTextToText.from_pretrained(
            cfg["repo"], revision=cfg["revision"], dtype=self.dtype
        ).to(device).eval()  # fmt: skip
        self.image_token_id = self.model.config.image_token_id

    def set_pixel_budget(self, max_pixels: int, min_pixels: int | None = None) -> None:
        """Bound the pixels per image fed to the vision encoder (the visual-token count).

        The processor resizes every image (keeping its aspect ratio, sides rounded to
        multiples of 32 px) so that ``min_pixels <= H * W <= max_pixels``; one visual
        token covers 32 x 32 px. Raising ``min_pixels`` above the image area upsamples it.
        """
        from transformers.image_utils import SizeDict

        ip = self.processor.image_processor
        hi = int(max_pixels)
        lo = int(min_pixels) if min_pixels is not None else min(int(ip.size["shortest_edge"]), hi)
        # shortest_edge / longest_edge hold the min / max pixel count for this processor.
        ip.size = SizeDict(**{**dict(ip.size), "longest_edge": hi, "shortest_edge": lo})

    @property
    def pixel_budget(self) -> tuple[int, int]:
        """Current ``(min_pixels, max_pixels)`` of the image processor."""
        size = self.processor.image_processor.size
        return int(size["shortest_edge"]), int(size["longest_edge"])

    def inputs(self, image, question: str, system: str | None = None):
        msgs = []
        if system:
            msgs.append({"role": "system", "content": [{"type": "text", "text": system}]})
        content = ([{"type": "image", "image": image}] if image is not None else []) + [
            {"type": "text", "text": question}
        ]
        msgs.append({"role": "user", "content": content})
        # enable_thinking=False: answer directly (no <think> block) - the model card warns
        # that the small Qwen3.5 models can loop in thinking mode.
        return self.processor.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors="pt", enable_thinking=False,
        ).to(self.device)  # fmt: skip

    def ask(
        self,
        image,
        question: str,
        max_new_tokens: int = 64,
        min_new_tokens: int | None = None,
        system: str | None = None,
    ) -> VLMAnswer:
        """Greedy-decode an answer and time it (CUDA-synchronised)."""
        import torch

        enc = self.inputs(image, question, system)
        n_prompt = int(enc["input_ids"].shape[1])
        n_img = int((enc["input_ids"] == self.image_token_id).sum())
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = self.model.generate(
                **enc, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens,
                do_sample=False, temperature=None, top_p=None, top_k=None,
            )  # fmt: skip
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        new = out[0, n_prompt:]
        text = self.processor.decode(new, skip_special_tokens=True).strip()
        return VLMAnswer(text, n_prompt, n_img, int(new.shape[0]), dt)


# --------------------------------------------------------------------- answer parsing
_NUMBER_WORDS = {
    "zero": 0, "no": 0, "none": 0, "one": 1, "a": 1, "an": 1, "single": 1, "two": 2,
    "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12,
}  # fmt: skip


def parse_yes_no(answer: str) -> bool | None:
    """``True`` / ``False`` from the first yes/no word of an answer (``None`` if neither)."""
    m = re.search(r"\b(yes|no)\b", answer.lower())
    return None if m is None else m.group(1) == "yes"


def parse_count(answer: str) -> int | None:
    """First integer in an answer, digits first (``"3"``), then number words (``"three"``)."""
    s = answer.lower()
    m = re.search(r"\b\d+\b", s)
    if m:
        return int(m.group(0))
    for w in re.findall(r"[a-z]+", s):
        if w in _NUMBER_WORDS and w not in {"a", "an", "no"}:
            return _NUMBER_WORDS[w]
    if re.search(r"\b(no|none)\b", s):
        return 0
    return None


def parse_left_right(answer: str) -> str | None:
    """``"left"`` / ``"right"`` from the first occurrence in an answer."""
    m = re.search(r"\b(left|right)\b", answer.lower())
    return None if m is None else m.group(1)


def parse_above_below(answer: str) -> str | None:
    """``"above"`` / ``"below"`` from the first occurrence in an answer."""
    m = re.search(r"\b(above|below)\b", answer.lower())
    return None if m is None else m.group(1)


# -------------------------------------------------------- VQA probes from ground truth
def _article(name: str) -> str:
    return "an" if name[0] in "aeiou" else "a"


def presence_probes(
    items: Sequence[CocoItem], n_pos: int = 2, n_neg: int = 2, seed: int = 0
) -> list[dict[str, Any]]:
    """POPE-style yes/no questions: classes present in the GT vs randomly drawn absent ones."""
    rng = np.random.default_rng(seed)
    probes = []
    for k, it in enumerate(items):
        present = sorted(set(it.names()))
        absent = [c for c in COCO_NAMES if c not in present]
        pos = rng.choice(present, size=min(n_pos, len(present)), replace=False) if present else []
        neg = rng.choice(absent, size=n_neg, replace=False)
        for c, ans in [(c, True) for c in pos] + [(c, False) for c in neg]:
            q = f"Is there {_article(c)} {c} in the image? Answer yes or no."
            probes.append({"item": k, "kind": "presence", "object": c, "question": q,
                           "truth": ans})  # fmt: skip
    return probes


_PLURALS = {
    "person": "people", "mouse": "computer mice", "knife": "knives", "sheep": "sheep",
    "skis": "pairs of skis", "scissors": "pairs of scissors",
    "tv": "TVs", "bus": "buses", "couch": "couches", "bench": "benches",
    "sandwich": "sandwiches", "toothbrush": "toothbrushes", "wine glass": "wine glasses",
    "sports ball": "balls", "broccoli": "pieces of broccoli",
}  # fmt: skip


def plural(name: str) -> str:
    """English plural of a COCO class name (``"person"`` -> ``"people"``)."""
    return _PLURALS.get(name, name + "s")


def count_probes(items: Sequence[CocoItem], max_count: int = 6) -> list[dict[str, Any]]:
    """'How many X?' for every class with 1..``max_count`` GT instances in an image."""
    probes = []
    for k, it in enumerate(items):
        names, counts = np.unique(it.names(), return_counts=True)
        for c, n in zip(names, counts, strict=True):
            if 1 <= n <= max_count:
                q = f"How many {plural(str(c))} are in the image? Answer with a single number."
                probes.append({"item": k, "kind": "count", "object": str(c), "question": q,
                               "truth": int(n)})  # fmt: skip
    return probes


def spatial_probes(
    items: Sequence[CocoItem], min_gap: float = 0.05, seed: int = 0
) -> list[dict[str, Any]]:
    """Left/right and above/below questions between two single-instance objects.

    For each image and each axis, the pair of single-instance objects whose boxes are
    separated the most along that axis is used, provided the gap between the boxes is at
    least ``min_gap`` of the image size (so the relation is unambiguous: the boxes do not
    overlap along that axis). The question order is randomised so that each answer word
    is correct about half of the time. Image coordinates: x to the right, y downwards.
    """
    rng = np.random.default_rng(seed)
    probes = []
    for k, it in enumerate(items):
        names, counts = np.unique(it.names(), return_counts=True)
        single = [str(n) for n, c in zip(names, counts, strict=True) if c == 1]
        box = {COCO_NAMES[c]: b for c, b in zip(it.classes, it.boxes, strict=True)}
        axes = ((0, it.width, ("left", "right")), (1, it.height, ("above", "below")))
        for axis, size, words in axes:
            best = None
            for i, a in enumerate(single):
                for b in single[i + 1 :]:
                    ba, bb = box[a], box[b]
                    gap = max(bb[axis] - ba[axis + 2], ba[axis] - bb[axis + 2]) / size
                    if gap >= min_gap and (best is None or gap > best[0]):
                        best = (gap, a, b)
            if best is None:
                continue
            _, a, b = best
            if rng.random() < 0.5:
                a, b = b, a
            truth = words[0] if box[a][axis + 2] <= box[b][axis] else words[1]
            if axis == 0:
                q = (f"Is the {a} to the left or to the right of the {b} in the image? "
                     "Answer with one word: left or right.")  # fmt: skip
            else:
                q = (f"Is the {a} above or below the {b} in the image? "
                     "Answer with one word: above or below.")  # fmt: skip
            probes.append({"item": k, "kind": "spatial", "object": f"{a} vs {b}", "question": q,
                           "truth": truth})  # fmt: skip
    return probes


def parse_on_off(answer: str) -> str | None:
    """``"on"`` / ``"off"`` from the first occurrence in an answer."""
    m = re.search(r"\b(on|off)\b", answer.lower())
    return None if m is None else m.group(1)


def score_probe(kind: str, truth: Any, answer: str) -> tuple[Any, bool]:
    """Parse a free-text answer for a question kind and say whether it is correct.

    Kinds: ``presence`` (yes/no), ``count`` (integer), ``spatial`` (left/right or
    above/below, chosen from the truth word), ``state`` (on/off) and ``keyword``
    (correct if any of the ``truth`` keywords occurs in the lower-cased answer).
    """
    if kind == "presence":
        pred = parse_yes_no(answer)
    elif kind == "count":
        pred = parse_count(answer)
    elif kind == "spatial":
        pred = parse_left_right(answer) if truth in ("left", "right") else parse_above_below(answer)
    elif kind == "state":
        pred = parse_on_off(answer)
    elif kind == "keyword":
        keys = [truth] if isinstance(truth, str) else list(truth)
        hit = [k for k in keys if k.lower() in answer.lower()]
        return (hit[0] if hit else None), bool(hit)
    else:
        raise ValueError(kind)
    return pred, pred == truth


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval of a binomial proportion ``k / n`` (robust for small n)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z**2 / n
    mid = (p + z**2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return (float(mid - half), float(mid + half))
