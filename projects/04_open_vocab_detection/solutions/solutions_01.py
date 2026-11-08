#!/usr/bin/env python
"""Solutions to the exercises of notebook 01 (text-prompted detection).

Run (GPU, about 1 minute)::

    python projects/04_open_vocab_detection/solutions/solutions_01.py

Figures go to ``runs/04_open_vocab_detection/solutions/`` (gitignored).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["YOLO_AUTOINSTALL"] = "false"
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ovd  # noqa: E402

from cvlab import io as cvio  # noqa: E402
from cvlab import viz  # noqa: E402
from cvlab.device import setup  # noqa: E402
from cvlab.metrics import box_iou  # noqa: E402
from cvlab.tum import load_sequence  # noqa: E402

OUT = cvio.runs_dir("04_open_vocab_detection", "solutions", create=True)
SEQ = load_sequence(cvio.data_dir("tum", "fr1_desk", "rgbd_dataset_freiburg1_desk"))


def exercise_1() -> None:
    """Box count vs threshold for desk frame 500; lowest 'clean' threshold per model.

    'Clean' = no prompt returns more boxes than there are instances in the frame (one game
    controller, one cereal box, one telephone, one red toy camera). This is a proxy for "no
    false positives" that needs no box annotations (duplicate or wrong boxes raise the count).
    """
    frame = SEQ.rgb(500)
    expected = {"game controller": 1, "cereal box": 1, "telephone": 1, "red toy camera": 1}
    prompts = list(expected)
    ts = np.round(np.arange(0.10, 0.90, 0.05), 2)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, key in zip(axes, ["yoloe26s", "llmdet_t"]):
        det = ovd.load_detector(key).set_vocabulary(prompts)
        d_all = det.predict(frame, min_score=0.05)
        counts = {
            p: [(d_all.above(t).labels == i).sum() for t in ts] for i, p in enumerate(prompts)
        }
        clean = next(
            (t for k, t in enumerate(ts) if all(counts[p][k] <= expected[p] for p in prompts)), None
        )
        found = {p: int(counts[p][list(ts).index(clean)]) for p in prompts} if clean else {}
        print(f"[ex1] {det.label}: lowest clean threshold {clean}; boxes per prompt there {found}")
        for i, p in enumerate(prompts):
            ax.plot(ts, counts[p], marker="o", ms=3, label=p, color=viz.color(i, unit=True))
        if clean is not None:
            ax.axvline(clean, color="k", ls=":", lw=1)
        ax.set_title(f"{det.label}: boxes vs threshold (frame 500)")
        ax.set_xlabel("score threshold")
        ovd.free_detector(det)
    axes[0].set_ylabel("number of boxes")
    axes[1].legend()
    fig.savefig(OUT / "ex1_threshold_sweep.png", dpi=100)
    plt.close(fig)


def exercise_2() -> None:
    """YOLOE visual prompt: telephone box in frame 100 used to find telephones in frame 300."""
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

    ref, target = SEQ.rgb(100), SEQ.rgb(300)
    # 1) A slower but stronger text-prompted model finds the reference box once (YOLOE's own
    #    text prompt misses the telephone in frame 100 at its F1-optimal threshold of 0.30).
    det = ovd.load_detector("llmdet_t").set_vocabulary(["telephone"])
    d = det.predict(ref, min_score=0.35)  # LLMDet-T F1-optimal threshold on the COCO subset
    ovd.free_detector(det)
    if len(d) == 0:
        print("[ex2] LLMDet-T found no telephone in the reference frame")
        return
    box = d.topk(1).boxes  # (1, 4) xyxy in reference-image pixels
    print(f"[ex2] reference box from LLMDet-T (score {d.topk(1).scores[0]:.3f})")
    # Visual prompts need the unfused promptable head: reload a fresh model.
    from ultralytics import YOLOE
    from ultralytics.utils import SETTINGS

    model = YOLOE(str(Path(SETTINGS["weights_dir"]) / "yoloe-26s-seg.pt"))
    res = model.predict(
        target[..., ::-1].copy(),
        refer_image=ref[..., ::-1].copy(),
        visual_prompts={"bboxes": box, "cls": np.array([0])},
        predictor=YOLOEVPSegPredictor,
        conf=0.1,
        verbose=False,
    )[0]
    b, s = res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()
    print(
        f"[ex2] visual prompt box {box.round(0).tolist()} -> {len(b)} detections in frame 300, "
        f"scores {np.round(s, 3).tolist()}"
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].imshow(viz.draw_boxes(ref, box, ["visual prompt"]))
    axes[0].set_title("reference (frame 100)")
    axes[1].imshow(viz.draw_boxes(target, b, ["telephone (visual)"] * len(b), s))
    axes[1].set_title("target (frame 300)")
    for ax in axes:
        ax.axis("off")
    fig.savefig(OUT / "ex2_visual_prompt.png", dpi=100)
    plt.close(fig)
    del model
    ovd.free_detector(None)  # gc + empty the CUDA cache


def exercise_3() -> None:
    """OWLv2 prompt ensembling vs single template on the cats image (present/absent scores)."""
    from PIL import Image

    templates = [
        "a photo of a {}",
        "a close-up photo of a {}",
        "a photo of the {}",
        "a {} in a scene",
        "a blurry photo of a {}",
        "a good photo of a {}",
    ]
    present = {"cat": "cat", "remote control": "remote"}
    absent = ["dog", "laptop", "giraffe", "fire hydrant", "person", "pizza"]
    names = list(present) + absent
    img = cvio.read_image(cvio.data_dir("coco", "val2017", "000000039769.jpg"))
    gt = cvio.load_json(cvio.data_dir("coco", "annotations", "instances_val2017.json"))
    cat_name = {c["id"]: c["name"] for c in gt["categories"]}
    gt_boxes: dict[str, list] = {}
    for a in gt["annotations"]:
        if a["image_id"] == 39769:
            x, y, w, h = a["bbox"]
            gt_boxes.setdefault(cat_name[a["category_id"]], []).append([x, y, x + w, y + h])

    det = ovd.load_detector("owlv2_b")
    m, proc = det.model, det.processor
    with torch.inference_mode():
        pix = proc(images=Image.fromarray(img), return_tensors="pt")["pixel_values"]
        fmap = m.image_embedder(pixel_values=pix.to(det.device, det.dtype))[0]  # (1, 60, 60, D)
        feats = fmap.reshape(1, -1, fmap.shape[-1])
        boxes = m.box_predictor(feats, fmap)[0].float()  # (3600, 4) cxcywh, padded-square units

        def embed(texts):
            tok = proc(text=texts, return_tensors="pt").to(det.device)
            e = m.owlv2.get_text_features(
                input_ids=tok["input_ids"], attention_mask=tok["attention_mask"]
            ).pooler_output
            return torch.nn.functional.normalize(e.float(), dim=-1)

        single = embed([templates[0].format(n) for n in names])
        ens = torch.stack([embed([t.format(n) for n in names]) for t in templates]).mean(0)
        ens = torch.nn.functional.normalize(ens, dim=-1)
        side = float(max(img.shape[:2]))
        cx, cy, bw, bh = boxes.unbind(-1)
        xyxy = (
            (torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], -1) * side)
            .cpu()
            .numpy()
        )
        for tag, q in [("single template", single), ("ensemble of 6", ens)]:
            logits, _ = m.class_predictor(
                feats,
                q[None].to(det.dtype),
                torch.ones(1, len(names), dtype=torch.bool, device=det.device),
            )
            sc = logits[0].float().sigmoid().cpu().numpy()  # (3600, n_names)
            pos = []
            for j, g in enumerate(present.values()):
                iou = box_iou(np.asarray(gt_boxes[g], float), xyxy)
                pos.append(np.mean([(sc[:, j] * (row >= 0.5)).max() for row in iou]))
            neg = sc[:, len(present) :].max(0)
            print(
                f"[ex3] OWLv2 {tag:16s}: present-object score {np.mean(pos):.3f} | "
                f"absent-object max score {neg.mean():.3f} | margin {np.mean(pos) - neg.mean():.3f}"
            )
    ovd.free_detector(det)


if __name__ == "__main__":
    setup(seed=0, verbose=False)
    exercise_1()
    exercise_2()
    exercise_3()
    print(f"figures in {OUT.relative_to(cvio.repo_root())}")
