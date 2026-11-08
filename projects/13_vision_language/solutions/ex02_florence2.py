#!/usr/bin/env python
"""Solutions to the exercises of notebooks/02_florence2_caption_detect_ground.ipynb.

1. ``<OCR_WITH_REGION>`` on the desk close-up, full frame vs an upsampled crop of the book.
2. ``<OD>`` on coco128 with greedy decoding vs 3-beam search (precision, recall, latency).
3. A "where is X?" robot primitive: phrase -> grounded box -> SigLIP 2 check -> 3D point.

Figures go to ``outputs/13_vision_language/`` (git-ignored). Run (GPU, about 1 minute)::

    python projects/13_vision_language/solutions/ex02_florence2.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import transformers  # noqa: E402
import vl_common as vl  # noqa: E402
from PIL import Image  # noqa: E402

from cvlab.device import setup  # noqa: E402
from cvlab.io import data_dir, outputs_dir  # noqa: E402
from cvlab.tum import load_sequence  # noqa: E402

transformers.logging.set_verbosity_error()


def draw_quads(ax, img, res, title):
    """Show an image with OCR quadrilaterals and their text."""
    ax.imshow(img)
    for q, text in zip(res.quads, res.labels):
        pts = q.reshape(4, 2)
        ax.add_patch(plt.Polygon(pts, fill=False, ec="tab:orange", lw=1.5))
        ax.text(
            pts[0, 0],
            pts[0, 1] - 2,
            text,
            fontsize=7,
            color="w",
            bbox={"fc": "tab:orange", "pad": 1},
        )
    ax.set_title(title)
    ax.axis("off")


def robust_depth(depth: np.ndarray, box: np.ndarray, u: int, v: int) -> tuple[float, str]:
    """Depth (metres) at pixel (u, v): median of the valid pixels in a 9 x 9 window.

    Dark or shiny objects (the black game controller) often leave holes (depth 0) in a
    structured-light sensor. If the whole window is a hole, fall back to the median valid depth
    in the central half of the box, which still lies on the object for a tight box.
    """
    win = depth[max(0, v - 4) : v + 5, max(0, u - 4) : u + 5]
    if np.any(win > 0):
        return float(np.median(win[win > 0])), "9x9 window"
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    inner = depth[int(y1 + h / 4) : int(y2 - h / 4) + 1, int(x1 + w / 4) : int(x2 - w / 4) + 1]
    if np.any(inner > 0):
        return float(np.median(inner[inner > 0])), "window is a depth hole; central half of box"
    return float("nan"), "no valid depth"


def main() -> None:
    device = setup(seed=0)
    out = outputs_dir("13_vision_language", create=True)
    fl = vl.Florence2(device)
    frames, _ = vl.desk_questions()
    img = vl.load_rgb(frames["desk_e"])

    # ---- Exercise 1: OCR on the full frame and on an upsampled crop of the grounded book
    full = fl.run(img, "<OCR_WITH_REGION>")
    book = fl.run(img, "<CAPTION_TO_PHRASE_GROUNDING>", "the blue book").boxes[0]
    x1, y1, x2, y2 = book.round().astype(int)
    crop = img.crop((x1, y1, x2, y2))
    crop = crop.resize((crop.width * 3, crop.height * 3), Image.Resampling.BICUBIC)
    zoom = fl.run(crop, "<OCR_WITH_REGION>")
    print("Exercise 1 - OCR lines, full frame:", full.labels)
    print("Exercise 1 - OCR lines, 3x crop of the blue book:", zoom.labels)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    draw_quads(axes[0], img, full, "<OCR_WITH_REGION> on the full frame")
    draw_quads(axes[1], crop, zoom, "same task on a 3x upsampled crop of 'the blue book'")
    fig.savefig(out / "ex02_ocr.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

    # ---- Exercise 2: greedy vs beam search for <OD> on coco128
    coco = data_dir("ultralytics", "coco128")
    items = vl.load_yolo_split(coco / "images" / "train2017", coco / "labels" / "train2017")
    fl.run(img, "<OD>")  # warm-up
    rows = []
    for beams in (1, 3):
        tp = fp = fn = 0
        lat = []
        for it in items:
            r = fl.run(vl.load_rgb(it.path), "<OD>", num_beams=beams)
            m = vl.match_by_label(r.boxes, r.labels, it.boxes, it.names())
            tp, fp, fn = tp + m["tp"], fp + m["fp"], fn + m["fn"]
            lat.append(r.latency_ms)
        rows.append(
            {
                "num_beams": beams,
                "precision@0.5": tp / (tp + fp),
                "recall@0.5": tp / (tp + fn),
                "predictions": tp + fp,
                "mean latency (ms)": np.mean(lat),
            }
        )
    print("\nExercise 2 - <OD> on coco128, greedy vs beam search")
    print(pd.DataFrame(rows).to_string(index=False, float_format="{:.3f}".format))

    # ---- Exercise 3: phrase -> verified box -> 3D point (camera frame, metres)
    siglip = vl.SigLIP2Encoder(device)
    seq = load_sequence(data_dir("tum", "fr1_desk", "rgbd_dataset_freiburg1_desk"))
    i = next(k for k, p in enumerate(seq.rgb_paths) if p.name == frames["desk_e"].name)
    depth = seq.depth(i)  # metres, 0 = missing
    K = seq.camera.K
    print("\nExercise 3 - where is X? (TUM fr1_desk, frame", frames["desk_e"].name + ")")
    for phrase in ["the game controller", "the telephone", "a banana"]:
        box = fl.run(img, "<CAPTION_TO_PHRASE_GROUNDING>", phrase).boxes[0]
        bx1, by1, bx2, by2 = box
        crop = img.crop((bx1, by1, bx2, by2))
        p = vl.zero_shot_scores(
            siglip.encode_images([crop]),
            siglip.encode_texts([f"This is a photo of {phrase}."]),
            siglip.scale,
            siglip.bias,
            "sigmoid",
        )[0, 0]
        u, v = int(round((bx1 + bx2) / 2)), int(round((by1 + by2) / 2))
        z, src = robust_depth(depth, box, u, v)
        X = z * np.linalg.inv(K) @ np.array([u, v, 1.0])
        verdict = "accept" if p >= 0.5 else "reject"
        print(
            f"  {phrase:20s} box {np.round(box).astype(int).tolist()} "
            f"SigLIP p={p:.3f} -> {verdict}; "
            f"pixel ({u}, {v}), z={z:.3f} m ({src}) -> X_cam = {np.round(X, 3).tolist()} m"
        )
    print(f"\nfigure: {vl.rel(out / 'ex02_ocr.png')}")


if __name__ == "__main__":
    main()
