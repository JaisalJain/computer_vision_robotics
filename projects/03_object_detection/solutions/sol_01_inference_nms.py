#!/usr/bin/env python
"""Solutions to the exercises of notebook 01 (YOLO26 inference, NMS, one-to-one head).

Run from the repository root under the GPU lock::

    flock /tmp/cv-gpu.lock python projects/03_object_detection/solutions/sol_01_inference_nms.py

Figures are written to ``outputs/03_object_detection/solutions/`` (gitignored).
"""

from __future__ import annotations

import logging
import os
import sys

os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.utils import LOGGER  # noqa: E402

from cvlab import io as cvio  # noqa: E402
from cvlab import viz  # noqa: E402
from cvlab.metrics import box_iou  # noqa: E402
from cvlab.tum import load_sequence  # noqa: E402

LOGGER.setLevel(logging.ERROR)
sys.path.insert(0, str(cvio.project_dir("03_object_detection") / "scripts"))
import detlib  # noqa: E402

OUT = cvio.outputs_dir("03_object_detection", "solutions", create=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def letterbox(img: np.ndarray, size: int = 640) -> tuple[np.ndarray, float, tuple[int, int]]:
    h, w = img.shape[:2]
    r = size / max(h, w)
    nh, nw = round(h * r), round(w * r)
    out = np.full((size, size, 3), 114, np.uint8)
    px, py = (size - nw) // 2, (size - nh) // 2
    out[py : py + nh, px : px + nw] = cv2.resize(img, (nw, nh))
    return out, r, (px, py)


def one_to_many_candidates(net, img: np.ndarray, conf: float = 0.25):
    """Decoded one-to-many candidates (xyxy in letterboxed px, best score, best class)."""
    lb, _, _ = letterbox(img)
    x = torch.from_numpy(lb).permute(2, 0, 1)[None].to(DEVICE).float() / 255
    with torch.inference_mode():
        y = net(x)[0][0]  # (4 + nc, 8400)
    s, c = y[4:].max(0)
    k = s > conf
    xywh = y[:4, k].T.cpu().numpy()
    boxes = np.concatenate([xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2], 1)
    return boxes, s[k].cpu().numpy(), c[k].cpu().numpy()


def class_aware_nms(boxes, scores, classes, iou=0.7):
    keep = [
        np.flatnonzero(classes == c)[
            detlib.nms_numpy(boxes[classes == c], scores[classes == c], iou)
        ]
        for c in np.unique(classes)
    ]
    return np.concatenate(keep) if keep else np.zeros(0, int)


def soft_nms(boxes, scores, sigma=0.5, score_thr=0.25):
    """Gaussian Soft-NMS: decay overlapping scores by exp(-IoU^2 / sigma) instead of deleting."""
    scores = scores.astype(np.float64).copy()
    idx = np.arange(len(scores))
    keep = []
    while idx.size:
        i = idx[np.argmax(scores[idx])]
        if scores[i] < score_thr:
            break
        keep.append(int(i))
        idx = idx[idx != i]
        if idx.size:
            iou = box_iou(boxes[i : i + 1], boxes[idx])[0]
            scores[idx] *= np.exp(-(iou**2) / sigma)
    return np.asarray(keep, int), scores


def main() -> None:
    model = YOLO("yolo26s.pt")
    net = model.model.to(DEVICE).eval()
    net.model[-1].end2end = False  # one-to-many head
    names = model.names
    seq = load_sequence(cvio.data_dir("tum", "fr1_desk", "rgbd_dataset_freiburg1_desk"))
    images = {
        p.stem: cvio.read_image(p)
        for p in sorted(cvio.data_dir("ultralytics", "coco8", "images", "val").glob("*.jpg"))
    }
    images.update({f"tum_desk_{i:03d}": seq.rgb(i) for i in (60, 330)})

    # Exercise 1: class-aware vs class-agnostic NMS.
    rows = []
    for k, img in images.items():
        b, s, c = one_to_many_candidates(net, img)
        aware = class_aware_nms(b, s, c)
        agnostic = detlib.nms_numpy(b, s, 0.7)
        removed = sorted({names[int(c[i])] for i in set(aware) - set(agnostic)})
        rows.append(
            {
                "image": k,
                "candidates": len(b),
                "class-aware": len(aware),
                "agnostic": len(agnostic),
                "removed by agnostic": ", ".join(removed),
            }
        )
    print("Exercise 1 - class-aware vs agnostic NMS (YOLO26s, conf 0.25, IoU 0.7)")
    print(pd.DataFrame(rows).to_string(index=False))

    # Exercise 2: imgsz sweep with YOLO26n on 30 TUM frames.
    small = YOLO("yolo26n.pt")
    frames = [np.ascontiguousarray(seq.rgb(i)[..., ::-1]) for i in range(0, 300, 10)]
    rows = []
    for size in (320, 480, 640, 960):
        for f in frames[:3]:
            small.predict(f, imgsz=size, verbose=False)
        res = [small.predict(f, imgsz=size, verbose=False)[0] for f in frames]
        lat = np.median([sum(r.speed.values()) for r in res])
        areas = np.concatenate(
            [(r.boxes.xywh[:, 2] * r.boxes.xywh[:, 3]).cpu().numpy() for r in res]
        )
        rows.append(
            {
                "imgsz": size,
                "predict median [ms]": lat,
                "boxes": len(areas),
                "boxes < 32x32 px": int((areas < 32**2).sum()),
            }
        )
    df = pd.DataFrame(rows)
    print("\nExercise 2 - imgsz sweep (YOLO26n, 30 TUM frames)")
    print(df.round(2).to_string(index=False))
    viz.set_style()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(df["imgsz"], df["predict median [ms]"], "o-", label="latency")
    ax2 = ax.twinx()
    ax2.plot(
        df["imgsz"],
        df["boxes < 32x32 px"],
        "s--",
        color=viz.color(1, unit=True),
        label="small boxes",
    )
    (
        ax.set_xlabel("imgsz [px]"),
        ax.set_ylabel("predict median [ms]"),
        ax2.set_ylabel("boxes < 32x32 px"),
    )
    ax.set_title("YOLO26n: input size vs latency and small detections")
    fig.savefig(OUT / "ex01_2_imgsz.png", dpi=100)

    # Exercise 3: Gaussian Soft-NMS vs hard NMS on the crowded COCO8 image.
    b, s, c = one_to_many_candidates(net, images["000000000049"], conf=0.05)
    hard = class_aware_nms(b, s, c)
    hard = hard[s[hard] >= 0.25]
    soft_keep = []
    for cls in np.unique(c):
        m = np.flatnonzero(c == cls)
        k, _ = soft_nms(b[m], s[m])
        soft_keep += list(m[k])
    print(
        "\nExercise 3 - hard vs Soft-NMS on 000000000049 (candidates from conf 0.05, keep >= 0.25)"
    )
    print(f"  candidates {len(b)} | hard NMS keeps {len(hard)} | Soft-NMS keeps {len(soft_keep)}")
    print("  hard:", sorted(names[int(c[i])] for i in hard))
    print("  soft:", sorted(names[int(c[i])] for i in soft_keep))


if __name__ == "__main__":
    main()
