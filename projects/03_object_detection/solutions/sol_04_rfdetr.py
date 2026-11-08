#!/usr/bin/env python
"""Solutions to the exercises of notebook 04 (YOLO26 vs RF-DETR).

Run from the repository root under the GPU lock::

    flock /tmp/cv-gpu.lock python projects/03_object_detection/solutions/sol_04_rfdetr.py

The comparison figure for exercise 2 goes to ``outputs/03_object_detection/solutions/``.
"""

from __future__ import annotations

import logging
import os
import sys
from copy import deepcopy

from cvlab import io as cvio

os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("RF_HOME", str(cvio.weights_dir("rfdetr")))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from rfdetr import RFDETRNano, RFDETRSmall  # noqa: E402
from rfdetr.assets.coco_classes import COCO_CLASSES  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.utils import LOGGER  # noqa: E402

from cvlab import viz  # noqa: E402
from cvlab.bench import benchmark  # noqa: E402
from cvlab.tum import load_sequence  # noqa: E402

LOGGER.setLevel(logging.ERROR)
logging.getLogger("rf-detr").setLevel(logging.ERROR)
sys.path.insert(0, str(cvio.project_dir("03_object_detection") / "scripts"))
import detlib  # noqa: E402

OUT = cvio.outputs_dir("03_object_detection", "solutions", create=True)
NAME_TO_ID = {v: k for k, v in COCO_CLASSES.items()}


def two_class_val():
    remap = {1: 0, 3: 1}  # wildlife elephant, zebra -> 0, 1
    out = []
    for s in detlib.load_split(cvio.data_dir("ultralytics", "african-wildlife"), "val"):
        m = np.isin(s.classes, list(remap))
        out.append(
            detlib.Sample(
                s.image,
                s.width,
                s.height,
                s.boxes[m],
                np.array([remap[c] for c in s.classes[m]], int),
            )
        )
    return out


def best_f1(samples, preds):
    scores, _, tp = detlib.detection_flags(samples, preds, 0.5)
    n_gt = sum(len(s.classes) for s in samples)
    best = (0.0, 0.0, 0.0, 0.0)
    for t in np.linspace(0.01, 0.95, 95):
        k = scores >= t
        if not k.any():
            continue
        p, r = tp[k].mean(), tp[k].sum() / n_gt
        f = 2 * p * r / max(p + r, 1e-9)
        if f > best[3]:
            best = (t, p, r, f)
    return best


def main() -> None:
    val2 = two_class_val()
    paths = [str(s.image) for s in val2]

    # Exercise 1: best-F1 thresholds, RF-DETR Small vs YOLO26s (zero-shot, elephant + zebra).
    yolo_s = YOLO("yolo26s.pt")
    yp = []
    for i in range(0, len(paths), 32):
        for r in yolo_s.predict(paths[i : i + 32], conf=0.001, batch=16, verbose=False):
            yp.append(detlib.ultralytics_to_detections(r, {20: 0, 22: 1}))
    rf_small = RFDETRSmall().inference(compile=False)
    keep = {NAME_TO_ID["elephant"]: 0, NAME_TO_ID["zebra"]: 1}
    rp = []
    for i in range(0, len(paths), 16):
        for d in rf_small.predict(paths[i : i + 16], threshold=0.001):
            m = np.isin(d.class_id, list(keep))
            rp.append(
                detlib.Detections(
                    d.xyxy[m].astype(float),
                    d.confidence[m].astype(float),
                    np.array([keep[c] for c in d.class_id[m]], int),
                )
            )
    rows = []
    for name, preds in [("yolo26s@640", yp), ("rfdetr-small@512", rp)]:
        t, p, r, f = best_f1(val2, preds)
        rows.append({"model": name, "best-F1 threshold": t, "precision": p, "recall": r, "F1": f})
    print("Exercise 1 - best-F1 operating points (zero-shot, IoU 0.5)")
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    # Exercise 2: TUM desk frames.
    seq = load_sequence(cvio.data_dir("tum", "fr1_desk", "rgbd_dataset_freiburg1_desk"))
    panels, titles = [], []
    print("\nExercise 2 - TUM fr1_desk frames (score >= 0.3)")
    for i in (60, 330):
        img = seq.rgb(i)
        r = yolo_s.predict(np.ascontiguousarray(img[..., ::-1]), conf=0.3, verbose=False)[0]
        d = detlib.ultralytics_to_detections(r)
        yn = [yolo_s.names[c] for c in d.classes]
        rd = rf_small.predict(img, threshold=0.3)
        rn = [COCO_CLASSES[c] for c in rd.class_id]
        print(f"  frame {i}: YOLO26s {sorted(yn)}")
        print(f"  frame {i}: RF-DETR-S {sorted(rn)}")
        panels += [
            viz.draw_boxes(img, d.boxes, yn, d.scores, d.classes),
            viz.draw_boxes(img, rd.xyxy, rn, rd.confidence, rd.class_id),
        ]
        titles += [f"YOLO26s frame {i}", f"RF-DETR Small frame {i}"]
    viz.set_style()
    viz.show_images(panels, titles, ncols=2, size=4).savefig(OUT / "ex04_2_tum.png", dpi=90)

    # Exercise 3: batching throughput (fp16 forward pass).
    yolo_n = deepcopy(YOLO("yolo26n.pt").model).cuda().eval()
    yolo_n.model[-1].end2end = True
    yolo_n = yolo_n.fuse(verbose=False).half()
    rf_n = deepcopy(RFDETRNano().model.model).cuda().eval()
    rf_n.export()
    rf_n = rf_n.half()
    rows = []
    with torch.inference_mode():
        for name, net, size in [("yolo26n@640", yolo_n, 640), ("rfdetr-nano@384", rf_n, 384)]:
            for b in (1, 4, 16):
                x = torch.rand(b, 3, size, size, device="cuda", dtype=torch.float16)
                r = benchmark(net, x, warmup=10, iters=50)
                rows.append(
                    {
                        "model": name,
                        "batch": b,
                        "latency [ms]": r["p50_ms"],
                        "images/s": 1000 * b / r["p50_ms"],
                    }
                )
    print("\nExercise 3 - fp16 forward throughput vs batch size")
    print(pd.DataFrame(rows).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
