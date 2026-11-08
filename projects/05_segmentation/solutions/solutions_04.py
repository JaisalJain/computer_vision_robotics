#!/usr/bin/env python
"""Solutions to the exercises of notebook 04 (YOLO26-seg and SAM refinement).

Run from the repository root (GPU; hold the lock)::

    flock /tmp/cv-gpu.lock python projects/05_segmentation/solutions/solutions_04.py

The fine-tuning run goes to ``runs/05_segmentation/finetune_yolo26n_seg`` (gitignored).
"""

from __future__ import annotations

import logging
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch

os.environ["YOLO_AUTOINSTALL"] = "false"  # never pip-install at runtime

from ultralytics import YOLO  # noqa: E402
from ultralytics.utils import LOGGER  # noqa: E402
from ultralytics.utils.ops import scale_masks  # noqa: E402

from cvlab.device import setup  # noqa: E402
from cvlab.io import project_dir, runs_dir, weights_dir  # noqa: E402
from cvlab.metrics import box_iou  # noqa: E402

sys.path.insert(0, str(project_dir("05_segmentation") / "scripts"))
import segkit as sk  # noqa: E402

PROJECT = "05_segmentation"
YOLO_N = weights_dir("ultralytics", "yolo26n-seg.pt")


def exercise_1() -> None:
    """Fine-tune YOLO26n-seg on coco8-seg for 20 epochs and compare validation mask mAP."""
    base = YOLO(str(YOLO_N)).val(
        data="coco8-seg.yaml",
        imgsz=640,
        batch=4,
        plots=False,
        verbose=False,
        project=str(runs_dir(PROJECT)),
        name="sol_val_base",
        exist_ok=True,
    )
    model = YOLO(str(YOLO_N))
    t0 = time.time()
    model.train(
        data="coco8-seg.yaml",
        epochs=20,
        imgsz=640,
        batch=4,
        workers=2,
        plots=False,
        verbose=False,
        project=str(runs_dir(PROJECT)),
        name="finetune_yolo26n_seg",
        exist_ok=True,
        seed=0,
    )
    minutes = (time.time() - t0) / 60
    tuned = model.val(
        data="coco8-seg.yaml",
        imgsz=640,
        batch=4,
        plots=False,
        verbose=False,
        project=str(runs_dir(PROJECT)),
        name="sol_val_tuned",
        exist_ok=True,
    )
    print(
        f"[ex1] coco8-seg val mask mAP50-95: pretrained {base.seg.map:.4f} -> "
        f"fine-tuned {tuned.seg.map:.4f} (20 epochs, {minutes:.1f} min). The 4 val images are "
        "COCO train2017 images already seen in pre-training and the 4 train images are a "
        "different set, so neither number estimates real accuracy."
    )


def matched_masks(yolo: YOLO, samples, retina: bool):
    """YOLO detections matched to GT (same class, box IoU >= 0.5) with masks at image size."""
    out, times = [], []
    for s in samples:
        img = np.ascontiguousarray(s.image[..., ::-1])
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = yolo.predict(img, conf=0.25, retina_masks=retina, verbose=False)[0]
        torch.cuda.synchronize()
        times.append(1000 * (time.perf_counter() - t0))
        if r.masks is None or len(s.masks) == 0:
            continue
        m = r.masks.data
        if m.shape[-2:] != s.image.shape[:2]:  # letterboxed prototype resolution -> image size
            m = scale_masks(m[None].float(), s.image.shape[:2])[0]
        masks = m.cpu().numpy() > 0.5
        boxes, cls = r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)
        ious = box_iou(boxes, s.boxes) * (cls[:, None] == s.classes[None, :])
        taken = np.zeros(len(s.masks), bool)
        for i in np.argsort(-r.boxes.conf.cpu().numpy()):
            cand = np.where(taken, -1, ious[i])
            j = int(np.argmax(cand))
            if cand[j] >= 0.5:
                taken[j] = True
                out.append((s, i, j, masks[i], boxes[i]))
    return out, float(np.median(times[1:]))


def exercise_2(samples) -> None:
    """retina_masks=False vs True: mask quality on matched detections and latency."""
    yolo = YOLO(str(YOLO_N))
    rows = []
    for retina in (False, True):
        pairs, ms = matched_masks(yolo, samples, retina)
        rows.append(
            {
                "retina_masks": retina,
                "matched": len(pairs),
                "mean J": np.mean([sk.j_measure(m, s.masks[j]) for s, _, j, m, _ in pairs]),
                "mean F": np.mean([sk.f_measure(m, s.masks[j]) for s, _, j, m, _ in pairs]),
                "median ms/image": ms,
            }
        )
    print(
        f"[ex2] YOLO26n-seg on coco128-seg:\n{pd.DataFrame(rows).round(4).to_string(index=False)}"
    )


def exercise_3(samples) -> None:
    """Refine YOLO26n-seg boxes with SAM 2.1-Tiny instead of SAM 3."""
    yolo = YOLO(str(YOLO_N))
    pairs, _ = matched_masks(yolo, samples, retina=True)
    rows = []
    for mid in [sk.SAM2_IDS["tiny"], sk.SAM3_ID]:
        seg = sk.PromptableSegmenter(mid)
        js, fs, enc, dec = [], [], [], []
        by_image: dict[str, list] = {}
        for p in pairs:
            by_image.setdefault(p[0].name, []).append(p)
        for items in by_image.values():
            s = items[0][0]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            seg.set_image(s.image)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            masks = seg.predict(boxes=[b.tolist() for *_, b in items], multimask=False)[0][:, 0]
            torch.cuda.synchronize()
            enc.append(1000 * (t1 - t0))
            dec.append(1000 * (time.perf_counter() - t1))
            for (s, _, j, _, _), m in zip(items, masks):
                js.append(sk.j_measure(m, s.masks[j]))
                fs.append(sk.f_measure(m, s.masks[j]))
        rows.append(
            {
                "refiner": seg.label,
                "matched": len(js),
                "mean J": np.mean(js),
                "mean F": np.mean(fs),
                "median encoder ms": np.median(enc[1:]),
                "median decoder ms (all boxes)": np.median(dec[1:]),
            }
        )
        del seg
        torch.cuda.empty_cache()
    yolo_j = np.mean([sk.j_measure(m, s.masks[j]) for s, _, j, m, _ in pairs])
    yolo_f = np.mean([sk.f_measure(m, s.masks[j]) for s, _, j, m, _ in pairs])
    print(
        f"[ex3] YOLO26n-seg own masks: J {yolo_j:.4f}, F {yolo_f:.4f}\n"
        f"{pd.DataFrame(rows).round(4).to_string(index=False)}"
    )


def main() -> None:
    setup(seed=0)
    warnings.filterwarnings("ignore")
    LOGGER.setLevel(logging.WARNING)
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    samples = sk.load_coco_seg("coco128-seg", splits=("train2017",))
    exercise_1()
    exercise_2(samples)
    exercise_3(samples)


if __name__ == "__main__":
    main()
