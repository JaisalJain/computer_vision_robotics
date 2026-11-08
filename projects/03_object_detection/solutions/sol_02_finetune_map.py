#!/usr/bin/env python
"""Solutions to the exercises of notebook 02 (fine-tuning and mAP).

Exercise 1 needs the medical-pills run first (about one minute)::

    flock /tmp/cv-gpu.lock python projects/03_object_detection/scripts/train_yolo.py \
        --model yolo26n.pt --data medical-pills.yaml --epochs 30 --name yolo26n_pills
    flock /tmp/cv-gpu.lock python projects/03_object_detection/solutions/sol_02_finetune_map.py
"""

from __future__ import annotations

import logging
import os
import sys

os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.utils import LOGGER  # noqa: E402

from cvlab import io as cvio  # noqa: E402

LOGGER.setLevel(logging.ERROR)
PROJ = cvio.project_dir("03_object_detection")
sys.path.insert(0, str(PROJ / "scripts"))
import detlib  # noqa: E402

CLASSES = ["buffalo", "elephant", "rhino", "zebra"]
ROOT = cvio.repo_root()


def predict_split(model, samples, conf=0.001):
    preds = []
    for i in range(0, len(samples), 32):
        paths = [str(s.image) for s in samples[i : i + 32]]
        for r in model.predict(paths, conf=conf, max_det=300, batch=16, verbose=False):
            preds.append(detlib.ultralytics_to_detections(r))
    return preds


def bucket(samples, preds, lo, hi):
    """Keep GT boxes and predictions whose area lies in [lo, hi) (simplified COCO size split)."""

    def area(b):
        return (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    s2, p2 = [], []
    for s, p in zip(samples, preds, strict=True):
        ms = (area(s.boxes) >= lo) & (area(s.boxes) < hi)
        mp = (area(p.boxes) >= lo) & (area(p.boxes) < hi)
        s2.append(detlib.Sample(s.image, s.width, s.height, s.boxes[ms], s.classes[ms]))
        p2.append(detlib.Detections(p.boxes[mp], p.scores[mp], p.classes[mp]))
    return s2, p2


def main() -> None:
    # Exercise 1: medical-pills fine-tune (trained by scripts/train_yolo.py) evaluated with detlib.
    pills = cvio.load_json(PROJ / "results" / "yolo26n_pills_summary.json")
    pval = detlib.load_split(cvio.data_dir("ultralytics", "medical-pills"), "val")
    res = detlib.evaluate(
        pval, predict_split(YOLO(ROOT / pills["best_checkpoint"]), pval), ["pill"]
    )
    print("Exercise 1 - medical-pills (YOLO26n, 30 epochs)")
    print(f"  train time {pills['train_minutes']} min, peak VRAM {pills['peak_vram_mb']} MiB")
    print(
        f"  val: {len(pval)} images, {int(res.n_gt[0])} pills | detlib mAP50 {res.map50:.3f} "
        f"mAP50-95 {res.map50_95:.3f} | Ultralytics (last val during training) mAP50-95 "
        f"{pills['final_val_metrics']['metrics/mAP50-95(B)']:.3f}"
    )
    # A crude reliability estimate: bootstrap the 23 val images.
    rng = np.random.default_rng(0)
    preds = predict_split(YOLO(ROOT / pills["best_checkpoint"]), pval)
    boot = []
    for _ in range(200):
        idx = rng.integers(0, len(pval), len(pval))
        boot.append(
            detlib.evaluate([pval[i] for i in idx], [preds[i] for i in idx], ["pill"]).map50_95
        )
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  bootstrap 95% interval of mAP50-95 over images: [{lo:.3f}, {hi:.3f}]")

    # Exercise 2: val vs test for both wildlife models.
    data = cvio.data_dir("ultralytics", "african-wildlife")
    splits = {sp: detlib.load_split(data, sp) for sp in ("val", "test")}
    rows = []
    for k in ("yolo26n", "yolo26s"):
        ckpt = (
            ROOT
            / cvio.load_json(PROJ / "results" / f"{k}_wildlife_summary.json")["best_checkpoint"]
        )
        model = YOLO(ckpt)
        for sp, samples in splits.items():
            r = detlib.evaluate(samples, predict_split(model, samples), CLASSES)
            rows.append(
                {
                    "model": k,
                    "split": sp,
                    "mAP50": r.map50,
                    "mAP50-95": r.map50_95,
                    **{c: v["AP50-95"] for c, v in r.per_class().items()},
                }
            )
        if k == "yolo26n":
            val_preds = predict_split(model, splits["val"])
    print("\nExercise 2 - val vs test (AP50-95 per class)")
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    # Exercise 3: AP by object size for the fine-tuned YOLO26n on val.
    rows = []
    for name, lo_a, hi_a in [("small", 0, 32**2), ("medium", 32**2, 96**2), ("large", 96**2, 1e12)]:
        s2, p2 = bucket(splits["val"], val_preds, lo_a, hi_a)
        r = detlib.evaluate(s2, p2, CLASSES)
        rows.append(
            {
                "bucket": name,
                "GT boxes": int(r.n_gt.sum()),
                "mAP50": r.map50,
                "mAP50-95": r.map50_95,
            }
        )
    print("\nExercise 3 - size-stratified AP (YOLO26n fine-tuned, val)")
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
