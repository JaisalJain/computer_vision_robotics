#!/usr/bin/env python
"""Solutions to the exercises of notebook 01 (text -> boxes -> masks). Needs the GPU.

flock /tmp/cv-gpu.lock python projects/15_robotics_capstone/solutions/nb01_solutions.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import transformers

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import capstone as cap  # noqa: E402

from cvlab.metrics import mask_iou  # noqa: E402

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()
DEV = "cuda"
FRAMES = [100, 300, 460, 540]


def exercise1(frames: cap.Frames) -> None:
    """Grounding DINO box-threshold sweep with two extra queries on frame 540."""
    queries = cap.QUERIES + ["monitor", "pen"]
    rgb = frames.rgb(540)
    rows = {}
    for thr in [0.25, 0.30, 0.35, 0.45]:
        det = cap.GroundingDinoDetector(DEV, box_threshold=thr)
        d = det.detect(rgb, queries)
        rows[f"box_threshold={thr}"] = {q: d.labels.count(q) for q in queries}
        del det
        cap.free()
    print("\nExercise 1 - detections per query on frame 540 vs Grounding DINO box threshold")
    print(pd.DataFrame(rows).to_string())


def exercise2(frames: cap.Frames) -> None:
    """OWLv2 prompt templates: number of boxes and mean score per query on 4 frames."""
    imgs = [frames.rgb(i) for i in FRAMES]
    rows = []
    for tpl in ["a photo of a {}", "{}", "a {} on an office desk"]:
        owl = cap.Owlv2Detector(DEV, template=tpl)
        dets = [owl.detect(img, cap.QUERIES) for img in imgs]
        for q in cap.QUERIES:
            s = [float(x) for d in dets for lab, x in zip(d.labels, d.scores) if lab == q]
            rows.append(
                {
                    "template": tpl,
                    "query": q,
                    "boxes": len(s),
                    "mean score": round(float(np.mean(s)), 3) if s else None,
                }
            )
        del owl
        cap.free()
    df = pd.DataFrame(rows).pivot(index="query", columns="template", values=["boxes", "mean score"])
    print("\nExercise 2 - OWLv2 prompt templates (4 frames)")
    print(df.to_string())


def exercise3(frames: cap.Frames) -> None:
    """Mask stability of SAM 3 and SAM 2.1-S under +-10 px box jitter (frame 540)."""
    rgb = frames.rgb(540)
    det = cap.GroundingDinoDetector(DEV)
    boxes = det.detect(rgb, cap.QUERIES).boxes
    del det
    cap.free()
    rng = np.random.default_rng(0)
    jitters = [rng.uniform(-10, 10, size=boxes.shape) for _ in range(10)]
    rows = []
    for key in ["sam3", "sam2.1-s"]:
        seg = cap.BoxSegmenter(key, DEV)
        seg.set_image(rgb)
        ref, _ = seg.segment(boxes)
        ious = []
        for j in jitters:
            jb = np.clip(boxes + j, 0, [639, 479, 639, 479])
            m, _ = seg.segment(jb)
            ious += [mask_iou(a, b) for a, b in zip(m, ref)]
        rows.append(
            {
                "segmenter": cap.MODEL_LABELS[key],
                "mean IoU": np.mean(ious),
                "min IoU": np.min(ious),
                "share >= 0.9": np.mean(np.array(ious) >= 0.9),
            }
        )
        del seg
        cap.free()
    print(f"\nExercise 3 - IoU of jittered-box masks vs original ({len(boxes)} boxes x 10 jitters)")
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    fr = cap.load_frames()
    exercise1(fr)
    exercise2(fr)
    exercise3(fr)
