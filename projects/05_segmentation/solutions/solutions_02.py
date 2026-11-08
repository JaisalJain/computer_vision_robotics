#!/usr/bin/env python
"""Solutions to the exercises of notebook 02 (text prompts).

Run from the repository root (GPU; hold the lock)::

    flock /tmp/cv-gpu.lock python projects/05_segmentation/solutions/solutions_02.py

Figures go to ``outputs/05_segmentation/solutions/`` (gitignored).
"""

from __future__ import annotations

import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from cvlab.device import setup  # noqa: E402
from cvlab.io import data_dir, outputs_dir, project_dir  # noqa: E402
from cvlab.tum import load_sequence  # noqa: E402

sys.path.insert(0, str(project_dir("05_segmentation") / "scripts"))
import segkit as sk  # noqa: E402

OUT = outputs_dir("05_segmentation", "solutions", create=True)
PHRASES = ["book", "blue book", "white book", "laptop", "open laptop", "closed laptop"]


def exercise_1(concept: sk.ConceptSegmenter, grounded: sk.GroundedSam) -> None:
    """Attribute phrases on the TUM desk frame."""
    desk = load_sequence(data_dir("tum", "fr1_desk", "rgbd_dataset_freiburg1_desk")).rgb(300)
    concept.set_image(desk)
    grounded.set_image(desk)
    rows = []
    for p in PHRASES:
        rows.append(
            {
                "phrase": p,
                "SAM 3": len(concept.segment(p, threshold=0.5)["scores"]),
                "Grounded-SAM 2.1": len(grounded.segment(p)["scores"]),
            }
        )
    df = pd.DataFrame(rows).set_index("phrase")
    print(f"[ex1] instances per phrase on the desk frame:\n{df.to_string()}")
    ax = df.plot.bar(figsize=(7, 3.5), rot=20)
    ax.set_ylabel("instances returned")
    ax.set_title("Attribute phrases (TUM fr1_desk frame 300)")
    ax.figure.savefig(OUT / "ex02_1_attribute_phrases.png", dpi=100, bbox_inches="tight")
    plt.close(ax.figure)


def exercise_2(concept: sk.ConceptSegmenter, samples: list[sk.InstanceSample]) -> None:
    """SAM 3 precision / recall at IoU 0.5 for several score thresholds (coco8-seg)."""
    names = sk.coco_names()
    preds = []  # per image: list of (class, mask, score)
    for s in samples:
        concept.set_image(s.image)
        items = []
        for c in sorted(set(s.classes.tolist())):
            r = concept.segment(names[c], threshold=0.05)
            items += [(c, m, sc) for m, sc in zip(r["masks"], r["scores"])]
        preds.append(items)
    rows = []
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7):
        tp = n_pred = 0
        for s, items in zip(samples, preds):
            for c in set(s.classes.tolist()):
                kept = sorted(
                    [(sc, m) for cc, m, sc in items if cc == c and sc >= thr], key=lambda t: -t[0]
                )
                n_pred += len(kept)
                g = s.masks[s.classes == c]
                if not kept:
                    continue
                iou = sk.mask_iou_matrix(np.stack([m for _, m in kept]), g)
                taken = np.zeros(len(g), bool)
                for k in range(len(kept)):
                    cand = np.where(taken, -1.0, iou[k])
                    j = int(np.argmax(cand))
                    if cand[j] >= 0.5:
                        taken[j] = True
                        tp += 1
        n_gt = sum(len(s.classes) for s in samples)
        rows.append(
            {
                "threshold": thr,
                "precision": tp / max(n_pred, 1),
                "recall": tp / n_gt,
                "predictions": n_pred,
            }
        )
    df = pd.DataFrame(rows).set_index("threshold").round(3)
    print(
        f"[ex2] SAM 3 on coco8-seg ({sum(len(s.classes) for s in samples)} GT instances), "
        f"IoU >= 0.5:\n{df.to_string()}"
    )
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(df.recall, df.precision, "o-")
    for t, r in df.iterrows():
        ax.annotate(f"{t:.1f}", (r.recall, r.precision), textcoords="offset points", xytext=(4, 4))
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("SAM 3 operating points (coco8-seg, IoU 0.5)")
    fig.savefig(OUT / "ex02_2_threshold_sweep.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def exercise_3(grounded: sk.GroundedSam, samples: list[sk.InstanceSample]) -> None:
    """Grounding DINO + SAM 2.1-S vs Grounding DINO + SAM 3 tracker: mask AP on coco8-seg."""
    names = sk.coco_names()
    res = {}
    for label, sam_id in [
        ("GDINO-T + SAM 2.1-S", sk.SAM2_IDS["small"]),
        ("GDINO-T + SAM 3 tracker", sk.SAM3_ID),
    ]:
        if grounded.sam.model_id != sam_id:
            del grounded.sam
            torch.cuda.empty_cache()
            grounded.sam = sk.PromptableSegmenter(sam_id)
        preds = []
        for s in samples:
            h, w = s.image.shape[:2]
            grounded.set_image(s.image)
            dets = []
            for c in sorted(set(s.classes.tolist())):
                r = grounded.segment(names[c], 0.10, 0.10)
                dets.append(sk.Detections(r["masks"], r["scores"], np.full(len(r["scores"]), c)))
            preds.append(sk.Detections.concat(dets, h, w))
        res[label] = sk.mask_ap(preds, samples)
    df = pd.DataFrame(res).T[["AP", "AP50", "AP75"]].round(3)
    print(f"[ex3] mask AP on coco8-seg (oracle vocabulary):\n{df.to_string()}")


def main() -> None:
    setup(seed=0)
    warnings.filterwarnings("ignore")
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    samples = sk.load_coco_seg("coco8-seg")
    concept = sk.ConceptSegmenter()
    grounded = sk.GroundedSam(sk.SAM2_IDS["small"])
    exercise_1(concept, grounded)
    exercise_2(concept, samples)
    del concept
    torch.cuda.empty_cache()
    exercise_3(grounded, samples)


if __name__ == "__main__":
    main()
