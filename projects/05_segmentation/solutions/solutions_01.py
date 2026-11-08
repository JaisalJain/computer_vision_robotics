#!/usr/bin/env python
"""Solutions to the exercises of notebook 01 (promptable segmentation).

Run from the repository root (GPU; hold the lock)::

    flock /tmp/cv-gpu.lock python projects/05_segmentation/solutions/solutions_01.py

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

from cvlab import viz  # noqa: E402
from cvlab.device import setup  # noqa: E402
from cvlab.io import outputs_dir, project_dir  # noqa: E402

sys.path.insert(0, str(project_dir("05_segmentation") / "scripts"))
import segkit as sk  # noqa: E402

OUT = outputs_dir("05_segmentation", "solutions", create=True)


def exercise_1(samples: list[sk.InstanceSample]) -> None:
    """Automatic mask generation with the transformers mask-generation pipeline."""
    from PIL import Image
    from transformers import pipeline

    s = samples[0]
    gen = pipeline("mask-generation", model=sk.SAM2_IDS["small"], device=0)
    out = gen(Image.fromarray(s.image), points_per_batch=64)
    masks = np.stack([np.asarray(m, dtype=bool) for m in out["masks"]])
    iou = sk.mask_iou_matrix(masks, s.masks)  # generated x GT
    best = iou.max(axis=0)
    print(
        f"[ex1] {len(masks)} masks generated on {s.name}; "
        f"{int((best > 0.5).sum())}/{len(s.masks)} GT objects covered with IoU > 0.5; "
        f"best IoU per GT object: {np.round(best, 3)}"
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].imshow(viz.overlay_masks(s.image, s.masks))
    axes[0].set_title(f"GT ({len(s.masks)} objects)")
    order = np.argsort([-m.sum() for m in masks])  # draw large masks first
    axes[1].imshow(viz.overlay_masks(s.image, masks[order], alpha=0.45, contour=False))
    axes[1].set_title(f"SAM 2.1-S automatic masks ({len(masks)})")
    for ax in axes:
        ax.axis("off")
    fig.savefig(OUT / "ex01_1_mask_generation.png", dpi=100)
    plt.close(fig)
    del gen
    torch.cuda.empty_cache()


def exercise_2(samples: list[sk.InstanceSample]) -> None:
    """Boundary F-measure of 1-click (top mask) vs box masks on all coco8-seg instances."""
    rows = []
    for mid in [sk.SAM3_ID, sk.SAM2_IDS["small"]]:
        seg = sk.PromptableSegmenter(mid)
        for s in samples:
            seg.set_image(s.image)
            for g in s.masks:
                m, sc, _ = seg.predict([[sk.interior_point(g)]], [[1]])
                click = m[0, int(np.argmax(sc[0]))]
                box = seg.predict(boxes=[sk.mask_to_box(g)], multimask=False)[0][0, 0]
                rows.append(
                    {
                        "model": seg.label,
                        "F click": sk.f_measure(click, g),
                        "F box": sk.f_measure(box, g),
                        "J click": sk.j_measure(click, g),
                        "J box": sk.j_measure(box, g),
                    }
                )
        del seg
        torch.cuda.empty_cache()
    df = pd.DataFrame(rows).groupby("model").mean().round(3)
    print(f"[ex2] mean over {len(rows) // 2} coco8-seg instances:\n{df.to_string()}")


def exercise_3() -> None:
    """How often does the predicted IoU pick a mask that is not the best of the three?"""
    inst = pd.read_csv(sk.results_dir() / "prompts_coco128-seg_instances.csv")
    inst["wrong_pick"] = inst.click_top < inst.click_oracle - 1e-6
    inst["iou_lost"] = np.where(inst.wrong_pick, inst.click_oracle - inst.click_top, np.nan)
    tab = inst.groupby(["model", "size"]).agg(
        wrong_pick_rate=("wrong_pick", "mean"), mean_iou_lost_when_wrong=("iou_lost", "mean")
    )
    print(f"[ex3] top-scored mask is not the best of 3:\n{tab.round(3).to_string()}")


def main() -> None:
    setup(seed=0)
    warnings.filterwarnings("ignore")
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    samples = sk.load_coco_seg("coco8-seg")
    exercise_1(samples)
    exercise_2(samples)
    exercise_3()


if __name__ == "__main__":
    main()
