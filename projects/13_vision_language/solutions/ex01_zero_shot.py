#!/usr/bin/env python
"""Solutions to the exercises of notebooks/01_zero_shot_classification.ipynb.

1. Descriptive class names for each model's two weakest classes.
2. k-shot logistic-regression probes on frozen embeddings vs zero-shot.
3. Open-set rejection of robot-desk frames with a max-score threshold.

Run (GPU, about 2.5 minutes)::

    python projects/13_vision_language/solutions/ex01_zero_shot.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import transformers  # noqa: E402
import vl_common as vl  # noqa: E402
from open_clip import zero_shot_metadata as zsm  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from cvlab.device import setup  # noqa: E402
from cvlab.io import data_dir  # noqa: E402

transformers.logging.set_verbosity_error()
pd.set_option("display.width", 160)

# A short description per class; only the weakest classes of each model get replaced.
DESCRIPTIONS = {
    "tench": "tench, a type of freshwater fish",
    "English Springer Spaniel": "English Springer Spaniel, a breed of dog",
    "cassette player": "cassette player, a portable audio device",
    "chainsaw": "chainsaw, a motorised saw",
    "church": "church, a building for worship",
    "French horn": "French horn, a brass musical instrument",
    "garbage truck": "garbage truck, a waste collection vehicle",
    "gas pump": "gas pump at a petrol station",
    "golf ball": "golf ball, a small white dimpled ball",
    "parachute": "parachute in the sky",
}
# The prompt variants of the notebook; each model uses its best one (chosen on validation,
# exactly as in the notebook).
TEMPLATES = {
    "class name only": ["{}"],
    "a photo of a {}.": ["a photo of a {}."],
    "This is a photo of {}.": ["This is a photo of {}."],
    "7-template ensemble": [t("{}") for t in zsm.SIMPLE_IMAGENET_TEMPLATES],
    "80-template ensemble": [t("{}") for t in zsm.OPENAI_IMAGENET_TEMPLATES],
}


def classifier(enc, names, templates):
    """Prompt-ensembled class weights (n_classes, D)."""
    return vl.ensemble_text_embeddings(
        enc.encode_texts(vl.build_prompts(names, templates)), len(names)
    )


def main() -> None:
    device = setup(seed=0)
    root = data_dir("imagenette", "imagenette2-160")
    val_paths, val_y = vl.imagenette_split(root, "val")
    tr_paths, tr_y = vl.imagenette_split(root, "train")
    classes = list(vl.IMAGENETTE_CLASSES.values())
    tum = sorted(data_dir("tum", "fr1_desk", "rgbd_dataset_freiburg1_desk", "rgb").glob("*.png"))
    print(f"val {len(val_paths)} | train {len(tr_paths)} | fr1_desk frames {len(tum)}")

    ex1, ex2, ex3 = [], [], []
    for Enc in (vl.SigLIP2Encoder, vl.OpenCLIPEncoder):
        enc = Enc(device)
        xv = enc.encode_images(val_paths)
        acc = {
            k: np.mean((xv @ classifier(enc, classes, t).T).argmax(1) == val_y)
            for k, t in TEMPLATES.items()
        }
        best = max(acc, key=acc.get)
        tpl = TEMPLATES[best]
        print(f"{enc.name}: best prompt variant '{best}' ({acc[best]:.4f})")

        # ---- Exercise 1: descriptive names for the two weakest classes
        w = classifier(enc, classes, tpl)
        pred = (xv @ w.T).argmax(1)
        per_cls = np.array([np.mean(pred[val_y == k] == k) for k in range(len(classes))])
        weakest = np.argsort(per_cls)[:2]
        names = [DESCRIPTIONS[c] if k in weakest else c for k, c in enumerate(classes)]
        pred2 = (xv @ classifier(enc, names, tpl).T).argmax(1)
        for k in weakest:
            ex1.append(
                {
                    "model": enc.name,
                    "class": classes[k],
                    "new name": names[k],
                    "acc before": per_cls[k],
                    "acc after": np.mean(pred2[val_y == k] == k),
                }
            )
        ex1.append(
            {
                "model": enc.name,
                "class": "(all 10 classes)",
                "new name": "",
                "acc before": np.mean(pred == val_y),
                "acc after": np.mean(pred2 == val_y),
            }
        )

        # ---- Exercise 2: k-shot logistic regression on frozen embeddings
        xt = enc.encode_images(tr_paths)
        zero_shot = float(np.mean(pred == val_y))
        for k in (1, 2, 4, 8, 16):
            accs = []
            for seed in range(3):
                rng = np.random.default_rng(seed)
                idx = np.concatenate(
                    [
                        rng.choice(np.flatnonzero(tr_y == c), k, replace=False)
                        for c in range(len(classes))
                    ]
                )
                clf = LogisticRegression(C=10.0, max_iter=2000).fit(xt[idx], tr_y[idx])
                accs.append(clf.score(xv, val_y))
            ex2.append(
                {
                    "model": enc.name,
                    "k shots/class": k,
                    "probe acc (mean of 3)": np.mean(accs),
                    "probe acc (std)": np.std(accs),
                    "zero-shot acc": zero_shot,
                }
            )

        # ---- Exercise 3: max-score threshold keeping 95% of in-distribution images
        p_in = vl.zero_shot_scores(xv, w, enc.scale, enc.bias, enc.kind).max(1)
        p_out = vl.zero_shot_scores(enc.encode_images(tum), w, enc.scale, enc.bias, enc.kind).max(1)
        thr = float(np.quantile(p_in, 0.05))
        ex3.append(
            {
                "model": enc.name,
                "score": enc.kind,
                "threshold (5% FRR)": thr,
                "desk frames rejected": float(np.mean(p_out < thr)),
                "n desk frames": len(p_out),
            }
        )
        del enc

    print("\nExercise 1 - descriptive class names (top-1 accuracy)")
    print(pd.DataFrame(ex1).to_string(index=False, float_format="{:.4f}".format))
    print("\nExercise 2 - k-shot logistic regression vs zero-shot (validation accuracy)")
    print(pd.DataFrame(ex2).to_string(index=False, float_format="{:.4f}".format))
    print("\nExercise 3 - open-set rejection with a max-score threshold")
    print(pd.DataFrame(ex3).to_string(index=False, float_format="{:.4f}".format))


if __name__ == "__main__":
    main()
