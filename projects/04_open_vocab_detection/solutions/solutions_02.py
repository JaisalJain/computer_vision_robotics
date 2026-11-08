#!/usr/bin/env python
"""Solutions to the exercises of notebook 02 (closed vs open-set benchmark).

Run (GPU, about 3 minutes; exercise 1 downloads MM-Grounding-DINO-T, about 0.7 GB, into the
Hugging Face cache on first use)::

    python projects/04_open_vocab_detection/solutions/solutions_02.py

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
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ovd  # noqa: E402

from cvlab import io as cvio  # noqa: E402
from cvlab import viz  # noqa: E402
from cvlab.bench import benchmark  # noqa: E402
from cvlab.device import setup  # noqa: E402

PROJECT = "04_open_vocab_detection"
OUT = cvio.runs_dir(PROJECT, "solutions", create=True)
RESULTS = cvio.project_dir(PROJECT) / "results"
MMGDINO = ("openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det",
           "ef2053aa1e0cc9d9cee1df31ca919ee9aae4fc8e")  # fmt: skip
WL_NAMES = ["buffalo", "elephant", "rhino", "zebra"]


def exercise_1() -> None:
    """MM-Grounding-DINO-T on african-wildlife val, compared with the stored GDINO/LLMDet runs."""
    wl = cvio.data_dir("ultralytics", "african-wildlife")
    gt = ovd.yolo_labels_to_coco(wl / "images" / "val", wl / "labels" / "val", WL_NAMES)
    coco_gt = ovd.coco_from_dict(gt)
    det = ovd.GroundingDinoDetector(*MMGDINO).set_vocabulary(WL_NAMES)
    results = []
    for im in gt["images"]:
        d = det.predict(cvio.read_image(wl / "images" / "val" / im["file_name"]), min_score=0.01)
        results += ovd.detections_to_coco(d, im["id"], [1, 2, 3, 4])
    ovd.free_detector(det)
    m = ovd.coco_evaluate(coco_gt, results)
    rows = {"MM-Grounding-DINO-T (this run)": {"AP": m["AP"], **m["per_class_AP"]}}
    for k in ("gdino_t", "llmdet_t"):
        r = cvio.load_json(RESULTS / f"wildlife_{k}.json")
        rows[f"{r['label']} (results/)"] = {"AP": r["metrics"]["AP"], **r["per_class_AP"]}
    print("[ex1] african-wildlife val AP:\n", pd.DataFrame(rows).T.round(3).to_string())


def exercise_2() -> None:
    """Latency vs vocabulary size for late fusion, caption fusion and one-pass-per-prompt."""
    img = cvio.read_image(cvio.data_dir("coco", "val2017", "000000039769.jpg"))
    sizes = [1, 5, 20, 80]
    lat = {}
    for key in ["yoloe26s", "gdino_t", "sam3"]:
        kw = {"dtype": "bfloat16"} if key == "sam3" else {}
        det = ovd.load_detector(key, **kw)
        lat[det.label] = []
        for n in sizes:
            det.set_vocabulary(ovd.COCO_NAMES[:n])
            lat[det.label].append(benchmark(det.predict, img, warmup=2, iters=5)["mean_ms"])
        ovd.free_detector(det)
    df = pd.DataFrame(lat, index=sizes).round(1)
    df.index.name = "vocabulary size"
    print("[ex2] latency [ms] vs vocabulary size:\n", df.to_string())
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, c in enumerate(df.columns):
        ax.plot(sizes, df[c], "o-", label=c, color=viz.color(i, unit=True))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of class prompts")
    ax.set_ylabel("latency per image [ms]")
    ax.set_title("Latency vs vocabulary size (RTX 5060 Laptop)")
    ax.legend()
    fig.savefig(OUT / "ex2_vocab_latency.png", dpi=100)
    plt.close(fig)


def exercise_3() -> None:
    """AP by object size (COCO small / medium / large) from the stored results."""
    rows = {}
    for k in ovd.MODEL_INFO:
        p = RESULTS / f"coco_{k}.json"
        if p.exists():
            r = cvio.load_json(p)
            rows[r["label"]] = {s: r["metrics"][s] for s in ("APs", "APm", "APl")}
    df = pd.DataFrame(rows).T
    print("[ex3] COCO AP by object size:\n", df.round(3).to_string())
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(df))
    for i, s in enumerate(["APs", "APm", "APl"]):
        ax.bar(x + (i - 1) * 0.27, df[s], 0.27, label=s, color=viz.color(i, unit=True))
    ax.set_xticks(x, df.index, rotation=20, ha="right")
    ax.set_ylabel("AP (IoU 0.50:0.95)")
    ax.set_title("COCO subset: AP by object size")
    ax.legend()
    fig.savefig(OUT / "ex3_ap_by_size.png", dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    setup(seed=0, verbose=False)
    exercise_3()
    exercise_1()
    exercise_2()
    print(f"figures in {OUT.relative_to(cvio.repo_root())}")
