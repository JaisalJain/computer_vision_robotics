#!/usr/bin/env python
"""Solution, notebook 01, exercise 1: sweep ByteTrack's track_buffer and match_thresh.

Uses the cached YOLO26s detections of MOT17-05 written by scripts/run_mot_benchmark.py
(CPU only). Saves a figure to outputs/06_tracking/nb01_ex1_bytetrack_tuning.png.

    python projects/06_tracking/solutions/nb01_ex1_bytetrack_tuning.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from cvlab.io import data_dir, outputs_dir, project_dir, repo_root, runs_dir  # noqa: E402
from cvlab.viz import set_style  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import motlib as ml  # noqa: E402

BUFFERS = [5, 15, 30, 60, 120]
MATCH = [0.6, 0.8, 0.9]


def main() -> None:
    seq = ml.load_sequence(data_dir("mot17", "MOT17-05-FRCNN"))
    gt = seq.gt()
    dets = ml.DetectionCache.load(runs_dir("06_tracking", "dets", "yolo26s_MOT17-05-FRCNN.npz"))
    cfg = project_dir("06_tracking") / "configs" / "bytetrack.yaml"
    rows = []
    for m in MATCH:
        for b in BUFFERS:
            tracker = ml.make_tracker(str(cfg), track_buffer=b, match_thresh=m)
            out, _ = ml.run_tracker(tracker, dets, seq.length)
            res = ml.summary_row(ml.evaluate(gt, out, seq.length))
            rows.append({"match_thresh": m, "track_buffer": b, **res})
    df = pd.DataFrame(rows)
    print(
        df[["match_thresh", "track_buffer", "HOTA", "MOTA", "IDF1", "IDSW", "Frag"]].to_string(
            index=False
        )
    )
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for m in MATCH:
        d = df[df.match_thresh == m]
        axes[0].plot(d.track_buffer, d.IDF1, marker="o", label=f"match_thresh {m}")
        axes[1].plot(d.track_buffer, d.IDSW, marker="o", label=f"match_thresh {m}")
    for ax, name in zip(axes, ["IDF1 [%]", "identity switches"]):
        ax.set_xscale("log")
        ax.set_xlabel("track_buffer [frames]")
        ax.set_ylabel(name)
        ax.legend()
    fig.suptitle("ByteTrack on MOT17-05 (YOLO26s detections)")
    path = outputs_dir("06_tracking", create=True) / "nb01_ex1_bytetrack_tuning.png"
    fig.savefig(path, dpi=110)
    print("figure:", path.relative_to(repo_root()))
    best = df.loc[df.IDF1.idxmax()]
    print(
        f"best IDF1 {best.IDF1} at match_thresh={best.match_thresh}, "
        f"track_buffer={int(best.track_buffer)}"
    )


if __name__ == "__main__":
    main()
