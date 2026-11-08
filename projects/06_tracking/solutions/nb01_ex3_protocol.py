#!/usr/bin/env python
"""Solution, notebook 01, exercise 3: scores with and without the MOT17 preprocessing.

Re-scores the BoT-SORT (GMC) tracker files of scripts/run_mot_benchmark.py (YOLO26s) with
the official protocol and with ``preprocess=False`` (every GT row counts, no distractor
removal), and lists the GT classes of each sequence. CPU only.

    python projects/06_tracking/solutions/nb01_ex3_protocol.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cvlab.io import data_dir, runs_dir

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import motlib as ml  # noqa: E402

CLASS_NAMES = {
    1: "pedestrian",
    2: "person on vehicle",
    3: "car",
    4: "bicycle",
    5: "motorbike",
    6: "non-motorized vehicle",
    7: "static person",
    8: "distractor",
    9: "occluder",
    10: "occluder on ground",
    11: "occluder full",
    12: "reflection",
}


def main() -> None:
    rows = []
    for name in ["MOT17-02-FRCNN", "MOT17-05-FRCNN"]:
        seq = ml.load_sequence(data_dir("mot17", name))
        gt = seq.gt()
        cls, n = np.unique(gt[:, 7].astype(int), return_counts=True)
        print(
            name,
            "GT rows per class:",
            {CLASS_NAMES[c]: int(k) for c, k in zip(cls, n)},
            "| consider_flag=0 rows:",
            int((gt[:, 6] == 0).sum()),
        )
        pred = ml.load_mot_txt(
            runs_dir("06_tracking", "trackers", "yolo26s", "botsort", "data", f"{name}.txt")
        )
        for proto in (True, False):
            r = ml.evaluate(gt, pred, seq.length, preprocess=proto)
            rows.append(
                {
                    "sequence": name,
                    "protocol": "MOT17 (official)" if proto else "no preprocessing",
                    "GT boxes": r["num_gt_dets"],
                    "tracker boxes": r["num_tr_dets"],
                    **{
                        k: ml.summary_row(r)[k]
                        for k in ("MOTA", "IDF1", "HOTA", "CLR_FP", "CLR_FN")
                    },
                }
            )
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
