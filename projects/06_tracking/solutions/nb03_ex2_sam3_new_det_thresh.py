#!/usr/bin/env python
"""Solution, notebook 03, exercise 2: effect of SAM 3's new-detection threshold.

Runs the "person" prompt on the first 50 MOT17-05 frames with new_det_thresh 0.5 / 0.7
(default) / 0.85 and scores each run with the MOT17 protocol. GPU: run under the lock
(about 4 minutes).

    flock /tmp/cv-gpu.lock python projects/06_tracking/solutions/nb03_ex2_sam3_new_det_thresh.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from cvlab.io import data_dir

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import motlib as ml  # noqa: E402
import sam3video  # noqa: E402

N = 50


def main() -> None:
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    seq = ml.load_sequence(data_dir("mot17", "MOT17-05-FRCNN"))
    gt = seq.gt()
    gt = gt[gt[:, 0] <= N]
    frames = [seq.frame(t) for t in range(1, N + 1)]
    model, proc = sam3video.load_sam3_video()
    default = model.new_det_thresh
    rows = []
    for thr in (0.5, default, 0.85):
        model.new_det_thresh = thr
        r = sam3video.track_concept(model, proc, frames, "person")
        res = ml.summary_row(ml.evaluate(gt, r["rows"], N))
        rows.append(
            {
                "new_det_thresh": thr,
                **{k: res[k] for k in ("HOTA", "MOTA", "IDF1", "IDSW", "CLR_FP", "CLR_FN")},
            }
        )
    model.new_det_thresh = default
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
