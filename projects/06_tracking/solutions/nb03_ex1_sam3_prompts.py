#!/usr/bin/env python
"""Solution, notebook 03, exercise 1: other text prompts for SAM 3 on MOT17-05 (30 frames).

GPU: run under the lock (about 3 minutes).

    flock /tmp/cv-gpu.lock python projects/06_tracking/solutions/nb03_ex1_sam3_prompts.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from cvlab.io import data_dir

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import motlib as ml  # noqa: E402
import sam3video  # noqa: E402

PROMPTS = ["person", "car", "bag", "traffic light"]
N = 30


def main() -> None:
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    seq = ml.load_sequence(data_dir("mot17", "MOT17-05-FRCNN"))
    frames = [seq.frame(t) for t in range(1, N + 1)]
    model, proc = sam3video.load_sam3_video()
    for prompt in PROMPTS:
        r = sam3video.track_concept(model, proc, frames, prompt)
        rows = r["rows"]
        n_ids = len(np.unique(rows[:, 1])) if len(rows) else 0
        print(
            f"{prompt!r:16s}: {n_ids:3d} distinct ids, {len(rows) / N:5.2f} objects per frame, "
            f"{r['fps']:.2f} FPS, peak VRAM {r['peak_vram_mb']:.0f} MiB"
        )


if __name__ == "__main__":
    main()
