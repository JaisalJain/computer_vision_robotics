#!/usr/bin/env python
"""Solution, notebook 01, exercise 4: calibrate BoT-SORT's appearance gate for DINOv3.

BoT-SORT accepts an appearance match when ``(1 - cos) / 2 <= 1 - appearance_thresh``, i.e.
``cos >= 2 * appearance_thresh - 1`` (0.6 for the default 0.8), a value chosen for a trained
ReID embedding. DINOv3 similarities are higher overall, so the gate is calibrated here on
MOT17-02 GT crops (cosine similarity that only 1 % of different-person pairs exceed) and the
tracker is then evaluated on the unseen MOT17-05 sequence, on the cached YOLO26s detections
of scripts/run_mot_benchmark.py. GPU: run under the lock.

    flock /tmp/cv-gpu.lock python projects/06_tracking/solutions/nb01_ex4_dinov3_gate.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

os.environ.setdefault("YOLO_AUTOINSTALL", "false")

from cvlab.io import data_dir, project_dir, runs_dir  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import dinoreid  # noqa: E402
import motlib as ml  # noqa: E402

FALSE_ACCEPT = 0.01  # share of different-person pairs allowed through the gate


def gt_crop_embeddings(seq: ml.MotSequence, enc, per_id: int = 8, seed: int = 0):
    """Embed up to ``per_id`` well-visible GT crops per pedestrian -> (ids, (N, D) unit rows)."""
    gt = seq.gt()
    gt = gt[(gt[:, 7] == ml.PEDESTRIAN) & (gt[:, 6] != 0) & (gt[:, 8] >= 0.7) & (gt[:, 5] >= 60)]
    rng = np.random.default_rng(seed)
    pids, n = np.unique(gt[:, 1], return_counts=True)
    sample = np.concatenate([gt[gt[:, 1] == p][rng.choice(k, min(per_id, k), replace=False)]
                             for p, k in zip(pids, n) if k >= 10])  # fmt: skip
    sample = sample[np.argsort(sample[:, 0], kind="stable")]
    feats = []
    for t in np.unique(sample[:, 0]).astype(int):
        r = sample[sample[:, 0] == t]
        xywh_c = np.column_stack([r[:, 2] + r[:, 4] / 2, r[:, 3] + r[:, 5] / 2, r[:, 4], r[:, 5]])
        feats += enc(cv2.imread(str(seq.frame_path(t))), xywh_c)
    return sample[:, 1], np.stack(feats)


def main() -> None:
    from transformers.utils import logging as hf_logging

    hf_logging.disable_progress_bar()
    enc = dinoreid.DinoV3ReID(device="cuda")
    # 1) calibration on MOT17-02 (static camera, different people than the test sequence)
    ids, e = gt_crop_embeddings(ml.load_sequence(data_dir("mot17", "MOT17-02-FRCNN")), enc)
    sim = e @ e.T
    diff = ~(ids[:, None] == ids[None, :]) & np.triu(np.ones_like(sim, dtype=bool), k=1)
    cos_star = float(np.quantile(sim[diff], 1 - FALSE_ACCEPT))
    thresh = (1 + cos_star) / 2
    print(f"MOT17-02: {len(ids)} crops of {len(np.unique(ids))} people; 99th percentile of "
          f"different-person cosine = {cos_star:.3f} -> appearance_thresh = {thresh:.3f}")
    # 2) evaluation on MOT17-05 with the cached YOLO26s detections
    seq = ml.load_sequence(data_dir("mot17", "MOT17-05-FRCNN"))
    gt = seq.gt()
    dets = ml.DetectionCache.load(runs_dir("06_tracking", "dets", "yolo26s_MOT17-05-FRCNN.npz"))
    cfg = project_dir("06_tracking") / "configs" / "botsort_reid.yaml"

    def load(t: int) -> np.ndarray:
        return cv2.imread(str(seq.frame_path(t)))

    rows = []
    for label, kw in [("BoT-SORT (GMC), no appearance", None),
                      ("BoT-SORT + DINOv3, default gate (0.8)", {"appearance_thresh": 0.8}),
                      (f"BoT-SORT + DINOv3, calibrated gate ({thresh:.3f})",
                       {"appearance_thresh": thresh})]:  # fmt: skip
        if kw is None:
            tracker = ml.make_tracker(str(project_dir("06_tracking") / "configs" / "botsort.yaml"))
        else:
            tracker = ml.make_tracker(str(cfg), device="cuda:0", encoder=enc,
                                      encoder_name=dinoreid.MODEL_ID, **kw)  # fmt: skip
        out, _ = ml.run_tracker(tracker, dets, seq.length, frame_loader=load)
        res = ml.summary_row(ml.evaluate(gt, out, seq.length))
        rows.append({"tracker": label, **{k: res[k] for k in ("HOTA", "AssA", "MOTA", "IDF1",
                                                              "IDSW")}})  # fmt: skip
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
