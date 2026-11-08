#!/usr/bin/env python
"""Solution, notebook 01, exercise 2: add ByteTrack's low-score association to SimpleSORT.

Tracks left unmatched by the high-score stage are matched (IoU >= 0.5) to boxes with
0.1 < score < 0.25; those boxes extend tracks but never start new ones. CPU only, uses the
cached YOLO26s detections from scripts/run_mot_benchmark.py.

    python projects/06_tracking/solutions/nb01_ex2_sort_low_score.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from cvlab.io import data_dir, runs_dir
from cvlab.metrics import box_iou

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import motlib as ml  # noqa: E402


class SortWithLowScore(ml.SimpleSORT):
    """SimpleSORT + a second association stage on low-confidence detections."""

    def __init__(self, low_thresh: float = 0.1, low_iou: float = 0.5, **kw):
        super().__init__(**kw)
        self.low_thresh, self.low_iou = low_thresh, low_iou

    def update(self, dets: ml.Detections, img=None) -> np.ndarray:
        self.frame_id += 1
        high = dets.conf >= self.det_thresh
        low = (dets.conf > self.low_thresh) & ~high
        boxes, scores, idx = dets.xyxy[high].astype(float), dets.conf[high], np.flatnonzero(high)
        lboxes, lscores, lidx = dets.xyxy[low].astype(float), dets.conf[low], np.flatnonzero(low)
        pred = np.array([ml._cxcywh_to_xyxy(tr["kf"].predict())[0] for tr in self.tracks]).reshape(
            -1, 4
        )
        matched_t, matched_d = np.zeros(0, int), np.zeros(0, int)
        if len(pred) and len(boxes):  # stage 1: high-score boxes
            iou = box_iou(pred, boxes)
            r, c = linear_sum_assignment(-iou)
            ok = iou[r, c] >= self.iou_thresh
            matched_t, matched_d = r[ok], c[ok]
        for ti, di in zip(matched_t, matched_d):
            tr = self.tracks[ti]
            tr["kf"].update(ml._xyxy_to_cxcywh(boxes[di])[0])
            tr.update(hits=tr["hits"] + 1, age=0, score=float(scores[di]), idx=int(idx[di]))
        # stage 2: confirmed tracks still unmatched vs low-score boxes (extend only)
        rest = [
            t
            for t in range(len(self.tracks))
            if t not in set(matched_t.tolist()) and self.tracks[t]["hits"] >= self.min_hits
        ]
        matched2 = set()
        if rest and len(lboxes):
            iou = box_iou(pred[rest], lboxes)
            r, c = linear_sum_assignment(-iou)
            for ri, ci in zip(r, c):
                if iou[ri, ci] >= self.low_iou:
                    tr = self.tracks[rest[ri]]
                    tr["kf"].update(ml._xyxy_to_cxcywh(lboxes[ci])[0])
                    tr.update(
                        hits=tr["hits"] + 1, age=0, score=float(lscores[ci]), idx=int(lidx[ci])
                    )
                    matched2.add(rest[ri])
        for ti in set(range(len(self.tracks))) - set(matched_t.tolist()) - matched2:
            self.tracks[ti]["age"] += 1
        for di in set(range(len(boxes))) - set(matched_d.tolist()):
            self.tracks.append(
                {
                    "kf": ml.KalmanBoxFilter(ml._xyxy_to_cxcywh(boxes[di])[0]),
                    "hits": 1,
                    "age": 0,
                    "id": None,
                    "score": float(scores[di]),
                    "idx": int(idx[di]),
                }
            )
        self.tracks = [tr for tr in self.tracks if tr["age"] <= self.max_age]
        out = []
        for tr in self.tracks:
            if tr["age"] == 0 and (tr["hits"] >= self.min_hits or self.frame_id <= self.min_hits):
                if tr["id"] is None:
                    tr["id"] = self._next_id
                    self._next_id += 1
                x1, y1, x2, y2 = ml._cxcywh_to_xyxy(tr["kf"].x[:4])[0]
                out.append([x1, y1, x2, y2, tr["id"], tr["score"], 0.0, tr["idx"]])
        return np.asarray(out, dtype=np.float32).reshape(-1, 8)


def main() -> None:
    rows = []
    for name in ["MOT17-02-FRCNN", "MOT17-05-FRCNN"]:
        seq = ml.load_sequence(data_dir("mot17", name))
        gt = seq.gt()
        dets = ml.DetectionCache.load(runs_dir("06_tracking", "dets", f"yolo26s_{name}.npz"))
        cfg = Path(__file__).resolve().parents[1] / "configs" / "bytetrack.yaml"
        for label, tracker in [
            ("SORT (scratch)", ml.SimpleSORT()),
            ("SORT + low-score stage", SortWithLowScore()),
            ("ByteTrack (Ultralytics)", ml.make_tracker(str(cfg))),
        ]:
            out, _ = ml.run_tracker(tracker, dets, seq.length)
            rows.append(
                {
                    "sequence": name,
                    "tracker": label,
                    **ml.summary_row(ml.evaluate(gt, out, seq.length)),
                }
            )
    df = pd.DataFrame(rows)
    print(
        df[["sequence", "tracker", "HOTA", "MOTA", "IDF1", "IDSW", "CLR_FP", "CLR_FN"]].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
