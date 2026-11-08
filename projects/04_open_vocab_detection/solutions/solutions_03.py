#!/usr/bin/env python
"""Solutions to the exercises of notebook 03 (open-vocabulary object map).

Run (GPU, about 1 minute)::

    python projects/04_open_vocab_detection/solutions/solutions_03.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["YOLO_AUTOINSTALL"] = "false"
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ovd  # noqa: E402

from cvlab import io as cvio  # noqa: E402
from cvlab.device import setup  # noqa: E402
from cvlab.geometry import transform_points  # noqa: E402
from cvlab.pointcloud import depth_to_points, pca_obb, remove_statistical_outliers  # noqa: E402
from cvlab.tum import load_sequence  # noqa: E402

SEQ = load_sequence(cvio.data_dir("tum", "fr1_desk", "rgbd_dataset_freiburg1_desk"))
K = SEQ.camera.K
FRAMES = list(range(0, len(SEQ), 4))
VOCAB = [
    "telephone",
    "game controller",
    "laptop",
    "stapler",
    "cereal box",
    "computer mouse",
    "keyboard",
]


def box_to_world(box, depth, T_wc, frac=0.3):
    x1, y1, x2, y2 = box
    cx, cy, hw, hh = (x1 + x2) / 2, (y1 + y2) / 2, frac * (x2 - x1) / 2, frac * (y2 - y1) / 2
    patch = depth[int(cy - hh) : int(cy + hh) + 1, int(cx - hw) : int(cx + hw) + 1]
    valid = patch[patch > 0]
    if valid.size < 5:
        return None
    X_c = float(np.median(valid)) * np.linalg.solve(K, np.array([cx, cy, 1.0]))
    return transform_points(T_wc, X_c[None])[0]


def voted(
    obs: pd.DataFrame, window: int = 4, min_votes: int = 2, radius: float = 0.15
) -> pd.DataFrame:
    """Keep an observation if the same class was seen within ``radius`` metres (world frame)
    in at least ``min_votes`` of the previous ``window`` processed frames."""
    frames = sorted(obs.frame.unique())
    pos = {f: k for k, f in enumerate(frames)}
    keep = []
    for idx, r in obs.iterrows():
        k = pos[r.frame]
        prev = set(frames[max(0, k - window) : k])
        cand = obs[(obs.cls == r.cls) & obs.frame.isin(prev)]
        d = np.linalg.norm(cand[["x", "y", "z"]].to_numpy() - [r.x, r.y, r.z], axis=1)
        if cand[d < radius].frame.nunique() >= min_votes:
            keep.append(idx)
    return obs.loc[keep]


def exercise_1() -> None:
    """Temporal voting on YOLOE-26-s observations: spurious rate and recall before/after."""
    det = ovd.load_detector("yoloe26s").set_vocabulary(VOCAB)
    rows = []
    for i in FRAMES:
        d = det.predict(SEQ.rgb(i), min_score=0.25)
        depth = SEQ.depth(i)
        for box, s, n in zip(d.boxes, d.scores, d.label_names):
            X = box_to_world(box, depth, SEQ.gt_poses[i])
            if X is not None:
                rows.append(
                    {"frame": i, "cls": n, "score": float(s), "x": X[0], "y": X[1], "z": X[2]}
                )
    ovd.free_detector(det)
    obs = pd.DataFrame(rows)

    # Map objects = per-class medians of dense clusters (same rule as the notebook, simplified:
    # an observation is "consistent" if >= 5 other frames saw the class within 15 cm of it).
    def consistent(o: pd.DataFrame) -> np.ndarray:
        out = []
        for _, r in o.iterrows():
            c = o[(o.cls == r.cls) & (o.frame != r.frame)]
            d = np.linalg.norm(c[["x", "y", "z"]].to_numpy() - [r.x, r.y, r.z], axis=1)
            out.append(c[d < 0.15].frame.nunique() >= 5)
        return np.array(out, bool)

    base_ok = consistent(obs)
    v = voted(obs)
    v_ok = base_ok[obs.index.get_indexer(v.index)]
    print(
        f"[ex1] before voting: {len(obs)} observations, spurious {100 * (1 - base_ok.mean()):.1f} %"
    )
    print(
        f"[ex1] after voting (2 of previous 4 frames): {len(v)} observations "
        f"({100 * len(v) / len(obs):.1f} % kept), spurious {100 * (1 - v_ok.mean()):.1f} %, "
        f"consistent observations kept {100 * v_ok.sum() / base_ok.sum():.1f} %"
    )


def exercise_2() -> None:
    """SAM 3 mask of the telephone -> point cloud -> PCA oriented bounding box."""
    frame = FRAMES[100]
    det = ovd.load_detector("sam3", dtype="bfloat16").set_vocabulary(["telephone"])
    d = det.predict(SEQ.rgb(frame), min_score=0.5, masks=True)
    ovd.free_detector(det)
    if len(d) == 0:
        print(f"[ex2] no telephone detected in frame {frame}")
        return
    d = d.topk(1)
    pts, _ = depth_to_points(SEQ.depth(frame), K, mask=d.masks[0], max_depth=4.0)
    pts_in = pts[remove_statistical_outliers(pts)]  # boolean inlier mask
    center, R, ext = pca_obb(pts_in)
    print(
        f"[ex2] frame {frame}: SAM 3 telephone score {d.scores[0]:.3f}, "
        f"mask {int(d.masks[0].sum())} px, "
        f"{len(pts)} depth points ({len(pts_in)} after outlier removal)"
    )
    print(
        f"[ex2] OBB centre (camera frame) {np.round(center, 3).tolist()} m, "
        f"extents {np.round(100 * ext, 1).tolist()} cm (longest first)"
    )


if __name__ == "__main__":
    setup(seed=0, verbose=False)
    exercise_1()
    exercise_2()
