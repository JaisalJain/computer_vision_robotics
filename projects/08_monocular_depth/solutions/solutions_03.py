#!/usr/bin/env python
"""Solutions to the exercises of notebook 03 (depth maps to point clouds).

Run (GPU, about 1 minute)::

    flock /tmp/cv-gpu.lock python projects/08_monocular_depth/solutions/solutions_03.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import mono_depth as md  # noqa: E402

from cvlab.device import seed_everything  # noqa: E402
from cvlab.geometry import transform_points  # noqa: E402
from cvlab.pointcloud import depth_to_points, voxel_downsample  # noqa: E402

FRAME = 450
FUSE = list(range(FRAME - 40, FRAME + 41, 10))


def exercise_1(seq) -> None:
    """Flying-pixel filter on DA3-BASE (median scale): relative depth-gradient threshold."""
    rgb, gt, K = seq.rgb(FRAME), seq.depth(FRAME), seq.camera.K
    model = md.DepthModel("da3-base")
    p = model.predict(rgb)
    del model
    md.free_cuda()
    depth = md.evaluate_frame(p.values, "depth", gt, ["median"])["median"]["depth"]
    gy, gx = np.gradient(depth)
    rel_grad = np.hypot(gx, gy) / depth  # per-pixel relative depth change
    gt_vox, _ = voxel_downsample(depth_to_points(gt, K, min_depth=md.MIN_DEPTH)[0], 0.01)
    base = md.valid_mask(gt)
    rows = []
    for tau in (None, 0.1, 0.05, 0.02):
        keep = base if tau is None else base & (rel_grad < tau)
        pts, _ = depth_to_points(depth, K, mask=keep)
        s = md.chamfer_stats(voxel_downsample(pts, 0.01)[0], gt_vox)
        rows.append({
            "tau": "none" if tau is None else tau, "removed share": 1 - keep.sum() / base.sum(),
            "accuracy [m]": s["accuracy_mean_m"], "completeness [m]": s["completeness_mean_m"],
            "precision@5cm": s["precision@5cm"], "recall@5cm": s["recall@5cm"],
        })  # fmt: skip
    print("Exercise 1 - flying-pixel filter, DA3-BASE (median scale), frame", FRAME)
    print(pd.DataFrame(rows).to_string(index=False, float_format="{:.3f}".format))


def exercise_2(seq) -> None:
    """Fusion with raw, per-frame-oracle and single-global scale for DA3METRIC-LARGE."""
    K = seq.camera.K
    model = md.DepthModel("da3metric-large")
    preds, gts, poses, scales = [], [], [], []
    for i in FUSE:
        gt = seq.depth(i)
        z = model.predict(seq.rgb(i), K).values
        preds.append(z)
        gts.append(gt)
        poses.append(seq.gt_poses[i])
        scales.append(md.evaluate_frame(z, "metric", gt, ["median"])["median"]["scale"])
    del model
    md.free_cuda()

    def fuse(depths: list[np.ndarray]) -> np.ndarray:
        pts = [transform_points(T, depth_to_points(d, K, stride=2, mask=md.valid_mask(g))[0])
               for d, g, T in zip(depths, gts, poses)]  # fmt: skip
        return voxel_downsample(np.concatenate(pts), 0.01)[0]

    ref = fuse(gts)
    variants = {
        "raw metric": preds,
        "one global scale (mean of per-frame)": [np.mean(scales) * z for z in preds],
        "per-frame median scale (oracle)": [s * z for s, z in zip(scales, preds)],
    }
    rows = []
    for name, depths in variants.items():
        s = md.chamfer_stats(fuse(depths), ref)
        rows.append({"variant": name, "chamfer [m]": s["chamfer_m"], "fscore@5cm": s["fscore@5cm"]})
    print(f"\nExercise 2 - fused {len(FUSE)} frames vs fused Kinect; per-frame scales "
          f"{np.round(scales, 3).tolist()}")  # fmt: skip
    print(pd.DataFrame(rows).to_string(index=False, float_format="{:.3f}".format))


def exercise_3(seq) -> None:
    """Back-project the same Kinect depth with calibrated vs ROS-default intrinsics."""
    gt = seq.depth(FRAME)
    K_cal = seq.camera.K
    K_ros = np.array([[525.0, 0, 319.5], [0, 525.0, 239.5], [0, 0, 1]])
    a, _ = depth_to_points(gt, K_cal, min_depth=md.MIN_DEPTH)  # same pixel order in both
    b, _ = depth_to_points(gt, K_ros, min_depth=md.MIN_DEPTH)
    disp = np.linalg.norm(a - b, axis=1)
    v, u = np.nonzero(gt > md.MIN_DEPTH)
    worst = np.argmax(disp)
    s = md.chamfer_stats(voxel_downsample(b, 0.01)[0], voxel_downsample(a, 0.01)[0])
    print(f"\nExercise 3 - calibrated vs ROS-default K on frame {FRAME}")
    print(f"per-point displacement: mean {disp.mean() * 100:.2f} cm, median "
          f"{np.median(disp) * 100:.2f} cm, max {disp.max() * 100:.2f} cm at pixel "
          f"(u={u[worst]}, v={v[worst]}), depth {gt[v[worst], u[worst]]:.2f} m")  # fmt: skip
    print(f"Chamfer between the two clouds: {s['chamfer_m'] * 100:.2f} cm. The shift grows with "
          "the distance from the principal point and with depth: x = (u - cx) z / fx.")


def main() -> None:
    seed_everything(0)
    seq = md.load_tum("fr1_desk")
    exercise_1(seq)
    exercise_2(seq)
    exercise_3(seq)


if __name__ == "__main__":
    main()
