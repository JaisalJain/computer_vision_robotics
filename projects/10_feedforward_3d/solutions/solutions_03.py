#!/usr/bin/env python
"""Solutions to the exercises of notebook 03 (scaling and metric geometry on TUM fr1_desk).

CPU only: everything is computed from the cached predictions of ``benchmark.py``::

    python projects/10_feedforward_3d/solutions/solutions_03.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ff3d  # noqa: E402

from cvlab.geometry import invert_T  # noqa: E402
from cvlab.pointcloud import depth_to_points, voxel_downsample  # noqa: E402

VOXEL = 0.01  # 1 cm, as in the notebook
MAX_DEPTH = 4.0  # Kinect returns beyond 4 m are too noisy to serve as reference


def setup():
    """TUM scene, cached VGGT prediction, its Sim(3) to mocap and the reference cloud."""
    scene = ff3d.load_scene("tum_desk")
    rec = ff3d.predict_cached(scene, "vggt")
    al = ff3d.sim3_align(rec.T_cw, scene.ref_T_cw)
    gt_depth = [scene.read_depth(i) for i in range(len(scene))]
    ref = []
    for i in range(len(scene)):
        P, _ = depth_to_points(gt_depth[i], scene.ref_K[i], stride=2, max_depth=MAX_DEPTH)
        T_wc = invert_T(scene.ref_T_cw[i])
        ref.append(P @ T_wc[:3, :3].T + T_wc[:3, 3])
    ref_cloud, _ = voxel_downsample(np.concatenate(ref), VOXEL)
    gt_valid = [(lambda g: (g > 0) & (g < MAX_DEPTH))(ff3d.resize_depth_nearest(d, rec.proc_hw))
                for d in gt_depth]  # fmt: skip
    return scene, rec, al, gt_valid, ref_cloud


def score(points: np.ndarray, ref_cloud: np.ndarray) -> dict[str, float]:
    P, _ = voxel_downsample(points, VOXEL)
    ch = ff3d.chamfer(P, ref_cloud)
    return {"accuracy median (cm)": 100 * ch["acc_median"],
            "completeness median (cm)": 100 * ch["comp_median"],
            "Chamfer mean (cm)": 100 * ch["chamfer_mean"], "points": len(P)}  # fmt: skip


def unproject(rec, i: int, mask: np.ndarray, T_wc: np.ndarray, scale: float = 1.0) -> np.ndarray:
    P, _ = depth_to_points(scale * rec.depth[i], rec.K[i], mask=mask, stride=2)
    return P @ T_wc[:3, :3].T + T_wc[:3, 3]


def exercise_1(scene, rec, al, gt_valid, ref_cloud) -> None:
    print("\n=== Exercise 1: confidence threshold vs accuracy / completeness (VGGT) ===")
    rows = []
    for q in [0.0, 0.1, 0.3, 0.5, 0.7]:
        thr = np.quantile(rec.conf, q)
        pts = [unproject(rec, i, gt_valid[i] & (rec.conf[i] >= thr), invert_T(rec.T_cw[i]))
               for i in range(len(scene))]  # fmt: skip
        P = ff3d.sim3_apply(np.concatenate(pts), al["s"], al["R"], al["t"])
        rows.append({"kept fraction": 1 - q, **score(P, ref_cloud)})
    print(pd.DataFrame(rows).set_index("kept fraction").round(2).to_string())
    print("Dropping low-confidence pixels improves accuracy (the remaining points sit closer to")
    print("the true surface) at the cost of completeness (holes appear): the confidence ranks")
    print("pixels by expected error, so it trades one for the other.")


def exercise_2(scene, rec, al, gt_valid, ref_cloud) -> None:
    print("\n=== Exercise 2: point-map head vs depth unprojection, in metres (VGGT) ===")
    pm = np.concatenate([rec.world_points[i][gt_valid[i]][::4] for i in range(len(scene))])
    dp = np.concatenate([unproject(rec, i, gt_valid[i], invert_T(rec.T_cw[i]))
                         for i in range(len(scene))])  # fmt: skip
    rows = []
    for name, P in [("depth + camera (section 4)", dp), ("point-map head", pm)]:
        rows.append({"route": name, **score(ff3d.sim3_apply(P, al["s"], al["R"], al["t"]),
                                            ref_cloud)})  # fmt: skip
    df = pd.DataFrame(rows).set_index("route")
    print(df.round(2).to_string())
    print("The point map lives in the same world frame as the predicted cameras, so the same")
    print("Sim(3) applies. Both routes keep ~1/4 of the masked pixels before the 1 cm voxels.")
    acc = df["accuracy median (cm)"]
    best = acc.idxmin()
    print(f"Here '{best}' has the lower median accuracy distance ({acc.min():.2f} vs "
          f"{acc.max():.2f} cm): on this scene the paper's finding that depth + camera beats the")
    print("point map is not universal, so it is worth checking both routes on your own data.")


def exercise_3(scene, rec, al, gt_valid, ref_cloud) -> None:
    print("\n=== Exercise 3: VGGT depth fused with the mocap trajectory ===")
    pred = np.concatenate([unproject(rec, i, gt_valid[i], invert_T(rec.T_cw[i]))
                           for i in range(len(scene))])  # fmt: skip
    pred = ff3d.sim3_apply(pred, al["s"], al["R"], al["t"])
    known = np.concatenate([unproject(rec, i, gt_valid[i], invert_T(scene.ref_T_cw[i]), al["s"])
                            for i in range(len(scene))])  # fmt: skip
    rows = [{"poses": "predicted + Sim(3)", **score(pred, ref_cloud)},
            {"poses": "mocap (known trajectory)", **score(known, ref_cloud)}]
    df = pd.DataFrame(rows).set_index("poses")
    print(df.round(2).to_string())
    print("With the true poses only the depth error (and the single scale s) remains; the")
    print("difference between the rows is the share of the error that came from the cameras.")
    d = df.iloc[0] - df.iloc[1]  # predicted minus known trajectory
    acc, comp = -d["accuracy median (cm)"], -d["completeness median (cm)"]
    print(f"Known poses change the median accuracy distance by {acc:+.2f} cm,")
    print(f"the median completeness distance by {comp:+.2f} cm and the "
          f"Chamfer mean by {-d['Chamfer mean (cm)']:+.2f} cm (negative = better).")  # fmt: skip
    print("The predicted cameras (ATE ~1.5 cm) are a minor part of the error; the depth maps")
    print("themselves dominate it.")


if __name__ == "__main__":
    ctx = setup()
    exercise_1(*ctx)
    exercise_2(*ctx)
    exercise_3(*ctx)
