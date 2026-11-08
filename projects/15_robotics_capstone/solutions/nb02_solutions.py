#!/usr/bin/env python
"""Solutions to the exercises of notebook 02 (masks + depth -> boxes -> grasps). Needs the GPU.

flock /tmp/cv-gpu.lock python projects/15_robotics_capstone/solutions/nb02_solutions.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import transformers

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import capstone as cap  # noqa: E402
import grasp3d as g3  # noqa: E402

from cvlab.metrics import align_depth_scale  # noqa: E402

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()
DEV = "cuda"


def grasp_str(g: g3.Grasp | None) -> str:
    return "-" if g is None else f"{g.axis}, w={100 * g.width:.1f} cm, score {g.score:.2f}"


def main() -> None:
    frames = cap.load_frames()
    rgb, depth = frames.rgb(540), frames.depth(540)
    det = cap.GroundingDinoDetector(DEV)
    dets = det.detect(rgb, cap.QUERIES)
    seg = cap.BoxSegmenter("sam3", DEV)
    seg.set_image(rgb)
    masks, _ = seg.segment(dets.boxes)
    mono_model = cap.MetricDepth(DEV)
    mono = mono_model.predict(rgb)
    del det, seg, mono_model
    cap.free()
    K, D = cap.CAMERAS[cap.DEFAULT_CAMERA]
    R = g3.pixel_rays(K, D, 640, 480)
    names = [f"{k}:{lab}" for k, lab in enumerate(dets.labels)]

    # Exercise 1: an 85 mm vs a 140 mm stroke
    small = cap.estimate_objects(rgb, depth, dets, masks, R)
    big = cap.estimate_objects(rgb, depth, dets, masks, R, gripper=g3.Gripper(max_opening=0.14))
    print("\nExercise 1 - best grasp with an 85 mm and a 140 mm gripper")
    print(
        pd.DataFrame(
            {
                "85 mm": [grasp_str(r.best_grasp) for r in small],
                "140 mm": [grasp_str(r.best_grasp) for r in big],
                "feasible candidates 85 / 140 mm": [
                    f"{sum(g.feasible for g in a.grasps)} / {sum(g.feasible for g in b.grasps)}"
                    for a, b in zip(small, big)
                ],
            },
            index=names,
        ).to_string()
    )

    # Exercise 2: no erosion and no depth gate
    rows = []
    for k, r in enumerate(small):
        if r.plane is None or r.cloud is None:
            continue
        raw = g3.object_cloud(depth, masks[k], R, rgb, erode_px=0, gate_k=1e9, gate_min=1e9)
        b0, b1 = r.box, g3.table_box(raw.points, r.plane)
        rows.append(
            {
                "object": names[k],
                "default [cm]": np.round(100 * b0.extents, 1),
                "no erosion / gate [cm]": np.round(100 * b1.extents, 1),
                "centre shift [cm]": round(100 * float(np.linalg.norm(b1.center - b0.center)), 1),
                "points default / raw": f"{len(r.cloud.points)} / {len(raw.points)}",
            }
        )
    print("\nExercise 2 - table-box extents (L x W x H) with and without cloud cleaning")
    print(pd.DataFrame(rows).set_index("object").to_string())

    # Exercise 3: grasps from median-scaled monocular depth
    scale = align_depth_scale(mono, depth, depth > 0)
    mono_res = cap.estimate_objects(rgb, scale * mono, dets, masks, R)
    rows = []
    for k, (a, b) in enumerate(zip(small, mono_res)):
        ga, gb = a.best_grasp, b.best_grasp
        d = None
        if ga is not None and gb is not None:
            d = round(100 * float(np.linalg.norm(ga.position - gb.position)), 1)
        rows.append(
            {
                "object": names[k],
                "sensor": grasp_str(ga),
                "mono (scaled)": grasp_str(gb),
                "position offset [cm]": d,
            }
        )
    print(f"\nExercise 3 - grasps from monocular depth x{scale:.3f} vs sensor depth")
    print(pd.DataFrame(rows).set_index("object").to_string())


if __name__ == "__main__":
    main()
