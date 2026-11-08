#!/usr/bin/env python
"""Solutions to the exercises of notebook 01 (VGGT feed-forward reconstruction).

Needs the GPU (VGGT-1B) and the cached mixed-precision prediction of ``benchmark.py``::

    flock /tmp/cv-gpu.lock python projects/10_feedforward_3d/solutions/solutions_01.py
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ff3d  # noqa: E402

from cvlab.geometry import rotation_error_deg  # noqa: E402
from cvlab.metrics import align_depth_scale, depth_metrics  # noqa: E402


def release_cuda_cache() -> None:
    """Return freed blocks to the driver (call after the last reference to a model is gone)."""
    gc.collect()
    torch.cuda.empty_cache()


def exercise_2(runner, scene: ff3d.Scene) -> None:
    print("\n=== Exercise 2: image 12.png as the reference view ===")
    base = ff3d.predict_cached(scene, "vggt")  # original order, mixed precision
    perm = [12] + [i for i in range(len(scene)) if i != 12]
    sub = scene.subset(perm)  # reorders the images AND the COLMAP reference poses
    rec = runner.predict(sub.image_paths)
    m0 = ff3d.evaluate_poses(base.T_cw, scene.ref_T_cw)
    m1 = ff3d.evaluate_poses(rec.T_cw, sub.ref_T_cw)
    print(pd.DataFrame([{"reference view": "00.png", **{k: m0[k] for k in ("auc@5", "auc@15")}},
                        {"reference view": "12.png", **{k: m1[k] for k in ("auc@5", "auc@15")}}])
          .set_index("reference view").round(4).to_string())  # fmt: skip
    # Same relative poses? Compare pair (i, j) of both runs (mapped back to the original order).
    inv = np.argsort(perm)
    T_back = rec.T_cw[inv]
    pe = ff3d.relative_pose_errors(T_back, base.T_cw)
    print(f"relative poses of the two runs differ by median {np.median(pe['rot_deg']):.3f} deg "
          f"(max {pe['rot_deg'].max():.3f} deg) in rotation")  # fmt: skip
    print("The relative poses are close but not identical: the model is not exactly equivariant")
    print("to the choice of reference frame (the first frame gets its own camera/register tokens).")


def exercise_3(runner, scene: ff3d.Scene) -> None:
    print("\n=== Exercise 3: a single image ===")
    multi = ff3d.predict_cached(scene, "vggt")
    single = runner.predict(scene.image_paths[:1])
    d_single, d_multi = single.depth[0].astype(np.float64), multi.depth[0].astype(np.float64)
    s = align_depth_scale(d_single, d_multi)
    m = depth_metrics(s * d_single, d_multi)
    ok = scene.has_ref
    fov_colmap = ff3d.fov_deg(scene.ref_K[ok][0], scene.ref_size[0])
    print(f"single-view vs multi-view depth of 00.png after median scaling: AbsRel "
          f"{m['abs_rel']:.4f}, delta1 {m['delta1']:.4f}")  # fmt: skip
    print(f"horizontal FoV: single view {ff3d.fov_deg(single.K[0], single.proc_hw[1]):.2f} deg, "
          f"multi-view {ff3d.fov_deg(multi.K[0], multi.proc_hw[1]):.2f} deg, "
          f"COLMAP {fov_colmap:.2f} deg")  # fmt: skip


def exercise_1(scene: ff3d.Scene) -> None:
    print("\n=== Exercise 1: all-bf16 vs mixed precision ===")
    mixed = ff3d.predict_cached(scene, "vggt")
    runner = ff3d.VGGTRunner(precision="bf16")
    runner.predict(scene.image_paths[:2])  # warm-up
    bf16 = runner.predict(scene.image_paths)
    del runner
    release_cuda_cache()
    rot = np.array([rotation_error_deg(a[:3, :3], b[:3, :3])
                    for a, b in zip(mixed.T_cw, bf16.T_cw)])  # fmt: skip
    rows = []
    for name, rec in [("mixed (bf16 aggregator, fp32 heads)", mixed), ("all bf16", bf16)]:
        m = ff3d.evaluate_poses(rec.T_cw, scene.ref_T_cw)
        rows.append({"precision": name, "weights VRAM (MiB)": rec.weights_vram_mb,
                     "peak VRAM (MiB)": rec.peak_vram_mb, "forward (s)": rec.runtime_s,
                     "AUC@5": m["auc@5"], "AUC@15": m["auc@15"]})  # fmt: skip
    print(pd.DataFrame(rows).set_index("precision").round(4).to_string())
    print(f"per-camera rotation difference mixed vs bf16: median {np.median(rot):.3f} deg, "
          f"max {rot.max():.3f} deg")  # fmt: skip


if __name__ == "__main__":
    scene = ff3d.load_scene("kitchen")
    runner = ff3d.VGGTRunner(precision="mixed")
    runner.predict(scene.image_paths[:2])  # warm-up
    exercise_2(runner, scene)
    exercise_3(runner, scene)
    del runner  # drop the only reference so the 2.8 GB of weights are freed before exercise 1
    release_cuda_cache()
    exercise_1(scene)
