#!/usr/bin/env python
"""Run one visual-odometry variant on a TUM RGB-D sequence and report ATE / RPE.

Examples (from the repository root)::

    python projects/12_visual_odometry/scripts/run_vo.py --sequence fr1_xyz --method rgbd-kf
    python projects/12_visual_odometry/scripts/run_vo.py --sequence fr1_desk --method mono-gt \\
        --feature sift

The estimated trajectory is written in TUM format to
``runs/12_visual_odometry/<sequence>_<method>_<feature>.txt`` so it can be checked with evo::

    evo_ape tum data/tum/fr1_xyz/rgbd_dataset_freiburg1_xyz/groundtruth.txt \\
        runs/12_visual_odometry/fr1_xyz_rgbd-kf_orb.txt -a
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import vo  # noqa: E402

from cvlab import tum  # noqa: E402
from cvlab.io import data_dir, repo_root, runs_dir  # noqa: E402

SEQUENCES = {
    "fr1_xyz": ("fr1_xyz", "rgbd_dataset_freiburg1_xyz"),
    "fr1_desk": ("fr1_desk", "rgbd_dataset_freiburg1_desk"),
}
METHODS = {
    # name: (runner, kwargs, alignment used for the ATE)
    "rgbd-f2f": ("pnp", {"keyframes": False}, "se3"),
    "rgbd-kf": ("pnp", {"keyframes": True}, "se3"),
    "mono-gt": ("mono", {"scale": "gt"}, "se3"),
    "mono-unit": ("mono", {"scale": "unit"}, "sim3"),
    "mono-map": ("pnp", {"depth": "triangulated"}, "sim3"),
}


def run(sequence: str, method: str, feature: str = "orb", n_features: int = 2000):
    seq = tum.load_sequence(data_dir("tum", *SEQUENCES[sequence]))
    frontend = vo.FeatureFrontend(feature, n_features)
    kind, kwargs, align = METHODS[method]
    t0 = time.perf_counter()
    if kind == "pnp":
        result = vo.run_pnp_vo(seq, frontend, **kwargs)
    else:
        result = vo.run_mono_essential(seq, frontend, **kwargs)
    wall = time.perf_counter() - t0
    metrics = vo.evaluate(result, seq, align=align)
    metrics.update(sequence=sequence, wall_s=wall)
    return seq, result, metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sequence", choices=sorted(SEQUENCES), default="fr1_xyz")
    ap.add_argument("--method", choices=list(METHODS), default="rgbd-kf")
    ap.add_argument("--feature", choices=["orb", "sift"], default="orb")
    ap.add_argument("--n-features", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=None, help="TUM trajectory output path")
    args = ap.parse_args()

    _, result, metrics = run(args.sequence, args.method, args.feature, args.n_features)
    out = args.out or runs_dir(
        "12_visual_odometry", f"{args.sequence}_{args.method}_{args.feature}.txt"
    )
    result.save_tum(out)
    try:
        shown = out.resolve().relative_to(repo_root())
    except ValueError:
        shown = out
    print(f"{result.name} on {args.sequence}: {len(result.frame_ids)} poses -> {shown}")
    keys = [
        "align",
        "ate_rmse_m",
        "rpe_trans_rmse_m",
        "rpe_rot_rmse_deg",
        "drift_pct",
        "keyframes",
        "lost",
        "ms_per_frame",
        "wall_s",
    ]
    print(json.dumps({k: metrics[k] for k in keys}, indent=2, default=float))


if __name__ == "__main__":
    main()
