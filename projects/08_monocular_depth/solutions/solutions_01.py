#!/usr/bin/env python
"""Solutions to the exercises of notebook 01 (models, outputs and scale alignment).

Run (GPU, about 1 minute)::

    flock /tmp/cv-gpu.lock python projects/08_monocular_depth/solutions/solutions_01.py

Writes ``outputs/08_monocular_depth/solutions_01_wrong_space.png`` (gitignored).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import mono_depth as md  # noqa: E402

from cvlab.device import seed_everything  # noqa: E402
from cvlab.io import outputs_dir, repo_root  # noqa: E402

FRAME = 450


def exercise_1(rgb: np.ndarray, gt: np.ndarray) -> None:
    """LSQ scale+shift in depth space vs inverse-depth space for DA-V2 relative."""
    model = md.DepthModel("da2-small")
    disp = model.predict(rgb).values
    del model
    right = md.evaluate_frame(disp, "disparity", gt)["lsq"]
    wrong = md.evaluate_frame(1.0 / np.maximum(disp, 1e-6), "depth", gt, ["lsq"])["lsq"]
    keys = ("abs_rel", "rmse", "delta1")
    df = pd.DataFrame(
        [{"fit space": "inverse depth (correct)", **{k: right[k] for k in keys}},
         {"fit space": "depth of 1/d (wrong)", **{k: wrong[k] for k in keys}}]
    )  # fmt: skip
    print("Exercise 1 - DA-V2 Small, frame", FRAME)
    print(df.to_string(index=False, float_format="%.3f"))

    # Error vs GT depth for both fits: the wrong fit bends at the near/far ends.
    valid = md.valid_mask(gt)
    bins = np.linspace(0.5, 3.0, 11)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for name, res in [("inverse-depth fit", right), ("depth-space fit", wrong)]:
        err = md.abs_rel_map(res["depth"], gt)[valid]
        idx = np.digitize(gt[valid], bins)
        means = [np.mean(err[idx == b]) if np.any(idx == b) else np.nan
                 for b in range(1, len(bins))]  # fmt: skip
        ax.plot(0.5 * (bins[1:] + bins[:-1]), means, "o-", label=name)
    ax.set(xlabel="GT depth [m]", ylabel="AbsRel", title="DA-V2 Small: alignment space")
    ax.legend()
    path = outputs_dir(md.PROJECT, create=True) / "solutions_01_wrong_space.png"
    fig.savefig(path, dpi=100)
    print("figure:", path.relative_to(repo_root()))


def exercise_2(rgb: np.ndarray, gt: np.ndarray, K: np.ndarray) -> None:
    """DA3METRIC-LARGE: metric depth scales linearly with the focal length."""
    model = md.DepthModel("da3metric-large")
    rows = []
    for name, (fx, fy) in {"calibrated fr1": (K[0, 0], K[1, 1]), "ROS default": (525.0, 525.0),
                           "wrong (400 px)": (400.0, 400.0)}.items():  # fmt: skip
        Kv = K.copy()
        Kv[0, 0], Kv[1, 1] = fx, fy
        p = model.predict(rgb, Kv)
        res = md.evaluate_frame(p.values, "metric", gt)
        rows.append({"intrinsics": name, "f_px": 0.5 * (fx + fy),
                     "median depth [m]": float(np.median(p.values)),
                     "raw AbsRel": res["none"]["abs_rel"], "median scale s": res["median"]["scale"],
                     "AbsRel after median scale": res["median"]["abs_rel"]})  # fmt: skip
    del model
    print("\nExercise 2 - DA3METRIC-LARGE focal-length sensitivity, frame", FRAME)
    print(pd.DataFrame(rows).to_string(index=False, float_format="%.3f"))
    print("The prediction is z = f * y / 300: changing f rescales every depth by the same factor,")
    print("so the median-scaled AbsRel (shape) is unchanged and only the global scale moves.")


def exercise_3(rgb: np.ndarray, gt: np.ndarray) -> None:
    """DA3-BASE accuracy and latency vs processing resolution."""
    rows = []
    for res_px in (280, 504, 756):
        model = md.DepthModel("da3-base", process_res=res_px)
        model.predict(rgb)  # warm-up
        times = []
        for _ in range(5):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            p = model.predict(rgb)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
        r = md.evaluate_frame(p.values, "depth", gt, ["median"])["median"]
        rows.append({"process_res": res_px, "network input": str(p.proc_hw),
                     "AbsRel (median scale)": r["abs_rel"], "delta1": r["delta1"],
                     "latency median [ms]": float(np.median(times))})  # fmt: skip
        del model
        md.free_cuda()
    print("\nExercise 3 - DA3-BASE vs processing resolution, frame", FRAME)
    print(pd.DataFrame(rows).to_string(index=False, float_format="%.3f"))


def main() -> None:
    seed_everything(0)
    seq = md.load_tum("fr1_desk")
    rgb, gt, K = seq.rgb(FRAME), seq.depth(FRAME), seq.camera.K
    exercise_1(rgb, gt)
    exercise_2(rgb, gt, K)
    exercise_3(rgb, gt)


if __name__ == "__main__":
    main()
