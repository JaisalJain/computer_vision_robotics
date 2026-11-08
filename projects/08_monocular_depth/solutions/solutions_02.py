#!/usr/bin/env python
"""Solutions to the exercises of notebook 02 (TUM benchmark).

Needs the benchmark results (``scripts/eval_tum.py``) for the frame list and the per-frame
scales. Run (GPU, about 1-2 minutes)::

    flock /tmp/cv-gpu.lock python projects/08_monocular_depth/solutions/solutions_02.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import mono_depth as md  # noqa: E402

from cvlab.device import seed_everything  # noqa: E402
from cvlab.io import load_json, project_dir  # noqa: E402
from cvlab.metrics import depth_metrics  # noqa: E402

RESULTS = project_dir(md.PROJECT) / "results"


def exercise_1(seq, frames: list[int]) -> None:
    """Per-frame vs per-sequence vs first-frame scale for DA3-BASE on fr1_desk."""
    model = md.DepthModel("da3-base")
    preds, gts = [], []
    for i in frames:
        preds.append(model.predict(seq.rgb(i)).values)
        gts.append(seq.depth(i))
    del model
    md.free_cuda()
    scales = np.array([md.align_prediction(p, "depth", g, md.valid_mask(g), "median")[1]
                       for p, g in zip(preds, gts)])  # fmt: skip
    variants = {
        "per-frame median scale (oracle every frame)": scales,
        "one median scale for the whole sequence": np.full(len(frames), np.median(scales)),
        "scale calibrated on the first frame only": np.full(len(frames), scales[0]),
    }
    rows = []
    for name, s in variants.items():
        m = [md.evaluate_frame(si * p, "metric", g, ["none"])["none"]
             for si, p, g in zip(s, preds, gts)]  # fmt: skip
        rows.append({"scale strategy": name, "AbsRel": np.mean([r["abs_rel"] for r in m]),
                     "delta1": np.mean([r["delta1"] for r in m])})  # fmt: skip
    cv = scales.std() / scales.mean()
    print(f"Exercise 1 - DA3-BASE on {len(frames)} fr1_desk frames "
          f"(per-frame scale: mean {scales.mean():.3f}, coeff. of variation {cv:.3f})")
    print(pd.DataFrame(rows).to_string(index=False, float_format="%.3f"))


def edge_mask(gt: np.ndarray, thresh: float = 0.05, dilate: int = 2) -> np.ndarray:
    """Pixels near GT depth discontinuities: ||grad z*|| / z* > thresh, dilated."""
    g = np.where(md.valid_mask(gt), gt, np.nan)
    gy, gx = np.gradient(g)
    rel = np.hypot(gx, gy) / g
    edges = np.nan_to_num(rel, nan=0.0) > thresh
    kernel = np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8)
    return cv2.dilate(edges.astype(np.uint8), kernel).astype(bool) & md.valid_mask(gt)


def exercise_2(seq, frames: list[int]) -> None:
    """AbsRel on depth edges vs flat regions (12 fr1_desk frames)."""
    rows = []
    for key, proto in [("da2-small", "lsq"), ("da3-base", "median")]:
        model = md.DepthModel(key)
        acc = {"edge": [], "flat": []}
        err_edge = err_total = 0.0
        n_edge = n_total = 0
        for i in frames:
            gt = seq.depth(i)
            p = model.predict(seq.rgb(i))
            depth = md.evaluate_frame(p.values, p.output, gt, [proto])[proto]["depth"]
            edges = edge_mask(gt)
            flat = md.valid_mask(gt) & ~edges
            acc["edge"].append(depth_metrics(depth, gt, edges)["abs_rel"])
            acc["flat"].append(depth_metrics(depth, gt, flat)["abs_rel"])
            err = md.abs_rel_map(depth, gt)
            err_edge += np.nansum(err[edges])
            err_total += np.nansum(err)
            n_edge += edges.sum()
            n_total += md.valid_mask(gt).sum()
        del model
        md.free_cuda()
        rows.append({"model": md.MODELS[key].label, "protocol": md.PROTOCOL_LABELS[proto],
                     "AbsRel edges": np.mean(acc["edge"]), "AbsRel flat": np.mean(acc["flat"]),
                     "edge pixel share": n_edge / n_total,
                     "edge share of total error": err_edge / err_total})  # fmt: skip
    print(f"\nExercise 2 - edges vs flat regions on {len(frames)} fr1_desk frames")
    print(pd.DataFrame(rows).to_string(index=False, float_format="%.3f"))


def exercise_3(per_frame: pd.DataFrame, seq, frames: list[int]) -> None:
    """Calibrate DA3METRIC-LARGE's scale on fr1_desk, apply it on fr1_xyz."""
    sel = (per_frame["model"] == "da3metric-large") & (per_frame["protocol"] == "median")
    c_desk = per_frame[sel & (per_frame["sequence"] == "fr1_desk")]["scale"].median()
    c_xyz = per_frame[sel & (per_frame["sequence"] == "fr1_xyz")]["scale"].median()
    model = md.DepthModel("da3metric-large")
    raw, corrected = [], []
    K = seq.camera.K
    for i in frames:
        gt = seq.depth(i)
        z = model.predict(seq.rgb(i), K).values
        raw.append(md.evaluate_frame(z, "metric", gt, ["none"])["none"]["abs_rel"])
        corrected.append(md.evaluate_frame(c_desk * z, "metric", gt, ["none"])["none"]["abs_rel"])
    del model
    md.free_cuda()
    print(f"\nExercise 3 - DA3METRIC-LARGE scale correction c = {c_desk:.3f} learned on fr1_desk "
          f"(median scale on fr1_xyz itself would be {c_xyz:.3f})")  # fmt: skip
    print(f"fr1_xyz ({len(frames)} frames) raw AbsRel {np.mean(raw):.3f} -> "
          f"corrected AbsRel {np.mean(corrected):.3f}")  # fmt: skip


def main() -> None:
    seed_everything(0)
    meta = load_json(RESULTS / "tum_eval_meta.json")
    per_frame = pd.read_csv(RESULTS / "tum_per_frame.csv")
    desk, xyz = md.load_tum("fr1_desk"), md.load_tum("fr1_xyz")
    exercise_1(desk, meta["frames"]["fr1_desk"])
    exercise_2(desk, meta["frames"]["fr1_desk"][::4])
    exercise_3(per_frame, xyz, meta["frames"]["fr1_xyz"])


if __name__ == "__main__":
    main()
