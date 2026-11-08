#!/usr/bin/env python
"""Solutions to the exercises of notebook 03 (DISK + LightGlue vs SIFT).

Run on the GPU (about 2 minutes; needs ``make data-09_multiview_geometry``)::

    flock /tmp/cv-gpu.lock python projects/09_multiview_geometry/solutions/solutions_03.py

All three exercises use the South Building pairs of the notebook (pairs ``(i, i + gap)``
every 8 images, reference poses from the COLMAP model shipped with the dataset).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import kornia.feature as KF
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import mvgeo as M  # noqa: E402

from cvlab.device import setup, track_vram  # noqa: E402
from cvlab.io import read_image  # noqa: E402

DEVICE = setup(seed=0)
REF = M.load_reference_cameras(scale_to_width=1024)
NAMES = sorted(REF)
K = REF[NAMES[0]]["K_cv"]  # OpenCV pixel convention
DIST = np.array([REF[NAMES[0]]["params"][3], 0.0, 0.0, 0.0])  # SIMPLE_RADIAL k -> OpenCV k1
IMAGES = {n: read_image(M.south_building_dir() / "images" / n) for n in NAMES}


def pairs(gaps: tuple[int, ...]) -> list[dict]:
    """Same pair list as the notebook: (i, i + gap) every 8 images, with GT relative pose."""
    out = []
    for gap in gaps:
        for i in range(0, len(NAMES) - gap, 8):
            a, b = NAMES[i], NAMES[i + gap]
            R, t = M.relative_pose_from_c2w(REF[a]["T_wc"], REF[b]["T_wc"])
            out.append({"gap": gap, "a": a, "b": b, "R": R, "t": t})
    return out


def pose_error(m: M.Matches, p: dict, thr: float = 1.0) -> float:
    """max(rotation error, translation-direction error) of 5-pt MAGSAC++ on the matches."""
    est = M.ransac_essential(m, K, thresh_px=thr, dist1=DIST)
    if est["R"] is None:
        return 180.0
    return max(M.pose_errors(est["R"], est["t"], p["R"], p["t"]))


def sync() -> None:
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


# ----------------------------------------------------------------------------- exercise 1
def exercise_1() -> None:
    plist = pairs((4,))
    rows = []
    for n in (512, 1024, 2048, 4096):
        dl = M.DiskLightGlue(device=DEVICE, max_keypoints=n, fp16=True)
        for _ in range(2):  # warm-up (cuDNN autotuning for this keypoint count)
            dl.match(IMAGES[plist[0]["a"]], IMAGES[plist[0]["b"]])
        errs, times = [], []
        with track_vram() as mem:
            for p in plist:
                m = dl.match(IMAGES[p["a"]], IMAGES[p["b"]])
                times.append(m.time_ms)
                errs.append(pose_error(m, p))
        rows.append({"max_keypoints": n, "pairs": len(plist),
                     "AUC@10 [%]": 100 * M.pose_auc(np.array(errs))["auc@10"],
                     "median pose err [deg]": float(np.median(errs)),
                     "time per pair [ms]": float(np.mean(times)),
                     "peak VRAM [MiB]": mem.peak_mb})  # fmt: skip
        del dl
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    print("Exercise 1 - keypoint budget, South Building gap-4 pairs (DISK + LightGlue, fp16)")
    print(df.round(2).to_string(index=False))
    best = df.loc[df["AUC@10 [%]"].idxmax()]
    gain = df["AUC@10 [%]"].diff().iloc[1:].to_numpy()
    cost = df["time per pair [ms]"].diff().iloc[1:].to_numpy()
    print(f"Highest AUC@10 with {int(best.max_keypoints)} keypoints. Per doubling of the budget:")
    for n, g, c in zip(df.max_keypoints.iloc[1:], gain, cost):
        print(f"  -> {n:5d} keypoints: AUC@10 {g:+.2f} points for {c:+.1f} ms per pair")
    print("The knee is the last doubling whose AUC gain is still worth its extra runtime.\n")


# ----------------------------------------------------------------------------- exercise 2
def exercise_2() -> None:
    dl = M.DiskLightGlue(device=DEVICE, max_keypoints=2048, fp16=True)
    fixed = KF.LightGlue("disk", mp=dl.fp16, depth_confidence=-1, width_confidence=-1)
    fixed = fixed.to(DEVICE).eval()
    variants = {"adaptive (default)": dl.lightglue, "fixed depth + width": fixed}
    feats = {n: dl.features(IMAGES[n]) for n in NAMES}  # extraction is not part of the timing
    rows = []
    for gap in (1, 8):
        plist = pairs((gap,))
        for name, lg in variants.items():
            dl.lightglue = lg
            for _ in range(3):  # warm-up
                dl.match_features(feats[plist[0]["a"]], feats[plist[0]["b"]])
            errs, times = [], []
            for p in plist:
                sync()
                t0 = time.perf_counter()
                idx, _ = dl.match_features(feats[p["a"]], feats[p["b"]])
                sync()
                times.append(1000 * (time.perf_counter() - t0))
                k1 = feats[p["a"]]["keypoints"].cpu().numpy().astype(np.float64)
                k2 = feats[p["b"]]["keypoints"].cpu().numpy().astype(np.float64)
                m = M.Matches(k1[idx[:, 0]], k2[idx[:, 1]], len(k1), len(k2))
                errs.append(pose_error(m, p))
            rows.append({"gap": gap, "LightGlue": name, "pairs": len(plist),
                         "matching time [ms]": float(np.mean(times)),
                         "AUC@10 [%]": 100 * M.pose_auc(np.array(errs))["auc@10"]})  # fmt: skip
    dl.lightglue = variants["adaptive (default)"]
    df = pd.DataFrame(rows)
    print("Exercise 2 - LightGlue adaptive depth/width vs a fixed 9-layer, no-pruning matcher")
    print(df.round(2).to_string(index=False))
    for gap in (1, 8):
        d = df[df.gap == gap].set_index("LightGlue")
        t, auc = d["matching time [ms]"], d["AUC@10 [%]"]
        print(f"  gap {gap}: adaptive matcher takes {t.iloc[0] / t.iloc[1]:.0%} of the fixed "
              f"matcher's time; AUC@10 {auc.iloc[0]:.1f} vs {auc.iloc[1]:.1f} %")
    print("Early exit stops once the matcher is confident (easy pairs need fewer layers) and")
    print("pruning drops points that cannot be matched (more of them on hard, low-overlap")
    print("pairs). Both trade a little accuracy for speed: compare the AUC columns per gap;")
    print("on a robot, pick the variant that fits the latency budget.\n")


# ----------------------------------------------------------------------------- exercise 3
def exercise_3() -> None:
    dl = M.DiskLightGlue(device=DEVICE, max_keypoints=2048, fp16=True)
    plist = pairs((1, 2, 4, 8))
    matches = {"SIFT + ratio": [], "DISK + LightGlue": []}
    for p in plist:
        a, b = IMAGES[p["a"]], IMAGES[p["b"]]
        matches["SIFT + ratio"].append(M.sift_match(a, b, n_features=4096))
        matches["DISK + LightGlue"].append(dl.match(a, b))
    rows = []
    for name, ms in matches.items():
        for thr in (0.5, 1.0, 2.0):
            errs = np.array([pose_error(m, p, thr) for m, p in zip(ms, plist)])
            auc = M.pose_auc(errs)
            rows.append({"method": name, "threshold [px]": thr,
                         "AUC@5 [%]": 100 * auc["auc@5"], "AUC@10 [%]": 100 * auc["auc@10"],
                         "median pose err [deg]": float(np.median(errs))})  # fmt: skip
    df = pd.DataFrame(rows)
    print(f"Exercise 3 - MAGSAC++ threshold, South Building pairs (gaps 1/2/4/8, {len(plist)})")
    print(df.round(2).to_string(index=False))
    best = {}
    for name in matches:
        d = df[df.method == name].set_index("threshold [px]")["AUC@10 [%]"]
        best[name] = d.idxmax()
        print(f"  {name}: best AUC@10 at a {d.idxmax():g} px threshold "
              f"(0.5 px: {d[0.5]:.1f} %, 2 px: {d[2.0]:.1f} %)")
    print("DISK keypoints lie on the integer pixel grid (quantisation error up to 0.5 px per")
    print("coordinate), which suggests DISK should need a looser threshold than sub-pixel SIFT.")
    if best["DISK + LightGlue"] <= best["SIFT + ratio"]:
        print("The data say otherwise here: DISK + LightGlue does not prefer a larger threshold.")
        print("In OpenCV's MAGSAC++ the threshold is the upper bound of the noise scale that is")
        print("marginalised over, not a hard inlier cut-off; with hundreds of matches per pair a")
        print("tighter bound plausibly just down-weights the least precise correspondences (an")
        print("interpretation, not tested further here).")
    else:
        print("The data agree: DISK + LightGlue prefers a larger threshold than SIFT.")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
