#!/usr/bin/env python
"""Solutions to the exercises of notebook 02 (stereo rectification and SGBM).

Run (CPU only, about 1 minute)::

    python projects/09_multiview_geometry/solutions/solutions_02.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import mvgeo as M  # noqa: E402


# ----------------------------------------------------------------------------- exercise 1
def sgbm_custom(left, right, ndisp, uniqueness, p2_ratio, block=5):
    cn = 3
    P1 = 8 * cn * block**2
    sgbm = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=M.num_disparities(ndisp), blockSize=block, P1=P1,
        P2=int(p2_ratio * P1), disp12MaxDiff=1, preFilterCap=63, uniquenessRatio=uniqueness,
        speckleWindowSize=100, speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )  # fmt: skip

    def run(a, b):
        d = sgbm.compute(cv2.cvtColor(a, cv2.COLOR_RGB2BGR), cv2.cvtColor(b, cv2.COLOR_RGB2BGR))
        d = d.astype(np.float32) / 16.0
        d[d < 0] = np.nan
        return d

    dl = run(left, right)
    dr = run(right[:, ::-1].copy(), left[:, ::-1].copy())[:, ::-1]
    return M.fill_holes(np.where(M.lr_consistency(dl, dr, 1.0), dl, np.nan))


def exercise_1() -> None:
    rows = []
    for scene in ["Motorcycle", "Piano", "Jadeplant"]:
        p = M.load_middlebury(scene)
        for u in (0, 5, 10, 15):
            for ratio in (2, 4, 8):
                d = sgbm_custom(p.left, p.right, p.ndisp, u, ratio)
                bad = M.disparity_metrics(d, p.gt, p.nocc)["bad-2 (all)"]
                rows.append({"scene": scene, "uniqueness": u, "P2/P1": ratio, "bad-2 (all)": bad})
    df = pd.DataFrame(rows)
    table = df.pivot_table(index=["uniqueness", "P2/P1"], columns="scene", values="bad-2 (all)")
    table["mean"] = table.mean(axis=1)
    print("Exercise 1 - SGBM grid search, bad-2 (all) [%] after LR check + hole filling")
    print(table.round(2).to_string())
    best = table.idxmin()
    print("best (uniqueness, P2/P1) per scene:", {k: v for k, v in best.items()})
    best_u = {k: v[0] for k, v in best.items() if k != "mean"}
    best_r = {k: v[1] for k, v in best.items() if k != "mean"}
    print(f"best P2/P1 per scene: {best_r} | best uniqueness ratio per scene: {best_u}")
    print("Where the optimum differs between scenes (texture, disparity range, thin")
    print("structures), a single global setting is a compromise: tune on some scenes and")
    print("report on held-out ones.\n")


# ----------------------------------------------------------------------------- exercise 2
def wta_block_matching(left, right, ndisp, block=9, cost="sad"):
    """Winner-takes-all block matching on a cost volume + parabola sub-pixel refinement."""
    h, w = left.shape[:2]
    if cost == "sad":
        gl = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gr = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY).astype(np.float32)
        cl, cr = gl, gr
    else:
        cl, cr = census(left), census(right)
    vol = np.full((ndisp, h, w), np.inf, np.float32)
    for d in range(ndisp):
        if cost == "sad":
            c = np.abs(cl[:, d:] - cr[:, : w - d])
        else:
            c = np.bitwise_count(cl[:, d:] ^ cr[:, : w - d]).astype(np.float32)
        vol[d, :, d:] = cv2.blur(c, (block, block))  # box-filter aggregation
    disp = np.argmin(vol, axis=0)
    # Sub-pixel: parabola through C(d-1), C(d), C(d+1).
    di = np.clip(disp, 1, ndisp - 2)
    rows, cols = np.mgrid[0:h, 0:w]
    c0, c1, c2 = vol[di - 1, rows, cols], vol[di, rows, cols], vol[di + 1, rows, cols]
    with np.errstate(invalid="ignore", divide="ignore"):  # inf costs at the left border
        denom = c0 - 2 * c1 + c2
        delta = np.where(np.isfinite(denom) & (denom > 0), 0.5 * (c0 - c2) / denom, 0.0)
    out = disp.astype(np.float32) + np.where((disp >= 1) & (disp <= ndisp - 2), delta, 0.0)
    out[:, :ndisp] = np.nan  # no full search range at the left border
    return out


def census(img, r=3):
    """7x7 census transform packed into uint64 (48 comparison bits)."""
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.int16)
    pad = np.pad(g, r, mode="edge")
    h, w = g.shape
    code = np.zeros((h, w), np.uint64)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy == 0 and dx == 0:
                continue
            nb = pad[r + dy : r + dy + h, r + dx : r + dx + w]
            code = (code << np.uint64(1)) | (nb < g).astype(np.uint64)
    return code


def exercise_2() -> None:
    p = M.load_middlebury("Teddy")
    rows = []
    for name, fn in [
        ("NumPy SAD WTA (9x9)", lambda: wta_block_matching(p.left, p.right, p.ndisp, 9)),
        ("cv2.StereoBM (15x15)", lambda: M.stereo_bm(p.left, p.right, p.ndisp)),
        ("cv2.StereoSGBM", lambda: M.stereo_sgbm(p.left, p.right, p.ndisp)),
    ]:
        t0 = time.perf_counter()
        d = fn()
        ms = 1000 * (time.perf_counter() - t0)
        rows.append({"method": name, **M.disparity_metrics(d, p.gt, p.nocc), "time_ms": ms})
    print("Exercise 2 - own block matcher on Teddy")
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("WTA without a uniqueness/LR check keeps every pixel (density ~1 inside the search")
    print("range) but makes many errors in textureless and occluded regions.\n")


# ----------------------------------------------------------------------------- exercise 3
def exercise_3() -> None:
    rows = []
    for scene in ["Motorcycle", "MotorcycleE", "Piano", "PianoL"]:
        p = M.load_middlebury(scene)
        for name, d in [
            ("SGBM (BT cost)", M.stereo_sgbm(p.left, p.right, p.ndisp)),
            ("SAD WTA 9x9", wta_block_matching(p.left, p.right, p.ndisp, 9, "sad")),
            ("census WTA 9x9", wta_block_matching(p.left, p.right, p.ndisp, 9, "census")),
        ]:
            m = M.disparity_metrics(d, p.gt, p.nocc)
            rows.append({"scene": scene, "method": name, "bad-2": m["bad-2"]})
    df = pd.DataFrame(rows).pivot(index="method", columns="scene", values="bad-2")
    print("Exercise 3 - photometric changes, bad-2 [%] on valid pixels")
    print(df.round(2).to_string())
    print("* Raw-intensity SAD collapses under both changes: the two views no longer agree on")
    print("  absolute intensities.")
    print("* Census only compares the *order* of intensities in a window, so it is invariant to")
    print("  the exposure change of MotorcycleE, but not to PianoL's different lighting, which")
    print("  moves shading and shadows and changes the local intensity order.")
    print("* OpenCV SGBM stays robust: it pre-filters the images with a clipped horizontal")
    print("  derivative (preFilterCap), which removes most of a global exposure change, and its")
    print("  smoothness term regularises the remaining ambiguous pixels.")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
