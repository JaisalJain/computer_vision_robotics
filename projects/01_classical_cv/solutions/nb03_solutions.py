#!/usr/bin/env python
"""Solutions to the exercises of notebook 03 (checkerboard camera calibration).

Run from the repository root (CPU only, about half a minute)::

    python projects/01_classical_cv/solutions/nb03_solutions.py
"""

from __future__ import annotations

import textwrap

import cv2
import numpy as np

from cvlab import io

CALIB = io.data_dir("01_classical_cv", "calib")
PATTERN = (9, 6)
HALF_WIN = 7  # cornerSubPix half-window selected by leave-one-out error in notebook 03
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
OBJ = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
OBJ[:, :2] = np.mgrid[0 : PATTERN[0], 0 : PATTERN[1]].T.reshape(-1, 2)


def note(text: str) -> None:
    """Print an explanatory paragraph, wrapped and indented."""
    print(textwrap.fill(text, 96, initial_indent="  ", subsequent_indent="  "))


def detect(prefix: str):
    names, imgs, corners = [], [], []
    for f in sorted(CALIB.glob(f"{prefix}*.jpg")):
        g = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        ok, c = cv2.findChessboardCorners(
            g, PATTERN, flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if not ok:
            print(f"  board not found in {f.name}")
            continue
        names.append(f.stem)
        imgs.append(g)
        corners.append(cv2.cornerSubPix(g, c, (HALF_WIN, HALF_WIN), (-1, -1), CRITERIA))
    return names, imgs, corners


def calibrate(corners, size, flags=0):
    rms, K, d, _, _, std, _, _ = cv2.calibrateCameraExtended(
        [OBJ] * len(corners), corners, size, None, None, flags=flags
    )
    return rms, K, d.ravel(), std.ravel()


def holdout(K, d, c):
    _, rv, tv = cv2.solvePnP(OBJ, c, K, d)
    p, _ = cv2.projectPoints(OBJ, rv, tv, K, d)
    return float(np.sqrt(((c - p.reshape(-1, 2)) ** 2).sum(1).mean()))


def exercise1(size, corners_left):
    print("Exercise 1: calibrate the right camera of the stereo pair")
    names_r, _, corners_r = detect("right")
    rms_l, K_l, d_l, s_l = calibrate(corners_left, size)
    rms_r, K_r, d_r, s_r = calibrate(corners_r, size)
    print(f"  views: left {len(corners_left)}, right {len(corners_r)} | RMS left {rms_l:.3f} px, "
          f"right {rms_r:.3f} px")  # fmt: skip
    print(f"  {'':4} {'left':>16} {'right':>16} {'diff / combined sigma':>22}")
    left = [K_l[0, 0], K_l[1, 1], K_l[0, 2], K_l[1, 2]]
    right = [K_r[0, 0], K_r[1, 1], K_r[0, 2], K_r[1, 2]]
    for i, (name, a, b) in enumerate(zip(["fx", "fy", "cx", "cy"], left, right)):
        sig = np.hypot(s_l[i], s_r[i])
        print(f"  {name:4} {a:9.2f} +- {s_l[i]:.2f} {b:9.2f} +- {s_r[i]:.2f} {(b - a) / sig:22.1f}")
    print(f"  k1: left {d_l[0]:.4f}, right {d_r[0]:.4f}")
    note(
        "Same camera model, but focal lengths and especially principal points differ by many "
        "sigma: two physical cameras are never identical, so a stereo rig needs both "
        "calibrations plus the extrinsics between them."
    )


def exercise2(size, names, corners):
    print(
        "\nExercise 2: distortion-model selection with leave-one-out hold-out error (left camera)"
    )
    models = {
        "k1 k2 p1 p2 k3 (5 params)": 0,
        "k1 k2 p1 p2 (FIX_K3)": cv2.CALIB_FIX_K3,
        "k1 k2 (FIX_K3 + ZERO_TANGENT)": cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST,
        "rational (8 params)": cv2.CALIB_RATIONAL_MODEL,
    }
    for label, flags in models.items():
        rms, *_ = calibrate(corners, size, flags)
        errs = []
        for i in range(len(corners)):
            rest = [c for j, c in enumerate(corners) if j != i]
            _, K, d, _ = calibrate(rest, size, flags)
            errs.append(holdout(K, d, corners[i]))
        print(f"  {label:32s} train RMS {rms:.4f} px | LOO mean {np.mean(errs):.4f} px, "
              f"max {np.max(errs):.4f} px")  # fmt: skip
    note(
        "Extra parameters lower the training error; the hold-out error shows whether they "
        "help. Here k3 is not needed (FIX_K3 generalises as well), the rational model "
        "overfits slightly and dropping the tangential terms underfits."
    )


def exercise3(size, names, imgs, corners):
    print("\nExercise 3: SB corners vs classic sub-pixel corners on the same views")
    sb, classic, used = [], [], []
    for n, g, c in zip(names, imgs, corners):
        ok, s = cv2.findChessboardCornersSB(
            g, PATTERN, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        )
        if not ok:
            continue
        if np.linalg.norm(s[::-1] - c, axis=1).mean() < np.linalg.norm(s - c, axis=1).mean():
            s = s[::-1].copy()  # align the enumeration order with the classic detector
        sb.append(s)
        classic.append(c)
        used.append(n)
    print(f"  SB found the board in {len(sb)}/{len(names)} views; comparing on these views")
    for label, cl in [("classic + cornerSubPix(7)", classic), ("findChessboardCornersSB", sb)]:
        rms, K, d, std = calibrate(cl, size)
        print(f"  {label:26s} RMS {rms:.4f} px | fx {K[0, 0]:.2f} +- {std[0]:.2f} | "
              f"cx {K[0, 2]:.2f} +- {std[2]:.2f} | k1 {d[0]:.4f}")  # fmt: skip
    diff = np.mean([np.linalg.norm(a - b, axis=1).mean() for a, b in zip(sb, classic)])
    print(f"  mean distance between the two corner sets: {diff:.3f} px")


if __name__ == "__main__":
    names, imgs, corners = detect("left")
    size = imgs[0].shape[::-1]
    exercise1(size, corners)
    exercise2(size, names, corners)
    exercise3(size, names, imgs, corners)
