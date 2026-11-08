#!/usr/bin/env python
"""Calibrate a camera from checkerboard photos and write a ROS ``camera_info`` YAML.

Pipeline (see notebook ``03_camera_calibration.ipynb`` for the theory and diagnostics):
``findChessboardCorners`` -> ``cornerSubPix`` -> ``calibrateCameraExtended`` -> optional
leave-one-out hold-out error per view (flags views that the other views cannot predict).

Usage (from the repository root)::

    python projects/01_classical_cv/scripts/calibrate_camera.py \
        --images "data/01_classical_cv/calib/left*.jpg" \
        --config projects/01_classical_cv/configs/checkerboard_9x6.yaml \
        --camera-name left --loo --out projects/01_classical_cv/results/left_camera_info.yaml

The YAML follows the ``camera_calibration_parsers`` layout used by ROS camera drivers
(``plumb_bob`` = OpenCV's k1, k2, p1, p2, k3 model); ``P`` is the ``alpha = 0`` optimal new
camera matrix (every undistorted pixel valid).
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import cv2
import numpy as np
import yaml

DEFAULTS = {
    "pattern": [9, 6],
    "square_size": 1.0,
    "subpix_half_window": 7,
    "subpix_max_iter": 30,
    "subpix_eps": 1e-3,
    "loo_flag_factor": 2.0,
}


def board_points(pattern: tuple[int, int], square: float = 1.0) -> np.ndarray:
    """``(cols*rows, 3)`` float32 inner-corner coordinates (z = 0, x varies fastest)."""
    cols, rows = pattern
    obj = np.zeros((cols * rows, 3), np.float32)
    obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
    return obj


def detect_corners(
    gray: np.ndarray,
    pattern: tuple[int, int],
    half_window: int = 7,
    max_iter: int = 30,
    eps: float = 1e-3,
) -> np.ndarray | None:
    """Classic chessboard detection + sub-pixel refinement: ``(N, 2)`` float32 or ``None``."""
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, tuple(pattern), flags=flags)
    if not ok:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, eps)
    win = (half_window, half_window)
    refined = cv2.cornerSubPix(gray, corners.copy(), win, (-1, -1), criteria)
    return np.asarray(refined, np.float32).reshape(-1, 2)  # OpenCV 5 already returns (N, 2)


def holdout_rms(obj: np.ndarray, corners: np.ndarray, K: np.ndarray, dist: np.ndarray) -> float:
    """Reprojection RMS of one view whose pose is fitted with ``solvePnP`` (intrinsics fixed)."""
    _, rvec, tvec = cv2.solvePnP(obj, corners, K, dist)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    return float(np.sqrt(((corners - proj.reshape(-1, 2)) ** 2).sum(1).mean()))


def calibrate_images(
    images: list[np.ndarray],
    pattern: tuple[int, int] = (9, 6),
    square: float = 1.0,
    half_window: int = 7,
    max_iter: int = 30,
    eps: float = 1e-3,
    loo: bool = False,
) -> dict:
    """Calibrate from greyscale images.

    Returns K, dist, RMS, std devs of the intrinsics, per-view errors and optionally LOO errors.
    """
    if not images:
        raise ValueError("no images given")
    h, w = images[0].shape[:2]
    obj = board_points(pattern, square)
    corners, used = [], []
    for i, img in enumerate(images):
        if img.shape[:2] != (h, w):
            raise ValueError(f"image {i} has size {img.shape[:2]}, expected {(h, w)}")
        c = detect_corners(img, pattern, half_window, max_iter, eps)
        if c is not None:
            corners.append(c)
            used.append(i)
    if len(corners) < 3:
        raise RuntimeError(f"board found in only {len(corners)} images; need at least 3")
    rms, K, dist, rvecs, tvecs, std_int, _, per_view = cv2.calibrateCameraExtended(
        [obj] * len(corners), corners, (w, h), None, None
    )
    out = {
        "image_size": (w, h),
        "rms": float(rms),
        "K": K,
        "dist": dist.ravel(),
        "std_intrinsics": std_int.ravel()[:9],
        "per_view_rms": per_view.ravel(),
        "used": used,
        "skipped": [i for i in range(len(images)) if i not in used],
        "rvecs": rvecs,
        "tvecs": tvecs,
    }
    if loo:
        errs = []
        for k in range(len(corners)):
            rest = [c for j, c in enumerate(corners) if j != k]
            _, Kk, dk, _, _ = cv2.calibrateCamera([obj] * len(rest), rest, (w, h), None, None)
            errs.append(holdout_rms(obj, corners[k], Kk, dk))
        out["loo_rms"] = np.array(errs)
    return out


def camera_info(result: dict, camera_name: str = "camera") -> dict:
    """ROS ``camera_info`` dict (plumb_bob distortion, P = alpha-0 optimal new camera matrix)."""
    w, h = result["image_size"]
    K, dist = result["K"], result["dist"][:5]
    P3, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0)
    P = np.hstack([P3, np.zeros((3, 1))])

    def mat(a: np.ndarray, rows: int, cols: int) -> dict:
        return {"rows": rows, "cols": cols, "data": [float(x) for x in np.asarray(a).ravel()]}

    return {
        "image_width": int(w),
        "image_height": int(h),
        "camera_name": camera_name,
        "camera_matrix": mat(K, 3, 3),
        "distortion_model": "plumb_bob",
        "distortion_coefficients": mat(dist, 1, 5),
        "rectification_matrix": mat(np.eye(3), 3, 3),
        "projection_matrix": mat(P, 3, 4),
    }


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return path.name


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--images", required=True, help='glob pattern, e.g. "data/calib/*.jpg" (quote it)'
    )
    ap.add_argument(
        "--config", type=Path, help="YAML with pattern / square_size / subpix_* settings"
    )
    ap.add_argument("--camera-name", default="camera")
    ap.add_argument(
        "--loo", action="store_true", help="also compute leave-one-out hold-out error per view"
    )
    ap.add_argument("--out", type=Path, help="write a ROS camera_info YAML here")
    args = ap.parse_args(argv)

    cfg = dict(DEFAULTS)
    if args.config is not None:
        cfg.update(yaml.safe_load(args.config.read_text()) or {})
    paths = sorted(Path(p) for p in glob.glob(args.images))
    if not paths:
        raise SystemExit(f"no images match {args.images!r}")
    images = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in paths]
    res = calibrate_images(
        images,
        tuple(cfg["pattern"]),
        float(cfg["square_size"]),
        int(cfg["subpix_half_window"]),
        int(cfg["subpix_max_iter"]),
        float(cfg["subpix_eps"]),
        loo=args.loo,
    )
    names = [p.stem for p in paths]
    w, h = res["image_size"]
    cols, rows = cfg["pattern"]
    print(
        f"{len(res['used'])}/{len(paths)} views with a detected {cols}x{rows} board, image {w}x{h}"
    )
    if res["skipped"]:
        print("skipped (board not found):", ", ".join(names[i] for i in res["skipped"]))
    K, d, s = res["K"], res["dist"], res["std_intrinsics"]
    print(f"RMS reprojection error: {res['rms']:.4f} px")
    for label, val, std in zip(["fx", "fy", "cx", "cy"], [K[0, 0], K[1, 1], K[0, 2], K[1, 2]], s):
        print(f"  {label} = {val:9.3f} +- {std:.3f} px")
    print("  dist (k1 k2 p1 p2 k3) =", np.array2string(d[:5], precision=5, suppress_small=True))
    header = f"{'view':>10} {'train RMS':>10}" + (f" {'LOO RMS':>9}" if args.loo else "")
    print(header)
    loo = res.get("loo_rms")
    flag = float(cfg["loo_flag_factor"]) * np.median(loo) if loo is not None else None
    for k, i in enumerate(res["used"]):
        line = f"{names[i]:>10} {res['per_view_rms'][k]:10.3f}"
        if loo is not None:
            note = "  <- inconsistent with the other views" if loo[k] > flag else ""
            line += f" {loo[k]:9.3f}{note}"
        print(line)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(yaml.safe_dump(camera_info(res, args.camera_name), sort_keys=False))
        print("wrote", _rel(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
