"""Helpers for project 09 (multi-view geometry).

Everything here is plain NumPy / OpenCV / pycolmap and is imported by the notebooks,
the scripts and the unit tests (``tests/test_p09_mvgeo.py``):

* two-view geometry: Hartley-normalised 8-point, Sampson distance, E decomposition and
  the cheirality check, relative-pose errors;
* stereo: Middlebury PFM/calib readers, StereoBM/SGBM wrappers, left-right check,
  hole filling, disparity metrics (bad-x, EPE) and disparity -> depth;
* features: SIFT + ratio test, DISK + LightGlue (kornia), RANSAC on matches;
* SfM: COLMAP (pycolmap 4.x) pipelines with SIFT or imported DISK+LightGlue matches,
  reconstruction statistics and pose comparison against a reference model.

Conventions follow ``cvlab.geometry``: OpenCV camera frame, ``x2 = R x1 + t`` for the
relative pose of camera 2 w.r.t. camera 1, pixel centres at integer coordinates.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from cvlab.geometry import angle_between_deg, rotation_error_deg, skew

PROJECT = "09_multiview_geometry"


# =============================================================== two-view geometry
def to_h(x: np.ndarray) -> np.ndarray:
    """Nx2 -> Nx3 homogeneous coordinates."""
    x = np.asarray(x, dtype=np.float64).reshape(-1, 2)
    return np.hstack([x, np.ones((x.shape[0], 1))])


def normalize_points(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hartley normalisation: centroid to the origin, mean distance sqrt(2).

    Returns ``(x_normalised Nx2, T 3x3)`` with ``x_n = T @ x`` in homogeneous form.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1, 2)
    c = x.mean(axis=0)
    d = np.sqrt(((x - c) ** 2).sum(axis=1)).mean()
    s = np.sqrt(2.0) / max(d, 1e-12)
    T = np.array([[s, 0.0, -s * c[0]], [0.0, s, -s * c[1]], [0.0, 0.0, 1.0]])
    return (to_h(x) @ T.T)[:, :2], T


def eight_point(x1: np.ndarray, x2: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Fundamental matrix from >= 8 correspondences (linear 8-point algorithm).

    Solves ``x2^T F x1 = 0`` in the least-squares sense with an SVD, then enforces
    rank 2 by zeroing the smallest singular value. With ``normalize=True`` this is
    Hartley's normalised 8-point algorithm (much better conditioned).
    """
    x1 = np.asarray(x1, dtype=np.float64).reshape(-1, 2)
    x2 = np.asarray(x2, dtype=np.float64).reshape(-1, 2)
    if x1.shape[0] < 8:
        raise ValueError("the 8-point algorithm needs at least 8 correspondences")
    T1 = T2 = np.eye(3)
    if normalize:
        x1, T1 = normalize_points(x1)
        x2, T2 = normalize_points(x2)
    u1, v1 = x1[:, 0], x1[:, 1]
    u2, v2 = x2[:, 0], x2[:, 1]
    A = np.stack([u2 * u1, u2 * v1, u2, v2 * u1, v2 * v1, v2, u1, v1, np.ones_like(u1)], 1)
    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)
    U, S, Vt = np.linalg.svd(F)
    F = U @ np.diag([S[0], S[1], 0.0]) @ Vt  # closest rank-2 matrix (Frobenius norm)
    F = T2.T @ F @ T1  # undo the normalisation
    F /= np.linalg.norm(F)
    return F if F[2, 2] >= 0 else -F


def algebraic_residual(F: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """``x2^T F x1`` per correspondence (zero for perfect data)."""
    return np.einsum("ni,ij,nj->n", to_h(x2), F, to_h(x1))


def sampson_distance(F: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """First-order geometric (Sampson) error per correspondence, in pixels^2."""
    X1, X2 = to_h(x1), to_h(x2)
    Fx1 = X1 @ F.T
    Ftx2 = X2 @ F
    num = np.einsum("ni,ni->n", X2, Fx1) ** 2
    den = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    return num / np.maximum(den, 1e-18)


def point_line_distance(lines: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Distance (pixels) from points ``x`` (Nx2) to lines ``a u + b v + c = 0`` (Nx3)."""
    lines = np.asarray(lines, dtype=np.float64).reshape(-1, 3)
    return np.abs(np.einsum("ni,ni->n", lines, to_h(x))) / np.linalg.norm(lines[:, :2], axis=1)


def essential_from_fundamental(F: np.ndarray, K1: np.ndarray, K2: np.ndarray) -> np.ndarray:
    """``E = K2^T F K1`` projected onto the essential manifold (singular values 1, 1, 0)."""
    E = K2.T @ F @ K1
    U, _, Vt = np.linalg.svd(E)
    return U @ np.diag([1.0, 1.0, 0.0]) @ Vt


def decompose_essential(E: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """The four ``(R, t)`` candidates of an essential matrix (``||t|| = 1``).

    ``E = U diag(1,1,0) V^T`` -> ``R in {U W V^T, U W^T V^T}``, ``t = +-u_3``.
    """
    U, _, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
        U = -U
    if np.linalg.det(Vt) < 0:
        Vt = -Vt
    W = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    R1, R2 = U @ W @ Vt, U @ W.T @ Vt
    t = U[:, 2]
    return [(R1, t), (R1, -t), (R2, t), (R2, -t)]


def projection_matrix(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """``P = K [R | t]`` (3x4)."""
    return K @ np.hstack([R, np.asarray(t, dtype=np.float64).reshape(3, 1)])


def cheirality_count(
    R: np.ndarray, t: np.ndarray, x1n: np.ndarray, x2n: np.ndarray
) -> tuple[int, np.ndarray]:
    """Number of normalised correspondences triangulated in front of both cameras.

    ``x1n, x2n`` are normalised image coordinates (``K^-1 x``). Returns
    ``(count, X Nx3 in camera-1 frame)``.
    """
    from cvlab.geometry import triangulate_dlt

    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = np.hstack([R, np.asarray(t).reshape(3, 1)])
    X = triangulate_dlt(P1, P2, x1n, x2n)
    z1 = X[:, 2]
    z2 = (X @ R.T + np.asarray(t).reshape(1, 3))[:, 2]
    return int(np.sum((z1 > 0) & (z2 > 0))), X


def relative_pose_from_c2w(T_wc1: np.ndarray, T_wc2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Relative pose ``x2 = R x1 + t`` between two camera-to-world poses."""
    T_21 = np.linalg.inv(T_wc2) @ T_wc1
    return T_21[:3, :3], T_21[:3, 3]


def pose_errors(
    R_est: np.ndarray, t_est: np.ndarray, R_gt: np.ndarray, t_gt: np.ndarray
) -> tuple[float, float]:
    """Rotation error and translation-direction error (degrees; t is only known up to scale)."""
    return rotation_error_deg(R_est, R_gt), angle_between_deg(t_est, t_gt)


def pose_auc(errors_deg: np.ndarray, thresholds: tuple[float, ...] = (5.0, 10.0, 20.0)) -> dict:
    """Area under the cumulative pose-error curve up to each threshold (as in SuperGlue/LoFTR).

    ``errors_deg`` is typically ``max(rotation error, translation-angle error)`` per pair;
    failures should be passed as ``np.inf``.
    """
    errors = np.sort(np.asarray(errors_deg, dtype=np.float64))
    recall = (np.arange(len(errors)) + 1) / max(len(errors), 1)
    errors = np.concatenate([[0.0], errors])
    recall = np.concatenate([[0.0], recall])
    out = {}
    for thr in thresholds:
        i = np.searchsorted(errors, thr)
        e = np.concatenate([errors[:i], [thr]])
        r = np.concatenate([recall[:i], [recall[i - 1]]])
        out[f"auc@{thr:g}"] = float(np.trapezoid(r, x=e) / thr)
    return out


def triangulation_angle_deg(X: np.ndarray, C1: np.ndarray, C2: np.ndarray) -> np.ndarray:
    """Angle (degrees) between the two viewing rays of each 3D point."""
    r1 = np.asarray(X) - np.asarray(C1).reshape(1, 3)
    r2 = np.asarray(X) - np.asarray(C2).reshape(1, 3)
    cos = np.einsum("ni,ni->n", r1, r2) / (
        np.linalg.norm(r1, axis=1) * np.linalg.norm(r2, axis=1) + 1e-12
    )
    return np.rad2deg(np.arccos(np.clip(cos, -1.0, 1.0)))


def essential_check(E: np.ndarray) -> dict:
    """Singular values and the cubic constraint ``2 E E^T E - tr(E E^T) E = 0``."""
    s = np.linalg.svd(E, compute_uv=False)
    cubic = 2 * E @ E.T @ E - np.trace(E @ E.T) * E
    return {"singular_values": s / s[0], "cubic_residual": float(np.linalg.norm(cubic))}


# ======================================================================= stereo
def read_pfm(path: str | Path) -> np.ndarray:
    """Read a PFM image (Middlebury disparities) as float32, top row first.

    PFM stores rows bottom-to-top; a negative scale means little-endian data.
    Middlebury marks unknown disparities with ``inf``.
    """
    with open(path, "rb") as f:
        header = f.readline().decode("ascii").strip()
        if header not in ("PF", "Pf"):
            raise ValueError(f"not a PFM file: {path}")
        channels = 3 if header == "PF" else 1
        dims = f.readline().decode("ascii").split()
        while len(dims) < 2:  # some writers split width/height over two lines
            dims += f.readline().decode("ascii").split()
        width, height = int(dims[0]), int(dims[1])
        scale = float(f.readline().decode("ascii").strip())
        dtype = "<f4" if scale < 0 else ">f4"
        data = np.frombuffer(f.read(), dtype=dtype, count=width * height * channels)
    shape = (height, width, channels) if channels == 3 else (height, width)
    return np.flipud(data.reshape(shape)).astype(np.float32)


def write_pfm(path: str | Path, image: np.ndarray, scale: float = 1.0) -> Path:
    """Write a float32 single-channel PFM (little-endian), used by the tests."""
    img = np.asarray(image, dtype="<f4")
    path = Path(path)
    with open(path, "wb") as f:
        f.write(b"Pf\n")
        f.write(f"{img.shape[1]} {img.shape[0]}\n".encode())
        f.write(f"{-abs(scale)}\n".encode())
        f.write(np.flipud(img).tobytes())
    return path


def read_middlebury_calib(path: str | Path) -> dict:
    """Parse a Middlebury ``calib.txt`` (cam0/cam1 matrices, doffs, baseline, ndisp, ...)."""
    out: dict = {}
    for line in Path(path).read_text().splitlines():
        if "=" not in line:
            continue
        key, val = (s.strip() for s in line.split("=", 1))
        if val.startswith("["):
            rows = [r.split() for r in val.strip("[]").split(";")]
            out[key] = np.array(rows, dtype=np.float64)
        else:
            out[key] = float(val) if re.search(r"[.eE]", val) else int(val)
    return out


@dataclass
class StereoPair:
    """One rectified Middlebury pair with ground truth (left view)."""

    name: str
    left: np.ndarray  # HxWx3 RGB uint8
    right: np.ndarray
    calib: dict
    gt: np.ndarray | None = None  # HxW float32 disparity, NaN where unknown
    nocc: np.ndarray | None = None  # HxW bool, non-occluded pixels with GT
    extra: dict = field(default_factory=dict)

    @property
    def f(self) -> float:
        return float(self.calib["cam0"][0, 0])

    @property
    def ndisp(self) -> int:
        return int(self.calib["ndisp"])


def middlebury_root() -> Path:
    from cvlab.io import data_dir

    return data_dir("middlebury", "MiddEval3")


def list_middlebury_scenes(split: str = "trainingQ") -> list[str]:
    root = middlebury_root() / split
    return sorted(p.name for p in root.iterdir() if (p / "im0.png").exists())


def load_middlebury(scene: str, split: str = "trainingQ") -> StereoPair:
    """Load ``data/middlebury/MiddEval3/<split>/<scene>`` (images, calib, GT if present)."""
    from cvlab.io import read_image

    d = middlebury_root() / split / scene
    pair = StereoPair(
        name=scene,
        left=read_image(d / "im0.png"),
        right=read_image(d / "im1.png"),
        calib=read_middlebury_calib(d / "calib.txt"),
    )
    if (d / "disp0GT.pfm").exists():
        gt = read_pfm(d / "disp0GT.pfm")
        gt[~np.isfinite(gt)] = np.nan
        pair.gt = gt
        mask = cv2.imread(str(d / "mask0nocc.png"), cv2.IMREAD_GRAYSCALE)
        pair.nocc = (mask == 255) & np.isfinite(gt)
    return pair


def num_disparities(ndisp: int) -> int:
    """SGBM needs ``numDisparities`` to be a positive multiple of 16."""
    return int(np.ceil(ndisp / 16.0) * 16)


def stereo_bm(left: np.ndarray, right: np.ndarray, ndisp: int, block: int = 15) -> np.ndarray:
    """Classic block matching (SAD window); returns float32 disparity, NaN = invalid."""
    gl = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
    gr = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)
    bm = cv2.StereoBM_create(numDisparities=num_disparities(ndisp), blockSize=block)
    d = bm.compute(gl, gr).astype(np.float32) / 16.0  # fixed point with 4 fractional bits
    d[d < 0] = np.nan
    return d


def stereo_sgbm(
    left: np.ndarray,
    right: np.ndarray,
    ndisp: int,
    block: int = 5,
    mode: int = cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    uniqueness: int = 10,
    speckle_window: int = 100,
    speckle_range: int = 2,
    disp12_max_diff: int = 1,
    min_disp: int = 0,
    p1: int | None = None,
    p2: int | None = None,
) -> np.ndarray:
    """Semi-global matching (OpenCV) on RGB images; returns float32 disparity, NaN = invalid.

    ``P1``/``P2`` default to the usual smoothness penalties ``8*3*b^2`` and ``32*3*b^2``
    (small vs large disparity changes between neighbouring pixels); pass ``p1``/``p2``
    to override them. The search covers ``[min_disp, min_disp + num_disparities(ndisp))``.
    Pass ``uniqueness=0, speckle_window=0, disp12_max_diff=-1`` for "raw" SGM.
    """
    cn = 3
    sgbm = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disparities(ndisp),
        blockSize=block,
        P1=8 * cn * block**2 if p1 is None else p1,
        P2=32 * cn * block**2 if p2 is None else p2,
        disp12MaxDiff=disp12_max_diff,
        preFilterCap=63,
        uniquenessRatio=uniqueness,
        speckleWindowSize=speckle_window,
        speckleRange=speckle_range,
        mode=mode,
    )
    bgr_l = cv2.cvtColor(left, cv2.COLOR_RGB2BGR)
    bgr_r = cv2.cvtColor(right, cv2.COLOR_RGB2BGR)
    raw = sgbm.compute(bgr_l, bgr_r)
    d = raw.astype(np.float32) / 16.0  # fixed point with 4 fractional bits
    d[raw < min_disp * 16] = np.nan  # invalid pixels are set to (min_disp - 1) * 16
    return d


def disparity_search_range(x1: np.ndarray, x2: np.ndarray, margin: int = 8) -> tuple[int, int]:
    """``(min_disp, ndisp)`` covering the horizontal disparities ``u_l - u_r`` of matches."""
    dx = np.asarray(x1)[:, 0] - np.asarray(x2)[:, 0]
    lo = int(np.floor(np.percentile(dx, 0.5))) - margin
    hi = int(np.ceil(np.percentile(dx, 99.5))) + margin
    return lo, num_disparities(hi - lo)


def right_disparity(fn, left: np.ndarray, right: np.ndarray, *args, **kwargs) -> np.ndarray:
    """Disparity of the *right* view with a left-view matcher: flip both images horizontally."""
    d = fn(right[:, ::-1].copy(), left[:, ::-1].copy(), *args, **kwargs)
    return d[:, ::-1].copy()


def lr_consistency(d_left: np.ndarray, d_right: np.ndarray, max_diff: float = 1.0) -> np.ndarray:
    """Left-right check: keep pixel x if ``|d_L(x) - d_R(x - d_L(x))| <= max_diff``."""
    h, w = d_left.shape
    xs = np.arange(w)[None, :].repeat(h, 0)
    xr = np.round(xs - np.nan_to_num(d_left, nan=-1e6)).astype(np.int64)
    ok = np.isfinite(d_left) & (xr >= 0) & (xr < w)
    rows = np.arange(h)[:, None].repeat(w, 1)
    d_r = np.full_like(d_left, np.nan)
    d_r[ok] = d_right[rows[ok], xr[ok]]
    return ok & np.isfinite(d_r) & (np.abs(d_left - d_r) <= max_diff)


def fill_holes(disp: np.ndarray) -> np.ndarray:
    """Fill invalid pixels per scanline with the *smaller* of the nearest valid left/right
    disparities (holes are mostly occlusions, which belong to the background)."""
    out = disp.copy()
    h, w = disp.shape
    idx = np.arange(w)
    for r in range(h):
        row = disp[r]
        valid = np.isfinite(row)
        if not valid.any():
            continue
        vi = idx[valid]
        # nearest valid index to the left / right of each pixel
        left_i = np.maximum.accumulate(np.where(valid, idx, -1))
        right_i = np.flip(np.minimum.accumulate(np.flip(np.where(valid, idx, w))))
        lv = np.where(left_i >= 0, row[np.clip(left_i, 0, w - 1)], np.inf)
        rv = np.where(right_i < w, row[np.clip(right_i, 0, w - 1)], np.inf)
        fill = np.minimum(lv, rv)
        out[r, ~valid] = fill[~valid]
        if not np.isfinite(out[r]).all():  # only possible if vi is empty (handled above)
            out[r, ~np.isfinite(out[r])] = row[vi].min()
    return out


def disparity_metrics(
    pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, thresholds: tuple[float, ...] = (1.0, 2.0)
) -> dict:
    """Middlebury-style disparity metrics on ``mask`` pixels (e.g. non-occluded).

    * ``density`` - fraction of mask pixels with a valid prediction;
    * ``bad-x`` - % of *valid predicted* mask pixels with ``|d - d_gt| > x``;
    * ``bad-x (all)`` - same, but pixels without a prediction count as bad;
    * ``epe`` - mean absolute disparity error on valid predicted pixels;
    * ``rmse`` - root mean squared error on valid predicted pixels.
    """
    m = mask & np.isfinite(gt)
    valid = m & np.isfinite(pred)
    err = np.abs(pred[valid] - gt[valid])
    out = {"density": float(valid.sum() / max(m.sum(), 1))}
    for t in thresholds:
        out[f"bad-{t:g}"] = float(100.0 * np.mean(err > t)) if err.size else float("nan")
        n_bad = np.sum(err > t) + (m.sum() - valid.sum())
        out[f"bad-{t:g} (all)"] = float(100.0 * n_bad / max(m.sum(), 1))
    out["epe"] = float(err.mean()) if err.size else float("nan")
    out["rmse"] = float(np.sqrt(np.mean(err**2))) if err.size else float("nan")
    return out


def disparity_to_depth(disp: np.ndarray, f: float, baseline: float, doffs: float = 0.0):
    """``Z = f * B / (d + doffs)`` (same unit as the baseline; NaN where ``d + doffs <= 0``)."""
    d = np.asarray(disp, dtype=np.float64) + doffs
    with np.errstate(divide="ignore", invalid="ignore"):
        z = f * baseline / d
    z[~(d > 0)] = np.nan
    return z


def vertical_disparity(left: np.ndarray, right: np.ndarray, n_features: int = 4000) -> dict:
    """Row misalignment of a stereo pair: ``dy = y_left - y_right`` of verified SIFT matches.

    Matches are verified with a generic fundamental-matrix RANSAC (which does *not* assume
    the pair is rectified). In a perfectly rectified pair ``dy == 0`` for every match.
    """
    m = sift_match(left, right, n_features=n_features)
    F, mask = cv2.findFundamentalMat(m.x1, m.x2, cv2.USAC_MAGSAC, 1.0, 0.9999)
    inl = mask.ravel().astype(bool) if mask is not None else np.ones(len(m), bool)
    dy = m.x1[inl, 1] - m.x2[inl, 1]
    return {
        "n": int(inl.sum()),
        "median_abs_dy": float(np.median(np.abs(dy))),
        "p90_abs_dy": float(np.percentile(np.abs(dy), 90)),
        "dy": dy,
        "x1": m.x1[inl],
        "x2": m.x2[inl],
        "F": F,
    }


def rectify_calibrated(
    left: np.ndarray,
    right: np.ndarray,
    K0: np.ndarray,
    K1: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
    alpha: float = 0.0,
) -> dict:
    """Bouguet rectification (``cv2.stereoRectify``) from known intrinsics and relative pose.

    ``R, T`` map left-camera coordinates to right-camera coordinates (``X_r = R X_l + T``).
    Both cameras are rotated (``R1``, ``R2``) so that their image planes become coplanar and
    rows align; ``P1, P2`` are the new projection matrices and ``Q`` maps ``(u, v, d, 1)``
    to homogeneous 3D points in the rectified left frame.
    """
    h, w = left.shape[:2]
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K0, None, K1, None, (w, h), R, np.asarray(T, dtype=np.float64).reshape(3, 1),
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=alpha,
    )  # fmt: skip
    m1 = cv2.initUndistortRectifyMap(K0, None, R1, P1, (w, h), cv2.CV_32FC1)
    m2 = cv2.initUndistortRectifyMap(K1, None, R2, P2, (w, h), cv2.CV_32FC1)
    return {
        "left": cv2.remap(left, *m1, cv2.INTER_LINEAR),
        "right": cv2.remap(right, *m2, cv2.INTER_LINEAR),
        "R1": R1, "R2": R2, "P1": P1, "P2": P2, "Q": Q,
    }  # fmt: skip


def rectify_uncalibrated(
    left: np.ndarray, right: np.ndarray, x1: np.ndarray, x2: np.ndarray, F: np.ndarray
) -> dict:
    """Hartley rectification from ``F`` and matches only (``cv2.stereoRectifyUncalibrated``).

    Returns the two warping homographies and the warped images; no metric depth can be
    recovered because the intrinsics are unknown (projective ambiguity).
    """
    h, w = left.shape[:2]
    ok, H1, H2 = cv2.stereoRectifyUncalibrated(x1.reshape(-1, 2), x2.reshape(-1, 2), F, (w, h))
    if not ok:
        raise RuntimeError("stereoRectifyUncalibrated failed")
    return {
        "left": cv2.warpPerspective(left, H1, (w, h)),
        "right": cv2.warpPerspective(right, H2, (w, h)),
        "H1": H1,
        "H2": H2,
    }


def rectified_disparity_to_left_depth(
    disp_rect: np.ndarray, Q: np.ndarray, R1: np.ndarray, K0: np.ndarray, shape: tuple[int, int]
) -> np.ndarray:
    """Depth map in the *original* left camera from a disparity map of the rectified pair.

    ``X_rect ~ Q [u, v, d, 1]^T`` (rectified-left frame), ``X_left = R1^T X_rect``, then a
    z-buffered projection with ``K0``. Returns HxW depth (NaN where empty).
    """
    from cvlab.pointcloud import points_to_depth

    valid = np.isfinite(disp_rect) & (disp_rect > 0)
    v, u = np.nonzero(valid)
    hom = np.stack([u, v, disp_rect[valid], np.ones_like(u)], axis=1).astype(np.float64) @ Q.T
    X_rect = hom[:, :3] / hom[:, 3:4]
    X_left = X_rect @ R1  # row-vector form of R1^T @ X
    depth = points_to_depth(X_left, K0, shape[1], shape[0])
    depth[depth <= 0] = np.nan
    return depth


def depth_metrics_simple(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> dict:
    """AbsRel and the fraction of pixels within 1 % / 5 % of the GT depth."""
    m = mask & np.isfinite(pred) & np.isfinite(gt) & (gt > 0)
    rel = np.abs(pred[m] - gt[m]) / gt[m]
    return {
        "coverage": float(m.sum() / max(mask.sum(), 1)),
        "abs_rel": float(rel.mean()) if rel.size else float("nan"),
        "within_1pct": float(np.mean(rel < 0.01)) if rel.size else float("nan"),
        "within_5pct": float(np.mean(rel < 0.05)) if rel.size else float("nan"),
    }


def orbit_pose(theta_deg: float, pivot: np.ndarray, axis: int = 1) -> np.ndarray:
    """World-to-camera transform of a virtual camera orbiting ``pivot`` by ``theta_deg``.

    The original camera is at the origin (``T_cw = I``); the new camera sees
    ``X' = R (X - pivot) + pivot`` with ``R`` a rotation about camera axis ``axis``.
    """
    from cvlab.geometry import so3_exp

    rotvec = np.zeros(3)
    rotvec[axis] = np.deg2rad(theta_deg)
    R = so3_exp(rotvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(pivot, dtype=np.float64) - R @ np.asarray(pivot, dtype=np.float64)
    return T


def render_point_cloud(
    points: np.ndarray,
    colors: np.ndarray,
    K: np.ndarray,
    T_cw: np.ndarray,
    width: int,
    height: int,
    splat: int = 2,
    background: int = 255,
) -> np.ndarray:
    """Z-buffered rendering of a coloured point cloud into a virtual pinhole camera.

    Every point is drawn as a ``splat x splat`` pixel square; the nearest point wins.
    ``colors`` are floats in [0, 1] or uint8. Returns an HxWx3 uint8 image.
    """
    from cvlab.geometry import project, transform_points

    X = transform_points(T_cw, points)
    keep = X[:, 2] > 1e-6
    X = X[keep]
    col = np.asarray(colors)[keep]
    col = (col * 255).round().astype(np.uint8) if col.dtype != np.uint8 else col
    uv, z = project(X, K)
    u0 = np.round(uv[:, 0]).astype(np.int64)
    v0 = np.round(uv[:, 1]).astype(np.int64)
    us, vs, zs, cs = [], [], [], []
    for du in range(splat):
        for dv in range(splat):
            us.append(u0 + du - splat // 2)
            vs.append(v0 + dv - splat // 2)
            zs.append(z)
            cs.append(col)
    u, v, z, col = map(np.concatenate, (us, vs, zs, cs))
    ok = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, z, col = u[ok], v[ok], z[ok], col[ok]
    zbuf = np.full((height, width), np.inf)
    np.minimum.at(zbuf, (v, u), z)
    win = z <= zbuf[v, u]
    img = np.full((height, width, 3), background, np.uint8)
    img[v[win], u[win]] = col[win]
    return img


def warp_rotation(img: np.ndarray, K: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Re-render an image as seen by a camera rotated by ``R`` about its centre.

    For a pure rotation the mapping is the homography ``H = K R K^-1`` (``x' ~ H x``).
    """
    H = K @ R @ np.linalg.inv(K)
    h, w = img.shape[:2]
    return cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR)


# ======================================================================= features
@dataclass
class Matches:
    """Putative correspondences between two images (pixel coordinates, Nx2 each)."""

    x1: np.ndarray
    x2: np.ndarray
    n_kpts1: int
    n_kpts2: int
    time_ms: float = 0.0
    extra: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.x1.shape[0])


def to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img


def sift_features(img: np.ndarray, n_features: int = 4096):
    """SIFT keypoints (Nx2, pixel centres at integers) and RootSIFT descriptors (NxD)."""
    sift = cv2.SIFT_create(nfeatures=n_features)
    kps, desc = sift.detectAndCompute(to_gray(img), None)
    xy = np.array([k.pt for k in kps], dtype=np.float64).reshape(-1, 2)
    if desc is None:
        return xy, np.zeros((0, 128), np.float32)
    desc = desc / (np.abs(desc).sum(axis=1, keepdims=True) + 1e-8)  # RootSIFT: L1 then sqrt
    return xy, np.sqrt(desc).astype(np.float32)


def match_ratio(desc1: np.ndarray, desc2: np.ndarray, ratio: float = 0.8, mutual: bool = True):
    """Nearest-neighbour matching with Lowe's ratio test (+ optional mutual check).

    Returns an Mx2 array of ``(index in 1, index in 2)``.
    """
    if len(desc1) < 2 or len(desc2) < 2:
        return np.zeros((0, 2), np.int64)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn = matcher.knnMatch(desc1, desc2, k=2)
    pairs = [(m.queryIdx, m.trainIdx) for m, n in knn if m.distance < ratio * n.distance]
    pairs = np.array(pairs, dtype=np.int64).reshape(-1, 2)
    if mutual and len(pairs):
        back = matcher.match(desc2, desc1)
        nn21 = np.full(len(desc2), -1)
        for m in back:
            nn21[m.queryIdx] = m.trainIdx
        pairs = pairs[nn21[pairs[:, 1]] == pairs[:, 0]]
    return pairs


def sift_match(img1: np.ndarray, img2: np.ndarray, n_features: int = 4096, ratio: float = 0.8):
    """SIFT + ratio test + mutual nearest neighbour (CPU)."""
    t0 = time.perf_counter()
    k1, d1 = sift_features(img1, n_features)
    k2, d2 = sift_features(img2, n_features)
    idx = match_ratio(d1, d2, ratio)
    ms = (time.perf_counter() - t0) * 1000.0
    return Matches(k1[idx[:, 0]], k2[idx[:, 1]], len(k1), len(k2), ms)


def set_torch_hub_dir() -> Path:
    """Cache kornia/torch.hub checkpoints under ``weights/torch_hub`` (gitignored)."""
    import torch

    from cvlab.io import weights_dir

    path = weights_dir("torch_hub", create=True)
    torch.hub.set_dir(str(path))
    return path


class DiskLightGlue:
    """DISK keypoints/descriptors + LightGlue matcher from kornia (both Apache-2.0/MIT).

    Weights: ``kornia/disk`` (depth checkpoint) and ``kornia/lightglue`` (disk_lightglue)
    on the Hugging Face Hub, cached in ``weights/torch_hub/checkpoints``.
    """

    def __init__(self, device=None, max_keypoints: int = 2048, fp16: bool = True) -> None:
        import kornia.feature as KF
        import torch

        set_torch_hub_dir()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.max_keypoints = max_keypoints
        self.fp16 = fp16 and self.device.type == "cuda"
        self.disk = KF.DISK.from_pretrained("depth", device=self.device).eval()
        self.lightglue = KF.LightGlue("disk", mp=self.fp16).to(self.device).eval()

    def features(self, img: np.ndarray) -> dict:
        """Extract DISK features from an RGB uint8 image (any size; padded to /16)."""
        import torch

        x = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0)[None].to(self.device)
        with torch.inference_mode(), torch.autocast(self.device.type, enabled=self.fp16):
            f = self.disk(x, n=self.max_keypoints, pad_if_not_divisible=True)[0]
        return {
            "keypoints": f.keypoints.float(),  # Nx2 (x, y) pixels, integer = pixel centre
            "descriptors": f.descriptors.float(),  # Nx128, L2-normalised
            "scores": f.detection_scores.float(),
            "image_size": torch.tensor([img.shape[1], img.shape[0]], device=self.device),
        }

    def match_features(self, f1: dict, f2: dict) -> tuple[np.ndarray, np.ndarray]:
        """LightGlue matches -> ``(Mx2 index pairs, M scores)``."""
        import torch

        data = {
            "image0": {
                "keypoints": f1["keypoints"][None],
                "descriptors": f1["descriptors"][None],
                "image_size": f1["image_size"][None],
            },
            "image1": {
                "keypoints": f2["keypoints"][None],
                "descriptors": f2["descriptors"][None],
                "image_size": f2["image_size"][None],
            },
        }
        with torch.inference_mode():
            out = self.lightglue(data)
        idx = out["matches"][0].cpu().numpy().astype(np.int64)
        scores = out["scores"][0].float().cpu().numpy()
        return idx, scores

    def match_features_mnn(self, f1: dict, f2: dict) -> tuple[np.ndarray, np.ndarray]:
        """Mutual nearest neighbours of the DISK descriptors (no learned matcher)."""
        import kornia.feature as KF
        import torch

        with torch.inference_mode():
            dists, idx = KF.match_mnn(f1["descriptors"], f2["descriptors"])
        return idx.cpu().numpy().astype(np.int64), dists.ravel().float().cpu().numpy()

    def match(self, img1: np.ndarray, img2: np.ndarray, matcher: str = "lightglue") -> Matches:
        """Extract DISK features from both images and match them (``lightglue`` or ``mnn``)."""
        import torch

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        f1, f2 = self.features(img1), self.features(img2)
        if matcher == "lightglue":
            idx, scores = self.match_features(f1, f2)
        else:
            idx, scores = self.match_features_mnn(f1, f2)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0
        k1 = f1["keypoints"].cpu().numpy().astype(np.float64)
        k2 = f2["keypoints"].cpu().numpy().astype(np.float64)
        return Matches(k1[idx[:, 0]], k2[idx[:, 1]], len(k1), len(k2), ms, {"scores": scores})


def ransac_essential(
    m: Matches,
    K1: np.ndarray,
    K2: np.ndarray | None = None,
    thresh_px: float = 1.0,
    dist1: np.ndarray | None = None,
    dist2: np.ndarray | None = None,
    method: int = cv2.USAC_MAGSAC,
) -> dict:
    """5-point essential matrix inside a robust estimator + cheirality-checked pose.

    Points are first undistorted and normalised with their own intrinsics (so two different
    cameras work); the threshold is given in pixels and converted with the mean focal
    length. ``method`` defaults to MAGSAC++ (local optimisation + final inlier refinement).
    """
    K2 = K1 if K2 is None else K2
    dist2 = dist1 if dist2 is None else dist2
    out = {"n_matches": len(m), "n_inliers": 0, "inlier_ratio": 0.0, "R": None, "t": None}
    if len(m) < 5:
        return out
    x1n = cv2.undistortPoints(m.x1.reshape(-1, 1, 2), K1, dist1).reshape(-1, 2)
    x2n = cv2.undistortPoints(m.x2.reshape(-1, 1, 2), K2, dist2).reshape(-1, 2)
    f = np.mean([K1[0, 0], K1[1, 1], K2[0, 0], K2[1, 1]])
    E, mask = cv2.findEssentialMat(
        x1n, x2n, np.eye(3), method=method, prob=0.9999, threshold=thresh_px / f
    )
    if E is None or E.shape[0] < 3 or mask is None:
        return out
    E = E[:3]  # several solutions may be stacked; the first is the best-scoring one
    n_ok, R, t, mask_pose = cv2.recoverPose(E, x1n, x2n, np.eye(3), mask=mask.copy())
    inl = mask.ravel().astype(bool)
    out.update(
        n_inliers=int(inl.sum()),
        inlier_ratio=float(inl.mean()),
        R=R,
        t=t.ravel(),
        inliers=inl,
        n_cheirality=int(n_ok),
    )
    return out


def draw_matches(
    ax,
    img1: np.ndarray,
    img2: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
    inliers: np.ndarray | None = None,
    max_lines: int = 150,
    seed: int = 0,
) -> None:
    """Side-by-side match plot: inliers in green, outliers in orange (random subset)."""
    h = max(img1.shape[0], img2.shape[0])
    canvas = np.zeros((h, img1.shape[1] + img2.shape[1], 3), np.uint8)
    canvas[: img1.shape[0], : img1.shape[1]] = img1
    canvas[: img2.shape[0], img1.shape[1] :] = img2
    ax.imshow(canvas)
    ax.axis("off")
    n = len(x1)
    inl = np.ones(n, bool) if inliers is None else np.asarray(inliers, bool)
    sel = np.random.default_rng(seed).permutation(n)[:max_lines]
    off = img1.shape[1]
    for i in sel:
        c = (0.0, 0.62, 0.45) if inl[i] else (0.84, 0.37, 0.0)
        ax.plot([x1[i, 0], x2[i, 0] + off], [x1[i, 1], x2[i, 1]], color=c, lw=0.6, alpha=0.8)


# ======================================================================== SfM
@dataclass
class SfmStats:
    """Summary numbers of a COLMAP reconstruction."""

    num_images_input: int
    num_reg_images: int
    num_points3D: int  # noqa: N815 - COLMAP naming
    mean_track_length: float
    mean_observations_per_image: float
    mean_reprojection_error: float

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def reconstruction_stats(rec, num_images_input: int) -> SfmStats:
    """Statistics of a ``pycolmap.Reconstruction``."""
    return SfmStats(
        num_images_input=num_images_input,
        num_reg_images=int(rec.num_reg_images()),
        num_points3D=int(rec.num_points3D()),
        mean_track_length=float(rec.compute_mean_track_length()),
        mean_observations_per_image=float(rec.compute_mean_observations_per_reg_image()),
        mean_reprojection_error=float(rec.compute_mean_reprojection_error()),
    )


def colmap_K_to_opencv(K: np.ndarray) -> np.ndarray:
    """COLMAP puts the top-left pixel *centre* at (0.5, 0.5), OpenCV at (0, 0): shift c by -0.5."""
    K = np.array(K, dtype=np.float64)
    K[0, 2] -= 0.5
    K[1, 2] -= 0.5
    return K


def rec_cameras(rec) -> dict[str, dict]:
    """Per registered image: ``{name: {"T_wc", "K" (COLMAP), "K_cv" (OpenCV), ...}}``."""
    out = {}
    for img in rec.images.values():
        if not img.has_pose:
            continue
        T_cw = np.eye(4)
        T_cw[:3, :] = img.cam_from_world().matrix()  # world -> camera (3x4)
        cam = img.camera
        out[img.name] = {
            "T_wc": np.linalg.inv(T_cw),
            "K": cam.calibration_matrix(),
            "K_cv": colmap_K_to_opencv(cam.calibration_matrix()),
            "width": int(cam.width),
            "height": int(cam.height),
            "params": np.array(cam.params, dtype=np.float64),  # copy: pycolmap returns a view
            "model": str(cam.model.name),
        }
    return out


def rec_points(rec) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(xyz Nx3, rgb Nx3 uint8, reprojection error N, track length N)`` of all 3D points."""
    pts = list(rec.points3D.values())
    if not pts:
        z = np.zeros((0, 3))
        return z, z.astype(np.uint8), np.zeros(0), np.zeros(0, int)
    xyz = np.array([np.array(p.xyz, dtype=np.float64) for p in pts])
    rgb = np.array([p.color for p in pts], dtype=np.uint8)
    err = np.array([p.error for p in pts])
    tl = np.array([p.track.length() for p in pts])
    return xyz, rgb, err, tl


def compare_to_reference(est: dict[str, dict], ref: dict[str, dict]) -> dict:
    """Align estimated camera centres to a reference with Sim(3) (Umeyama) and compare.

    Returns per-image rotation errors (deg), centre errors (in reference units, after
    alignment) and the similarity transform. Images missing in either model are skipped.
    """
    from cvlab.geometry import umeyama_alignment

    names = sorted(set(est) & set(ref))
    C_est = np.array([est[n]["T_wc"][:3, 3] for n in names])
    C_ref = np.array([ref[n]["T_wc"][:3, 3] for n in names])
    s, R, t = umeyama_alignment(C_est, C_ref, with_scale=True)
    C_al = s * C_est @ R.T + t
    rot_err = []
    for n in names:
        R_est_w = R @ est[n]["T_wc"][:3, :3]  # camera orientation in the reference world
        rot_err.append(rotation_error_deg(R_est_w, ref[n]["T_wc"][:3, :3]))
    return {
        "names": names,
        "scale": s,
        "R": R,
        "t": t,
        "rot_err_deg": np.array(rot_err),
        "center_err": np.linalg.norm(C_al - C_ref, axis=1),
        "C_est_aligned": C_al,
        "C_ref": C_ref,
    }


def colmap_sift_sfm(
    image_dir: Path,
    image_names: list[str],
    work_dir: Path,
    max_num_features: int = 8192,
    camera_model: str = "SIMPLE_RADIAL",
    num_threads: int = -1,
) -> tuple[object | None, dict]:
    """COLMAP incremental SfM with SIFT features and exhaustive matching (CPU).

    Writes ``work_dir/database.db`` and ``work_dir/sparse/<k>/``; returns the largest
    reconstruction and per-stage timings in seconds.
    """
    import pycolmap

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    db = work_dir / "database.db"
    if db.exists():
        db.unlink()
    timings = {}
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = camera_model
    extract = pycolmap.FeatureExtractionOptions()
    extract.num_threads = num_threads
    extract.sift.max_num_features = max_num_features
    t0 = time.perf_counter()
    pycolmap.extract_features(
        db, image_dir, image_names=image_names, camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader, extraction_options=extract, device=pycolmap.Device.cpu,
    )  # fmt: skip
    timings["features_s"] = time.perf_counter() - t0
    matching = pycolmap.FeatureMatchingOptions()
    matching.num_threads = num_threads
    t0 = time.perf_counter()
    pycolmap.match_exhaustive(db, matching_options=matching, device=pycolmap.Device.cpu)
    timings["matching_s"] = time.perf_counter() - t0
    rec, t_map = _map(db, image_dir, work_dir, image_names, num_threads)
    timings["mapping_s"] = t_map
    return rec, timings


def _map(
    db: Path, image_dir: Path, work_dir: Path, image_names: list[str], num_threads: int,
    random_seed: int = 0,
):  # fmt: skip
    """Incremental mapping into ``work_dir/sparse``; returns (largest model or None, seconds).

    ``random_seed`` seeds COLMAP's RANSAC/mapper. Multi-threaded mapping is still not
    bit-reproducible: identical inputs can give (slightly or, near a failure mode, very)
    different models, so judge a configuration over several runs.
    """
    import pycolmap

    sparse = work_dir / "sparse"
    if sparse.exists():
        import shutil

        shutil.rmtree(sparse)
    sparse.mkdir(parents=True)
    opts = pycolmap.IncrementalPipelineOptions()
    opts.num_threads = num_threads
    opts.random_seed = random_seed
    opts.mapper.random_seed = random_seed
    t0 = time.perf_counter()
    recs = pycolmap.incremental_mapping(db, image_dir, sparse, options=opts)
    dt = time.perf_counter() - t0
    if not recs:
        return None, dt
    best = max(recs.values(), key=lambda r: r.num_reg_images())
    return best, dt


def colmap_import_matches_sfm(
    image_dir: Path,
    image_names: list[str],
    work_dir: Path,
    keypoints: dict[str, np.ndarray],
    matches: dict[tuple[str, str], np.ndarray],
    camera_model: str = "SIMPLE_RADIAL",
    num_threads: int = -1,
    random_seed: int = 0,
) -> tuple[object | None, dict]:
    """COLMAP incremental SfM from *externally computed* keypoints and matches.

    ``keypoints[name]`` are Nx2 pixel coordinates with pixel centres at integers
    (OpenCV/kornia convention); COLMAP puts the top-left pixel centre at (0.5, 0.5),
    so 0.5 is added before writing. ``matches[(name1, name2)]`` are Mx2 index pairs.
    COLMAP runs its own geometric verification (RANSAC on F/E/H) on the imported matches.
    """
    import pycolmap

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    db_path = work_dir / "database.db"
    if db_path.exists():
        db_path.unlink()
    timings = {}
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = camera_model
    t0 = time.perf_counter()
    pycolmap.Database.open(db_path).close()  # create an empty database with the schema
    # Creates one shared camera (EXIF/size-based initial focal) + one image row per file.
    pycolmap.import_images(
        db_path, image_dir, camera_mode=pycolmap.CameraMode.SINGLE,
        image_names=image_names, options=reader,
    )  # fmt: skip
    pairs_file = work_dir / "pairs.txt"
    with pycolmap.Database.open(db_path) as db:
        name_to_id = {im.name: im.image_id for im in db.read_all_images()}
        for name, kp in keypoints.items():
            kp_colmap = np.asarray(kp, dtype=np.float32) + 0.5  # pixel-centre convention
            db.write_keypoints(name_to_id[name], kp_colmap)
        lines = []
        for (n1, n2), idx in matches.items():
            db.write_matches(name_to_id[n1], name_to_id[n2], np.asarray(idx, dtype=np.uint32))
            lines.append(f"{n1} {n2}")
    pairs_file.write_text("\n".join(lines) + "\n")
    timings["import_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    verify = pycolmap.TwoViewGeometryOptions()
    verify.ransac.random_seed = random_seed
    pycolmap.verify_matches(db_path, pairs_file, options=verify)
    timings["verification_s"] = time.perf_counter() - t0
    rec, t_map = _map(db_path, image_dir, work_dir, image_names, num_threads, random_seed)
    timings["mapping_s"] = t_map
    return rec, timings


def sequential_pairs(image_names: list[str], window: int | None = None) -> list[tuple[str, str]]:
    """Image pairs ``(names[i], names[j])`` with ``0 < j - i <= window`` (all pairs if None).

    For an ordered capture (video, a walk around a building) this is the pairing prior of
    COLMAP's sequential matcher: only nearby images are matched.
    """
    import itertools

    n = len(image_names)
    w = n if window is None or window <= 0 else window
    return [(image_names[i], image_names[j]) for i, j in itertools.combinations(range(n), 2)
            if j - i <= w]  # fmt: skip


def disk_lightglue_match(
    matcher: DiskLightGlue,
    image_dir: Path,
    image_names: list[str],
    pairs: list[tuple[str, str]] | None = None,
) -> tuple[dict[str, np.ndarray], dict[tuple[str, str], np.ndarray], dict]:
    """DISK features for every image + LightGlue matches for ``pairs`` (default: all pairs).

    Returns ``(keypoints {name: Nx2}, matches {(a, b): Mx2 indices}, timings)``.
    """
    import torch

    from cvlab.io import read_image

    pairs = sequential_pairs(image_names) if pairs is None else pairs
    timings = {}
    sync = torch.cuda.synchronize if matcher.device.type == "cuda" else (lambda: None)
    t0 = time.perf_counter()
    feats = {n: matcher.features(read_image(Path(image_dir) / n)) for n in image_names}
    sync()
    timings["features_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    matches = {(a, b): matcher.match_features(feats[a], feats[b])[0] for a, b in pairs}
    sync()
    timings["matching_s"] = time.perf_counter() - t0
    kps = {n: f["keypoints"].cpu().numpy() for n, f in feats.items()}
    return kps, matches, timings


def disk_lightglue_sfm(
    matcher: DiskLightGlue,
    image_dir: Path,
    image_names: list[str],
    work_dir: Path,
    camera_model: str = "SIMPLE_RADIAL",
    window: int | None = None,
    random_seed: int = 0,
) -> tuple[object | None, dict]:
    """DISK features + LightGlue matching, imported into COLMAP for mapping.

    ``window=None`` matches all pairs (exhaustive); ``window=w`` only pairs at most ``w``
    images apart in ``image_names`` (sequential prior for ordered captures).
    """
    pairs = sequential_pairs(image_names, window)
    kps, matches, timings = disk_lightglue_match(matcher, image_dir, image_names, pairs)
    rec, t_colmap = colmap_import_matches_sfm(
        image_dir, image_names, work_dir, kps, matches, camera_model=camera_model,
        random_seed=random_seed,
    )  # fmt: skip
    timings.update(t_colmap)
    return rec, timings


def reference_covisibility(image_names: list[str]) -> np.ndarray:
    """NxN number of 3D points of the South Building reference model seen by both images.

    Zero means the reference reconstruction (all 128 full-resolution photos) found no common
    point: such pairs most likely do not overlap, so verified matches between them are false.
    """
    import pycolmap

    rec = pycolmap.Reconstruction(south_building_dir() / "reference_sparse")
    pts = {img.name: {p.point3D_id for p in img.points2D if p.has_point3D()}
           for img in rec.images.values()}  # fmt: skip
    n = len(image_names)
    cov = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(i + 1, n):
            c = len(pts.get(image_names[i], set()) & pts.get(image_names[j], set()))
            cov[i, j] = cov[j, i] = c
    return cov


def view_graph(db_path: Path, image_names: list[str]) -> np.ndarray:
    """NxN matrix of geometrically verified inlier matches between the given images."""
    import pycolmap

    idx = {n: k for k, n in enumerate(image_names)}
    G = np.zeros((len(image_names), len(image_names)), dtype=np.int64)
    with pycolmap.Database.open(db_path) as db:
        id_to_name = {im.image_id: im.name for im in db.read_all_images()}
        pair_ids, geoms = db.read_two_view_geometries()
        for pid, g in zip(pair_ids, geoms):
            i1, i2 = pycolmap.pair_id_to_image_pair(pid)
            a, b = id_to_name[i1], id_to_name[i2]
            if a in idx and b in idx and g.inlier_matches is not None:
                n = len(g.inlier_matches)
                G[idx[a], idx[b]] = G[idx[b], idx[a]] = n
    return G


def gravity_align(cams: dict[str, dict]) -> np.ndarray:
    """Rotation (3x3) that maps the mean camera "up" direction (-y of OpenCV cameras) to +z.

    Handheld / walking captures keep the cameras roughly upright, so this gives a natural
    top-down view of an SfM model whose world frame is arbitrary.
    """
    up = -np.mean([c["T_wc"][:3, 1] for c in cams.values()], axis=0)
    up /= np.linalg.norm(up)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(up, z)
    s, c = np.linalg.norm(v), float(up @ z)
    if s < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    return np.eye(3) + skew(v) + skew(v) @ skew(v) * ((1 - c) / s**2)


def database_stats(db_path: Path) -> dict:
    """Keypoints / raw matches / verified pairs stored in a COLMAP database."""
    import pycolmap

    with pycolmap.Database.open(db_path) as db:
        return {
            "num_images": int(db.num_images()),
            "num_keypoints": int(db.num_keypoints()),
            "num_matched_pairs": int(db.num_matched_image_pairs()),
            "num_verified_pairs": int(db.num_verified_image_pairs()),
            "num_matches": int(db.num_matches()),
            "num_inlier_matches": int(db.num_inlier_matches()),
        }


def south_building_dir() -> Path:
    from cvlab.io import data_dir

    return data_dir("south_building")


def south_building_subset(step: int = 4, max_images: int = 32) -> list[str]:
    """Every ``step``-th image of the (sorted) South Building sequence."""
    names = sorted(p.name for p in (south_building_dir() / "images").glob("*.JPG"))
    return names[::step][:max_images]


def load_reference_cameras(scale_to_width: int | None = None) -> dict[str, dict]:
    """Cameras of the reference COLMAP model shipped with South Building (if extracted)."""
    import pycolmap

    ref_dir = south_building_dir() / "reference_sparse"
    if not (ref_dir / "cameras.bin").exists() and not (ref_dir / "cameras.txt").exists():
        return {}
    cams = rec_cameras(pycolmap.Reconstruction(ref_dir))
    if scale_to_width is not None:
        for c in cams.values():
            s = scale_to_width / c["width"]
            K = c["K"].copy()
            K[:2] *= s  # exact in COLMAP's pixel convention (coordinates scale with the image)
            c.update(K=K, K_cv=colmap_K_to_opencv(K), width=round(c["width"] * s),
                     height=round(c["height"] * s))
    return cams
