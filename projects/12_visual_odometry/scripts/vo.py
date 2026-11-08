"""Feature-based visual odometry building blocks (OpenCV + NumPy, CPU only).

Three pipelines share one front end (ORB or SIFT keypoints + ratio-test matching):

* :func:`run_mono_essential` - monocular **2D-2D** VO: essential matrix (5-point + RANSAC /
  MAGSAC++) and ``recoverPose`` between keyframes. The translation scale is unobservable, so it
  is either taken from ground truth (an *oracle*, for illustration) or fixed to 1 per step.
* :func:`run_pnp_vo` with ``depth="sensor"`` - **RGB-D 3D-2D** VO: keypoints of the reference
  frame are back-projected with the Kinect depth and the current pose is found with
  ``solvePnPRansac`` + Levenberg-Marquardt refinement (metric scale). With
  ``keyframes=False`` the reference is always the previous frame (frame-to-frame); with
  ``keyframes=True`` the reference is kept until tracking weakens (frame-to-keyframe).
* :func:`run_pnp_vo` with ``depth="triangulated"`` - monocular VO with a tiny map: the first
  two keyframes are initialised from the essential matrix, later 3D points are triangulated
  between keyframes, and every frame is tracked with PnP. The scale is consistent along the
  run (up to one global factor and slow scale drift).

Conventions (same as :mod:`cvlab.geometry`): OpenCV camera axes (x right, y down, z forward),
``T_ab`` maps points from frame *b* to frame *a*, poses are camera-to-world ``T_wc``. A relative
motion estimated between a reference frame *r* and the current frame *c* is ``T_cr``
(``X_c = T_cr @ X_r``), so the new pose is ``T_wc = T_wr @ inv(T_cr)``.

fr1 RGB images of TUM RGB-D are *not* undistorted. Keypoints are detected on the raw image and
their coordinates are undistorted (``cv2.undistortPoints``) before any geometry; depth, which
the Kinect driver registered to the raw RGB pixel grid, is looked up at the raw coordinates.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from cvlab.geometry import invert_T, make_T, umeyama_alignment, unproject
from cvlab.metrics import ate as cvlab_ate
from cvlab.metrics import rpe as cvlab_rpe

PNP_FLAGS = {"epnp": cv2.SOLVEPNP_EPNP, "sqpnp": cv2.SOLVEPNP_SQPNP, "iterative": 0}
ESSENTIAL_METHODS = {"ransac": cv2.RANSAC, "magsac": cv2.USAC_MAGSAC}


# ------------------------------------------------------------------------------ front end
class FeatureFrontend:
    """Keypoint detector/descriptor plus a brute-force matcher with Lowe's ratio test.

    Args:
        kind: ``"orb"`` (FAST corners + rotated BRIEF, binary, Hamming distance) or ``"sift"``
            (DoG blobs + 128-D gradient histograms, L2 distance).
        n_features: Maximum number of keypoints per image.
        ratio: Ratio-test threshold (best / second-best distance); default 0.8 for ORB and
            0.75 for SIFT.
    """

    def __init__(self, kind: str = "orb", n_features: int = 2000, ratio: float | None = None):
        self.kind = kind.lower()
        if self.kind == "orb":
            self.detector = cv2.ORB_create(nfeatures=n_features)
            norm, default_ratio = cv2.NORM_HAMMING, 0.8
        elif self.kind == "sift":
            self.detector = cv2.SIFT_create(nfeatures=n_features)
            norm, default_ratio = cv2.NORM_L2, 0.75
        else:
            raise ValueError(f"unknown feature type: {kind}")
        self.n_features = n_features
        self.ratio = default_ratio if ratio is None else ratio
        self.matcher = cv2.BFMatcher(norm)

    def detect(self, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """Detect keypoints -> ``(uv Nx2 float64 raw pixel coordinates, descriptors)``."""
        kps, desc = self.detector.detectAndCompute(gray, None)
        uv = np.array([k.pt for k in kps], dtype=np.float64).reshape(-1, 2)
        if desc is None:  # no keypoints (e.g. a black or completely blurred frame)
            width = 32 if self.kind == "orb" else 128
            desc = np.zeros((0, width), np.uint8 if self.kind == "orb" else np.float32)
        return uv, desc

    def match(self, desc1: np.ndarray | None, desc2: np.ndarray | None) -> np.ndarray:
        """Ratio-test matches as an ``Mx2`` array of ``(index in 1, index in 2)``.

        Each keypoint of image 2 is used at most once (the closest match wins).
        """
        if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
            return np.zeros((0, 2), dtype=np.int64)
        knn = self.matcher.knnMatch(desc1, desc2, k=2)
        good = [
            (m[0].queryIdx, m[0].trainIdx, m[0].distance)
            for m in knn
            if len(m) == 2 and m[0].distance < self.ratio * m[1].distance
        ]
        if not good:
            return np.zeros((0, 2), dtype=np.int64)
        arr = np.array(good)
        arr = arr[np.argsort(arr[:, 2], kind="stable")]
        _, first = np.unique(arr[:, 1], return_index=True)  # drop many-to-one matches
        return arr[np.sort(first), :2].astype(np.int64)


# ------------------------------------------------------------------------ geometry helpers
def undistort_pixels(uv: np.ndarray, K: np.ndarray, dist: np.ndarray | None) -> np.ndarray:
    """Map raw (distorted) pixels to ideal pinhole pixels of the same ``K``."""
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    if dist is None or not np.any(dist) or len(uv) == 0:
        return uv.copy()
    # The default 5 fixed-point iterations leave up to ~0.1 px error in the image corners for
    # the strong fr1 distortion (round trip through cv2.projectPoints); 20 iterations converge.
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, 1e-10)
    out = cv2.undistortPoints(uv.reshape(-1, 1, 2), K, np.asarray(dist), P=K, criteria=crit)
    return out.reshape(-1, 2)


def depth_edge_mask(depth: np.ndarray, rel_thresh: float = 0.05, ksize: int = 5) -> np.ndarray:
    """``True`` where the depth is locally smooth (no occlusion boundary or hole nearby).

    Keypoints love corners, and corners often sit on depth discontinuities where the Kinect
    returns a mix of foreground and background depth. A pixel is kept when
    ``max - min`` of the depth in a ``ksize x ksize`` window is below ``rel_thresh * depth``.
    """
    kernel = np.ones((ksize, ksize), np.uint8)
    d = depth.astype(np.float32)
    dmax = cv2.dilate(d, kernel)
    dmin = cv2.erode(d, kernel)  # zeros (holes) propagate and invalidate their surroundings
    return (dmin > 0) & ((dmax - dmin) < rel_thresh * d)


def lookup_depth(
    depth: np.ndarray,
    uv_raw: np.ndarray,
    z_range: tuple[float, float] = (0.2, 5.0),
    smooth_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Depth (m) at raw pixel coordinates (nearest pixel); ``NaN`` when invalid."""
    h, w = depth.shape
    u = np.clip(np.rint(uv_raw[:, 0]).astype(np.int64), 0, w - 1)
    v = np.clip(np.rint(uv_raw[:, 1]).astype(np.int64), 0, h - 1)
    z = depth[v, u].astype(np.float64)
    bad = (z < z_range[0]) | (z > z_range[1])
    if smooth_mask is not None:
        bad |= ~smooth_mask[v, u]
    z[bad] = np.nan
    return z


def relative_pose_essential(
    uv1: np.ndarray,
    uv2: np.ndarray,
    K: np.ndarray,
    method: str = "magsac",
    threshold_px: float = 1.0,
    prob: float = 0.999,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Relative pose ``T_21`` (``X_2 = R X_1 + t``, ``||t|| = 1``) from matched pixels.

    Fits the essential matrix with the 5-point solver inside RANSAC or MAGSAC++ and picks the
    one of its four ``(R, t)`` decompositions that puts the points in front of both cameras.

    Returns:
        ``(T_21 or None, inlier mask)``; the mask marks matches that are epipolar inliers *and*
        pass the cheirality check.
    """
    n = len(uv1)
    if n < 8:
        return None, np.zeros(n, dtype=bool)
    E, mask = cv2.findEssentialMat(
        uv1, uv2, K, method=ESSENTIAL_METHODS[method], prob=prob, threshold=threshold_px
    )
    if E is None or E.shape[0] < 3:
        return None, np.zeros(n, dtype=bool)
    E = E[:3]  # the 5-point solver can return up to 10 stacked candidates
    n_good, R, t, mask_pose = cv2.recoverPose(E, uv1, uv2, K, mask=mask.copy())
    inl = mask_pose.ravel() > 0
    if n_good < 8:
        return None, inl
    return make_T(R, t.ravel()), inl


def pose_pnp(
    X_ref: np.ndarray,
    uv_cur: np.ndarray,
    K: np.ndarray,
    reproj_px: float = 3.0,
    iterations: int = 200,
    confidence: float = 0.999,
    method: str = "epnp",
    refine: bool = True,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Pose ``T_cr`` of the current camera from 3D points in the reference frame.

    ``solvePnPRansac`` hypothesises poses from minimal samples, keeps the one with the most
    points whose reprojection error is below ``reproj_px``, and ``solvePnPRefineLM`` then
    minimises the reprojection error of all inliers (Levenberg-Marquardt).
    """
    n = len(X_ref)
    if n < 6:
        return None, np.zeros(n, dtype=bool)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        X_ref.astype(np.float64),
        uv_cur.astype(np.float64),
        K,
        None,
        iterationsCount=iterations,
        reprojectionError=reproj_px,
        confidence=confidence,
        flags=PNP_FLAGS[method],
    )
    mask = np.zeros(n, dtype=bool)
    if not ok or inliers is None or len(inliers) < 6:
        return None, mask
    mask[inliers.ravel()] = True
    if refine:
        rvec, tvec = cv2.solvePnPRefineLM(X_ref[mask], uv_cur[mask], K, None, rvec, tvec)
    return make_T(cv2.Rodrigues(rvec)[0], tvec.ravel()), mask


def triangulate(
    uv_ref: np.ndarray,
    uv_cur: np.ndarray,
    T_cr: np.ndarray,
    K: np.ndarray,
    min_angle_deg: float = 1.0,
    max_reproj_px: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Linear triangulation of matches; returns ``(X in the current frame Nx3, valid mask)``.

    A point is valid when it is in front of both cameras, its two viewing rays intersect at
    more than ``min_angle_deg`` (otherwise depth is ill-conditioned) and it reprojects within
    ``max_reproj_px`` in both images.
    """
    P_ref = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P_cur = K @ T_cr[:3]
    Xh = cv2.triangulatePoints(P_ref, P_cur, uv_ref.T.astype(float), uv_cur.T.astype(float))
    w = Xh[3]
    with np.errstate(divide="ignore", invalid="ignore"):
        X_ref = (Xh[:3] / w).T
    X_cur = X_ref @ T_cr[:3, :3].T + T_cr[:3, 3]
    ok = np.isfinite(X_ref).all(1) & (X_ref[:, 2] > 0) & (X_cur[:, 2] > 0)
    # parallax angle between the two viewing rays (camera centres: 0 and -R^T t in ref frame)
    c_cur = -T_cr[:3, :3].T @ T_cr[:3, 3]
    r1 = X_ref
    r2 = X_ref - c_cur
    with np.errstate(divide="ignore", invalid="ignore"):
        cosang = np.sum(r1 * r2, 1) / (np.linalg.norm(r1, axis=1) * np.linalg.norm(r2, axis=1))
    ok &= np.degrees(np.arccos(np.clip(cosang, -1, 1))) > min_angle_deg
    with np.errstate(divide="ignore", invalid="ignore"):
        e1 = np.linalg.norm(X_ref[:, :2] / X_ref[:, 2:3] @ K[:2, :2].T + K[:2, 2] - uv_ref, axis=1)
        e2 = np.linalg.norm(X_cur[:, :2] / X_cur[:, 2:3] @ K[:2, :2].T + K[:2, 2] - uv_cur, axis=1)
    ok &= (e1 < max_reproj_px) & (e2 < max_reproj_px)
    X_cur[~ok] = np.nan
    return X_cur, ok


# ------------------------------------------------------------------------------- results
@dataclass
class VOResult:
    """Estimated trajectory plus per-frame diagnostics of one VO run."""

    name: str
    frame_ids: np.ndarray  # indices into the sequence that received a pose
    timestamps: np.ndarray  # their RGB timestamps (s)
    poses: np.ndarray  # Nx4x4 camera-to-world
    log: dict[str, np.ndarray] = field(default_factory=dict)  # per *processed* frame
    meta: dict = field(default_factory=dict)

    def save_tum(self, path: str | Path) -> Path:
        """Write the trajectory in TUM format (``t tx ty tz qx qy qz qw``), readable by evo."""
        from cvlab.tum import write_trajectory

        return write_trajectory(path, self.timestamps, self.poses)

    @property
    def ms_per_frame(self) -> float:
        """Mean processing time per input frame (front end + pose), image I/O excluded."""
        return float(np.mean(self.log["ms"])) if "ms" in self.log else float("nan")


class _Log:
    def __init__(self) -> None:
        self.rows: dict[str, list] = {}

    def add(self, **kw) -> None:
        for k, v in kw.items():
            self.rows.setdefault(k, []).append(v)

    def arrays(self) -> dict[str, np.ndarray]:
        return {k: np.asarray(v) for k, v in self.rows.items()}


def _gray(seq, i: int) -> np.ndarray:
    img = cv2.imread(str(seq.rgb_paths[i]), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(seq.rgb_paths[i])
    return img


def _frame_range(seq, start: int, max_frames: int | None) -> range:
    stop = len(seq) if max_frames is None else min(len(seq), start + max_frames)
    return range(start, stop)


# ----------------------------------------------------------------- monocular 2D-2D (E matrix)
def run_mono_essential(
    seq,
    frontend: FeatureFrontend,
    scale: str = "gt",
    min_parallax_px: float = 20.0,
    method: str = "magsac",
    threshold_px: float = 1.0,
    min_inliers: int = 30,
    min_inlier_ratio: float = 0.5,
    min_matches: int = 100,
    start: int = 0,
    max_frames: int | None = None,
    undistort: bool = True,
    seed: int = 0,
    name: str | None = None,
) -> VOResult:
    """Monocular keyframe-to-keyframe VO from essential matrices.

    A frame becomes the next keyframe once the median displacement of its matches to the
    current keyframe exceeds ``min_parallax_px`` (too little parallax makes ``t`` unstable).
    If the essential matrix is rejected (fewer than ``min_inliers`` cheirality inliers, or
    less than ``min_inlier_ratio`` of the matches) the keyframe is kept and the next frame is
    tried; only when fewer than ``min_matches`` matches survive is the track declared lost.
    Only keyframes receive a pose.

    Args:
        scale: ``"gt"`` - each step's length is copied from ground truth (oracle scale, for
            illustration only); ``"unit"`` - every step has length 1 (true monocular VO, compare
            with a Sim(3) alignment).
    """
    cv2.setRNGSeed(seed)
    K, dist = seq.camera.K, (seq.dist if undistort else None)
    if scale == "gt" and seq.gt_poses is None:
        raise ValueError("scale='gt' needs ground truth")
    frames = _frame_range(seq, start, max_frames)
    log = _Log()
    ids, poses = [], []
    kf_id = frames[0]
    T_wkf = seq.gt_poses[kf_id].copy() if seq.gt_poses is not None else np.eye(4)
    gray = _gray(seq, kf_id)
    t0 = time.perf_counter()
    uv_raw, kf_desc = frontend.detect(gray)
    kf_uv = undistort_pixels(uv_raw, K, dist)
    ids.append(kf_id)
    poses.append(T_wkf.copy())
    ms = 1000 * (time.perf_counter() - t0)
    log.add(
        frame=kf_id, n_kps=len(kf_uv), n_matches=0, n_inliers=0, parallax_px=0.0, status="kf", ms=ms
    )
    for i in frames[1:]:
        gray = _gray(seq, i)
        t0 = time.perf_counter()
        uv_raw, desc = frontend.detect(gray)
        uv = undistort_pixels(uv_raw, K, dist)
        m = frontend.match(kf_desc, desc)
        p1, p2 = kf_uv[m[:, 0]], uv[m[:, 1]]
        parallax = float(np.median(np.linalg.norm(p2 - p1, axis=1))) if len(m) else 0.0
        status, n_inl = "skipped", 0
        if parallax >= min_parallax_px or len(m) < min_matches or i == frames[-1]:
            T_ck, inl = relative_pose_essential(p1, p2, K, method, threshold_px)
            n_inl = int(inl.sum())
            ok = T_ck is not None and n_inl >= max(min_inliers, min_inlier_ratio * len(m))
            if not ok and len(m) >= min_matches and i != frames[-1]:
                # usually too little translation (or a degenerate pose): wait for more baseline
                log.add(
                    frame=i,
                    n_kps=len(uv),
                    n_matches=len(m),
                    n_inliers=n_inl,
                    parallax_px=parallax,
                    status="rejected",
                    ms=1000 * (time.perf_counter() - t0),
                )
                continue
            if not ok:
                status = "lost"  # keep the old pose, restart from this frame
            else:
                if scale == "gt":
                    s = np.linalg.norm(seq.gt_poses[i][:3, 3] - seq.gt_poses[kf_id][:3, 3])
                else:
                    s = 1.0
                T_ck[:3, 3] *= s
                T_wkf = T_wkf @ invert_T(T_ck)
                status = "kf"
            kf_id, kf_uv, kf_desc = i, uv, desc
            ids.append(i)
            poses.append(T_wkf.copy())
        ms = 1000 * (time.perf_counter() - t0)
        log.add(
            frame=i,
            n_kps=len(uv),
            n_matches=len(m),
            n_inliers=n_inl,
            parallax_px=parallax,
            status=status,
            ms=ms,
        )
    ids = np.array(ids)
    return VOResult(
        name=name or f"mono-E ({frontend.kind}, {scale} scale)",
        frame_ids=ids,
        timestamps=seq.timestamps[ids],
        poses=np.stack(poses),
        log=log.arrays(),
        meta={
            "type": "mono_essential",
            "scale": scale,
            "feature": frontend.kind,
            "min_parallax_px": min_parallax_px,
            "method": method,
        },
    )


# --------------------------------------------------------- PnP tracking (RGB-D or mono map)
@dataclass
class _Keyframe:
    idx: int
    T_wc: np.ndarray
    uv: np.ndarray  # undistorted pixels, N x 2
    desc: np.ndarray
    X: np.ndarray  # N x 3 points in this camera frame, NaN when unknown

    @property
    def valid(self) -> np.ndarray:
        return np.isfinite(self.X[:, 0])


def run_pnp_vo(
    seq,
    frontend: FeatureFrontend,
    depth: str = "sensor",
    keyframes: bool = True,
    kf_min_inliers: int = 150,
    kf_max_parallax_px: float = 40.0,
    kf_min_parallax_px: float = 15.0,
    init_parallax_px: float = 20.0,
    reproj_px: float = 3.0,
    pnp_method: str = "epnp",
    min_inliers: int = 20,
    max_lost: int = 5,
    depth_edge_filter: bool = True,
    z_range: tuple[float, float] = (0.2, 5.0),
    start: int = 0,
    max_frames: int | None = None,
    undistort: bool = True,
    seed: int = 0,
    name: str | None = None,
) -> VOResult:
    """PnP-based VO against a reference keyframe whose keypoints carry 3D points.

    Args:
        depth: ``"sensor"`` - 3D points from the RGB-D depth image (metric scale);
            ``"triangulated"`` - monocular: points are triangulated between keyframes
            (scale fixed by the first baseline, set to 1).
        keyframes: ``False`` - the previous frame is always the reference (frame-to-frame);
            ``True`` - keep the reference until the inlier count drops below
            ``kf_min_inliers``, so small errors are not integrated at every frame. In
            monocular mode a keyframe also needs a median *translational* parallax (image
            motion after removing the rotation) above ``kf_min_parallax_px`` to triangulate
            new points, and is forced above ``kf_max_parallax_px``.
        min_inliers: Fewer PnP inliers than this counts as a tracking failure. The frame gets
            no pose and the next frames are matched against the same reference; after
            ``max_lost`` consecutive failures the current frame becomes the new reference
            with the last pose (RGB-D) or the map is re-initialised (monocular; the new map
            is scaled so that its median depth equals that of the lost map).
    """
    if depth not in ("sensor", "triangulated"):
        raise ValueError(depth)
    cv2.setRNGSeed(seed)
    K, dist = seq.camera.K, (seq.dist if undistort else None)
    mono = depth == "triangulated"
    frames = _frame_range(seq, start, max_frames)
    log = _Log()
    ids, poses = [], []
    T_w0 = seq.gt_poses[frames[0]].copy() if seq.gt_poses is not None else np.eye(4)

    def make_rgbd_kf(i: int, T_wc: np.ndarray, uv_raw, uv, desc) -> _Keyframe:
        d = seq.depth(i)
        smooth = depth_edge_mask(d) if depth_edge_filter else None
        z = lookup_depth(d, uv_raw, z_range, smooth)
        X = unproject(uv, np.nan_to_num(z), K)
        X[~np.isfinite(z)] = np.nan
        return _Keyframe(i, T_wc.copy(), uv, desc, X)

    kf: _Keyframe | None = None
    T_wc = T_w0.copy()
    ref_depth: float | None = None  # monocular: median map depth to restore after a reset
    n_lost = 0
    for i in frames:
        gray = _gray(seq, i)
        t0 = time.perf_counter()
        uv_raw, desc = frontend.detect(gray)
        uv = undistort_pixels(uv_raw, K, dist)
        status, n_match, n_inl, parallax = "tracked", 0, 0, 0.0
        if kf is None:
            # first frame (or monocular re-initialisation candidate)
            if mono:
                kf = _Keyframe(i, T_wc.copy(), uv, desc, np.full((len(uv), 3), np.nan))
                status = "init"
            else:
                kf = make_rgbd_kf(i, T_wc, uv_raw, uv, desc)
                status = "kf"
            if not mono or not ids:
                ids.append(i)
                poses.append(T_wc.copy())
        elif mono and not kf.valid.any():
            # monocular initialisation: wait for enough parallax, then E + triangulation
            m = frontend.match(kf.desc, desc)
            n_match = len(m)
            p1, p2 = kf.uv[m[:, 0]], uv[m[:, 1]]
            parallax = float(np.median(np.linalg.norm(p2 - p1, axis=1))) if n_match else 0.0
            status = "init"
            if parallax >= init_parallax_px:
                T_ck, inl = relative_pose_essential(p1, p2, K)
                if T_ck is not None and inl.sum() >= min_inliers:
                    Xc, ok = triangulate(p1[inl], p2[inl], T_ck, K)  # baseline length 1
                    n_inl = int(ok.sum())
                    if n_inl >= min_inliers:
                        # first map: the baseline defines the unit; after a reset: keep the
                        # scale of the lost map by matching its median scene depth
                        s = 1.0 if ref_depth is None else ref_depth / np.median(Xc[ok, 2])
                        T_ck[:3, 3] *= s
                        T_wc = kf.T_wc @ invert_T(T_ck)
                        X = np.full((len(uv), 3), np.nan)
                        X[m[inl, 1][ok]] = Xc[ok] * s
                        kf = _Keyframe(i, T_wc.copy(), uv, desc, X)
                        ids.append(i)
                        poses.append(T_wc.copy())
                        status = "kf"
                if status == "init":  # failed: restart the initialisation from this frame
                    kf = _Keyframe(i, kf.T_wc.copy(), uv, desc, np.full((len(uv), 3), np.nan))
        else:
            valid = np.flatnonzero(kf.valid)
            m = frontend.match(kf.desc[valid], desc)
            n_match = len(m)
            T_ck, inl = (None, np.zeros(0, bool))
            if n_match >= 6:
                T_ck, inl = pose_pnp(
                    kf.X[valid[m[:, 0]]], uv[m[:, 1]], K, reproj_px, method=pnp_method
                )
            n_inl = int(inl.sum())
            if T_ck is None or n_inl < min_inliers:
                # tracking failure (blur, fast rotation): no pose for this frame. Keep the
                # reference and retry with the next frames; after `max_lost` failures in a
                # row, re-anchor at the current frame with the last pose (RGB-D) or
                # re-initialise the map (monocular, scale seeded from the lost map's depth).
                status = "lost"
                n_lost += 1
                rescue = _mono_rescue(kf, uv, desc, frontend, K) if mono else None
                if rescue is not None:  # E-matrix pose, scale from the keyframe's points
                    T_ck, X = rescue
                    n_lost, status = 0, "rescued"
                    T_wc = kf.T_wc @ invert_T(T_ck)
                    kf = _Keyframe(i, T_wc.copy(), uv, desc, X)
                    ids.append(i)
                    poses.append(T_wc.copy())
                elif n_lost > max_lost:
                    n_lost = 0
                    if mono:
                        if kf.valid.any():
                            ref_depth = float(np.median(kf.X[kf.valid, 2]))
                        kf = _Keyframe(i, T_wc.copy(), uv, desc, np.full((len(uv), 3), np.nan))
                    else:
                        kf = make_rgbd_kf(i, T_wc, uv_raw, uv, desc)
            else:
                n_lost = 0
                T_wc = kf.T_wc @ invert_T(T_ck)
                pk = kf.uv[valid[m[inl, 0]]]
                if mono:  # rotation-compensated: only translation creates usable baseline
                    parallax = _translational_parallax(pk, uv[m[inl, 1]], T_ck[:3, :3], K)
                else:
                    parallax = float(np.median(np.linalg.norm(uv[m[inl, 1]] - pk, axis=1)))
                need_kf = (not keyframes) or n_inl < kf_min_inliers
                if mono:  # a new keyframe needs baseline, otherwise nothing can be triangulated
                    need_kf = (need_kf and parallax > kf_min_parallax_px) or (
                        parallax > kf_max_parallax_px
                    )
                if need_kf:
                    status = "kf"
                    if mono:
                        kf = _mono_keyframe(kf, i, T_wc, T_ck, uv, desc, frontend, K)
                    else:
                        kf = make_rgbd_kf(i, T_wc, uv_raw, uv, desc)
                ids.append(i)
                poses.append(T_wc.copy())
        ms = 1000 * (time.perf_counter() - t0)
        log.add(
            frame=i,
            n_kps=len(uv),
            n_matches=n_match,
            n_inliers=n_inl,
            parallax_px=parallax,
            status=status,
            ms=ms,
        )
    ids = np.array(ids)
    kind = "mono-map" if mono else "RGB-D PnP"
    ref = "keyframe" if keyframes else "frame-to-frame"
    return VOResult(
        name=name or f"{kind} ({frontend.kind}, {ref})",
        frame_ids=ids,
        timestamps=seq.timestamps[ids],
        poses=np.stack(poses),
        log=log.arrays(),
        meta={
            "type": "mono_map" if mono else "rgbd_pnp",
            "feature": frontend.kind,
            "keyframes": keyframes,
            "kf_min_inliers": kf_min_inliers,
        },
    )


def _mono_rescue(
    kf,
    uv,
    desc,
    frontend,
    K,
    min_shared: int = 15,
    min_inliers: int = 50,
    min_inlier_ratio: float = 0.5,
    min_parallax_px: float = 10.0,
):
    """Recover a monocular pose when PnP fails, keeping the map scale.

    The essential matrix between the keyframe and the current frame gives ``R`` and the
    *direction* of ``t``; triangulating with ``||t|| = 1`` gives depths that are off by one
    unknown factor ``s``. Keyframe points that already have a depth fix it:
    ``s = median(z_map / z_unit)`` (the classic relative-scale trick). The rescue is only
    accepted when the essential matrix is well supported (inlier count and ratio) and the
    rotation-compensated parallax is large enough for a stable translation direction.
    Returns ``(T_ck, X_cur)`` or ``None``.
    """
    m = frontend.match(kf.desc, desc)
    if len(m) < max(30, min_inliers):
        return None
    p1, p2 = kf.uv[m[:, 0]], uv[m[:, 1]]
    T_ck, inl = relative_pose_essential(p1, p2, K)
    if T_ck is None or inl.sum() < max(min_inliers, min_inlier_ratio * len(m)):
        return None
    if _translational_parallax(p1[inl], p2[inl], T_ck[:3, :3], K) < min_parallax_px:
        return None
    Xc, ok = triangulate(p1, p2, T_ck, K)
    ok &= inl
    X_k = (Xc - T_ck[:3, 3]) @ T_ck[:3, :3]  # the same points in the keyframe frame
    shared = ok & kf.valid[m[:, 0]]
    if shared.sum() < min_shared:
        return None
    s = float(np.median(kf.X[m[shared, 0], 2] / X_k[shared, 2]))
    if not np.isfinite(s) or s <= 0:
        return None
    T_ck[:3, 3] *= s
    X = np.full((len(uv), 3), np.nan)
    X[m[ok, 1]] = Xc[ok] * s
    return T_ck, X


def _mono_keyframe(kf, i, T_wc, T_ck, uv, desc, frontend, K, min_shared: int = 20) -> _Keyframe:
    """New monocular keyframe: carry over known points and triangulate new ones.

    The PnP translation between two nearby keyframes is only a few centimetres and its *length*
    can easily be off by tens of percent (a small sideways shift looks like a small rotation).
    Triangulating with it would give the new points a different scale than the old ones, and
    a map with mixed scales quickly collapses. So all matches are triangulated with the PnP
    pose, and the new points are rescaled by ``s = median(z_map / z_triangulated)`` over the
    matches that already had a map point (the relative-scale trick). Without enough such
    matches no new points are added.
    """
    X = np.full((len(uv), 3), np.nan)
    m = frontend.match(kf.desc, desc)
    if len(m):
        known = kf.valid[m[:, 0]]
        # existing map points, moved into the new camera frame
        Xk = kf.X[m[known, 0]]
        X[m[known, 1]] = Xk @ T_ck[:3, :3].T + T_ck[:3, 3]
        # triangulate every match with the PnP pose as the baseline
        Xc, ok = triangulate(kf.uv[m[:, 0]], uv[m[:, 1]], T_ck, K)
        shared = ok & known
        if shared.sum() >= min_shared:
            s = float(np.median(X[m[shared, 1], 2] / Xc[shared, 2]))
            new = ok & ~known
            if np.isfinite(s) and s > 0:
                X[m[new, 1]] = Xc[new] * s
    return _Keyframe(i, T_wc.copy(), uv, desc, X)


def _translational_parallax(uv_ref: np.ndarray, uv_cur: np.ndarray, R_cr: np.ndarray, K) -> float:
    """Median image motion left after removing the rotation (px).

    A pure rotation moves pixels by the infinite homography ``H = K R K^-1`` and creates no
    baseline. Warping the reference pixels with ``H`` first leaves only the translation-induced
    parallax, which is what triangulation needs.
    """
    H = K @ R_cr @ np.linalg.inv(K)
    ph = np.column_stack([uv_ref, np.ones(len(uv_ref))]) @ H.T
    warped = ph[:, :2] / ph[:, 2:3]
    return float(np.median(np.linalg.norm(uv_cur - warped, axis=1)))


# ------------------------------------------------------------------------------ evaluation
def gt_for(result: VOResult, seq) -> np.ndarray:
    """Ground-truth poses (Nx4x4) at the frames that received an estimate."""
    return seq.gt_poses[result.frame_ids]


def align_poses(est: np.ndarray, gt: np.ndarray, mode: str = "se3") -> tuple[np.ndarray, float]:
    """Align estimated poses to ground truth with Umeyama on the positions.

    ``mode``: ``"se3"`` (rotation + translation, metric pipelines), ``"sim3"`` (+ scale,
    monocular) or ``"none"``. Returns ``(aligned Nx4x4 poses, scale)``.
    """
    if mode == "none":
        return est.copy(), 1.0
    s, R, t = umeyama_alignment(est[:, :3, 3], gt[:, :3, 3], with_scale=(mode == "sim3"))
    out = est.copy()
    out[:, :3, :3] = R @ est[:, :3, :3]
    out[:, :3, 3] = s * est[:, :3, 3] @ R.T + t
    return out, s


def rpe_time(
    timestamps: np.ndarray,
    est: np.ndarray,
    gt: np.ndarray,
    delta_s: float = 1.0,
    tol_s: float = 0.1,
) -> dict[str, np.ndarray]:
    """Relative pose error over a fixed time interval (TUM ``evaluate_rpe.py`` style).

    For every pose *i* the partner *j* is the pose whose timestamp is closest to
    ``t_i + delta_s`` (skipped if it is more than ``tol_s`` away). The error of the pair is
    ``E = (Q_i^-1 Q_j)^-1 (P_i^-1 P_j)`` with ground truth ``Q`` and estimate ``P``.

    Returns per-pair arrays ``i``, ``j``, ``t`` (time of *j*), ``trans`` (m) and ``rot`` (deg).
    """
    ts = np.asarray(timestamps, dtype=np.float64)
    js = np.searchsorted(ts, ts + delta_s)
    out = {"i": [], "j": [], "t": [], "trans": [], "rot": []}
    for i, j in enumerate(js):
        cands = [c for c in (j - 1, j) if i < c < len(ts)]
        if not cands:
            continue
        j = min(cands, key=lambda c: abs(ts[c] - ts[i] - delta_s))
        if abs(ts[j] - ts[i] - delta_s) > tol_s:
            continue
        d_est = invert_T(est[i]) @ est[j]
        d_gt = invert_T(gt[i]) @ gt[j]
        E = invert_T(d_gt) @ d_est
        cos = np.clip((np.trace(E[:3, :3]) - 1) / 2, -1, 1)
        out["i"].append(i)
        out["j"].append(j)
        out["t"].append(ts[j])
        out["trans"].append(np.linalg.norm(E[:3, 3]))
        out["rot"].append(np.degrees(np.arccos(cos)))
    return {k: np.asarray(v) for k, v in out.items()}


def evaluate(result: VOResult, seq, align: str = "se3", delta_s: float = 1.0) -> dict:
    """ATE (after alignment) and RPE per ``delta_s`` seconds of one VO run.

    The ATE numbers come from :func:`cvlab.metrics.ate`; RPE uses :func:`rpe_time` on the
    aligned trajectory (for Sim(3) the fitted scale matters, the rigid part cancels).
    ``drift_pct`` is the RPE translation RMSE divided by the mean distance the camera travels
    in ``delta_s`` (ground truth), in percent.
    """
    gt = gt_for(result, seq)
    est_al, s = align_poses(result.poses, gt, align)
    a = cvlab_ate(result.poses[:, :3, 3], gt[:, :3, 3], align=align)
    r = rpe_time(result.timestamps, est_al, gt, delta_s=delta_s)
    has_pairs = len(r["trans"]) > 0
    rpe_t = float(np.sqrt(np.mean(r["trans"] ** 2))) if has_pairs else float("nan")
    rpe_r = float(np.sqrt(np.mean(r["rot"] ** 2))) if has_pairs else float("nan")
    gt_path = np.linalg.norm(np.diff(seq.gt_poses[:, :3, 3], axis=0), axis=1).sum()
    speed = gt_path / (seq.timestamps[-1] - seq.timestamps[0])  # mean GT speed, m/s
    status = result.log.get("status", np.array([]))
    return {
        "method": result.name,
        "align": align,
        "ate_rmse_m": a["rmse"],
        "ate_mean_m": a["mean"],
        "ate_median_m": a["median"],
        "ate_max_m": a["max"],
        "scale": s,
        "rpe_trans_rmse_m": rpe_t,
        "rpe_rot_rmse_deg": rpe_r,
        "rpe_pairs": len(r["trans"]),
        "drift_pct": 100 * rpe_t / (speed * delta_s),
        "poses": len(result.frame_ids),
        "frames": len(result.log.get("frame", [])),
        "keyframes": int(np.sum(status == "kf")),
        "lost": int(np.sum(status == "lost")),
        "ms_per_frame": result.ms_per_frame,
    }


def evo_metrics(
    timestamps: np.ndarray,
    est: np.ndarray,
    gt: np.ndarray,
    align: str = "se3",
    delta_frames: int = 1,
) -> dict[str, float]:
    """The same ATE / frame-based RPE computed with ``evo`` on already associated poses."""
    import copy

    from evo.core import metrics
    from evo.core.trajectory import PoseTrajectory3D
    from evo.core.units import Unit

    ref = PoseTrajectory3D(poses_se3=list(gt), timestamps=np.asarray(timestamps))
    traj = PoseTrajectory3D(poses_se3=list(est), timestamps=np.asarray(timestamps))
    traj_al = copy.deepcopy(traj)
    if align != "none":
        traj_al.align(ref, correct_scale=(align == "sim3"))
    ape = metrics.APE(metrics.PoseRelation.translation_part)
    ape.process_data((ref, traj_al))
    rpe_t = metrics.RPE(
        metrics.PoseRelation.translation_part,
        delta=delta_frames,
        delta_unit=Unit.frames,
        all_pairs=True,
    )
    rpe_t.process_data((ref, traj_al))
    rpe_r = metrics.RPE(
        metrics.PoseRelation.rotation_angle_deg,
        delta=delta_frames,
        delta_unit=Unit.frames,
        all_pairs=True,
    )
    rpe_r.process_data((ref, traj_al))
    return {
        "ate_rmse_m": float(ape.get_statistic(metrics.StatisticsType.rmse)),
        "rpe_trans_rmse_m": float(rpe_t.get_statistic(metrics.StatisticsType.rmse)),
        "rpe_rot_rmse_deg": float(rpe_r.get_statistic(metrics.StatisticsType.rmse)),
    }


def cvlab_frame_rpe(est: np.ndarray, gt: np.ndarray, align_scale: float = 1.0, delta: int = 1):
    """:func:`cvlab.metrics.rpe` after applying a Sim(3) scale to the estimated translations."""
    est = est.copy()
    est[:, :3, 3] *= align_scale
    return cvlab_rpe(est, gt, delta=delta)
