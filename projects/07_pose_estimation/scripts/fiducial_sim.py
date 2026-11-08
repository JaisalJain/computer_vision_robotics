"""Synthetic fiducial-marker renderer with exact ground truth, plus detection / PnP helpers.

The renderer treats a printed marker (or ChArUco board) as a textured plane ``Z = 0`` in
its own object frame. A pinhole camera with intrinsics ``K`` and pose ``(R, t)``
(``X_cam = R @ X_obj + t``, OpenCV camera axes: x right, y down, z forward) sees the plane
through the homography

    H = K [r1 r2 t] A,

where ``A`` maps texture pixels ``(u, v, 1)`` to metric plane coordinates ``(X, Y, 1)``.
Because the image is generated from ``(R, t)``, the ground-truth pose and the ground-truth
corner pixels are exact, which lets us measure detector and PnP errors separately.

Conventions (checked by ``tests/test_p07_fiducial_sim.py``):

* single marker (OpenCV ``SOLVEPNP_IPPE_SQUARE`` frame): origin at the marker centre,
  X right, Y **up**, Z **out of the marker** (towards the viewer); corners are ordered
  top-left, top-right, bottom-right, bottom-left, the same order ``detectMarkers`` returns;
* ChArUco board (``cv2.aruco.CharucoBoard``): origin at the top-left board corner, X right,
  Y **down**, Z **into the board**.
"""

from __future__ import annotations

import contextlib
import ctypes
import os
from dataclasses import dataclass, field

import cv2
import numpy as np

REFINE_METHODS = {
    "none": cv2.aruco.CORNER_REFINE_NONE,
    "subpix": cv2.aruco.CORNER_REFINE_SUBPIX,
    "contour": cv2.aruco.CORNER_REFINE_CONTOUR,
    "apriltag": cv2.aruco.CORNER_REFINE_APRILTAG,
}
PNP_METHODS = {
    "IPPE_SQUARE": cv2.SOLVEPNP_IPPE_SQUARE,
    "IPPE": cv2.SOLVEPNP_IPPE,
    "ITERATIVE": cv2.SOLVEPNP_ITERATIVE,
    "SQPNP": cv2.SOLVEPNP_SQPNP,
    "EPNP": cv2.SOLVEPNP_EPNP,
}
# Rotation that makes an object frame "face" the camera.
R_FRONT_MARKER = np.diag([1.0, -1.0, -1.0])  # marker Z (out) points back at the camera
R_FRONT_BOARD = np.eye(3)  # board Z (into the board) points away from the camera


def marker_object_points(length: float) -> np.ndarray:
    """4x3 corners of a square marker of side ``length`` [m] in the IPPE_SQUARE order."""
    h = length / 2.0
    return np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]])


# ----------------------------------------------------------------------------- textures
@dataclass
class PlanarTexture:
    """A grayscale printable pattern and its pixel -> object-plane mapping.

    Attributes:
        image: uint8 HxW texture (white = 255).
        A: 3x3 matrix mapping texture pixel centres ``(u, v, 1)`` to ``(X, Y, 1)`` [m].
        R_front: rotation (object -> camera) for which the pattern faces the camera.
        center: 3-vector, centre of the pattern in the object frame [m].
        size: ``(width, height)`` of the pattern in metres (without the white margin).
    """

    image: np.ndarray
    A: np.ndarray
    R_front: np.ndarray
    center: np.ndarray
    size: tuple[float, float]
    meta: dict = field(default_factory=dict)


def marker_texture(
    dictionary: cv2.aruco.Dictionary,
    marker_id: int,
    length: float,
    cell_px: int = 16,
    margin_cells: int = 2,
) -> PlanarTexture:
    """Square marker of side ``length`` [m] (outer edge of the black border) + white margin."""
    n_cells = dictionary.markerSize + 2  # data bits + one black border cell on each side
    side = n_cells * cell_px
    img = cv2.aruco.generateImageMarker(dictionary, marker_id, side, borderBits=1)
    pad = margin_cells * cell_px
    img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    s = length / side  # metres per texture pixel
    c = pad + side / 2.0 - 0.5  # texture pixel coordinate of the marker centre
    # Image rows grow downwards while the marker Y axis points up -> flip the sign of v.
    A = np.array([[s, 0.0, -s * c], [0.0, -s, s * c], [0.0, 0.0, 1.0]])
    return PlanarTexture(
        img,
        A,
        R_FRONT_MARKER,
        np.zeros(3),
        (length, length),
        {"id": marker_id, "side_px": side, "pad_px": pad},
    )


def charuco_texture(
    board: cv2.aruco.CharucoBoard, px_per_square: int = 48, margin_px: int = 48
) -> PlanarTexture:
    """Texture of a ChArUco board (origin top-left, X right, Y down, Z into the board)."""
    nx, ny = board.getChessboardSize()
    sq = board.getSquareLength()
    img = board.generateImage((nx * px_per_square, ny * px_per_square), marginSize=0, borderBits=1)
    img = cv2.copyMakeBorder(
        img, margin_px, margin_px, margin_px, margin_px, cv2.BORDER_CONSTANT, value=255
    )
    s = sq / px_per_square
    o = s * (0.5 - margin_px)  # texture pixel (margin - 0.5) is the board corner X = 0
    A = np.array([[s, 0.0, o], [0.0, s, o], [0.0, 0.0, 1.0]])
    w, h = nx * sq, ny * sq
    return PlanarTexture(img, A, R_FRONT_BOARD, np.array([w / 2, h / 2, 0.0]), (w, h))


def image_texture(image: np.ndarray, width: float) -> PlanarTexture:
    """Any image (e.g. a photo used as a table top) as a plane ``width`` metres wide.

    Same frame as a marker: origin at the image centre, X right, Y up, Z out of the image.
    """
    h, w = image.shape[:2]
    s = width / w
    A = np.array([[s, 0.0, -s * (w / 2 - 0.5)], [0.0, -s, s * (h / 2 - 0.5)], [0.0, 0.0, 1.0]])
    return PlanarTexture(image, A, R_FRONT_MARKER, np.zeros(3), (width, h * s))


# ----------------------------------------------------------------------------- poses
def rot_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation matrix for a rotation of ``angle`` [rad] about ``axis``."""
    axis = np.asarray(axis, dtype=np.float64)
    rvec = axis / np.linalg.norm(axis) * angle
    return cv2.Rodrigues(rvec)[0]


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation taking unit vector ``a`` to unit vector ``b``."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    v = np.cross(a, b)
    s, c = np.linalg.norm(v), float(a @ b)
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        perp = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(a, [0.0, 1.0, 0.0])
        return rot_axis_angle(perp, np.pi)
    return rot_axis_angle(v, np.arctan2(s, c))


def pose_facing(
    texture: PlanarTexture,
    K: np.ndarray,
    pixel: tuple[float, float],
    distance: float,
    tilt_deg: float = 0.0,
    tilt_dir_deg: float = 0.0,
    roll_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Pose ``(R, t)`` placing the pattern centre at ``distance`` [m] along the ray of ``pixel``.

    The *viewing angle* ``tilt_deg`` is measured between the pattern normal and the line of
    sight (not the optical axis), so it is exactly the angle the camera sees the pattern at.
    ``tilt_dir_deg`` picks the in-plane tilt axis, ``roll_deg`` spins the pattern about its normal.
    """
    ray = np.linalg.solve(K, np.array([pixel[0], pixel[1], 1.0]))
    ray /= np.linalg.norm(ray)
    R_align = rotation_between(np.array([0.0, 0.0, 1.0]), ray)  # optical axis -> line of sight
    phi = np.deg2rad(tilt_dir_deg)
    R_tilt = rot_axis_angle(np.array([np.cos(phi), np.sin(phi), 0.0]), np.deg2rad(tilt_deg))
    R_roll = rot_axis_angle(np.array([0.0, 0.0, 1.0]), np.deg2rad(roll_deg))
    R = R_align @ texture.R_front @ R_tilt @ R_roll
    t = distance * ray - R @ texture.center
    return R, t


def sample_pose(
    texture: PlanarTexture,
    K: np.ndarray,
    image_size: tuple[int, int],
    distance: float,
    tilt_deg: float,
    rng: np.random.Generator,
    center_frac: float = 0.5,
    margin_px: float = 4.0,
    max_tries: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Random pose at a fixed distance/viewing angle with the whole pattern inside the image.

    The pattern centre is drawn uniformly from the central ``center_frac`` of the image;
    the tilt axis and the roll about the normal are uniform.
    """
    w, h = image_size
    corners = pattern_corners(texture)
    for _ in range(max_tries):
        px = (
            w / 2 + (rng.uniform() - 0.5) * center_frac * w,
            h / 2 + (rng.uniform() - 0.5) * center_frac * h,
        )
        R, t = pose_facing(
            texture,
            K,
            px,
            distance,
            tilt_deg,
            tilt_dir_deg=rng.uniform(0, 360),
            roll_deg=rng.uniform(0, 360),
        )
        uv, z = project(corners, K, R, t)
        inside = (uv >= margin_px).all() and (uv[:, 0] <= w - 1 - margin_px).all()
        if (z > 0.05).all() and inside and (uv[:, 1] <= h - 1 - margin_px).all():
            return R, t
    raise RuntimeError("could not place the pattern inside the image (too close?)")


def pattern_corners(texture: PlanarTexture) -> np.ndarray:
    """4x3 outer corners of the pattern (without margin) in the object frame."""
    w, h = texture.size
    if texture.R_front is R_FRONT_MARKER:  # centred frame, Y up (markers, image textures)
        return np.array(
            [[-w / 2, h / 2, 0], [w / 2, h / 2, 0], [w / 2, -h / 2, 0], [-w / 2, -h / 2, 0]],
            dtype=np.float64,
        )
    return np.array([[0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0]], dtype=np.float64)


def project(points: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray):
    """Project Nx3 object points with pose ``(R, t)``; returns ``(uv Nx2, depth N)``."""
    Xc = np.asarray(points, float) @ R.T + t
    uvw = Xc @ K.T
    return uvw[:, :2] / uvw[:, 2:3], Xc[:, 2]


# ----------------------------------------------------------------------------- rendering
def scaled_K(K: np.ndarray, s: float) -> np.ndarray:
    """Intrinsics of the same camera rendered ``s`` times larger (pixel-centre convention)."""
    Ks = np.array(K, dtype=np.float64)
    Ks[0, 0] *= s
    Ks[1, 1] *= s
    Ks[0, 2] = (K[0, 2] + 0.5) * s - 0.5
    Ks[1, 2] = (K[1, 2] + 0.5) * s - 0.5
    return Ks


def paint(
    canvas: np.ndarray,
    texture: PlanarTexture,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    supersample: int = 3,
    black: float = 25.0,
    white: float = 230.0,
) -> np.ndarray:
    """Composite one textured plane onto a float32 HxWx3 ``canvas`` (returns a new array).

    The plane is warped at ``supersample`` x resolution and area-averaged down, which gives
    anti-aliased edges like a real sensor integrating light over each pixel. Grayscale
    textures are mapped to the intensity range ``[black, white]`` (printed paper is never
    perfectly black or white); colour textures are used as they are.
    """
    h, w = canvas.shape[:2]
    s = int(supersample)
    H = scaled_K(K, s) @ np.column_stack([R[:, 0], R[:, 1], t]) @ texture.A  # texture -> image
    tex = texture.image.astype(np.float32)
    if tex.ndim == 2:
        tex = black + (white - black) * (tex / 255.0)
    size = (w * s, h * s)
    warped = cv2.warpPerspective(tex, H, size, flags=cv2.INTER_LINEAR, borderValue=0)
    ones = np.ones(tex.shape[:2], np.float32)
    mask = cv2.warpPerspective(ones, H, size, flags=cv2.INTER_LINEAR, borderValue=0)
    if s > 1:
        warped = cv2.resize(warped, (w, h), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_AREA)
    if warped.ndim == 2:
        warped = warped[..., None]
    m = mask[..., None]
    return canvas * (1.0 - m) + warped * m


def degrade(
    image: np.ndarray,
    blur_sigma: float = 0.6,
    noise_sigma: float = 2.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Optical blur (Gaussian, ``blur_sigma`` px) + sensor noise (``noise_sigma`` grey levels)."""
    out = image.astype(np.float32)
    if blur_sigma > 0:
        out = cv2.GaussianBlur(out, (0, 0), blur_sigma)
    if noise_sigma > 0:
        rng = rng if rng is not None else np.random.default_rng()
        out = out + rng.normal(0.0, noise_sigma, out.shape).astype(np.float32)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def render(
    texture: PlanarTexture,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    background: np.ndarray,
    supersample: int = 3,
    blur_sigma: float = 0.6,
    noise_sigma: float = 2.0,
    rng: np.random.Generator | None = None,
    **paint_kw,
) -> np.ndarray:
    """Render one plane over ``background`` (HxWx3 uint8) and apply blur + noise."""
    canvas = paint(background.astype(np.float32), texture, K, R, t, supersample, **paint_kw)
    return degrade(canvas, blur_sigma, noise_sigma, rng)


def render_scene(
    items: list[tuple[PlanarTexture, np.ndarray]],
    K: np.ndarray,
    T_cam_world: np.ndarray,
    background: np.ndarray,
    supersample: int = 3,
    blur_sigma: float = 0.6,
    noise_sigma: float = 2.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Render several planes given their world poses ``T_world_obj`` (painted in list order).

    ``T_cam_world`` (4x4) maps world points into the camera frame, so each plane is drawn
    with ``T_cam_obj = T_cam_world @ T_world_obj``. Planes behind the camera are skipped.
    """
    canvas = background.astype(np.float32)
    for texture, T_world_obj in items:
        T = T_cam_world @ T_world_obj
        _, z = project(pattern_corners(texture), K, T[:3, :3], T[:3, 3])
        if (z <= 0.05).any():
            continue
        canvas = paint(canvas, texture, K, T[:3, :3], T[:3, 3], supersample)
    return degrade(canvas, blur_sigma, noise_sigma, rng)


# ----------------------------------------------------------------------------- detection
def make_detector(
    dictionary: cv2.aruco.Dictionary, refine: str = "subpix"
) -> cv2.aruco.ArucoDetector:
    """``ArucoDetector`` with the chosen corner refinement (none/subpix/contour/apriltag)."""
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = REFINE_METHODS[refine]
    return cv2.aruco.ArucoDetector(dictionary, params)


def detect(detector: cv2.aruco.ArucoDetector, image: np.ndarray) -> dict[int, np.ndarray]:
    """Detect markers; returns ``{id: 4x2 corners}`` (TL, TR, BR, BL in marker orientation)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2).astype(np.float64) for i, c in zip(ids.ravel(), corners)}


# ----------------------------------------------------------------------------- PnP
@dataclass
class PnPResult:
    R: np.ndarray  # best solution (object -> camera)
    t: np.ndarray
    reproj_rmse: float  # px, of the returned solution
    candidates: list[tuple[np.ndarray, np.ndarray, float]]  # all (R, t, rmse) from IPPE


def solve_marker_pose(
    corners: np.ndarray,
    length: float,
    K: np.ndarray,
    dist: np.ndarray | None = None,
    method: str = "IPPE_SQUARE",
    refine_lm: bool = False,
) -> PnPResult:
    """Pose of one square marker from its 4 image corners.

    ``IPPE_SQUARE`` / ``IPPE`` return *two* candidate poses (planar ambiguity) through
    ``solvePnPGeneric``; they are sorted by reprojection error and the best one is returned
    first. ``refine_lm`` polishes the winner with Levenberg-Marquardt (``solvePnPRefineLM``).
    """
    obj = marker_object_points(length)
    img = np.asarray(corners, dtype=np.float64).reshape(4, 1, 2)
    flags = PNP_METHODS[method]
    n, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, img, K, dist, flags=flags)
    if n == 0:
        raise RuntimeError("solvePnP failed")
    cands = []
    for rv, tv, e in zip(rvecs, tvecs, np.asarray(errs).ravel()):
        cands.append((cv2.Rodrigues(rv)[0], tv.reshape(3).astype(np.float64), float(e)))
    cands.sort(key=lambda c: c[2])
    R, t, e = cands[0]
    if refine_lm:
        rv, tv = cv2.Rodrigues(R)[0], t.reshape(3, 1).copy()
        rv, tv = cv2.solvePnPRefineLM(obj, img, K, dist, rv, tv)
        R, t = cv2.Rodrigues(rv)[0], tv.reshape(3)
        e = reprojection_rmse(obj, img.reshape(4, 2), K, R, t)
    return PnPResult(R, t, e, cands)


def reprojection_rmse(obj: np.ndarray, uv: np.ndarray, K: np.ndarray, R, t) -> float:
    """RMS reprojection error [px] of object points under pose ``(R, t)`` (no distortion)."""
    proj, _ = project(obj, K, R, t)
    return float(np.sqrt(np.mean(np.sum((proj - np.asarray(uv).reshape(-1, 2)) ** 2, axis=1))))


def rotation_error_deg(R_gt: np.ndarray, R_est: np.ndarray) -> float:
    """Geodesic angle ``arccos((tr(R_gt^T R_est) - 1) / 2)`` in degrees."""
    c = (np.trace(R_gt.T @ R_est) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def normal_error_deg(R_gt: np.ndarray, R_est: np.ndarray) -> float:
    """Angle between the two estimated plane normals (third rotation column), in degrees."""
    c = float(R_gt[:, 2] @ R_est[:, 2])
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def translation_errors(t_gt: np.ndarray, t_est: np.ndarray) -> dict[str, float]:
    """Translation error [m]: total, along the line of sight (depth) and lateral to it."""
    d = t_est - t_gt
    ray = t_gt / np.linalg.norm(t_gt)
    along = float(d @ ray)
    lateral = float(np.linalg.norm(d - along * ray))
    return {"t_err": float(np.linalg.norm(d)), "t_err_depth": abs(along), "t_err_lateral": lateral}


# ----------------------------------------------------------------------------- pupil-apriltags
# Two convention differences between the AprilTag 3 C library (pupil-apriltags) and cv2.aruco,
# both verified numerically in tests/test_p07_fiducial_sim.py:
# 1. Pixel origin: AprilTag puts the centre of the top-left pixel at (0.5, 0.5), OpenCV at (0, 0).
# 2. Tag frame: OpenCV's DICT_APRILTAG_36h11 marker image is the official AprilTag image rotated
#    by 180 degrees in-plane, and AprilTag's pose frame has Y down / Z into the tag. Together:
#    R_opencv = R_apriltag @ diag(-1, 1, -1), and AprilTag corner j is OpenCV corner
#    APRILTAG_TO_OPENCV_CORNER[j].
R_IPPE_FROM_APRILTAG = np.diag([-1.0, 1.0, -1.0])
APRILTAG_TO_OPENCV_CORNER = (1, 0, 3, 2)


@contextlib.contextmanager
def silence_c_output():
    """Temporarily redirect file descriptors 1 and 2 (C-level stdout/stderr) to /dev/null.

    The AprilTag C library prints diagnostics such as "more than one new minima found" from
    its pose solver straight to the process's output, bypassing Python; this keeps notebook
    outputs clean.
    """
    libc = ctypes.CDLL(None)
    libc.fflush(None)
    saved = {fd: os.dup(fd) for fd in (1, 2)}
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        for fd in saved:
            os.dup2(devnull, fd)
        yield
    finally:
        libc.fflush(None)
        for fd, keep in saved.items():
            os.dup2(keep, fd)
            os.close(keep)
        os.close(devnull)


def pupil_detect(
    detector, image: np.ndarray, K: np.ndarray, length: float, estimate_pose: bool = False
) -> dict[int, dict]:
    """Run a ``pupil_apriltags.Detector`` and return results in OpenCV conventions.

    Returns ``{id: {"corners": 4x2 (TL, TR, BR, BL, pixel-centre origin), "margin": decision
    margin}}`` plus ``"R"`` (3x3) and ``"t"`` (3,) when ``estimate_pose`` is set. The principal
    point is shifted by +0.5 px into AprilTag's pixel convention so that its pose is unbiased.

    Pose estimation is off by default: everything in this project computes the pose from the
    corners with OpenCV's IPPE, and the AprilTag pose solver (``estimate_tag_pose``) hit a
    heap-corruption abort in this environment in one long-running script (see the README).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    cam = (K[0, 0], K[1, 1], K[0, 2] + 0.5, K[1, 2] + 0.5)
    with silence_c_output():
        if estimate_pose:
            results = detector.detect(gray, estimate_tag_pose=True, camera_params=cam,
                                      tag_size=length)  # fmt: skip
        else:
            results = detector.detect(gray)
    out = {}
    for r in results:
        corners = np.empty((4, 2))
        for j, k in enumerate(APRILTAG_TO_OPENCV_CORNER):
            corners[k] = r.corners[j] - 0.5
        out[int(r.tag_id)] = {"corners": corners, "margin": float(r.decision_margin)}
        if estimate_pose:
            out[int(r.tag_id)]["R"] = r.pose_R @ R_IPPE_FROM_APRILTAG
            out[int(r.tag_id)]["t"] = np.asarray(r.pose_t, float).reshape(3)
    return out


def match_corner_order(corners: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Cyclically shift/reflect 4 corners so they best match ``reference`` (for error stats)."""
    best, best_err = corners, np.inf
    for seq in (corners, corners[::-1]):
        for k in range(4):
            cand = np.roll(seq, k, axis=0)
            err = np.linalg.norm(cand - reference, axis=1).mean()
            if err < best_err:
                best, best_err = cand, err
    return best
