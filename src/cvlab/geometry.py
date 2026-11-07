"""Camera and rigid-body geometry (NumPy, float64).

Conventions used throughout the repository:

* **Camera frame (OpenCV):** x right, y down, z forward (optical axis).
* **Poses:** ``T_ab`` is a 4x4 homogeneous transform mapping points expressed in
  frame *b* into frame *a*: ``p_a = T_ab @ p_b``. Hence ``T_wc`` is
  camera-to-world (the camera pose) and ``T_cw = inv(T_wc)`` is world-to-camera
  (the extrinsics used for projection).
* **Pixels:** ``(u, v)`` with the origin at the centre of the top-left pixel.
* **Quaternions:** ``(x, y, z, w)`` order (same as SciPy / ROS / TUM files).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Rotation from the OpenCV optical frame (x right, y down, z forward) to the ROS
# body frame of REP-103 (x forward, y left, z up): p_ros = R_ROS_FROM_OPENCV @ p_cv.
R_ROS_FROM_OPENCV = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


# ----------------------------------------------------------------------------- cameras
@dataclass
class PinholeCamera:
    """Ideal pinhole camera ``K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]``."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def K(self) -> np.ndarray:  # noqa: N802 - K is the standard symbol
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    @classmethod
    def from_K(cls, K: np.ndarray, width: int, height: int) -> PinholeCamera:  # noqa: N802
        K = np.asarray(K, dtype=np.float64)
        return cls(K[0, 0], K[1, 1], K[0, 2], K[1, 2], int(width), int(height))

    @classmethod
    def from_fov(cls, width: int, height: int, fov_x_deg: float) -> PinholeCamera:
        """Square-pixel camera with the principal point at the image centre."""
        f = 0.5 * width / np.tan(0.5 * np.deg2rad(fov_x_deg))
        return cls(f, f, (width - 1) / 2.0, (height - 1) / 2.0, width, height)

    @property
    def fov_deg(self) -> tuple[float, float]:
        """Horizontal and vertical field of view in degrees."""
        fov_x = 2 * np.rad2deg(np.arctan(0.5 * self.width / self.fx))
        fov_y = 2 * np.rad2deg(np.arctan(0.5 * self.height / self.fy))
        return float(fov_x), float(fov_y)

    def scaled(self, sx: float, sy: float | None = None) -> PinholeCamera:
        """Intrinsics after resizing the image by ``(sx, sy)``.

        Uses the pixel-centre convention: ``c' = (c + 0.5) * s - 0.5``.
        """
        sy = sx if sy is None else sy
        return PinholeCamera(
            self.fx * sx,
            self.fy * sy,
            (self.cx + 0.5) * sx - 0.5,
            (self.cy + 0.5) * sy - 0.5,
            round(self.width * sx),
            round(self.height * sy),
        )

    def project(self, points_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project Nx3 camera-frame points. Returns ``(uv Nx2, depth N)``."""
        return project(points_cam, self.K)

    def unproject(self, uv: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """Back-project pixels with metric depth (z) to Nx3 camera-frame points."""
        return unproject(uv, depth, self.K)

    def in_image(self, uv: np.ndarray) -> np.ndarray:
        """Boolean mask of pixel coordinates that fall inside the image."""
        uv = np.asarray(uv)
        return (
            (uv[:, 0] >= -0.5)
            & (uv[:, 0] < self.width - 0.5)
            & (uv[:, 1] >= -0.5)
            & (uv[:, 1] < self.height - 0.5)
        )


def project(points_cam: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pinhole projection ``u = fx * x / z + cx``. Returns ``(uv Nx2, z N)``."""
    X = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
    z = X[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uvw = (K @ X.T).T
        uv = uvw[:, :2] / uvw[:, 2:3]
    return uv, z


def unproject(uv: np.ndarray, depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Inverse of :func:`project` for depth measured along the optical axis (z)."""
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    depth = np.asarray(depth, dtype=np.float64).reshape(-1)
    ones = np.ones((uv.shape[0], 1))
    rays = (np.linalg.inv(K) @ np.hstack([uv, ones]).T).T  # z component == 1
    return rays * depth[:, None]


def project_points(
    points_world: np.ndarray, K: np.ndarray, T_cw: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points through extrinsics ``T_cw`` (world->camera) and ``K``."""
    return project(transform_points(T_cw, points_world), K)


# ---------------------------------------------------------------------------- SO(3)/SE(3)
def skew(v: np.ndarray) -> np.ndarray:
    """Cross-product matrix ``[v]_x`` such that ``[v]_x @ w == v x w``."""
    x, y, z = np.asarray(v, dtype=np.float64).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def so3_exp(rotvec: np.ndarray) -> np.ndarray:
    """Rodrigues formula: axis-angle vector (radians) -> rotation matrix."""
    rotvec = np.asarray(rotvec, dtype=np.float64).reshape(3)
    theta = np.linalg.norm(rotvec)
    if theta < 1e-12:
        return np.eye(3) + skew(rotvec)
    k = skew(rotvec / theta)
    return np.eye(3) + np.sin(theta) * k + (1 - np.cos(theta)) * (k @ k)


def so3_log(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> axis-angle vector (radians), robust near 0 and pi."""
    R = np.asarray(R, dtype=np.float64)
    cos = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos)
    if theta < 1e-8:
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2.0
    if np.pi - theta < 1e-6:
        # Near pi: axis from the dominant column of (R + I) / 2 = k k^T.
        B = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(B)))
        axis = B[:, i] / np.sqrt(B[i, i])
        return axis * theta
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w * theta / (2.0 * np.sin(theta))


def quat_to_R(q: np.ndarray) -> np.ndarray:  # noqa: N802
    """Unit quaternion ``(x, y, z, w)`` -> rotation matrix."""
    x, y, z, w = np.asarray(q, dtype=np.float64).reshape(4) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def R_to_quat(R: np.ndarray) -> np.ndarray:  # noqa: N802
    """Rotation matrix -> unit quaternion ``(x, y, z, w)`` with ``w >= 0``."""
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        w, x = 0.25 * s, (R[2, 1] - R[1, 2]) / s
        y, z = (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w, x = (R[2, 1] - R[1, 2]) / s, 0.25 * s
        y, z = (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w, x = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s
        y, z = 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w, x = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s
        y, z = (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    q /= np.linalg.norm(q)
    return -q if q[3] < 0 else q


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:  # noqa: N802
    """Build a 4x4 homogeneous transform from ``R`` (3x3) and ``t`` (3,)."""
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def invert_T(T: np.ndarray) -> np.ndarray:  # noqa: N802
    """Closed-form inverse of a rigid transform (or a stack of them, ...x4x4)."""
    T = np.asarray(T, dtype=np.float64)
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    Rt = np.swapaxes(R, -1, -2)
    out = np.zeros_like(T)
    out[..., :3, :3] = Rt
    out[..., :3, 3] = -np.einsum("...ij,...j->...i", Rt, t)
    out[..., 3, 3] = 1.0
    return out


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to Nx3 points."""
    P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return P @ T[:3, :3].T + T[:3, 3]


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray = (0.0, 0.0, 1.0)) -> np.ndarray:
    """Camera-to-world pose ``T_wc`` (OpenCV axes) looking from ``eye`` to ``target``."""
    eye = np.asarray(eye, dtype=np.float64)
    z = np.asarray(target, dtype=np.float64) - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, np.asarray(up, dtype=np.float64))
    if np.linalg.norm(x) < 1e-9:  # up parallel to viewing direction
        x = np.cross(z, np.array([1.0, 0.0, 0.0]))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return make_T(np.stack([x, y, z], axis=1), eye)


def rotation_error_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    """Geodesic angle between two rotations, in degrees."""
    cos = (np.trace(np.asarray(R1).T @ np.asarray(R2)) - 1.0) / 2.0
    return float(np.rad2deg(np.arccos(np.clip(cos, -1.0, 1.0))))


def angle_between_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle between two vectors in degrees (e.g. translation-direction error)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return float(np.rad2deg(np.arccos(np.clip(cos, -1.0, 1.0))))


# ------------------------------------------------------------------- two-view geometry
def essential_from_Rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:  # noqa: N802
    """``E = [t]_x R`` for the relative pose ``x2 = R x1 + t``."""
    return skew(t) @ np.asarray(R, dtype=np.float64)


def fundamental_from_essential(E: np.ndarray, K1: np.ndarray, K2: np.ndarray) -> np.ndarray:
    """``F = K2^-T E K1^-1`` (normalised so that ``||F||_F = 1``)."""
    F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
    return F / np.linalg.norm(F)


def triangulate_dlt(P1: np.ndarray, P2: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Linear (DLT) triangulation of N correspondences from 3x4 projection matrices.

    Args:
        P1, P2: 3x4 projection matrices ``K [R | t]``.
        x1, x2: Nx2 pixel coordinates in image 1 and 2.

    Returns:
        Nx3 points in the frame in which P1/P2 are expressed.
    """
    x1 = np.asarray(x1, dtype=np.float64).reshape(-1, 2)
    x2 = np.asarray(x2, dtype=np.float64).reshape(-1, 2)
    out = np.empty((x1.shape[0], 3))
    for i, ((u1, v1), (u2, v2)) in enumerate(zip(x1, x2, strict=True)):
        A = np.stack(
            [u1 * P1[2] - P1[0], v1 * P1[2] - P1[1], u2 * P2[2] - P2[0], v2 * P2[2] - P2[1]]
        )
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        out[i] = X[:3] / X[3]
    return out


def umeyama_alignment(
    src: np.ndarray, dst: np.ndarray, with_scale: bool = True
) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity transform with ``dst ~= s * R @ src + t`` (Umeyama 1991).

    Args:
        src, dst: Nx3 corresponding points.
        with_scale: Estimate the scale ``s`` (Sim(3)); otherwise ``s = 1`` (SE(3)).

    Returns:
        ``(s, R, t)``.
    """
    src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / src.shape[0]
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    var_s = (xs**2).sum() / src.shape[0]
    s = float(np.trace(np.diag(D) @ S) / var_s) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t
