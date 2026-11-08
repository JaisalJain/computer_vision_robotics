"""3D geometry of the capstone: pixel rays, object point clouds, support planes, boxes, grasps.

Only NumPy, SciPy and :mod:`cvlab` (pure NumPy) are required, so exactly the same code runs
in the ``cv`` conda env (notebooks) and under the system Python of ROS 2 Jazzy
(``ros2/capstone_node.py``). OpenCV is optional (used for lens undistortion only).

Conventions (see :mod:`cvlab.geometry`):

* camera optical frame: x right, y down, z forward, metres;
* ``T_ab`` maps points from frame *b* to frame *a*; ``T_wc`` is the camera pose in the world;
* gripper frame ``g``: origin between the finger pads, **z = approach direction** (pointing
  from the palm into the object), **x = closing direction** of the two fingers, y = z x x.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cvlab.geometry import R_to_quat, make_T, quat_to_R
from cvlab.pointcloud import estimate_normals, remove_statistical_outliers, voxel_downsample


# ============================================================================ camera rays
def pixel_rays(K: np.ndarray, dist: np.ndarray | None, width: int, height: int) -> np.ndarray:
    """Normalised ray direction ``(x/z, y/z)`` of every *raw* pixel, shape ``(H, W, 2)``.

    With lens distortion the ray of pixel ``(u, v)`` is found by inverting the distortion
    model (iteratively, as in ``cv2.undistortPoints``). Back-projecting through this lookup
    table is exact for every pixel, so the image never has to be resampled.
    """
    K = np.asarray(K, dtype=np.float64)
    u, v = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    if dist is None or not np.any(dist):
        return np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1]], axis=-1)
    import cv2

    pts = np.stack([u, v], axis=-1).reshape(-1, 1, 2)
    dist = np.asarray(dist, dtype=np.float64).reshape(-1)
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-12)
    try:  # OpenCV 5: criteria is an argument of undistortPoints
        und = cv2.undistortPoints(pts, K, dist, criteria=crit)
    except (TypeError, cv2.error):  # OpenCV 4.x
        und = cv2.undistortPointsIter(pts, K, dist, None, None, crit)
    return und.reshape(height, width, 2)


def project_points(
    points_cam: np.ndarray, K: np.ndarray, dist: np.ndarray | None = None
) -> np.ndarray:
    """Project camera-frame points to raw pixels (applies lens distortion if given)."""
    P = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
    K = np.asarray(K, dtype=np.float64)
    x, y = P[:, 0] / P[:, 2], P[:, 1] / P[:, 2]
    if dist is not None and np.any(dist):
        k1, k2, p1, p2, k3 = np.asarray(dist, dtype=np.float64).reshape(-1)[:5]
        r2 = x * x + y * y
        radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
        x, y = (
            x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
            y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y,
        )
    return np.stack([K[0, 0] * x + K[0, 2], K[1, 1] * y + K[1, 2]], axis=1)


def backproject(
    depth: np.ndarray,
    rays: np.ndarray,
    mask: np.ndarray | None = None,
    min_depth: float = 0.1,
    max_depth: float = 4.0,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project valid pixels: ``X = z * (x_n, y_n, 1)``.

    Returns ``(points Nx3 in the camera frame, pixel indices Nx2 as (v, u))``.
    """
    d = np.asarray(depth, dtype=np.float64)[::stride, ::stride]
    r = rays[::stride, ::stride]
    valid = np.isfinite(d) & (d > min_depth) & (d < max_depth)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)[::stride, ::stride]
    vv, uu = np.nonzero(valid)
    z = d[vv, uu]
    pts = np.stack([r[vv, uu, 0] * z, r[vv, uu, 1] * z, z], axis=1)
    return pts, np.stack([vv * stride, uu * stride], axis=1)


def _morph(mask: np.ndarray, px: int, erode: bool) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    if px <= 0:
        return m
    from scipy import ndimage

    st = np.ones((2 * px + 1, 2 * px + 1), dtype=bool)
    op = ndimage.binary_erosion if erode else ndimage.binary_dilation
    return op(m, structure=st)


def erode_mask(mask: np.ndarray, px: int) -> np.ndarray:
    """Shrink a mask by ``px`` pixels (removes the unreliable depth at object borders)."""
    return _morph(mask, px, erode=True)


def dilate_mask(mask: np.ndarray, px: int) -> np.ndarray:
    """Grow a mask by ``px`` pixels."""
    return _morph(mask, px, erode=False)


def box_to_mask(box: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Boolean mask filled inside an ``(x1, y1, x2, y2)`` box."""
    h, w = shape
    x1, y1, x2, y2 = np.round(np.asarray(box, dtype=np.float64)).astype(int)
    m = np.zeros((h, w), dtype=bool)
    m[max(y1, 0) : min(y2, h), max(x1, 0) : min(x2, w)] = True
    return m


# ============================================================================ object cloud
@dataclass
class ObjectCloud:
    """Cleaned 3D points of one object (camera frame) plus bookkeeping for the tutorial."""

    points: np.ndarray  # N x 3, metres, voxel-downsampled
    colors: np.ndarray | None  # N x 3 in [0, 1]
    n_mask_px: int  # pixels in the (eroded) mask
    n_valid: int  # mask pixels with valid depth
    n_gated: int  # after the depth-mode gate
    n_inliers: int  # after statistical outlier removal

    @property
    def centroid(self) -> np.ndarray:
        return self.points.mean(axis=0)


def object_cloud(
    depth: np.ndarray,
    mask: np.ndarray,
    rays: np.ndarray,
    rgb: np.ndarray | None = None,
    erode_px: int = 3,
    gate_k: float = 3.0,
    gate_min: float = 0.05,
    voxel: float = 0.005,
    sor_k: int = 16,
    sor_std: float = 2.0,
    min_points: int = 30,
) -> ObjectCloud | None:
    """Mask + depth -> cleaned object point cloud.

    1. erode the mask (RGB/depth misalignment and mixed pixels live at the border);
    2. back-project the pixels with valid depth through the ray lookup table;
    3. **depth-mode gate**: keep ``|z - median(z)| < max(gate_k * 1.4826 * MAD, gate_min)``,
       which removes background seen "through" mask leaks;
    4. statistical outlier removal (mean k-NN distance), then voxel down-sampling.
    """
    m = erode_mask(mask, erode_px)
    n_mask = int(m.sum())
    pts, pix = backproject(depth, rays, m)
    n_valid = len(pts)
    if n_valid < min_points:
        return None
    z = pts[:, 2]
    med = np.median(z)
    mad = 1.4826 * np.median(np.abs(z - med))
    keep = np.abs(z - med) < max(gate_k * mad, gate_min)
    pts, pix = pts[keep], pix[keep]
    n_gated = len(pts)
    if n_gated < min_points:
        return None
    inl = remove_statistical_outliers(pts, k=sor_k, std_ratio=sor_std)
    pts, pix = pts[inl], pix[inl]
    cols = None
    if rgb is not None:
        cols = np.asarray(rgb)[pix[:, 0], pix[:, 1]].astype(np.float64) / 255.0
    pts_v, cols_v = voxel_downsample(pts, voxel, cols)
    if len(pts_v) < min(min_points, 10):
        return None
    return ObjectCloud(pts_v, cols_v, n_mask, n_valid, n_gated, int(inl.sum()))


# =========================================================================== planes / boxes
@dataclass
class Plane:
    """Plane ``n . p + d = 0`` with the unit normal ``n`` pointing towards the camera."""

    normal: np.ndarray
    d: float
    n_inliers: int = 0
    inlier_ratio: float = 0.0

    def height(self, points: np.ndarray) -> np.ndarray:
        """Signed distance of points to the plane (> 0 on the camera side, i.e. 'above')."""
        return np.asarray(points, dtype=np.float64).reshape(-1, 3) @ self.normal + self.d


def fit_plane_ransac(
    points: np.ndarray,
    threshold: float = 0.01,
    iterations: int = 256,
    seed: int = 0,
    max_eval: int = 4000,
) -> tuple[Plane, np.ndarray] | None:
    """RANSAC plane fit followed by a total-least-squares refit on the inliers.

    All hypotheses are scored at once on a random subset of at most ``max_eval`` points
    (vectorised: one ``(n_eval x iterations)`` distance matrix). Returns the plane and the
    boolean inlier mask over ``points``, or ``None`` with fewer than 3 points.
    """
    P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n_pts = len(P)
    if n_pts < 3:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_pts, size=(iterations, 3))
    a, b, c = P[idx[:, 0]], P[idx[:, 1]], P[idx[:, 2]]
    nrm = np.cross(b - a, c - a)
    length = np.linalg.norm(nrm, axis=1)
    ok = length > 1e-12
    if not ok.any():
        return None
    nrm = nrm[ok] / length[ok, None]
    d = -np.einsum("ij,ij->i", nrm, a[ok])
    sub = P if n_pts <= max_eval else P[rng.choice(n_pts, max_eval, replace=False)]
    counts = (np.abs(sub @ nrm.T + d) < threshold).sum(axis=0)
    best = int(np.argmax(counts))
    normal, offset = nrm[best], d[best]
    inl = np.abs(P @ normal + offset) < threshold
    for _ in range(2):  # refit: smallest principal direction of the inliers
        if inl.sum() < 3:
            break
        Q = P[inl]
        mu = Q.mean(axis=0)
        normal = np.linalg.svd(Q - mu, full_matrices=False)[2][2]
        offset = -normal @ mu
        inl = np.abs(P @ normal + offset) < threshold
    if offset < 0:  # the camera (origin) must be on the positive side: n.0 + d > 0
        normal, offset = -normal, -offset
    return Plane(normal, float(offset), int(inl.sum()), float(inl.mean())), inl


def support_plane(
    depth: np.ndarray,
    rays: np.ndarray,
    box: np.ndarray,
    mask: np.ndarray,
    expand: float = 0.6,
    stride: int = 2,
    threshold: float = 0.01,
    min_inliers: int = 150,
    seed: int = 0,
) -> Plane | None:
    """Estimate the surface an object rests on from the pixels *around* it.

    The 2D box is enlarged by ``expand`` (relative) on every side; pixels of the (dilated)
    object mask are excluded; RANSAC finds the dominant plane in this ring.
    """
    h, w = depth.shape
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float64)
    bw, bh = x2 - x1, y2 - y1
    ring = box_to_mask(
        [x1 - expand * bw, y1 - expand * bh, x2 + expand * bw, y2 + expand * bh], (h, w)
    ) & ~dilate_mask(mask, 4)
    pts, _ = backproject(depth, rays, ring, stride=stride)
    fit = fit_plane_ransac(pts, threshold=threshold, seed=seed)
    if fit is None or fit[0].n_inliers < min_inliers:
        return None
    return fit[0]


def min_area_rect(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Minimum-area enclosing rectangle of 2D points (rotating calipers on the convex hull).

    Returns ``(center (2,), axes 2x2 with columns = rectangle axes, longest first,
    extents (2,) full side lengths)``.
    """
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    hull = xy
    try:
        from scipy.spatial import ConvexHull

        if len(xy) >= 3:
            hull = xy[ConvexHull(xy).vertices]
    except Exception:  # noqa: BLE001 - degenerate (collinear) input: use all points
        hull = xy
    edges = np.diff(np.vstack([hull, hull[:1]]), axis=0)
    angles = np.unique(np.round(np.arctan2(edges[:, 1], edges[:, 0]) % (np.pi / 2), 12))
    if angles.size == 0:
        angles = np.zeros(1)
    c, s = np.cos(angles), np.sin(angles)
    # rotate the hull by -angle: u = x c + y s, v = -x s + y c   (one row per angle)
    u = hull[None, :, 0] * c[:, None] + hull[None, :, 1] * s[:, None]
    v = -hull[None, :, 0] * s[:, None] + hull[None, :, 1] * c[:, None]
    areas = (u.max(1) - u.min(1)) * (v.max(1) - v.min(1))
    i = int(np.argmin(areas))
    ax_u = np.array([c[i], s[i]])
    ax_v = np.array([-s[i], c[i]])
    ext = np.array([u[i].max() - u[i].min(), v[i].max() - v[i].min()])
    mid = np.array([(u[i].max() + u[i].min()) / 2, (v[i].max() + v[i].min()) / 2])
    center = mid[0] * ax_u + mid[1] * ax_v
    axes = np.stack([ax_u, ax_v], axis=1)
    if ext[1] > ext[0]:
        axes, ext = axes[:, ::-1].copy(), ext[::-1].copy()
    return center, axes, ext


@dataclass
class Box3D:
    """Oriented box: ``center``, rotation ``R`` (columns = box axes), full ``extents``."""

    center: np.ndarray
    R: np.ndarray
    extents: np.ndarray
    kind: str = "pca"  # "pca" (visible surface only) or "table" (gravity-aligned, extruded)

    @property
    def T(self) -> np.ndarray:  # noqa: N802 - pose of the box frame in the cloud frame
        return make_T(self.R, self.center)

    @property
    def quat_xyzw(self) -> np.ndarray:
        return R_to_quat(self.R)

    def corners(self) -> np.ndarray:
        signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        return self.center + (signs * self.extents / 2.0) @ self.R.T

    def transformed(self, T: np.ndarray) -> Box3D:
        T = np.asarray(T, dtype=np.float64)
        return Box3D(
            T[:3, :3] @ self.center + T[:3, 3], T[:3, :3] @ self.R, self.extents.copy(), self.kind
        )


# Edges of the box returned by Box3D.corners() (index pairs), for drawing.
BOX_EDGES = [
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (0, 2),
    (1, 3),
    (4, 6),
    (5, 7),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
]


def pca_box(points: np.ndarray) -> Box3D:
    """Oriented box from the principal axes of the visible points (no prior about gravity)."""
    from cvlab.pointcloud import pca_obb

    c, R, e = pca_obb(points)
    return Box3D(c, R, e, "pca")


def in_plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors spanning the plane orthogonal to ``normal``."""
    n = np.asarray(normal, dtype=np.float64)
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = a - (a @ n) * n
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(n, e1)


def table_box(
    points: np.ndarray, plane: Plane, top_pct: float = 98.0, min_height: float = 0.005
) -> Box3D:
    """Gravity-aligned box: z = support-plane normal, footprint = min-area rectangle.

    A camera only sees the top and the camera-facing sides of an object, so the box is
    **extruded down to the support plane** (objects rest on the table). Its height is a
    robust maximum (``top_pct`` percentile) of the point heights above the plane.
    """
    P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = plane.normal
    h = plane.height(P)
    keep = h > -0.01  # points below the table are noise
    P, h = (P[keep], h[keep]) if keep.sum() >= 3 else (P, h)
    e1, e2 = in_plane_basis(n)
    uv = np.stack([P @ e1, P @ e2], axis=1)
    c2, A2, ext2 = min_area_rect(uv)
    major = A2[0, 0] * e1 + A2[1, 0] * e2
    if major[0] < 0:  # deterministic sign: major axis points to the image right
        major = -major
    minor = np.cross(n, major)  # right-handed: major x minor = n
    top = max(float(np.percentile(h, top_pct)), min_height)
    center = c2[0] * e1 + c2[1] * e2 + (-plane.d + top / 2.0) * n
    R = np.stack([major, minor, n], axis=1)
    return Box3D(center, R, np.array([ext2[0], ext2[1], top]), "table")


# ================================================================================ grasps
@dataclass
class Gripper:
    """Parallel-jaw gripper (defaults roughly match a Robotiq 2F-85)."""

    max_opening: float = 0.085  # stroke [m]
    finger_depth: float = 0.035  # pad length along the approach axis [m]
    finger_width: float = 0.022  # pad width along the grasp line [m]
    finger_thickness: float = 0.012  # finger thickness along the closing axis [m]
    clearance: float = 0.01  # free space on each side when approaching open [m]
    min_height: float = 0.012  # flatter objects cannot be pinched from above [m]
    tip_clearance: float = 0.003  # fingertips stay this far above the table [m]


@dataclass
class Grasp:
    """Top-down parallel-jaw grasp hypothesis.

    ``T`` is the gripper pose in the camera frame (z = approach, x = closing direction).
    """

    T: np.ndarray
    width: float  # object width between the pads [m]
    opening: float  # pre-grasp opening = width + 2 * clearance [m]
    score: float
    feasible: bool
    reason: str
    axis: str  # "minor" or "major" box axis used as closing direction
    offset: float  # position along the grasp line relative to the box centre [m]
    parts: dict = field(default_factory=dict)

    @property
    def position(self) -> np.ndarray:
        return self.T[:3, 3]

    @property
    def quat_xyzw(self) -> np.ndarray:
        return R_to_quat(self.T[:3, :3])

    def transformed(self, T_new_old: np.ndarray) -> Grasp:
        g = Grasp(**{k: getattr(self, k) for k in self.__dataclass_fields__})
        g.T = np.asarray(T_new_old) @ self.T
        return g


def _finger_hits(q: np.ndarray, opening: float, g: Gripper, z_min: float, z_max: float) -> int:
    """Number of points (gripper-frame coords) inside the two finger volumes."""
    ax = np.abs(q[:, 0])
    inside = (
        (ax >= opening / 2)
        & (ax <= opening / 2 + g.finger_thickness)
        & (np.abs(q[:, 1]) <= g.finger_width / 2)
        & (q[:, 2] >= z_min)
        & (q[:, 2] <= z_max)
    )
    return int(inside.sum())


def top_down_grasps(
    points: np.ndarray,
    box: Box3D,
    plane: Plane,
    scene_points: np.ndarray | None = None,
    gripper: Gripper | None = None,
    n_offsets: int = 5,
    normals: np.ndarray | None = None,
    max_collisions: int = 8,
) -> list[Grasp]:
    """Enumerate and score top-down antipodal grasps on a table-aligned box.

    For each closing axis (box minor axis, then major axis) and each offset along the other
    in-plane axis (the *grasp line*):

    * the object points under the finger pads (a slab of width ``finger_width``) give the
      local width ``w`` and top height; the gripper must open to ``w + 2 * clearance``;
    * the grasp centre sits ``finger_depth / 2`` below the local top, but the fingertips
      stay ``tip_clearance`` above the table;
    * scene points (other objects, the table edge...) inside the finger volumes = collision;
    * score (feasible, collision-free grasps only) =
      ``0.4 * width margin + 0.3 * centring + 0.3 * antipodal normal alignment``.

    Returns all hypotheses, best first (infeasible ones have ``score = 0``).
    """
    g = gripper or Gripper()
    P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = plane.normal
    if normals is None and len(P) >= 10:
        normals = estimate_normals(P, k=min(16, len(P)))
    base = box.center - n * (box.extents[2] / 2.0)  # box footprint centre on the table
    out: list[Grasp] = []
    for axis_name, ci, li in (("minor", 1, 0), ("major", 0, 1)):
        x = box.R[:, ci]  # closing direction
        line = box.R[:, li]  # fingers slide along this axis when we vary the offset
        z = -n  # approach: straight down onto the table
        y = np.cross(z, x)
        R_g = np.stack([x, y, z], axis=1)
        half = box.extents[li] / 2.0
        offsets = np.linspace(-0.35, 0.35, n_offsets) * box.extents[li]
        for s in offsets:
            if abs(s) > max(half - g.finger_width / 2, 0.0):
                continue
            c_line = base + s * line
            rel = P - c_line
            slab = np.abs(rel @ line) <= g.finger_width / 2
            parts: dict = {"n_slab": int(slab.sum())}
            T0 = make_T(R_g, c_line)
            if slab.sum() < 10:
                out.append(
                    Grasp(
                        T0,
                        0.0,
                        0.0,
                        0.0,
                        False,
                        "no points under the pads",
                        axis_name,
                        float(s),
                        parts,
                    )
                )
                continue
            xs = rel[slab] @ x
            hs = rel[slab] @ n
            lo, hi = np.percentile(xs, [1, 99])
            width = float(hi - lo)
            top = float(np.percentile(hs, 98))
            opening = width + 2 * g.clearance
            hg = max(top - g.finger_depth / 2, g.finger_depth / 2 + g.tip_clearance)
            center = c_line + 0.5 * (lo + hi) * x + hg * n
            T = make_T(R_g, center)
            parts.update(top=top, width=width)
            if top < g.min_height:
                out.append(
                    Grasp(
                        T,
                        width,
                        opening,
                        0.0,
                        False,
                        f"too flat ({top * 100:.1f} cm)",
                        axis_name,
                        float(s),
                        parts,
                    )
                )
                continue
            if opening > g.max_opening:
                out.append(
                    Grasp(
                        T,
                        width,
                        opening,
                        0.0,
                        False,
                        f"too wide ({width * 100:.1f} cm)",
                        axis_name,
                        float(s),
                        parts,
                    )
                )
                continue
            # finger volumes in the gripper frame: from 2 cm above the object top down to the tips
            z_min, z_max = -(top - hg) - 0.02, g.finger_depth / 2
            hits_obj = _finger_hits((P - center) @ R_g, opening, g, z_min, z_max)
            hits_scene = 0
            if scene_points is not None and len(scene_points):
                S = np.asarray(scene_points, dtype=np.float64).reshape(-1, 3)
                S = S[plane.height(S) > 0.015]  # the table itself is not an obstacle
                hits_scene = _finger_hits((S - center) @ R_g, opening, g, z_min, z_max)
            parts.update(hits_obj=hits_obj, hits_scene=hits_scene)
            # antipodal quality: normals of points on the two contact faces vs closing axis
            q_obj = rel[slab] @ x
            near = ((np.abs(q_obj - lo) < 0.01) | (np.abs(q_obj - hi) < 0.01)) & (hs < top - 0.005)
            antipodal = None
            if normals is not None and near.sum() >= 5:
                antipodal = float(np.mean(np.abs(normals[slab][near] @ x)))
            parts["antipodal"] = antipodal
            width_margin = 1.0 - opening / g.max_opening
            centring = 1.0 - abs(s) / max(half, 1e-6)
            parts.update(width_margin=width_margin, centring=centring)
            if hits_obj + hits_scene > max_collisions:
                out.append(
                    Grasp(
                        T,
                        width,
                        opening,
                        0.0,
                        False,
                        f"collision ({hits_obj + hits_scene} pts)",
                        axis_name,
                        float(s),
                        parts,
                    )
                )
                continue
            score = (
                0.4 * width_margin
                + 0.3 * centring
                + 0.3 * (0.5 if antipodal is None else antipodal)
            )
            out.append(
                Grasp(T, width, opening, float(score), True, "ok", axis_name, float(s), parts)
            )
    out.sort(key=lambda gr: (-gr.feasible, -gr.score))
    return out


# =========================================================================== pose helpers
def interpolate_pose(ts: np.ndarray, poses: np.ndarray, t: float) -> np.ndarray:
    """Pose at time ``t``: linear interpolation of translation, SLERP of rotation.

    ``ts`` must be sorted; ``t`` outside the range is clamped to the nearest end.
    """
    ts = np.asarray(ts, dtype=np.float64)
    i = int(np.clip(np.searchsorted(ts, t), 1, len(ts) - 1))
    t0, t1 = ts[i - 1], ts[i]
    a = float(np.clip((t - t0) / max(t1 - t0, 1e-12), 0.0, 1.0))
    q0, q1 = R_to_quat(poses[i - 1][:3, :3]), R_to_quat(poses[i][:3, :3])
    if q0 @ q1 < 0:  # take the short way round
        q1 = -q1
    cos = float(np.clip(q0 @ q1, -1.0, 1.0))
    if cos > 0.9995:
        q = (1 - a) * q0 + a * q1
    else:
        th = np.arccos(cos)
        q = (np.sin((1 - a) * th) * q0 + np.sin(a * th) * q1) / np.sin(th)
    trans = (1 - a) * poses[i - 1][:3, 3] + a * poses[i][:3, 3]
    return make_T(quat_to_R(q / np.linalg.norm(q)), trans)
