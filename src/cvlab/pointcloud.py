"""Depth maps -> point clouds, simple point-cloud processing and PLY I/O (NumPy/SciPy)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .geometry import PinholeCamera


def _as_K(camera: np.ndarray | PinholeCamera) -> np.ndarray:
    return camera.K if isinstance(camera, PinholeCamera) else np.asarray(camera, dtype=np.float64)


def depth_to_points(
    depth: np.ndarray,
    camera: np.ndarray | PinholeCamera,
    rgb: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    stride: int = 1,
    min_depth: float = 1e-6,
    max_depth: float = np.inf,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Back-project a metric depth map (z along the optical axis) to a point cloud.

    ``X = z * K^-1 [u, v, 1]^T`` for every valid pixel.

    Args:
        depth: HxW depth in metres (0 / NaN = invalid).
        camera: 3x3 intrinsics ``K`` or a :class:`PinholeCamera`.
        rgb: Optional HxWx3 image (uint8 or float) for per-point colours.
        mask: Optional HxW boolean mask of pixels to keep.
        stride: Subsample every ``stride`` pixels in u and v.
        min_depth, max_depth: Valid depth range.

    Returns:
        ``(points Nx3 float64 in the camera frame, colors Nx3 float in [0, 1] or None)``.
    """
    K = _as_K(camera)
    depth = np.asarray(depth, dtype=np.float64)
    h, w = depth.shape
    v, u = np.mgrid[0:h:stride, 0:w:stride]
    z = depth[::stride, ::stride]
    valid = np.isfinite(z) & (z > min_depth) & (z < max_depth)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)[::stride, ::stride]
    u, v, z = u[valid].astype(np.float64), v[valid].astype(np.float64), z[valid]
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    points = np.stack([x, y, z], axis=1)
    colors = None
    if rgb is not None:
        c = np.asarray(rgb)[::stride, ::stride][valid]
        colors = c.astype(np.float64) / 255.0 if c.dtype == np.uint8 else c.astype(np.float64)
    return points, colors


def points_to_depth(
    points_cam: np.ndarray, camera: np.ndarray | PinholeCamera, width: int, height: int
) -> np.ndarray:
    """Render a z-buffered depth image (0 = empty) from camera-frame points."""
    K = _as_K(camera)
    P = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
    P = P[P[:, 2] > 0]
    u = np.round(P[:, 0] / P[:, 2] * K[0, 0] + K[0, 2]).astype(int)
    v = np.round(P[:, 1] / P[:, 2] * K[1, 1] + K[1, 2]).astype(int)
    ok = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, z = u[ok], v[ok], P[ok, 2]
    depth = np.full((height, width), np.inf)
    np.minimum.at(depth, (v, u), z)
    depth[np.isinf(depth)] = 0.0
    return depth


def voxel_downsample(
    points: np.ndarray, voxel: float, colors: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray | None]:
    """Average all points (and colours) that fall into the same voxel of size ``voxel``."""
    P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if P.shape[0] == 0:
        return P, colors
    keys = np.floor(P / voxel).astype(np.int64)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    out = np.zeros((counts.size, 3))
    np.add.at(out, inv, P)
    out /= counts[:, None]
    out_c = None
    if colors is not None:
        C = np.asarray(colors, dtype=np.float64)
        out_c = np.zeros((counts.size, C.shape[1]))
        np.add.at(out_c, inv, C)
        out_c /= counts[:, None]
    return out, out_c


def remove_statistical_outliers(
    points: np.ndarray, k: int = 16, std_ratio: float = 2.0
) -> np.ndarray:
    """Boolean inlier mask: mean k-NN distance within ``mean + std_ratio * std``."""
    from scipy.spatial import cKDTree

    P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if P.shape[0] <= k:
        return np.ones(P.shape[0], dtype=bool)
    dists, _ = cKDTree(P).query(P, k=k + 1)
    mean_d = dists[:, 1:].mean(axis=1)
    return mean_d <= mean_d.mean() + std_ratio * mean_d.std()


def estimate_normals(
    points: np.ndarray, k: int = 20, viewpoint: np.ndarray | None = None
) -> np.ndarray:
    """Per-point normals from PCA of the k nearest neighbours.

    Normals are flipped to face ``viewpoint`` (default: the origin, i.e. the camera).
    """
    from scipy.spatial import cKDTree

    P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    k = min(k, P.shape[0])
    _, idx = cKDTree(P).query(P, k=k)
    nbrs = P[idx]  # N x k x 3
    centered = nbrs - nbrs.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centered, centered) / k
    _, eigvecs = np.linalg.eigh(cov)  # ascending eigenvalues
    normals = eigvecs[:, :, 0]
    vp = np.zeros(3) if viewpoint is None else np.asarray(viewpoint, dtype=np.float64)
    flip = np.einsum("ni,ni->n", normals, vp - P) < 0
    normals[flip] *= -1
    return normals


def pca_obb(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Oriented bounding box from principal axes.

    Returns:
        ``(center (3,), R (3x3, columns = box axes, longest first, right-handed),
        extents (3,) full side lengths along those axes)``.
    """
    P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    mean = P.mean(axis=0)
    _, _, Vt = np.linalg.svd(P - mean, full_matrices=False)
    R = Vt.T  # columns sorted by decreasing variance
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1
    local = (P - mean) @ R
    lo, hi = local.min(axis=0), local.max(axis=0)
    center = mean + R @ ((lo + hi) / 2.0)
    return center, R, hi - lo


def obb_corners(center: np.ndarray, R: np.ndarray, extents: np.ndarray) -> np.ndarray:
    """8x3 corners of an oriented box (see :func:`pca_obb`)."""
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    return center + (signs * (np.asarray(extents) / 2.0)) @ np.asarray(R).T


# ------------------------------------------------------------------------------ PLY I/O
_PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}  # fmt: skip


def write_ply(
    path: str | Path,
    points: np.ndarray,
    colors: np.ndarray | None = None,
    normals: np.ndarray | None = None,
    binary: bool = True,
) -> Path:
    """Write a point cloud as PLY (float32 xyz, optional normals, uint8 rgb)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    P = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    fields = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    if normals is not None:
        fields += [("nx", "f4"), ("ny", "f4"), ("nz", "f4")]
    if colors is not None:
        fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    arr = np.empty(P.shape[0], dtype=fields)
    arr["x"], arr["y"], arr["z"] = P.T
    if normals is not None:
        N = np.asarray(normals, dtype=np.float32).reshape(-1, 3)
        arr["nx"], arr["ny"], arr["nz"] = N.T
    if colors is not None:
        C = np.asarray(colors)
        C = (np.clip(C, 0, 1) * 255).round() if C.dtype.kind == "f" else C
        C = C.astype(np.uint8).reshape(-1, 3)
        arr["red"], arr["green"], arr["blue"] = C.T
    inv = {v: k for k, v in _PLY_TYPES.items() if k in ("float", "uchar")}
    fmt = "binary_little_endian" if binary else "ascii"
    header = ["ply", f"format {fmt} 1.0", f"element vertex {P.shape[0]}"]
    header += [f"property {inv[t]} {n}" for n, t in fields]
    header += ["end_header"]
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        if binary:
            f.write(arr.astype(arr.dtype.newbyteorder("<")).tobytes())
        else:
            for row in arr:
                f.write((" ".join(str(x) for x in row) + "\n").encode("ascii"))
    return path


def read_ply(path: str | Path) -> dict[str, np.ndarray]:
    """Read the vertex element of a PLY file (ascii or binary little endian).

    Returns a dict with ``points`` (Nx3) and, when present, ``colors`` (Nx3 uint8),
    ``normals`` (Nx3) plus every raw vertex property by name.
    """
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"Not a PLY file: {path}")
        fmt, n_vertex, props, in_vertex = None, 0, [], False
        while True:
            line = f.readline().decode("ascii").strip()
            if line == "end_header":
                break
            tok = line.split()
            if not tok:
                continue
            if tok[0] == "format":
                fmt = tok[1]
            elif tok[0] == "element":
                in_vertex = tok[1] == "vertex"
                if in_vertex:
                    n_vertex = int(tok[2])
            elif tok[0] == "property" and in_vertex:
                if tok[1] == "list":
                    raise ValueError("List properties in the vertex element are not supported")
                props.append((tok[2], _PLY_TYPES[tok[1]]))
        if fmt == "binary_little_endian":
            dtype = np.dtype([(n, "<" + t) for n, t in props])
            data = np.frombuffer(f.read(dtype.itemsize * n_vertex), dtype=dtype, count=n_vertex)
        elif fmt == "ascii":
            rows = [f.readline().split() for _ in range(n_vertex)]
            data = np.array([tuple(r) for r in rows], dtype=[(n, t) for n, t in props])
        else:
            raise ValueError(f"Unsupported PLY format: {fmt}")
    out: dict[str, np.ndarray] = {name: np.asarray(data[name]) for name, _ in props}
    out["points"] = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)
    if {"red", "green", "blue"} <= out.keys():
        out["colors"] = np.stack([data["red"], data["green"], data["blue"]], axis=1)
    if {"nx", "ny", "nz"} <= out.keys():
        out["normals"] = np.stack([data["nx"], data["ny"], data["nz"]], axis=1)
    return out
