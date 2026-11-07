"""Static matplotlib plotting style and drawing helpers.

Everything here renders to static images (no widgets / interactive backends)
so notebook outputs display on GitHub.
"""

from __future__ import annotations

import io as _io
from collections.abc import Sequence
from pathlib import Path

import numpy as np

# Okabe-Ito colour-blind-safe palette (RGB 0-255), used for boxes, masks and lines.
PALETTE: list[tuple[int, int, int]] = [
    (0, 114, 178),
    (230, 159, 0),
    (0, 158, 115),
    (204, 121, 167),
    (86, 180, 233),
    (213, 94, 0),
    (240, 228, 66),
    (120, 120, 120),
]


def color(i: int, unit: bool = False) -> tuple:
    """i-th palette colour as uint8 RGB (or floats in [0, 1] with ``unit=True``)."""
    c = PALETTE[i % len(PALETTE)]
    return tuple(x / 255.0 for x in c) if unit else c


def set_style() -> None:
    """Apply the repository-wide matplotlib style."""
    import matplotlib as mpl
    from cycler import cycler

    mpl.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": 120,
            "savefig.bbox": "tight",
            "figure.figsize": (7.0, 4.2),
            "figure.constrained_layout.use": True,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "image.cmap": "viridis",
            "axes.prop_cycle": cycler(color=[color(i, unit=True) for i in range(len(PALETTE))]),
        }
    )


def show_images(
    images: Sequence[np.ndarray],
    titles: Sequence[str] | None = None,
    ncols: int = 4,
    size: float = 3.2,
    cmap: str | None = None,
    suptitle: str | None = None,
):
    """Plot a grid of images (RGB uint8/float or single-channel) without axes.

    Returns the matplotlib ``Figure``.
    """
    import matplotlib.pyplot as plt

    n = len(images)
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))
    h, w = np.asarray(images[0]).shape[:2]
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(size * ncols, size * nrows * h / w + 0.4), squeeze=False
    )
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < n:
            img = np.asarray(images[i])
            ax.imshow(img, cmap=cmap if img.ndim == 2 else None)
            if titles is not None:
                ax.set_title(titles[i], fontsize=10)
    if suptitle:
        fig.suptitle(suptitle)
    return fig


def draw_boxes(
    img: np.ndarray,
    boxes: np.ndarray,
    labels: Sequence[str] | None = None,
    scores: Sequence[float] | None = None,
    class_ids: Sequence[int] | None = None,
    thickness: int = 2,
) -> np.ndarray:
    """Draw ``(x1, y1, x2, y2)`` boxes with optional labels on a copy of an RGB image."""
    import cv2

    out = np.ascontiguousarray(img.copy())
    boxes = np.asarray(boxes).reshape(-1, 4)
    scale = max(out.shape[:2]) / 640
    for i, (x1, y1, x2, y2) in enumerate(boxes.round().astype(int)):
        c = color(class_ids[i] if class_ids is not None else i)
        cv2.rectangle(out, (x1, y1), (x2, y2), c, max(1, round(thickness * scale)))
        text = ""
        if labels is not None:
            text = str(labels[i])
        if scores is not None:
            text = f"{text} {scores[i]:.2f}".strip()
        if text:
            fs = 0.5 * max(scale, 0.6)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
            y_text = max(y1, th + 4)
            cv2.rectangle(out, (x1, y_text - th - 4), (x1 + tw + 4, y_text), c, -1)
            cv2.putText(out, text, (x1 + 2, y_text - 3), cv2.FONT_HERSHEY_SIMPLEX, fs,
                        (255, 255, 255), 1, cv2.LINE_AA)  # fmt: skip
    return out


def overlay_masks(
    img: np.ndarray,
    masks: np.ndarray | Sequence[np.ndarray],
    alpha: float = 0.5,
    contour: bool = True,
    colors: Sequence[tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """Alpha-blend one or more boolean HxW masks onto an RGB uint8 image."""
    import cv2

    out = img.astype(np.float32).copy()
    masks = np.asarray(masks, dtype=bool)
    if masks.ndim == 2:
        masks = masks[None]
    for i, m in enumerate(masks):
        c = np.array(colors[i] if colors is not None else color(i), dtype=np.float32)
        out[m] = (1 - alpha) * out[m] + alpha * c
    out = out.clip(0, 255).astype(np.uint8)
    if contour:
        for i, m in enumerate(masks):
            c = tuple(int(x) for x in (colors[i] if colors is not None else color(i)))
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)  # fmt: skip
            cv2.drawContours(out, cnts, -1, c, 2)
    return out


def colorize_depth(
    depth: np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "turbo",
    invalid_color: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Map a depth (or disparity) image to an RGB uint8 colormap.

    Limits default to the 2nd/98th percentile of valid (finite, > 0) values.
    """
    import matplotlib as mpl

    d = np.asarray(depth, dtype=np.float64)
    valid = np.isfinite(d) & (d > 0)
    if not valid.any():
        return np.zeros((*d.shape, 3), dtype=np.uint8)
    lo = np.percentile(d[valid], 2) if vmin is None else vmin
    hi = np.percentile(d[valid], 98) if vmax is None else vmax
    norm = np.clip((d - lo) / max(hi - lo, 1e-12), 0, 1)
    rgb = (mpl.colormaps[cmap](norm)[..., :3] * 255).astype(np.uint8)
    rgb[~valid] = invalid_color
    return rgb


def set_axes_equal(ax) -> None:
    """Equal aspect ratio for a 3D matplotlib axis."""
    lims = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    center, radius = lims.mean(axis=1), 0.5 * np.max(lims[:, 1] - lims[:, 0])
    ax.set_xlim3d(center[0] - radius, center[0] + radius)
    ax.set_ylim3d(center[1] - radius, center[1] + radius)
    ax.set_zlim3d(center[2] - radius, center[2] + radius)


def plot_point_cloud(
    points: np.ndarray,
    colors: np.ndarray | None = None,
    ax=None,
    max_points: int = 30000,
    size: float = 0.5,
    elev: float = -70.0,
    azim: float = -90.0,
    title: str | None = None,
    seed: int = 0,
):
    """Static 3D scatter of a point cloud (randomly subsampled to ``max_points``).

    The default view looks down the camera's +z axis with y pointing down
    (OpenCV camera frame). Returns the 3D axis.
    """
    import matplotlib.pyplot as plt

    P = np.asarray(points).reshape(-1, 3)
    C = None if colors is None else np.asarray(colors)
    if P.shape[0] > max_points:
        idx = np.random.default_rng(seed).choice(P.shape[0], max_points, replace=False)
        P = P[idx]
        C = None if C is None else C[idx]
    if ax is None:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(projection="3d")
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=C if C is not None else P[:, 2], s=size,
               cmap=None if C is not None else "viridis",
               depthshade=False, linewidths=0)  # fmt: skip
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.view_init(elev=elev, azim=azim)
    set_axes_equal(ax)
    if title:
        ax.set_title(title)
    return ax


def plot_camera_frustums(
    ax,
    poses_c2w: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
    scale: float = 0.1,
    color_rgb: tuple[float, float, float] | None = None,
    label: str | None = None,
) -> None:
    """Draw camera frustums (OpenCV convention, camera-to-world poses) on a 3D axis."""
    K_inv = np.linalg.inv(np.asarray(K, dtype=np.float64))
    corners_px = np.array([[0, 0, 1], [width, 0, 1], [width, height, 1], [0, height, 1]], float)
    corners_cam = (K_inv @ corners_px.T).T * scale
    c = color_rgb if color_rgb is not None else color(0, unit=True)
    for k, T in enumerate(np.asarray(poses_c2w).reshape(-1, 4, 4)):
        center = T[:3, 3]
        world = corners_cam @ T[:3, :3].T + center
        for p in world:
            ax.plot(*np.stack([center, p]).T, color=c, lw=0.8)
        loop = np.vstack([world, world[:1]])
        ax.plot(*loop.T, color=c, lw=0.8, label=label if k == 0 else None)


def plot_trajectories(
    trajectories: dict[str, np.ndarray],
    ax=None,
    dims: tuple[int, int] = (0, 2),
    title: str | None = None,
):
    """Top-down 2D plot of several Nx3 trajectories (default: x-z plane). Returns the axis."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    names = "xyz"
    for i, (name, traj) in enumerate(trajectories.items()):
        t = np.asarray(traj).reshape(-1, 3)
        ax.plot(t[:, dims[0]], t[:, dims[1]], label=name, color=color(i, unit=True), lw=1.5)
        ax.plot(t[0, dims[0]], t[0, dims[1]], "o", color=color(i, unit=True), ms=4)
    ax.set_xlabel(f"{names[dims[0]]} [m]")
    ax.set_ylabel(f"{names[dims[1]]} [m]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend()
    if title:
        ax.set_title(title)
    return ax


def savefig(fig, project: str, name: str, dpi: int = 120, max_bytes: int = 1_000_000) -> Path:
    """Save a figure to ``assets/<project>/<name>.png`` (<= ``max_bytes``, else lower dpi)."""
    from .io import assets_dir

    path = assets_dir(project) / (name if name.endswith(".png") else f"{name}.png")
    cur = dpi
    while True:
        buf = _io.BytesIO()
        fig.savefig(buf, format="png", dpi=cur, bbox_inches="tight")
        if buf.tell() <= max_bytes or cur <= 40:
            break
        cur = int(cur * 0.8)
    path.write_bytes(buf.getvalue())
    return path


def save_gif(
    frames: Sequence[np.ndarray],
    path: str | Path,
    fps: float = 8.0,
    max_side: int = 480,
    max_bytes: int = 1_000_000,
) -> Path:
    """Write an animated GIF for READMEs, shrinking until it is <= ``max_bytes``."""
    from PIL import Image

    from .io import resize_max

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = list(frames)
    side, step = max_side, 1
    while True:
        sel = frames[::step]
        imgs = [Image.fromarray(resize_max(np.asarray(f), side)) for f in sel]
        buf = _io.BytesIO()
        imgs[0].save(buf, format="GIF", save_all=True, append_images=imgs[1:],
                     duration=int(1000 * step / fps), loop=0, optimize=True)  # fmt: skip
        if buf.tell() <= max_bytes or (side <= 160 and step >= 4):
            break
        if side > 160:
            side = int(side * 0.8)
        else:
            step += 1
    path.write_bytes(buf.getvalue())
    return path
