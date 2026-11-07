"""Evaluation metrics shared across projects (NumPy only, framework-agnostic)."""

from __future__ import annotations

import numpy as np

from .geometry import invert_T, rotation_error_deg, umeyama_alignment


# ----------------------------------------------------------------------- detection / seg
def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between Nx4 and Mx4 boxes in ``(x1, y1, x2, y2)`` format -> NxM."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(axis=2)
    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """IoU of two boolean masks (1.0 when both are empty)."""
    pred, gt = np.asarray(pred, dtype=bool), np.asarray(gt, dtype=bool)
    union = np.logical_or(pred, gt).sum()
    return 1.0 if union == 0 else float(np.logical_and(pred, gt).sum() / union)


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice / F1 overlap of two boolean masks (1.0 when both are empty)."""
    pred, gt = np.asarray(pred, dtype=bool), np.asarray(gt, dtype=bool)
    denom = pred.sum() + gt.sum()
    return 1.0 if denom == 0 else float(2 * np.logical_and(pred, gt).sum() / denom)


def average_precision(recall: np.ndarray, precision: np.ndarray, method: str = "coco") -> float:
    """Area under a precision-recall curve.

    Args:
        recall, precision: Curves ordered by decreasing detection score (recall ascending).
        method: ``"coco"`` - mean interpolated precision at 101 recall thresholds
            (``p_interp(r) = max_{r' >= r} p(r')``, 0 when ``r`` is never reached), or
            ``"continuous"`` - exact area under the interpolated curve (VOC2010+).
    """
    rec = np.asarray(recall, dtype=np.float64)
    prec = np.asarray(precision, dtype=np.float64)
    if rec.size == 0:
        return 0.0
    if method == "coco":
        envelope = np.flip(np.maximum.accumulate(np.flip(prec)))
        idx = np.searchsorted(rec, np.linspace(0.0, 1.0, 101), side="left")
        q = np.where(idx < rec.size, envelope[np.minimum(idx, rec.size - 1)], 0.0)
        return float(q.mean())
    if method == "continuous":
        r = np.concatenate([[0.0], rec, [1.0]])
        p = np.concatenate([[1.0], prec, [0.0]])
        p = np.flip(np.maximum.accumulate(np.flip(p)))
        i = np.where(r[1:] != r[:-1])[0]
        return float(np.sum((r[i + 1] - r[i]) * p[i + 1]))
    raise ValueError(f"Unknown method: {method}")


def precision_recall(
    scores: np.ndarray, is_tp: np.ndarray, n_gt: int
) -> tuple[np.ndarray, np.ndarray]:
    """Precision/recall curves from per-detection scores and TP flags (one class)."""
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    tp = np.asarray(is_tp, dtype=np.float64)[order]
    ctp, cfp = np.cumsum(tp), np.cumsum(1.0 - tp)
    recall = ctp / max(n_gt, 1)
    precision = ctp / np.maximum(ctp + cfp, 1e-12)
    return precision, recall


def match_detections(
    pred_boxes: np.ndarray, pred_scores: np.ndarray, gt_boxes: np.ndarray, iou_thr: float = 0.5
) -> np.ndarray:
    """Greedy score-ordered matching; returns a TP flag per prediction (input order)."""
    pred_boxes = np.asarray(pred_boxes, dtype=np.float64).reshape(-1, 4)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)
    is_tp = np.zeros(len(pred_boxes), dtype=bool)
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return is_tp
    ious = box_iou(pred_boxes, gt_boxes)
    taken = np.zeros(len(gt_boxes), dtype=bool)
    for i in np.argsort(-np.asarray(pred_scores, dtype=np.float64), kind="stable"):
        cand = np.where(~taken, ious[i], -1.0)
        j = int(np.argmax(cand))
        if cand[j] >= iou_thr:
            taken[j] = True
            is_tp[i] = True
    return is_tp


# ---------------------------------------------------------------------- classification
def topk_accuracy(logits: np.ndarray, labels: np.ndarray, k: int = 1) -> float:
    """Fraction of samples whose label is among the top-``k`` scores."""
    logits = np.asarray(logits)
    topk = np.argsort(-logits, axis=1)[:, :k]
    return float(np.mean(np.any(topk == np.asarray(labels)[:, None], axis=1)))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """``C[i, j]`` = number of samples with true class ``i`` predicted as ``j``."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm


# ---------------------------------------------------------------------------- images
def psnr(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    """Peak signal-to-noise ratio in dB."""
    mse = np.mean((np.asarray(pred, np.float64) - np.asarray(target, np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(data_range**2 / mse))


def ssim(
    pred: np.ndarray, target: np.ndarray, data_range: float = 1.0, sigma: float = 1.5
) -> float:
    """Structural similarity (Wang et al. 2004) with an 11x11 Gaussian window.

    Colour images (HxWxC) are averaged over channels.
    """
    from scipy.ndimage import gaussian_filter

    x = np.asarray(pred, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if x.ndim == 2:
        x, y = x[..., None], y[..., None]
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    vals = []
    for ch in range(x.shape[2]):
        a, b = x[..., ch], y[..., ch]

        def blur(img: np.ndarray) -> np.ndarray:
            return gaussian_filter(img, sigma, truncate=3.5)  # radius 5 -> 11x11

        mu_a, mu_b = blur(a), blur(b)
        s_aa = blur(a * a) - mu_a**2
        s_bb = blur(b * b) - mu_b**2
        s_ab = blur(a * b) - mu_a * mu_b
        num = (2 * mu_a * mu_b + c1) * (2 * s_ab + c2)
        den = (mu_a**2 + mu_b**2 + c1) * (s_aa + s_bb + c2)
        vals.append(np.mean(num / den))
    return float(np.mean(vals))


# ------------------------------------------------------------------------------ depth
def align_depth_scale(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Median scale ``s`` so that ``s * pred ~= gt`` (for scale-ambiguous depth)."""
    m = _valid(pred, gt, mask)
    return float(np.median(gt[m]) / np.median(pred[m]))


def align_depth_scale_shift(
    pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None
) -> tuple[float, float]:
    """Least-squares ``(s, t)`` minimising ``||s * pred + t - gt||^2`` over valid pixels.

    Use it in the space where the model is affine-invariant (often inverse depth).
    """
    m = _valid(pred, gt, mask)
    A = np.stack([pred[m], np.ones(m.sum())], axis=1)
    (s, t), *_ = np.linalg.lstsq(A, gt[m], rcond=None)
    return float(s), float(t)


def depth_metrics(
    pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None
) -> dict[str, float]:
    """Standard monocular-depth metrics over valid pixels (gt > 0 and finite).

    Returns ``abs_rel``, ``sq_rel``, ``rmse``, ``rmse_log``, ``delta1/2/3``
    (fraction with ``max(p/g, g/p) < 1.25^k``).
    """
    m = _valid(pred, gt, mask)
    p = np.clip(np.asarray(pred, np.float64)[m], 1e-6, None)
    g = np.asarray(gt, np.float64)[m]
    ratio = np.maximum(p / g, g / p)
    return {
        "abs_rel": float(np.mean(np.abs(p - g) / g)),
        "sq_rel": float(np.mean((p - g) ** 2 / g)),
        "rmse": float(np.sqrt(np.mean((p - g) ** 2))),
        "rmse_log": float(np.sqrt(np.mean((np.log(p) - np.log(g)) ** 2))),
        "delta1": float(np.mean(ratio < 1.25)),
        "delta2": float(np.mean(ratio < 1.25**2)),
        "delta3": float(np.mean(ratio < 1.25**3)),
    }


def _valid(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    pred, gt = np.asarray(pred), np.asarray(gt)
    m = np.isfinite(gt) & (gt > 0) & np.isfinite(pred)
    if mask is not None:
        m &= np.asarray(mask, dtype=bool)
    if not m.any():
        raise ValueError("No valid pixels for depth evaluation")
    return m


# ------------------------------------------------------------------------- trajectories
def ate(
    est_xyz: np.ndarray, gt_xyz: np.ndarray, align: str = "sim3"
) -> dict[str, float | np.ndarray]:
    """Absolute trajectory error of time-associated positions.

    Args:
        est_xyz, gt_xyz: Nx3 positions at matching timestamps.
        align: ``"sim3"`` (scale, for monocular), ``"se3"`` or ``"none"``.

    Returns:
        ``rmse``, ``mean``, ``median``, ``max`` (metres) and ``aligned`` (Nx3 estimate
        after alignment) plus the fitted ``scale``.
    """
    est = np.asarray(est_xyz, dtype=np.float64).reshape(-1, 3)
    gt = np.asarray(gt_xyz, dtype=np.float64).reshape(-1, 3)
    if align == "none":
        s, R, t = 1.0, np.eye(3), np.zeros(3)
    elif align in ("sim3", "se3"):
        s, R, t = umeyama_alignment(est, gt, with_scale=(align == "sim3"))
    else:
        raise ValueError(f"Unknown alignment: {align}")
    aligned = s * est @ R.T + t
    err = np.linalg.norm(aligned - gt, axis=1)
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mean": float(err.mean()),
        "median": float(np.median(err)),
        "max": float(err.max()),
        "scale": float(s),
        "aligned": aligned,
    }


def rpe(est_poses: np.ndarray, gt_poses: np.ndarray, delta: int = 1) -> dict[str, float]:
    """Relative pose error over a fixed frame offset ``delta``.

    Args:
        est_poses, gt_poses: Nx4x4 camera-to-world poses at matching timestamps.

    Returns:
        RMSE of the translational (m) and rotational (deg) drift per ``delta`` frames.
    """
    est = np.asarray(est_poses, dtype=np.float64)
    gt = np.asarray(gt_poses, dtype=np.float64)
    t_err, r_err = [], []
    for i in range(len(est) - delta):
        d_est = invert_T(est[i]) @ est[i + delta]
        d_gt = invert_T(gt[i]) @ gt[i + delta]
        e = invert_T(d_gt) @ d_est
        t_err.append(np.linalg.norm(e[:3, 3]))
        r_err.append(rotation_error_deg(np.eye(3), e[:3, :3]))
    t_err, r_err = np.array(t_err), np.array(r_err)
    return {
        "trans_rmse": float(np.sqrt(np.mean(t_err**2))),
        "rot_rmse_deg": float(np.sqrt(np.mean(r_err**2))),
    }
