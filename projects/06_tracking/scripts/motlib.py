"""Multi-object tracking toolkit for project 06 (NumPy + SciPy, CPU only).

* MOTChallenge I/O: sequences (``seqinfo.ini`` + ``img1/``), ``gt.txt`` and tracker files.
* MOT17 evaluation protocol (distractor removal, zero-marked GT) and the three metric
  families used by the benchmark: CLEAR MOT (MOTA, MOTP, IDSW, Frag, MT/PT/ML),
  Identity (IDF1, IDP, IDR) and HOTA (DetA, AssA, LocA). The implementations follow the
  reference TrackEval code step by step, so the numbers can be cross-checked against it.
* :class:`Detections`: a NumPy stand-in for Ultralytics ``Boxes`` so that ByteTrack /
  BoT-SORT can run on *cached* detections (same detections for every tracker).
* :class:`SimpleSORT`: a minimal SORT (Kalman filter + IoU + Hungarian) written from scratch.

Box conventions: MOT files store ``(x_left, y_top, w, h)`` in pixels with 1-based frames;
trackers and drawing use ``(x1, y1, x2, y2)``.
"""

from __future__ import annotations

import configparser
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from cvlab.metrics import box_iou

EPS = np.finfo(float).eps
# MOT17 classes that are *not* scored but also not penalised when a tracker covers them:
# 2 person_on_vehicle, 7 static_person, 8 distractor, 12 reflection.
DISTRACTOR_CLASSES = (2, 7, 8, 12)
PEDESTRIAN = 1
HOTA_ALPHAS = np.arange(0.05, 0.99, 0.05)  # 19 localisation thresholds 0.05 ... 0.95


# --------------------------------------------------------------------------- MOT I/O
@dataclass
class MotSequence:
    """One MOTChallenge sequence folder (``seqinfo.ini``, ``img1/``, ``gt/gt.txt``)."""

    root: Path
    name: str
    frame_rate: float
    length: int
    width: int
    height: int
    img_dir: str = "img1"
    img_ext: str = ".jpg"

    def frame_path(self, t: int) -> Path:
        """Path of frame ``t`` (1-based, like the MOT files)."""
        return self.root / self.img_dir / f"{t:06d}{self.img_ext}"

    def frame(self, t: int, rgb: bool = True) -> np.ndarray:
        """Read frame ``t`` as uint8 RGB (or BGR with ``rgb=False``)."""
        from cvlab.io import read_image

        return read_image(self.frame_path(t), rgb=rgb)

    def gt(self) -> np.ndarray:
        """Ground truth rows ``frame, id, x, y, w, h, consider_flag, class, visibility``."""
        return load_mot_txt(self.root / "gt" / "gt.txt")


def load_sequence(root: str | Path) -> MotSequence:
    """Parse ``seqinfo.ini`` of a MOTChallenge sequence directory."""
    root = Path(root)
    ini = configparser.ConfigParser()
    if not ini.read(root / "seqinfo.ini"):
        raise FileNotFoundError(f"{root}/seqinfo.ini not found")
    s = ini["Sequence"]
    return MotSequence(
        root=root,
        name=s["name"],
        frame_rate=float(s["frameRate"]),
        length=int(s["seqLength"]),
        width=int(s["imWidth"]),
        height=int(s["imHeight"]),
        img_dir=s.get("imDir", "img1"),
        img_ext=s.get("imExt", ".jpg"),
    )


def load_mot_txt(path: str | Path) -> np.ndarray:
    """Load a comma-separated MOT file as a float array (empty -> shape (0, 9))."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return np.zeros((0, 9))
    return np.atleast_2d(np.loadtxt(path, delimiter=",", dtype=np.float64))


def write_mot_txt(path: str | Path, rows: np.ndarray) -> Path:
    """Write tracker output rows ``frame, id, x, y, w, h, score`` in MOTChallenge format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = np.asarray(rows, dtype=np.float64).reshape(-1, 7)
    with open(path, "w") as f:
        for r in rows:
            f.write(
                f"{int(r[0])},{int(r[1])},{r[2]:.2f},{r[3]:.2f},{r[4]:.2f},{r[5]:.2f},"
                f"{r[6]:.4f},-1,-1,-1\n"
            )
    return path


def xywh_to_xyxy(b: np.ndarray) -> np.ndarray:
    """``(x_left, y_top, w, h)`` -> ``(x1, y1, x2, y2)``."""
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    return np.concatenate([b[:, :2], b[:, :2] + b[:, 2:]], axis=1)


def xyxy_to_xywh(b: np.ndarray) -> np.ndarray:
    """``(x1, y1, x2, y2)`` -> ``(x_left, y_top, w, h)``."""
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    return np.concatenate([b[:, :2], b[:, 2:] - b[:, :2]], axis=1)


def clip_rows(rows: np.ndarray, width: int, height: int) -> np.ndarray:
    """Clip MOT rows' boxes to the image, as Ultralytics ``Results`` does in ``model.track()``.

    MOT17 GT boxes are amodal and may extend past the image border, so clipping lowers IoU
    for people at the border.
    """
    out = np.array(rows, dtype=np.float64, copy=True)
    xyxy = xywh_to_xyxy(out[:, 2:6])
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, width)
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, height)
    out[:, 2:6] = xyxy_to_xywh(xyxy).round(2)
    return out


def _group_by_frame(rows: np.ndarray, length: int) -> list[np.ndarray]:
    """Split MOT rows into a list indexed by frame ``0..length-1`` (frame ``t`` -> ``t-1``)."""
    out = [np.zeros((0, rows.shape[1] if rows.ndim == 2 else 9)) for _ in range(length)]
    if len(rows) == 0:
        return out
    frames = rows[:, 0].astype(int)
    if frames.min() < 1 or frames.max() > length:
        raise ValueError(f"frames must lie in 1..{length}, got {frames.min()}..{frames.max()}")
    order = np.argsort(frames, kind="stable")
    rows, frames = rows[order], frames[order]
    bounds = np.searchsorted(frames, np.arange(1, length + 2))
    return [rows[bounds[i] : bounds[i + 1]] for i in range(length)]


# ----------------------------------------------------------------------- evaluation data
@dataclass
class EvalData:
    """Per-frame inputs of the metrics, after the MOT17 preprocessing.

    ``gt_ids[t]`` / ``tr_ids[t]`` hold *contiguous* ids (0..n-1); ``similarity[t]`` is the
    IoU matrix between the GT and tracker boxes of frame ``t`` (shape n_gt_t x n_tr_t).
    """

    gt_ids: list[np.ndarray]
    tr_ids: list[np.ndarray]
    similarity: list[np.ndarray]
    gt_boxes: list[np.ndarray]
    tr_boxes: list[np.ndarray]
    gt_id_map: np.ndarray  # contiguous id -> original GT id
    tr_id_map: np.ndarray  # contiguous id -> original tracker id
    removed_tracker_dets: int = 0

    @property
    def num_gt_ids(self) -> int:
        return len(self.gt_id_map)

    @property
    def num_tr_ids(self) -> int:
        return len(self.tr_id_map)

    @property
    def num_gt_dets(self) -> int:
        return int(sum(len(x) for x in self.gt_ids))

    @property
    def num_tr_dets(self) -> int:
        return int(sum(len(x) for x in self.tr_ids))


def prepare_mot17(
    gt: np.ndarray,
    pred: np.ndarray,
    length: int,
    preprocess: bool = True,
    distractors: Iterable[int] = DISTRACTOR_CLASSES,
) -> EvalData:
    """Build :class:`EvalData` with the MOT17 protocol (as in TrackEval's MotChallenge2DBox).

    1. Per frame, tracker boxes are matched to *all* GT boxes (Hungarian on IoU >= 0.5);
       tracker boxes matched to a distractor class (static person, reflection, person on a
       vehicle, ...) are removed, so they count neither as TP nor FP.
    2. Only GT rows with class 1 (pedestrian) and consider-flag 1 are scored.
    3. Ids are relabelled to contiguous integers.

    Args:
        gt: ``gt.txt`` rows (frame, id, x, y, w, h, flag, class, visibility).
        pred: Tracker rows (frame, id, x, y, w, h, ...).
        length: Number of frames in the sequence.
        preprocess: Apply steps 1-2 (False = score every GT row, keep every tracker box).
    """
    gt = np.asarray(gt, dtype=np.float64).reshape(-1, max(9, np.shape(gt)[-1] if len(gt) else 9))
    pred = np.asarray(pred, dtype=np.float64)
    pred = pred.reshape(-1, pred.shape[-1] if pred.size else 7)
    distractors = np.asarray(list(distractors))
    g_frames, p_frames = _group_by_frame(gt, length), _group_by_frame(pred, length)
    gt_raw_ids, tr_raw_ids, sims, gt_boxes, tr_boxes = [], [], [], [], []
    removed = 0
    for g, p in zip(g_frames, p_frames):
        if len(np.unique(p[:, 1])) != len(p):
            raise ValueError("a tracker id appears twice in one frame")
        sim = box_iou(xywh_to_xyxy(g[:, 2:6]), xywh_to_xyxy(p[:, 2:6]))
        keep_p = np.ones(len(p), dtype=bool)
        keep_g = np.ones(len(g), dtype=bool)
        if preprocess:
            if len(g) and len(p):
                score = sim.copy()
                score[score < 0.5 - EPS] = 0
                rows, cols = linear_sum_assignment(-score)
                ok = score[rows, cols] > 0 + EPS
                rows, cols = rows[ok], cols[ok]
                keep_p[cols[np.isin(g[rows, 7], distractors)]] = False
            keep_g = (g[:, 6] != 0) & (g[:, 7] == PEDESTRIAN)
        removed += int((~keep_p).sum())
        gt_raw_ids.append(g[keep_g, 1].astype(int))
        tr_raw_ids.append(p[keep_p, 1].astype(int))
        sims.append(sim[np.ix_(keep_g, keep_p)])
        gt_boxes.append(g[keep_g, 2:6])
        tr_boxes.append(p[keep_p, 2:6])
    gt_map = np.unique(np.concatenate(gt_raw_ids)) if gt_raw_ids else np.zeros(0, int)
    tr_map = np.unique(np.concatenate(tr_raw_ids)) if tr_raw_ids else np.zeros(0, int)
    return EvalData(
        gt_ids=[np.searchsorted(gt_map, x) for x in gt_raw_ids],
        tr_ids=[np.searchsorted(tr_map, x) for x in tr_raw_ids],
        similarity=sims,
        gt_boxes=gt_boxes,
        tr_boxes=tr_boxes,
        gt_id_map=gt_map,
        tr_id_map=tr_map,
        removed_tracker_dets=removed,
    )


# ----------------------------------------------------------------------- CLEAR MOT
def clear_metrics(data: EvalData, threshold: float = 0.5, return_matches: bool = False) -> dict:
    """CLEAR MOT metrics (Bernardin & Stiefelhagen 2008), TrackEval semantics.

    Per frame, GT and tracker boxes are matched with the Hungarian algorithm on IoU
    (>= ``threshold``), preferring to keep the match of the previous frame. An identity
    switch (IDSW) is counted when a GT object is matched to a different tracker id than
    the last time it was matched.

    ``MOTA = 1 - (FN + FP + IDSW) / num_GT`` and ``MOTP = mean IoU of matched pairs``.
    With ``return_matches=True`` the dict also holds ``matches``: rows
    ``(frame_index, gt_id, tracker_id)`` with *original* ids, and ``idsw_events``.
    """
    n_g = data.num_gt_ids
    res = {k: 0 for k in ("CLR_TP", "CLR_FN", "CLR_FP", "IDSW", "MT", "PT", "ML", "Frag")}
    res["MOTP_sum"] = 0.0
    matches_out, idsw_events = [], []
    gt_id_count = np.zeros(n_g)
    gt_matched_count = np.zeros(n_g)
    gt_frag_count = np.zeros(n_g)
    prev_tracker_id = np.full(n_g, np.nan)  # last tracker id ever matched to each GT id
    prev_timestep_tracker_id = np.full(n_g, np.nan)  # tracker id matched in the previous frame
    if data.num_tr_dets == 0:
        res["CLR_FN"] = data.num_gt_dets
        res["ML"] = n_g
    elif data.num_gt_dets == 0:
        res["CLR_FP"] = data.num_tr_dets
    else:
        for t, (g_ids, t_ids) in enumerate(zip(data.gt_ids, data.tr_ids)):
            if len(g_ids) == 0:
                res["CLR_FP"] += len(t_ids)
                continue
            if len(t_ids) == 0:
                res["CLR_FN"] += len(g_ids)
                gt_id_count[g_ids] += 1
                continue
            sim = data.similarity[t]
            # Huge bonus for continuing last frame's match, then maximise IoU.
            score = (t_ids[None, :] == prev_timestep_tracker_id[g_ids[:, None]]) * 1000.0 + sim
            score[sim < threshold - EPS] = 0
            rows, cols = linear_sum_assignment(-score)
            ok = score[rows, cols] > 0 + EPS
            rows, cols = rows[ok], cols[ok]
            m_g, m_t = g_ids[rows], t_ids[cols]
            prev = prev_tracker_id[m_g]
            is_idsw = ~np.isnan(prev) & (m_t != prev)
            res["IDSW"] += int(is_idsw.sum())
            if return_matches:
                for gg, tt, sw, pp in zip(m_g, m_t, is_idsw, prev):
                    matches_out.append((t, data.gt_id_map[gg], data.tr_id_map[tt]))
                    if sw:
                        idsw_events.append(
                            (t, data.gt_id_map[gg], data.tr_id_map[int(pp)], data.tr_id_map[tt])
                        )
            gt_id_count[g_ids] += 1
            gt_matched_count[m_g] += 1
            not_prev_tracked = np.isnan(prev_timestep_tracker_id)
            prev_tracker_id[m_g] = m_t
            prev_timestep_tracker_id[:] = np.nan
            prev_timestep_tracker_id[m_g] = m_t
            gt_frag_count += not_prev_tracked & ~np.isnan(prev_timestep_tracker_id)
            res["CLR_TP"] += len(m_g)
            res["CLR_FN"] += len(g_ids) - len(m_g)
            res["CLR_FP"] += len(t_ids) - len(m_g)
            res["MOTP_sum"] += float(sim[rows, cols].sum())
        valid = gt_id_count > 0
        ratio = gt_matched_count[valid] / gt_id_count[valid]
        res["MT"] = int((ratio > 0.8).sum())
        res["PT"] = int((ratio >= 0.2).sum()) - res["MT"]
        res["ML"] = n_g - res["MT"] - res["PT"]
        res["Frag"] = int((gt_frag_count[gt_frag_count > 0] - 1).sum())
    res = _clear_final(res)
    if return_matches:
        res["matches"] = np.asarray(matches_out, dtype=np.int64).reshape(-1, 3)
        res["idsw_events"] = np.asarray(idsw_events, dtype=np.int64).reshape(-1, 4)
    return res


def _clear_final(res: dict) -> dict:
    tp, fn, fp = res["CLR_TP"], res["CLR_FN"], res["CLR_FP"]
    n_gt = max(1, tp + fn)
    res["MOTA"] = (tp - fp - res["IDSW"]) / n_gt
    res["MODA"] = (tp - fp) / n_gt
    res["MOTP"] = res["MOTP_sum"] / max(1, tp)
    res["CLR_Re"] = tp / n_gt
    res["CLR_Pr"] = tp / max(1, tp + fp)
    return res


# ------------------------------------------------------------------------ Identity
def identity_metrics(data: EvalData, threshold: float = 0.5) -> dict:
    """IDF1 / IDP / IDR (Ristani et al. 2016).

    Each GT trajectory is assigned to at most one tracker trajectory for the *whole* video
    (bipartite matching that maximises the number of frames where the pair overlaps with
    IoU >= ``threshold``). ``IDTP`` counts those frames; everything else is IDFN / IDFP.
    ``IDF1 = 2 IDTP / (2 IDTP + IDFP + IDFN)``.
    """
    pot = np.zeros((data.num_gt_ids, data.num_tr_ids))
    for g_ids, t_ids, sim in zip(data.gt_ids, data.tr_ids, data.similarity):
        gi, ti = np.nonzero(sim >= threshold)
        pot[g_ids[gi], t_ids[ti]] += 1
    n_gt_dets, n_tr_dets = data.num_gt_dets, data.num_tr_dets
    if pot.size:
        rows, cols = linear_sum_assignment(pot, maximize=True)
        idtp = int(pot[rows, cols].sum())
    else:
        idtp = 0
    res = {"IDTP": idtp, "IDFN": n_gt_dets - idtp, "IDFP": n_tr_dets - idtp}
    return _identity_final(res)


def _identity_final(res: dict) -> dict:
    tp, fn, fp = res["IDTP"], res["IDFN"], res["IDFP"]
    res["IDR"] = tp / max(1, tp + fn)
    res["IDP"] = tp / max(1, tp + fp)
    res["IDF1"] = tp / max(1, tp + 0.5 * fp + 0.5 * fn)
    return res


# ---------------------------------------------------------------------------- HOTA
def hota_metrics(data: EvalData, alphas: np.ndarray = HOTA_ALPHAS) -> dict:
    """HOTA (Luiten et al. 2020): ``HOTA_a = sqrt(DetA_a * AssA_a)``, averaged over alpha.

    ``DetA = TP / (TP + FN + FP)`` measures detection; ``AssA`` averages, over all TP
    matches c, the Jaccard index of the two trajectories involved,
    ``|TPA(c)| / (|TPA(c)| + |FNA(c)| + |FPA(c)|)``; ``LocA`` is the mean IoU of matches.
    Matching maximises IoU weighted by a global trajectory-alignment score, then a match is
    kept when IoU >= alpha.
    """
    na = len(alphas)
    keys = ("HOTA_TP", "HOTA_FN", "HOTA_FP", "LocA", "AssA", "AssRe", "AssPr")
    res = {k: np.zeros(na) for k in keys}
    if data.num_tr_dets == 0:
        res["HOTA_FN"] += data.num_gt_dets
        res["LocA"] += 1.0
        return _hota_final(res)
    if data.num_gt_dets == 0:
        res["HOTA_FP"] += data.num_tr_dets
        res["LocA"] += 1.0
        return _hota_final(res)
    n_g, n_t = data.num_gt_ids, data.num_tr_ids
    pot = np.zeros((n_g, n_t))
    gt_id_count = np.zeros((n_g, 1))
    tr_id_count = np.zeros((1, n_t))
    for g_ids, t_ids, sim in zip(data.gt_ids, data.tr_ids, data.similarity):
        denom = sim.sum(0)[None, :] + sim.sum(1)[:, None] - sim
        sim_iou = np.zeros_like(sim)
        mask = denom > 0 + EPS
        sim_iou[mask] = sim[mask] / denom[mask]
        pot[g_ids[:, None], t_ids[None, :]] += sim_iou
        gt_id_count[g_ids] += 1
        tr_id_count[0, t_ids] += 1
    global_align = pot / (gt_id_count + tr_id_count - pot)
    match_counts = [np.zeros((n_g, n_t)) for _ in range(na)]
    for g_ids, t_ids, sim in zip(data.gt_ids, data.tr_ids, data.similarity):
        if len(g_ids) == 0:
            res["HOTA_FP"] += len(t_ids)
            continue
        if len(t_ids) == 0:
            res["HOTA_FN"] += len(g_ids)
            continue
        score = global_align[g_ids[:, None], t_ids[None, :]] * sim
        rows, cols = linear_sum_assignment(-score)
        for a, alpha in enumerate(alphas):
            ok = sim[rows, cols] >= alpha - EPS
            r, c = rows[ok], cols[ok]
            res["HOTA_TP"][a] += len(r)
            res["HOTA_FN"][a] += len(g_ids) - len(r)
            res["HOTA_FP"][a] += len(t_ids) - len(r)
            if len(r):
                res["LocA"][a] += sim[r, c].sum()
                match_counts[a][g_ids[r], t_ids[c]] += 1
    for a in range(na):
        mc = match_counts[a]
        tp = np.maximum(1, res["HOTA_TP"][a])
        res["AssA"][a] = np.sum(mc * mc / np.maximum(1, gt_id_count + tr_id_count - mc)) / tp
        res["AssRe"][a] = np.sum(mc * mc / np.maximum(1, gt_id_count)) / tp
        res["AssPr"][a] = np.sum(mc * mc / np.maximum(1, tr_id_count)) / tp
    res["LocA"] = np.maximum(1e-10, res["LocA"]) / np.maximum(1e-10, res["HOTA_TP"])
    return _hota_final(res)


def _hota_final(res: dict) -> dict:
    """Derived HOTA fields; per-alpha arrays are kept as ``<name>_alpha`` for combining."""
    tp, fn, fp = res["HOTA_TP"], res["HOTA_FN"], res["HOTA_FP"]
    res["DetRe"] = tp / np.maximum(1, tp + fn)
    res["DetPr"] = tp / np.maximum(1, tp + fp)
    res["DetA"] = tp / np.maximum(1, tp + fn + fp)
    res["HOTA"] = np.sqrt(res["DetA"] * res["AssA"])
    out = {k: res[k] for k in ("HOTA_TP", "HOTA_FN", "HOTA_FP")}
    for k in ("HOTA", "DetA", "AssA", "LocA", "DetRe", "DetPr", "AssRe", "AssPr"):
        out[k + "_alpha"] = res[k]
        out[k] = float(np.mean(res[k]))
    out["HOTA(0)"] = float(res["HOTA"][0])
    out["LocA(0)"] = float(res["LocA"][0])
    return out


# --------------------------------------------------------------------------- summary
SUMMARY_KEYS = (
    "HOTA", "DetA", "AssA", "MOTA", "IDF1", "MOTP", "IDSW", "Frag", "CLR_FP", "CLR_FN",
    "CLR_TP", "MT", "PT", "ML", "IDP", "IDR", "LocA",
)  # fmt: skip


def evaluate(gt: np.ndarray, pred: np.ndarray, length: int, preprocess: bool = True) -> dict:
    """All metrics for one sequence (MOT17 protocol). Raw counts are kept for combining."""
    data = prepare_mot17(gt, pred, length, preprocess=preprocess)
    res = {**clear_metrics(data), **identity_metrics(data), **hota_metrics(data)}
    res["num_gt_dets"], res["num_tr_dets"] = data.num_gt_dets, data.num_tr_dets
    res["num_gt_ids"], res["num_tr_ids"] = data.num_gt_ids, data.num_tr_ids
    res["removed_distractor_matches"] = data.removed_tracker_dets
    return res


def combine(results: list[dict]) -> dict:
    """Combine per-sequence results like TrackEval's ``COMBINED_SEQ`` row.

    Counts are summed; AssA/AssRe/AssPr/LocA are averaged per alpha, weighted by HOTA_TP.
    """
    out: dict = {}
    counts = ("CLR_TP", "CLR_FN", "CLR_FP", "IDSW", "MT", "PT", "ML", "Frag", "MOTP_sum", "IDTP",
              "IDFN", "IDFP", "num_gt_dets", "num_tr_dets", "num_gt_ids", "num_tr_ids")  # fmt: skip
    for k in counts:
        out[k] = sum(r[k] for r in results)
    out = _identity_final(_clear_final(out))
    h = {k: sum(np.asarray(r[k], dtype=float) for r in results)
         for k in ("HOTA_TP", "HOTA_FN", "HOTA_FP")}  # fmt: skip
    w = np.maximum(1e-10, h["HOTA_TP"])
    for k in ("AssA", "AssRe", "AssPr", "LocA"):
        h[k] = sum(np.asarray(r[k + "_alpha"]) * np.asarray(r["HOTA_TP"]) for r in results) / w
    out.update(_hota_final(h))
    return out


def summary_row(res: dict) -> dict:
    """Pick the headline numbers (percentages for ratios, ints for counts)."""
    row = {}
    for k in SUMMARY_KEYS:
        v = res[k]
        row[k] = round(100 * float(v), 2) if isinstance(v, float) else int(v)
    return row


# -------------------------------------------------------------- detections + trackers
class Detections:
    """NumPy-backed stand-in for Ultralytics ``Boxes`` (what ``tracker.update`` consumes).

    Exposes ``xyxy``, ``xywh`` (centre x, centre y, w, h), ``conf`` and ``cls`` and supports
    boolean / integer indexing and ``len``.
    """

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray | None = None):
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32).reshape(-1)
        self.cls = (
            np.zeros(len(self.conf), np.float32) if cls is None else np.asarray(cls, np.float32)
        )

    @property
    def xywh(self) -> np.ndarray:
        wh = self.xyxy[:, 2:] - self.xyxy[:, :2]
        return np.concatenate([self.xyxy[:, :2] + wh / 2, wh], axis=1)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, idx) -> Detections:
        return Detections(self.xyxy[idx], self.conf[idx], self.cls[idx])


@dataclass
class DetectionCache:
    """Per-frame detections of a whole sequence (frame index is 1-based)."""

    frame: np.ndarray  # (N,) int
    xyxy: np.ndarray  # (N, 4) float32, pixels
    conf: np.ndarray  # (N,) float32
    meta: dict = field(default_factory=dict)

    def at(self, t: int) -> Detections:
        m = self.frame == t
        return Detections(self.xyxy[m], self.conf[m])

    def save(self, path: str | Path) -> Path:
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, frame=self.frame, xyxy=self.xyxy, conf=self.conf,
                            meta=json.dumps(self.meta))  # fmt: skip
        return path

    @classmethod
    def load(cls, path: str | Path) -> DetectionCache:
        import json

        z = np.load(path, allow_pickle=False)
        return cls(z["frame"], z["xyxy"], z["conf"], json.loads(str(z["meta"])))


def load_tracker_cfg(cfg: str | Path, **overrides) -> dict:
    """Load an Ultralytics tracker YAML (built-in name such as ``bytetrack.yaml`` or a path)."""
    from ultralytics.utils import YAML
    from ultralytics.utils.checks import check_yaml

    d = dict(YAML.load(check_yaml(str(cfg))))
    d.update(overrides)
    return d


def make_tracker(cfg: str | Path | dict, device: str = "cpu", encoder=None, encoder_name: str = "",
                 **overrides):  # fmt: skip
    """Instantiate an Ultralytics tracker (ByteTrack, BoT-SORT, OC-SORT, ...) from a config.

    ``cfg="sort_scratch"`` returns this module's :class:`SimpleSORT` instead.
    ``encoder`` plugs a custom appearance (ReID) encoder into BoT-SORT: any callable
    ``(img_bgr, dets_xywh) -> list of feature vectors``, e.g. ``dinoreid.DinoV3ReID``.
    """
    if cfg == "sort_scratch":
        return SimpleSORT(**overrides)
    from ultralytics.trackers.track import TRACKER_MAP
    from ultralytics.utils import IterableSimpleNamespace

    d = dict(cfg) if isinstance(cfg, dict) else load_tracker_cfg(cfg)
    d.update(overrides)
    if encoder is not None:
        d["with_reid"] = False  # build without Ultralytics' own encoder, attach ours below
        d["model"] = encoder_name or type(encoder).__name__  # anything but "auto": pass frames
    ns = IterableSimpleNamespace(**d)
    ns.device = device
    tracker = TRACKER_MAP[d["tracker_type"]](args=ns)
    if encoder is not None:
        if not hasattr(tracker, "encoder"):
            raise ValueError(f"{d['tracker_type']} does not support an appearance encoder")
        tracker.args.with_reid = True
        tracker.encoder = encoder
    return tracker


def tracker_needs_image(tracker) -> bool:
    """True when ``update`` uses the frame (BoT-SORT GMC or an appearance/ReID encoder)."""
    gmc = getattr(tracker, "gmc", None)
    uses_gmc = gmc is not None and gmc.method is not None
    return uses_gmc or getattr(tracker, "encoder", None) is not None


def run_tracker(
    tracker,
    dets: DetectionCache,
    length: int,
    frame_loader: Callable[[int], np.ndarray] | None = None,
) -> tuple[np.ndarray, dict]:
    """Feed cached detections frame by frame into ``tracker.update``.

    Args:
        tracker: Object with ``update(Detections, img_bgr)`` returning rows
            ``[x1, y1, x2, y2, track_id, score, cls, det_idx]`` (Ultralytics convention).
        dets: Cached detections.
        length: Number of frames.
        frame_loader: ``t -> BGR image``; only called when the tracker needs images.

    Returns:
        MOT rows ``(frame, id, x, y, w, h, score)`` and a timing dict (tracker time only;
        image decoding is excluded).
    """
    need_img = frame_loader is not None and tracker_needs_image(tracker)
    rows, ms = [], []
    for t in range(1, length + 1):
        img = frame_loader(t) if need_img else None
        d = dets.at(t)
        t0 = time.perf_counter()
        out = tracker.update(d, img)
        ms.append((time.perf_counter() - t0) * 1000.0)
        out = np.asarray(out).reshape(-1, out.shape[-1] if np.size(out) else 8)
        for r in out:
            x1, y1, x2, y2, tid, score = r[:6]
            rows.append((t, int(tid), x1, y1, x2 - x1, y2 - y1, score))
    ms = np.asarray(ms)
    timing = {"tracker_ms_per_frame": float(ms.mean()), "tracker_fps": float(1000.0 / ms.mean())}
    out = np.asarray(rows, dtype=np.float64).reshape(-1, 7)
    out[:, 2:6] = out[:, 2:6].round(2)  # same precision as write_mot_txt, so in-memory rows
    out[:, 6] = out[:, 6].round(4)  # and the saved tracker files score identically
    return out, timing


# --------------------------------------------------------------------- SORT from scratch
class KalmanBoxFilter:
    """Constant-velocity Kalman filter on ``(cx, cy, w, h)`` boxes.

    State ``x = [cx, cy, w, h, vcx, vcy, vw, vh]``; measurement ``z = [cx, cy, w, h]``.
    Process/measurement noise scale with the box height (as in DeepSORT/ByteTrack), so the
    filter is equally confident for near (large) and far (small) people.
    """

    std_pos, std_vel = 1.0 / 20, 1.0 / 160

    def __init__(self, z: np.ndarray):
        z = np.asarray(z, dtype=np.float64)
        self.x = np.r_[z, np.zeros(4)]
        h = z[3]
        std = np.r_[2 * self.std_pos * h * np.ones(4), 10 * self.std_vel * h * np.ones(4)]
        self.P = np.diag(std**2)
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)  # x_{k+1} = x_k + v_k * (1 frame)
        self.H = np.eye(4, 8)

    def predict(self) -> np.ndarray:
        h = self.x[3]
        q = np.r_[self.std_pos * h * np.ones(4), self.std_vel * h * np.ones(4)]
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + np.diag(q**2)
        return self.x[:4]

    def update(self, z: np.ndarray) -> None:
        r = self.std_pos * self.x[3] * np.ones(4)
        S = self.H @ self.P @ self.H.T + np.diag(r**2)  # innovation covariance
        K = np.linalg.solve(S, self.H @ self.P).T  # Kalman gain P H^T S^-1 (S symmetric)
        self.x = self.x + K @ (np.asarray(z, dtype=np.float64) - self.H @ self.x)
        self.P = (np.eye(8) - K @ self.H) @ self.P


def _xyxy_to_cxcywh(b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    wh = b[:, 2:] - b[:, :2]
    return np.concatenate([b[:, :2] + wh / 2, wh], axis=1)


def _cxcywh_to_xyxy(b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    return np.concatenate([b[:, :2] - b[:, 2:] / 2, b[:, :2] + b[:, 2:] / 2], axis=1)


class SimpleSORT:
    """Minimal SORT (Bewley et al. 2016): Kalman prediction + IoU Hungarian matching.

    Only detections with ``score >= det_thresh`` are used (no ByteTrack low-score stage,
    no camera-motion compensation, no appearance). Tracks are reported once they have
    ``min_hits`` matches and are deleted after ``max_age`` frames without a match.
    """

    def __init__(self, det_thresh: float = 0.25, iou_thresh: float = 0.3, max_age: int = 30,
                 min_hits: int = 3):  # fmt: skip
        self.det_thresh, self.iou_thresh = det_thresh, iou_thresh
        self.max_age, self.min_hits = max_age, min_hits
        self.tracks: list[dict] = []
        self.frame_id = 0
        self._next_id = 1

    def update(self, dets: Detections, img: np.ndarray | None = None) -> np.ndarray:
        self.frame_id += 1
        keep = dets.conf >= self.det_thresh
        boxes, scores = dets.xyxy[keep].astype(np.float64), dets.conf[keep]
        det_idx = np.flatnonzero(keep)
        # 1) predict every track one frame ahead
        pred = np.array([_cxcywh_to_xyxy(tr["kf"].predict())[0] for tr in self.tracks])
        pred = pred.reshape(-1, 4)
        # 2) associate predictions and detections (maximise IoU, gate at iou_thresh)
        matched_t, matched_d = np.zeros(0, int), np.zeros(0, int)
        if len(pred) and len(boxes):
            iou = box_iou(pred, boxes)
            r, c = linear_sum_assignment(-iou)
            ok = iou[r, c] >= self.iou_thresh
            matched_t, matched_d = r[ok], c[ok]
        for ti, di in zip(matched_t, matched_d):
            tr = self.tracks[ti]
            tr["kf"].update(_xyxy_to_cxcywh(boxes[di])[0])
            tr.update(hits=tr["hits"] + 1, age=0, score=float(scores[di]), idx=int(det_idx[di]))
        for ti in set(range(len(self.tracks))) - set(matched_t.tolist()):
            self.tracks[ti]["age"] += 1
        # 3) unmatched detections start new (tentative) tracks
        for di in set(range(len(boxes))) - set(matched_d.tolist()):
            self.tracks.append({"kf": KalmanBoxFilter(_xyxy_to_cxcywh(boxes[di])[0]), "hits": 1,
                                "age": 0, "id": None, "score": float(scores[di]),
                                "idx": int(det_idx[di])})  # fmt: skip
        # 4) delete stale tracks; report confirmed tracks updated in this frame
        self.tracks = [tr for tr in self.tracks if tr["age"] <= self.max_age]
        out = []
        for tr in self.tracks:
            confirmed = tr["hits"] >= self.min_hits or self.frame_id <= self.min_hits
            if tr["age"] == 0 and confirmed:
                if tr["id"] is None:
                    tr["id"] = self._next_id
                    self._next_id += 1
                x1, y1, x2, y2 = _cxcywh_to_xyxy(tr["kf"].x[:4])[0]
                out.append([x1, y1, x2, y2, tr["id"], tr["score"], 0.0, tr["idx"]])
        return np.asarray(out, dtype=np.float32).reshape(-1, 8)


# -------------------------------------------------------------------------- drawing
def id_color(track_id: int) -> tuple[int, int, int]:
    """A stable, distinct RGB colour per track id (golden-ratio hue walk)."""
    import colorsys

    h = (int(track_id) * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.75, 0.95)
    return int(255 * r), int(255 * g), int(255 * b)


def draw_tracks(
    img: np.ndarray,
    rows: np.ndarray,
    trails: dict[int, list[tuple[float, float]]] | None = None,
    thickness: int = 2,
    font_scale: float | None = None,
) -> np.ndarray:
    """Draw MOT rows ``(frame, id, x, y, w, h, ...)`` of one frame onto a copy of an RGB image.

    ``trails`` maps track id -> list of past foot points to draw as a polyline.
    """
    import cv2

    out = np.ascontiguousarray(img.copy())
    scale = max(out.shape[:2]) / 640
    fs = font_scale or 0.45 * max(scale, 0.8)
    th = max(1, round(thickness * scale))
    if trails:
        for tid, pts in trails.items():
            if len(pts) > 1:
                p = np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(out, [p], False, id_color(tid), th, cv2.LINE_AA)
    for r in np.asarray(rows).reshape(-1, np.shape(rows)[-1] if np.size(rows) else 7):
        tid = int(r[1])
        x, y, w, h = r[2:6]
        c = id_color(tid)
        cv2.rectangle(out, (int(x), int(y)), (int(x + w), int(y + h)), c, th)
        label = str(tid)
        (tw, tht), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        y0 = max(int(y), tht + 4)
        cv2.rectangle(out, (int(x), y0 - tht - 4), (int(x) + tw + 4, y0), c, -1)
        cv2.putText(out, label, (int(x) + 2, y0 - 3), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 1,
                    cv2.LINE_AA)  # fmt: skip
    return out
