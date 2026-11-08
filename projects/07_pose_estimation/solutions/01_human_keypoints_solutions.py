#!/usr/bin/env python
"""Solutions to the exercises of notebook 01 (human keypoints with YOLO26-pose).

Uses the GPU, so run it under the repository's GPU lock (from the repository root)::

    flock /tmp/cv-gpu.lock python \
        projects/07_pose_estimation/solutions/01_human_keypoints_solutions.py
"""

from __future__ import annotations

import os

os.environ["YOLO_AUTOINSTALL"] = "false"  # never pip-install at runtime

import logging
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
from ultralytics.utils import ASSETS, LOGGER

from cvlab import io as cvio
from cvlab.metrics import average_precision, precision_recall

LOGGER.setLevel(logging.ERROR)
PROJECT = "07_pose_estimation"
sys.path.insert(0, str(cvio.project_dir(PROJECT) / "scripts"))
import keypoints as kp  # noqa: E402

COCO8 = cvio.data_dir("ultralytics", "coco8-pose")
WEIGHTS = cvio.weights_dir("ultralytics")


def load_split(split: str) -> list[dict]:
    """Images + YOLO-pose labels of one COCO8-Pose split."""
    out = []
    for f in sorted((COCO8 / "images" / split).glob("*.jpg")):
        img = cvio.read_image(f)
        h, w = img.shape[:2]
        boxes, kpts, areas = kp.load_yolo_pose_labels(
            COCO8 / "labels" / split / f"{f.stem}.txt", w, h
        )
        out.append({"name": f.name, "img": img, "kpts": kpts, "areas": areas})
    return out


def predict(model, img, imgsz=640, conf=0.25):
    """Keypoints (N,17,2), keypoint confidences (N,17) and box scores (N,) for one RGB image."""
    r = model.predict(img[..., ::-1], imgsz=imgsz, conf=conf, quantize=16, verbose=False)[0]
    return r.keypoints.xy.cpu().numpy(), r.keypoints.conf.cpu().numpy(), r.boxes.conf.cpu().numpy()


# ------------------------------------------------------------------ Exercise 1: pose AP by hand
def pose_ap(model, val, thr: float) -> float:
    """COCO-style AP at one OKS threshold (score-ordered greedy matching, 101-point interp.)."""
    scores, tps, n_gt = [], [], 0
    for v in val:
        k, _, sc = predict(model, v["img"], conf=0.001)  # low threshold, as in model.val()
        oks = np.nan_to_num(kp.oks_matrix(v["kpts"], v["areas"], k))  # (n_gt, n_pred)
        taken = np.zeros(len(v["kpts"]), dtype=bool)
        for j in np.argsort(-sc, kind="stable"):
            cand = np.where(~taken, oks[:, j], -1.0)
            i = int(np.argmax(cand))
            tp = bool(cand[i] >= thr)
            taken[i] |= tp
            scores.append(sc[j])
            tps.append(tp)
        n_gt += len(v["kpts"])
    precision, recall = precision_recall(np.array(scores), np.array(tps), n_gt)
    return average_precision(recall, precision, method="coco")


def exercise1(models, val):
    print("Exercise 1 - pose AP from our own OKS matching vs Ultralytics' model.val()")
    for name, model in models.items():
        ap50 = pose_ap(model, val, 0.5)
        ap = np.mean([pose_ap(model, val, t) for t in np.arange(0.5, 0.96, 0.05)])
        m = model.val(data="coco8-pose.yaml", imgsz=640, batch=4, quantize=16, plots=False,
                      verbose=False, project=str(cvio.runs_dir(PROJECT)), name=f"sol_{name}",
                      exist_ok=True)  # fmt: skip
        print(
            f"  {name}: ours AP50 {ap50:.3f} / AP50-95 {ap:.3f} | "
            f"ultralytics mAP50 {m.pose.map50:.3f} / mAP50-95 {m.pose.map:.3f}"
        )
    print("  (small differences come from the matching rule: COCO matches predictions in score")
    print("   order, Ultralytics matches by descending OKS per threshold)")


# ------------------------------------------------------------- Exercise 2: flip test-time aug
def predict_flip_tta(model, img, imgsz):
    """Average the keypoints of the original and the mirrored image (confidence-weighted)."""
    k1, c1, _ = predict(model, img, imgsz)
    k2, c2, _ = predict(model, np.ascontiguousarray(img[:, ::-1]), imgsz)
    if len(k1) == 0 or len(k2) == 0:
        return k1, c1
    w = img.shape[1]
    k2 = k2[:, kp.FLIP_IDX].copy()  # left <-> right joints
    c2 = c2[:, kp.FLIP_IDX]
    k2[..., 0] = w - k2[..., 0]  # mirror x back (continuous coordinates: image spans [0, W])
    cost = np.linalg.norm(k1[:, None] - k2[None], axis=-1).mean(-1)  # mean keypoint distance
    rows, cols = linear_sum_assignment(cost)
    out_k, out_c = k1.copy(), c1.copy()
    for i, j in zip(rows, cols):
        wa, wb = c1[i][:, None], c2[j][:, None]
        out_k[i] = (wa * k1[i] + wb * k2[j]) / np.maximum(wa + wb, 1e-6)
        out_c[i] = (c1[i] + c2[j]) / 2
    return out_k, out_c


def exercise2(models, val):
    print("Exercise 2 - flip test-time augmentation (mean OKS over the 14 val people)")
    for name, model in models.items():
        for imgsz in (320, 640):
            plain, tta = [], []
            for v in val:
                plain += list(
                    kp.match_by_oks(v["kpts"], v["areas"], predict(model, v["img"], imgsz)[0])
                )
                tta += list(
                    kp.match_by_oks(
                        v["kpts"], v["areas"], predict_flip_tta(model, v["img"], imgsz)[0]
                    )
                )
            print(
                f"  {name} @ {imgsz}px: plain {np.mean(plain):.3f} -> flip-TTA {np.mean(tta):.3f}"
            )


# ---------------------------------------------------------------- Exercise 3: stop gesture
def stop_gesture(k: np.ndarray, c: np.ndarray, thr: float = 0.5, min_elbow: float = 120.0):
    """True/False when both arms are visible, 'unknown' otherwise (COCO-17 keypoints)."""
    if min(c[[5, 6, 7, 8, 9, 10]]) < thr:
        return "unknown"
    wrists_up = k[9, 1] < k[5, 1] and k[10, 1] < k[6, 1]  # image y points down
    straight = (kp.joint_angle_deg(k[5], k[7], k[9]) > min_elbow
                and kp.joint_angle_deg(k[6], k[8], k[10]) > min_elbow)  # fmt: skip
    return bool(wrists_up and straight)


def exercise3(model):
    print("Exercise 3 - stop gesture on the 8 COCO8-Pose images + 2 samples (YOLO26s-pose)")
    images = {v["name"]: v["img"] for v in load_split("train") + load_split("val")}
    images |= {n: cvio.read_image(ASSETS / n) for n in ("bus.jpg", "zidane.jpg")}
    counts = {True: 0, False: 0, "unknown": 0}
    for name, img in images.items():
        k, c, _ = predict(model, img)
        res = [stop_gesture(k[i], c[i]) for i in range(len(k))]
        for r in res:
            counts[r] += 1
        print(f"  {name:18s} {len(k)} people -> {res}")
    print(f"  totals: stop={counts[True]}, no stop={counts[False]}, unknown={counts['unknown']}")


def main() -> None:
    names = {"YOLO26n-pose": "yolo26n-pose.pt", "YOLO26s-pose": "yolo26s-pose.pt"}
    models = {n: YOLO(str(WEIGHTS / f)) for n, f in names.items()}
    val = load_split("val")
    exercise1(models, val)
    exercise2(models, val)
    exercise3(models["YOLO26s-pose"])


if __name__ == "__main__":
    main()
