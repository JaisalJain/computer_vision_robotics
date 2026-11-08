#!/usr/bin/env python
"""COCO-subset benchmarks for project 05 (results -> ``projects/05_segmentation/results``).

Two tasks on Ultralytics' coco128-seg (128 COCO train2017 images, YOLO polygon labels):

* ``--task prompts`` - promptable segmentation with SAM 3 and SAM 2.1 (tiny/small/base-plus):
  one interior click (top-scored mask and best-of-3), a tight GT box, and a simulated
  interactive session of 5 error-driven clicks per instance. Also times the image encoder
  and the prompt decoder and records peak VRAM.
* ``--task text`` - text-prompted instance segmentation with the image's GT class names
  as prompts ("oracle vocabulary"): SAM 3 concept prompts vs Grounding DINO tiny + SAM 2.1-S
  (Grounded-SAM) vs YOLO26n/s-seg (closed-set, predictions filtered to the same names).
  Mask AP (COCO-style) plus a hallucination test with 3 absent class names per image.

Every model run is timed with CUDA synchronisation; peak VRAM comes from
``torch.cuda.max_memory_allocated``. Run under the GPU lock::

    flock /tmp/cv-gpu.lock python projects/05_segmentation/scripts/benchmark_coco.py --task prompts
    flock /tmp/cv-gpu.lock python projects/05_segmentation/scripts/benchmark_coco.py --task text
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import segkit as sk  # noqa: E402

from cvlab.device import env_summary, peak_vram_mb, reset_peak_vram, seed_everything  # noqa: E402
from cvlab.io import save_json, weights_dir  # noqa: E402

N_CLICKS = 5
MIN_AREA = 100  # px; drop annotation specks
SAM_THR_AP, SAM_THR_OP = 0.05, 0.5  # SAM 3 score threshold for AP / at the operating point
GD_THR_AP, GD_THR_OP = (0.10, 0.10), (0.35, 0.25)  # Grounding DINO (box, text) thresholds
N_ABSENT = 3


def _sync() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed(fn, *args, **kwargs):
    _sync()
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    _sync()
    return out, 1000 * (time.perf_counter() - t0)


def size_bucket(area: int) -> str:
    """COCO object-size buckets (pixel area)."""
    return "small" if area < 32**2 else "medium" if area < 96**2 else "large"


# ================================================================================ prompts
def run_prompts(dataset: str, model_keys: list[str]) -> None:
    import torch

    samples = sk.load_coco_seg(dataset, splits=("train2017",), min_area=MIN_AREA)
    ids = {"sam3": sk.SAM3_ID, **{f"sam2.1-{k}": v for k, v in sk.SAM2_IDS.items()}}
    summary, rows = {}, []
    for key in model_keys:
        mid = ids[key]
        torch.cuda.empty_cache()
        reset_peak_vram()
        seg, load_ms = _timed(sk.PromptableSegmenter, mid)
        weights_mb = torch.cuda.memory_allocated() / 2**20
        enc_ms, dec_ms = [], []
        for si, s in enumerate(samples):
            _, ms = _timed(seg.set_image, s.image)
            if si > 0:  # the first call includes CUDA warm-up
                enc_ms.append(ms)
            for oi, g in enumerate(s.masks):
                (masks, scores, _), ms = _timed(seg.predict, [[sk.interior_point(g)]], [[1]])
                dec_ms.append(ms)
                ious = [sk.j_measure(m, g) for m in masks[0]]
                top = int(np.argmax(scores[0]))
                (bmask, _, _), _ = _timed(seg.predict, boxes=[sk.mask_to_box(g)], multimask=False)
                curve = seg.interactive(g, N_CLICKS)
                area = int(g.sum())
                rows.append(
                    {
                        "model": seg.label,
                        "image": s.name,
                        "obj": oi,
                        "cls": int(s.classes[oi]),
                        "area": area,
                        "size": size_bucket(area),
                        "click_top": round(ious[top], 4),
                        "click_oracle": round(max(ious), 4),
                        "top_idx": top,
                        "box": round(sk.j_measure(bmask[0, 0], g), 4),
                        **{f"click{k + 1}": round(v, 4) for k, v in enumerate(curve)},
                    }
                )
        mine = [r for r in rows if r["model"] == seg.label]
        curve = np.array([[r[f"click{k + 1}"] for k in range(N_CLICKS)] for r in mine])

        def noc(thr: float, c: np.ndarray = curve) -> float:
            # clicks needed to reach thr (N_CLICKS + 1 when never reached), averaged
            hit = c >= thr
            return float(np.where(hit.any(1), hit.argmax(1) + 1, N_CLICKS + 1).mean())

        summary[seg.label] = {
            "model_id": mid,
            "params_M": round(seg.n_params_m, 1),
            "dtype": "bfloat16",
            "load_s": round(load_ms / 1000, 1),
            "weights_vram_MiB": round(weights_mb, 0),
            "peak_vram_MiB": round(peak_vram_mb(), 0),
            "encoder_ms": round(float(np.median(enc_ms)), 1),
            "decoder_ms": round(float(np.median(dec_ms)), 2),
            "n_instances": len(mine),
            "mIoU_click_top": round(float(np.mean([r["click_top"] for r in mine])), 4),
            "mIoU_click_oracle": round(float(np.mean([r["click_oracle"] for r in mine])), 4),
            "mIoU_box": round(float(np.mean([r["box"] for r in mine])), 4),
            "mIoU_per_click": [round(float(v), 4) for v in curve.mean(0)],
            "NoC@85": round(noc(0.85), 2),
            "NoC@90": round(noc(0.90), 2),
        }
        print(key, summary[seg.label], flush=True)
        del seg
    out = sk.results_dir()
    meta = {
        "dataset": dataset,
        "n_images": len(samples),
        "min_area_px": MIN_AREA,
        "n_clicks": N_CLICKS,
        "env": env_summary(),
    }
    save_json({"meta": meta, "models": summary}, out / f"prompts_{dataset}.json")
    with open(out / f"prompts_{dataset}_instances.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote results/prompts_{dataset}.json and results/prompts_{dataset}_instances.csv")


# =================================================================================== text
def absent_classes(sample: sk.InstanceSample, rng: np.random.Generator) -> list[int]:
    present = set(sample.classes.tolist())
    pool = [c for c in range(80) if c not in present]
    return [int(c) for c in rng.choice(pool, size=N_ABSENT, replace=False)]


def run_text(dataset: str) -> None:
    import torch

    names = sk.coco_names()
    samples = sk.load_coco_seg(dataset, splits=("train2017",), min_area=0)
    rng = np.random.default_rng(0)
    absent = [absent_classes(s, rng) for s in samples]
    results: dict[str, dict] = {}

    def finish(label: str, preds, lat_ms, extra) -> None:
        ap = sk.mask_ap(preds, samples)
        results[label] = {
            **{k: round(v, 4) if isinstance(v, float) else v for k, v in ap.items()},
            "ms_per_prompt": round(float(np.median(lat_ms)), 1),
            "peak_vram_MiB": round(peak_vram_mb(), 0),
            **extra,
        }
        print(label, results[label], flush=True)

    # ---- SAM 3 concept prompts
    torch.cuda.empty_cache()
    reset_peak_vram()
    cs = sk.ConceptSegmenter()
    preds, lat, fp_counts, img_ms = [], [], [], []
    for si, s in enumerate(samples):
        h, w = s.image.shape[:2]
        _, ms = _timed(cs.set_image, s.image)
        if si > 0:
            img_ms.append(ms)
        dets = []
        for c in sorted(set(s.classes.tolist())):
            r, ms = _timed(cs.segment, names[c], threshold=SAM_THR_AP)
            if si > 0:
                lat.append(ms)
            dets.append(sk.Detections(r["masks"], r["scores"], np.full(len(r["scores"]), c)))
        preds.append(sk.Detections.concat(dets, h, w))
        for c in absent[si]:
            r = cs.segment(names[c], threshold=SAM_THR_OP)
            fp_counts.append(len(r["scores"]))
    finish(
        "SAM 3 (text)",
        preds,
        lat,
        {
            "params_M": round(cs.n_params_m, 1),
            "image_encoder_ms": round(float(np.median(img_ms)), 1),
            "absent_fp_per_query": round(float(np.mean(fp_counts)), 3),
            "absent_query_hit_rate": round(float(np.mean(np.array(fp_counts) > 0)), 3),
            "threshold_ap": SAM_THR_AP,
            "threshold_op": SAM_THR_OP,
        },
    )
    del cs

    # ---- Grounding DINO tiny -> SAM 2.1-S boxes
    torch.cuda.empty_cache()
    reset_peak_vram()
    gs = sk.GroundedSam(sk.SAM2_IDS["small"])
    preds, lat, fp_counts = [], [], []
    for si, s in enumerate(samples):
        h, w = s.image.shape[:2]
        gs.set_image(s.image)
        dets = []
        for c in sorted(set(s.classes.tolist())):
            r, ms = _timed(gs.segment, names[c], *GD_THR_AP)
            if si > 0:
                lat.append(ms)
            dets.append(sk.Detections(r["masks"], r["scores"], np.full(len(r["scores"]), c)))
        preds.append(sk.Detections.concat(dets, h, w))
        for c in absent[si]:
            r = gs.detector.detect(s.image, names[c], *GD_THR_OP)
            fp_counts.append(len(r["scores"]))
    finish(
        "Grounding DINO-T + SAM 2.1-S",
        preds,
        lat,
        {
            "params_M": round(gs.detector.n_params_m + gs.sam.n_params_m, 1),
            "absent_fp_per_query": round(float(np.mean(fp_counts)), 3),
            "absent_query_hit_rate": round(float(np.mean(np.array(fp_counts) > 0)), 3),
            "threshold_ap": list(GD_THR_AP),
            "threshold_op": list(GD_THR_OP),
        },
    )
    del gs

    # ---- YOLO26-seg (closed set): keep only the image's GT class names, like the prompts
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    from ultralytics import YOLO

    for size in ("n", "s"):
        torch.cuda.empty_cache()
        reset_peak_vram()
        model = YOLO(str(weights_dir("ultralytics", f"yolo26{size}-seg.pt")))
        preds, lat = [], []
        for si, s in enumerate(samples):
            h, w = s.image.shape[:2]
            res, ms = _timed(
                model.predict,
                np.ascontiguousarray(s.image[..., ::-1]),  # Ultralytics expects BGR arrays
                conf=0.001,
                max_det=300,
                retina_masks=True,
                verbose=False,
            )
            if si > 0:
                lat.append(ms)
            r = res[0]
            if r.masks is None:
                preds.append(sk.Detections.empty(h, w))
                continue
            cls = r.boxes.cls.cpu().numpy().astype(int)
            keep = np.isin(cls, s.classes)
            preds.append(
                sk.Detections(
                    r.masks.data.cpu().numpy()[keep] > 0.5,
                    r.boxes.conf.cpu().numpy()[keep],
                    cls[keep],
                )
            )
        n_params = sum(p.numel() for p in model.model.parameters()) / 1e6
        finish(
            f"YOLO26{size}-seg (closed set)",
            preds,
            lat,
            {"params_M": round(n_params, 1), "note": "ms_per_prompt = ms per image (all classes)"},
        )
        del model

    meta = {
        "dataset": dataset,
        "n_images": len(samples),
        "protocol": "prompts = GT class names of each image (oracle vocabulary); "
        "YOLO predictions filtered to the same names",
        "n_absent_per_image": N_ABSENT,
        "env": env_summary(),
    }
    save_json({"meta": meta, "methods": results}, sk.results_dir() / f"text_{dataset}.json")
    print(f"wrote results/text_{dataset}.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", choices=["prompts", "text"], required=True)
    ap.add_argument("--dataset", default="coco128-seg")
    ap.add_argument(
        "--models",
        nargs="+",
        default=["sam3", "sam2.1-tiny", "sam2.1-small", "sam2.1-base-plus"],
        help="prompt task only",
    )
    args = ap.parse_args()
    seed_everything(0)
    import warnings

    warnings.filterwarnings("ignore")
    if args.task == "prompts":
        run_prompts(args.dataset, args.models)
    else:
        run_text(args.dataset)


if __name__ == "__main__":
    main()
