#!/usr/bin/env python
"""Evaluate closed-set and open-vocabulary detectors with COCO-style AP (project 04).

Datasets (see ``configs/eval.yaml``):

* ``coco``     - fixed-seed 200-image subset of COCO val2017, vocabulary = 80 COCO names.
* ``wildlife`` - Ultralytics african-wildlife val split (225 images), vocabulary =
  buffalo / elephant / rhino / zebra. Buffalo and rhino are not COCO classes, so the
  closed-set COCO detector cannot name them.

For every (dataset, model) the script writes

* ``runs/04_open_vocab_detection/<dataset>/<model>_dets.json`` (all detections, COCO format),
* ``projects/04_open_vocab_detection/results/<dataset>_<model>.json`` (AP, per-class AP,
  PR curve, F1-vs-threshold curve, latency, peak VRAM, a few showcase detections).

Usage::

    python projects/04_open_vocab_detection/scripts/eval_detectors.py --dataset coco
    python projects/04_open_vocab_detection/scripts/eval_detectors.py --dataset wildlife
    python projects/04_open_vocab_detection/scripts/eval_detectors.py --prompt-study
    # options: --models yoloe26s gdino_t   --max-images 20   --skip-existing
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ovd  # noqa: E402

from cvlab.bench import benchmark  # noqa: E402
from cvlab.device import env_summary, seed_everything  # noqa: E402
from cvlab.io import project_dir, read_image, repo_root, runs_dir, save_json  # noqa: E402

PROJECT = "04_open_vocab_detection"
CFG_PATH = project_dir(PROJECT) / "configs" / "eval.yaml"
RESULTS = project_dir(PROJECT) / "results"
SHOWCASE = {"coco": 4, "wildlife": 4}  # number of images whose detections are kept in results


def rel(p: Path) -> str:
    return str(Path(p).resolve().relative_to(repo_root()))


def load_dataset(name: str, cfg: dict, max_images: int | None):
    """Return (coco_gt, [(image_id, path)], names, cat_ids)."""
    d = cfg["datasets"][name]
    root = repo_root()
    if name == "coco":
        gt = json.loads((root / d["annotations"]).read_text())
        img_dir = root / d["images"]
    else:
        gt = ovd.yolo_labels_to_coco(root / d["images"], root / d["labels"], d["names"])
        img_dir = root / d["images"]
    coco_gt = ovd.coco_from_dict(gt)
    cat_ids = sorted(coco_gt.getCatIds())
    names = [coco_gt.cats[c]["name"] for c in cat_ids]
    images = sorted((im["id"], img_dir / im["file_name"]) for im in gt["images"])
    if max_images:
        images = images[:max_images]
    return coco_gt, images, names, cat_ids


def dtype_for(key: str, cfg: dict) -> str:
    return cfg["precision"].get(key, cfg["precision"]["default"])


def build(key: str, cfg: dict, template: str | None = None) -> ovd.Detector:
    kw = {}
    if key not in ("yolo26s", "yoloe26s"):
        kw["dtype"] = dtype_for(key, cfg)
    return ovd.load_detector(key, "cuda", template=template, **kw)


def run_model(key: str, dataset: str, cfg: dict, max_images: int | None) -> dict:
    import torch

    coco_gt, images, names, cat_ids = load_dataset(dataset, cfg, max_images)
    ev = cfg["eval"]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t_load = time.perf_counter()
    det = build(key, cfg)
    det.set_vocabulary(names)
    load_s = time.perf_counter() - t_load
    torch.cuda.synchronize()
    weights_mb = torch.cuda.memory_allocated() / 2**20
    torch.cuda.reset_peak_memory_stats()

    results, showcase, per_img_ms = [], {}, []
    t0 = time.perf_counter()
    for k, (img_id, path) in enumerate(images):
        img = read_image(path)
        t = time.perf_counter()
        dets = det.predict(img, min_score=ev["min_score"], max_det=ev["max_det"])
        torch.cuda.synchronize()
        per_img_ms.append((time.perf_counter() - t) * 1000)
        results += ovd.detections_to_coco(dets, img_id, cat_ids)
        if k < SHOWCASE[dataset]:
            top = dets.above(0.1).topk(30)
            showcase[str(img_id)] = {"file": path.name, "boxes": np.round(top.boxes, 1).tolist(),
                                     "scores": np.round(top.scores, 4).tolist(),
                                     "labels": top.label_names}  # fmt: skip
        if (k + 1) % 50 == 0:
            print(f"  {key}: {k + 1}/{len(images)} images", flush=True)
    eval_s = time.perf_counter() - t0
    peak_mb = torch.cuda.max_memory_allocated() / 2**20

    # Latency on a fixed image with the full vocabulary (end-to-end predict call,
    # including pre/post-processing; text embeddings are cached where the model allows).
    img0 = read_image(images[0][1])
    lat = benchmark(det.predict, img0, min_score=ev["min_score"], max_det=ev["max_det"],
                    warmup=cfg["latency"]["warmup"], iters=cfg["latency"]["iters"])  # fmt: skip

    # raw detections (gitignored); quick --max-images runs go to runs/<project>/debug/
    if max_images:
        dets_path = runs_dir(PROJECT, "debug", create=True) / f"{dataset}_{key}_dets.json"
    else:
        dets_path = runs_dir(PROJECT, dataset, create=True) / f"{key}_dets.json"
    dets_path.write_text(json.dumps(results))
    metrics = ovd.coco_evaluate(coco_gt, results, [i for i, _ in images])
    f1 = ovd.f1_vs_threshold(coco_gt, results, np.round(np.arange(0.05, 0.96, 0.05), 2),
                             iou_thr=ev["f1_iou"])  # fmt: skip
    info = env_summary()
    record = {
        "model": key, "label": det.label, "kind": det.kind, "dataset": dataset,
        "n_images": len(images), "vocab_size": len(names), "dtype": dtype_for(key, cfg)
        if key not in ("yolo26s", "yoloe26s") else "float16 (half=True)",
        "prompts": det.prompts(names)[:3] + ["..."],
        "metrics": {k: v for k, v in metrics.items() if not isinstance(v, (dict, list))},
        "per_class_AP": metrics["per_class_AP"], "per_class_AP50": metrics["per_class_AP50"],
        "pr_curve_iou50": metrics["pr_curve_iou50"], "f1_curve": f1,
        "latency_ms": {k: round(v, 2) for k, v in lat.items()},
        "eval_mean_ms_per_image": round(float(np.mean(per_img_ms)), 2),
        "weights_vram_mb": round(weights_mb, 1), "peak_vram_mb": round(peak_mb, 1),
        "load_s": round(load_s, 1), "eval_s": round(eval_s, 1),
        "n_detections": len(results), "showcase": showcase,
        "hardware": f"{info.get('gpu')} ({info.get('vram_gb')} GB)",
        "torch": info.get("torch"),
    }  # fmt: skip
    if dataset == "wildlife" and key == "yolo26s":
        record["closed_set_confusion"] = closed_set_confusion(det, coco_gt, images, cfg)
    ovd.free_detector(det)
    return record


def closed_set_confusion(det: ovd.Detector, coco_gt, images, cfg: dict) -> dict:
    """What does the COCO-only detector call the animals it has no class for?

    Runs YOLO26 with its full 80-class vocabulary and, for every ground-truth box of a
    wildlife class, records the label of the best-overlapping prediction (IoU >= 0.5,
    score >= 0.25) or 'missed'.
    """
    from collections import Counter

    from cvlab.metrics import box_iou

    det.set_vocabulary(ovd.COCO_NAMES)
    counts: dict[str, Counter] = {}
    for img_id, path in images:
        d = det.predict(read_image(path), min_score=0.25, max_det=100)
        for a in coco_gt.imgToAnns[img_id]:
            name = coco_gt.cats[a["category_id"]]["name"]
            x, y, w, h = a["bbox"]
            label = "missed"
            if len(d):
                iou = box_iou(np.array([[x, y, x + w, y + h]]), d.boxes)[0]
                j = int(np.argmax(iou))
                if iou[j] >= 0.5:
                    label = d.label_names[j]
            counts.setdefault(name, Counter())[label] += 1
    return {k: dict(v.most_common()) for k, v in counts.items()}


def prompt_study(cfg: dict, max_images: int | None, only: list[str] | None) -> None:
    """AP on african-wildlife for several wordings of the same four classes."""
    import torch

    ps = cfg["prompt_study"]
    coco_gt, images, names, cat_ids = load_dataset("wildlife", cfg, max_images)
    path = RESULTS / "wildlife_prompt_study.json"
    study = json.loads(path.read_text()) if path.exists() else {}
    for key in only or ps["models"]:
        det = build(key, cfg, template="{}")  # prompts are given verbatim here
        study[key] = {"label": det.label}
        for variant, prompts in ps["variants"].items():
            det.set_vocabulary(names, prompts=prompts)
            results = []
            for img_id, p in images:
                d = det.predict(read_image(p), min_score=cfg["eval"]["min_score"],
                                max_det=cfg["eval"]["max_det"])  # fmt: skip
                results += ovd.detections_to_coco(d, img_id, cat_ids)
            m = ovd.coco_evaluate(coco_gt, results, [i for i, _ in images])
            study[key][variant] = {"prompts": prompts, "AP": m["AP"], "AP50": m["AP50"],
                                   "per_class_AP": m["per_class_AP"]}  # fmt: skip
            print(f"  {key:9s} {variant:15s} AP={m['AP']:.3f} AP50={m['AP50']:.3f}", flush=True)
        ovd.free_detector(det)
        torch.cuda.empty_cache()
        save_json(study, path)
    print(f"saved {rel(path)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", choices=["coco", "wildlife"], default="coco")
    ap.add_argument("--models", nargs="*", help="subset of configs/eval.yaml models")
    ap.add_argument("--max-images", type=int, default=None, help="debug: first N images only")
    ap.add_argument("--skip-existing", action="store_true", help="skip models with results")
    ap.add_argument("--prompt-study", action="store_true", help="run the prompt-wording study")
    args = ap.parse_args()

    cfg = yaml.safe_load(CFG_PATH.read_text())
    seed_everything(0)
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.prompt_study:
        prompt_study(cfg, args.max_images, args.models)
        return
    failed: list[str] = []
    for key in args.models or cfg["models"]:
        out = RESULTS / f"{args.dataset}_{key}.json"
        if args.skip_existing and out.exists():
            print(f"[skip] {rel(out)}")
            continue
        print(f"[{args.dataset}] {key} ...", flush=True)
        try:
            rec = run_model(key, args.dataset, cfg, args.max_images)
        except Exception:  # noqa: BLE001 - report and continue with the next model
            traceback.print_exc()
            print(f"  FAILED: {key}", flush=True)
            failed.append(key)
            continue
        if args.max_images:
            out = runs_dir(PROJECT, "debug", create=True) / out.name  # never overwrite results
        save_json(rec, out)
        m = rec["metrics"]
        print(f"  AP={m['AP']:.3f} AP50={m['AP50']:.3f} | {rec['latency_ms']['mean_ms']:.1f} ms/img"
              f" | peak {rec['peak_vram_mb']:.0f} MiB | eval {rec['eval_s']:.0f} s -> {rel(out)}",
              flush=True)  # fmt: skip
    if failed:
        sys.exit(f"failed models: {failed}")


if __name__ == "__main__":
    main()
