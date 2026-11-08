#!/usr/bin/env python
"""Fine-tune a COCO-pretrained YOLO26 detector on a small Ultralytics dataset.

Artifacts (checkpoints, Ultralytics plots) go to ``runs/03_object_detection/<name>/``
(gitignored); a small summary JSON and the per-epoch ``results.csv`` are copied to
``projects/03_object_detection/results/`` so the notebooks can plot them.

Examples::

    python projects/03_object_detection/scripts/train_yolo.py --model yolo26n.pt \
        --data african-wildlife.yaml --epochs 40 --name yolo26n_wildlife
    python projects/03_object_detection/scripts/train_yolo.py --model yolo26n.pt \
        --data medical-pills.yaml --epochs 30 --name yolo26n_pills

Run it under the GPU lock (``flock /tmp/cv-gpu.lock python ...``). The model name is a
plain argument, so a newer family (e.g. ``yolo27n.pt`` once released) is a drop-in swap.
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path

os.environ.setdefault("YOLO_AUTOINSTALL", "false")  # never let Ultralytics pip-install anything

import torch  # noqa: E402

from cvlab.device import env_summary, reset_peak_vram  # noqa: E402
from cvlab.io import project_dir, repo_root, runs_dir, save_json  # noqa: E402

PROJECT = "03_object_detection"


def rel(path: Path | str) -> str:
    """Repo-relative path (results files must never contain absolute paths)."""
    try:
        return str(Path(path).resolve().relative_to(repo_root()))
    except ValueError:
        return Path(path).name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="yolo26n.pt", help="pretrained checkpoint name or path")
    ap.add_argument("--data", default="african-wildlife.yaml", help="Ultralytics dataset yaml")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None, help="run name (default: <model>_<data>)")
    args = ap.parse_args()

    from ultralytics import YOLO
    from ultralytics.utils import LOGGER

    name = args.name or f"{Path(args.model).stem}_{Path(args.data).stem}"
    project = runs_dir(PROJECT, create=True)
    model = YOLO(args.model)

    reset_peak_vram()
    t0 = time.perf_counter()
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        seed=args.seed,
        project=str(project),
        name=name,
        exist_ok=True,
        plots=True,
    )
    train_min = (time.perf_counter() - t0) / 60.0
    peak_mb = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0

    run = project / name
    best = run / "weights" / "best.pt"
    out = project_dir(PROJECT) / "results"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run / "results.csv", out / f"{name}_results.csv")
    trainer = model.trainer
    summary = {
        "name": name,
        "model": Path(args.model).name,
        "data": Path(args.data).name,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "seed": args.seed,
        "train_minutes": round(train_min, 2),
        "peak_vram_mb": round(peak_mb, 1),
        "best_checkpoint": rel(best),
        "best_fitness": float(trainer.best_fitness) if trainer.best_fitness is not None else None,
        "final_val_metrics": {k: float(v) for k, v in (trainer.metrics or {}).items()},
        "env": env_summary(),
    }
    save_json(summary, out / f"{name}_summary.json")
    LOGGER.info(f"done in {train_min:.1f} min, peak VRAM {peak_mb:.0f} MiB -> {rel(best)}")


if __name__ == "__main__":
    main()
