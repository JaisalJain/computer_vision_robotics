#!/usr/bin/env python
"""Linear probe vs full fine-tuning of small timm backbones on Imagenette-160.

For every (backbone, mode) pair this script trains a 10-class head (``linear_probe``:
frozen trunk; ``finetune``: everything trainable) with bf16 autocast, evaluates on the
validation split after every epoch and writes

* ``runs/02_representation_learning/transfer/<backbone>__<mode>.pt`` - final weights
  (gitignored, loaded by the notebooks for confusion matrices and Grad-CAM),
* ``projects/02_representation_learning/results/transfer/<backbone>__<mode>.json`` -
  per-epoch curves, final accuracy, wall time, throughput and peak VRAM,
* ``projects/02_representation_learning/results/transfer_summary.csv`` - one row per run.

Usage (GPU; hold the GPU lock)::

    flock /tmp/cv-gpu.lock python projects/02_representation_learning/scripts/train_transfer.py
    # a subset / quick smoke test:
    python projects/02_representation_learning/scripts/train_transfer.py \
        --models resnet18.a1_in1k --modes linear_probe --epochs 1 --max-steps 20 --dry-run
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import replearn as rl  # noqa: E402

from cvlab.device import env_summary, peak_vram_mb, reset_peak_vram, seed_everything  # noqa: E402
from cvlab.io import project_dir, repo_root, runs_dir, save_json  # noqa: E402

RESULTS = project_dir(rl.PROJECT) / "results"


def run_one(name: str, mode: str, args: argparse.Namespace, device: torch.device) -> dict:
    """Train + evaluate one (backbone, mode) pair and return the result record."""
    seed_everything(args.seed)
    train_ds = rl.make_dataset("train", rl.build_transform(True, args.img_size))
    val_ds = rl.make_dataset("val", rl.build_transform(False, args.img_size))
    train_dl = rl.make_loader(train_ds, args.batch_size, train=True, workers=args.workers,
                              seed=args.seed)  # fmt: skip
    val_dl = rl.make_loader(val_ds, 256, train=False, workers=args.workers)

    model = rl.build_model(name, mode).to(device).to(memory_format=torch.channels_last)
    total, trainable = rl.count_params(model)
    lr_head = args.lr_head_lp if mode == "linear_probe" else args.lr_head_ft
    opt = rl.make_optimizer(model, mode, lr_head, args.lr_backbone, args.weight_decay)
    steps_per_epoch = len(train_dl) if args.max_steps is None else args.max_steps
    total_steps = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, rl.cosine_warmup(total_steps, warmup_steps=steps_per_epoch // 2)
    )

    history = []
    train_peak = eval_peak = 0.0
    t_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        # Peak VRAM is tracked separately for training steps and for evaluation.
        reset_peak_vram()
        tr = rl.train_one_epoch(model, train_dl, opt, sched, device, mode,
                                max_steps=args.max_steps)  # fmt: skip
        train_peak = max(train_peak, peak_vram_mb())
        reset_peak_vram()
        ev = rl.evaluate(model, val_dl, device)
        eval_peak = max(eval_peak, peak_vram_mb())
        rec = {
            "epoch": epoch,
            "train_loss": tr["loss"],
            "train_acc": tr["acc"],
            "val_loss": ev["loss"],
            "val_acc": ev["acc"],
            "epoch_seconds": tr["seconds"],
            "train_imgs_per_s": tr["imgs_per_s"],
        }
        history.append(rec)
        print(f"  [{name} | {mode}] epoch {epoch}/{args.epochs} "
              f"train_loss {tr['loss']:.3f} train_acc {tr['acc']:.3f} "
              f"val_acc {ev['acc']:.4f} ({tr['seconds']:.1f} s, {tr['imgs_per_s']:.0f} img/s)",
              flush=True)  # fmt: skip
    train_time = time.perf_counter() - t_start

    ckpt = runs_dir("02_representation_learning", "transfer", create=True) / f"{name}__{mode}.pt"
    if not args.dry_run:
        torch.save(model.state_dict(), ckpt)
    return {
        "model": name,
        "mode": mode,
        "params_total": total,
        "params_trainable": trainable,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr_head": lr_head,
        "lr_backbone": args.lr_backbone if mode == "finetune" else None,
        "weight_decay": args.weight_decay,
        "label_smoothing": 0.1,
        "amp": "bf16",
        "final_val_acc": history[-1]["val_acc"],
        "best_val_acc": max(h["val_acc"] for h in history),
        "train_time_s": train_time,
        "mean_train_imgs_per_s": sum(h["train_imgs_per_s"] for h in history) / len(history),
        "peak_vram_train_mb": train_peak,
        "peak_vram_eval_mb": eval_peak,
        "eval_batch_size": 256,
        "checkpoint": ckpt.relative_to(repo_root()).as_posix(),
        "history": history,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--models", nargs="+", default=rl.BACKBONES)
    ap.add_argument("--modes", nargs="+", default=rl.MODES, choices=rl.MODES)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--img-size", type=int, default=160)
    ap.add_argument("--lr-head-lp", type=float, default=3e-3, help="head LR, linear probe")
    ap.add_argument("--lr-head-ft", type=float, default=2e-3, help="head LR, fine-tuning")
    ap.add_argument("--lr-backbone", type=float, default=2e-4, help="trunk LR, fine-tuning")
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=None, help="cap steps/epoch (smoke test)")
    ap.add_argument("--dry-run", action="store_true", help="do not write checkpoints/results")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    env = env_summary()
    print(f"device {device} | {env.get('gpu', 'cpu')} | torch {env['torch']}")
    rows = []
    for name in args.models:
        for mode in args.modes:
            res = run_one(name, mode, args, device)
            res["env"] = {k: env.get(k) for k in ("gpu", "vram_gb", "torch", "cuda")}
            script = Path(sys.argv[0]).resolve().relative_to(repo_root()).as_posix()
            res["command"] = " ".join(["python", script, *sys.argv[1:]])
            print(f"==> {name} {mode}: val_acc {res['final_val_acc']:.4f} | "
                  f"{res['train_time_s']:.0f} s | train VRAM {res['peak_vram_train_mb']:.0f} MiB",
                  flush=True)  # fmt: skip
            if not args.dry_run:
                save_json(res, RESULTS / "transfer" / f"{name}__{mode}.json")
            rows.append({k: v for k, v in res.items() if k not in ("history", "env")})
    if not args.dry_run:
        # Merge with existing rows so that partial re-runs keep the other results.
        path = RESULTS / "transfer_summary.csv"
        old = []
        if path.exists():
            new_keys = {(x["model"], x["mode"]) for x in rows}
            with open(path) as f:
                old = [r for r in csv.DictReader(f) if (r["model"], r["mode"]) not in new_keys]
        keys = list(rows[0].keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(old + rows)
        print(f"wrote {path.relative_to(repo_root()).as_posix()}")


if __name__ == "__main__":
    main()
