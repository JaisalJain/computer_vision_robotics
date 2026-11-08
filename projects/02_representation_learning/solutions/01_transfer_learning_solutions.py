#!/usr/bin/env python
"""Solutions to the exercises of notebook 01 (transfer learning).

Needs the checkpoints from ``scripts/train_transfer.py``. Run on the GPU lock::

    flock /tmp/cv-gpu.lock python \
        projects/02_representation_learning/solutions/01_transfer_learning_solutions.py

Figures are written to ``outputs/02_representation_learning/solutions/`` (gitignored).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import replearn as rl  # noqa: E402

from cvlab import viz  # noqa: E402
from cvlab.device import peak_vram_mb, reset_peak_vram, seed_everything  # noqa: E402
from cvlab.io import outputs_dir, runs_dir  # noqa: E402
from cvlab.metrics import confusion_matrix, topk_accuracy  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPTS = runs_dir(rl.PROJECT, "transfer")
OUT = outputs_dir(rl.PROJECT, "solutions", create=True)


def load(name: str, mode: str) -> torch.nn.Module:
    m = rl.build_model(name, mode, pretrained=False)
    state = torch.load(CKPTS / f"{name}__{mode}.pt", map_location="cpu", weights_only=True)
    m.load_state_dict(state)
    return m.to(DEVICE).to(memory_format=torch.channels_last).eval()


def exercise_1_partial_finetune(epochs: int = 3) -> None:
    """Fine-tune only layer4 + fc of ResNet-18, keeping frozen BatchNorm statistics fixed."""
    print("\n=== Exercise 1: partial fine-tuning (layer4 + head) ===")
    seed_everything(0)
    model = rl.build_model("resnet18.a1_in1k", "finetune").to(DEVICE)
    model = model.to(memory_format=torch.channels_last)
    model.requires_grad_(False)
    for part in (model.layer4, model.fc):
        part.requires_grad_(True)
    frozen = [model.conv1, model.bn1, model.layer1, model.layer2, model.layer3]
    total, trainable = rl.count_params(model)
    opt = torch.optim.AdamW(
        [{"params": model.fc.parameters(), "lr": 2e-3},
         {"params": model.layer4.parameters(), "lr": 2e-4}], weight_decay=0.05)  # fmt: skip
    train_dl = rl.make_loader(rl.make_dataset("train", rl.build_transform(True, 160)), 128,
                              train=True)  # fmt: skip
    val_dl = rl.make_loader(rl.make_dataset("val", rl.build_transform(False, 160)), 256)
    steps = epochs * len(train_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, rl.cosine_warmup(steps, len(train_dl) // 2))
    reset_peak_vram()
    t0 = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        for m in frozen:
            m.eval()  # frozen layers: BatchNorm uses (and keeps) its ImageNet running statistics
        for x, y in train_dl:
            x = x.to(DEVICE, non_blocking=True).contiguous(memory_format=torch.channels_last)
            y = y.to(DEVICE, non_blocking=True)
            with torch.autocast(DEVICE.type, dtype=torch.bfloat16):
                loss = F.cross_entropy(model(x).float(), y, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        print(f"  epoch {epoch + 1}: val acc {rl.evaluate(model, val_dl, DEVICE)['acc']:.4f}")
    dt = time.perf_counter() - t0
    acc = rl.evaluate(model, val_dl, DEVICE)["acc"]
    print(f"  trainable params {trainable:,} of {total:,} ({100 * trainable / total:.1f} %)")
    print(f"  final val acc {acc:.4f} | {dt:.0f} s for {epochs} epochs | "
          f"peak VRAM {peak_vram_mb():.0f} MiB")  # fmt: skip
    print("  (compare: linear probe 0.9707, full fine-tune 0.9743 after 10 epochs)")


def exercise_2_topk_and_confusions() -> None:
    """Top-1/top-3 accuracy of all six runs and the most confused pairs of ResNet-18."""
    print("\n=== Exercise 2: top-k accuracy and most confused pairs ===")
    val_dl = rl.make_loader(rl.make_dataset("val", rl.build_transform(False, 160)), 256)
    for name in rl.BACKBONES:
        for mode in rl.MODES:
            logits, y = rl.predict(load(name, mode), val_dl, DEVICE)
            print(f"  {name:28s} {mode:13s} top-1 {topk_accuracy(logits, y, 1):.4f} "
                  f"top-3 {topk_accuracy(logits, y, 3):.4f}")  # fmt: skip
            if (name, mode) == ("resnet18.a1_in1k", "finetune"):
                cm = confusion_matrix(y, logits.argmax(1), 10)
    off = cm.copy()
    np.fill_diagonal(off, 0)
    print("  most confused (true -> predicted) pairs, ResNet-18 fine-tuned:")
    for flat in np.argsort(-off.ravel())[:3]:
        i, j = np.unravel_index(flat, off.shape)
        print(f"    {rl.CLASS_NAMES[i]:>16s} -> {rl.CLASS_NAMES[j]:<16s} {off[i, j]} images")


def exercise_3_test_resolution() -> None:
    """Accuracy of the 160-px fine-tuned ResNet-18 at several test resolutions (FixRes effect)."""
    print("\n=== Exercise 3: train-test resolution discrepancy ===")
    model = load("resnet18.a1_in1k", "finetune")
    sizes, accs = [128, 160, 192, 224], []
    for s in sizes:
        dl = rl.make_loader(rl.make_dataset("val", rl.build_transform(False, s)), 256)
        accs.append(rl.evaluate(model, dl, DEVICE)["acc"])
        print(f"  test size {s:3d} px: val acc {accs[-1]:.4f}")
    viz.set_style()
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(sizes, 100 * np.asarray(accs), "o-")
    ax.axvline(160, color="0.6", ls="--", label="train crop size")
    ax.set(xlabel="test resolution [px]", ylabel="val top-1 accuracy [%]",
           title="ResNet-18 (trained at 160 px)")  # fmt: skip
    ax.legend()
    fig.savefig(OUT / "ex1_3_test_resolution.png")
    print(f"  best test size: {sizes[int(np.argmax(accs))]} px")


if __name__ == "__main__":
    exercise_1_partial_finetune()
    exercise_2_topk_and_confusions()
    exercise_3_test_resolution()
