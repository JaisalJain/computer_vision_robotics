#!/usr/bin/env python
"""Solutions to the exercises of notebook 03 (Grad-CAM).

Needs the checkpoints from ``scripts/train_transfer.py``. Run on the GPU lock::

    flock /tmp/cv-gpu.lock python \
        projects/02_representation_learning/solutions/03_grad_cam_solutions.py

Figures are written to ``outputs/02_representation_learning/solutions/`` (gitignored).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import replearn as rl  # noqa: E402

from cvlab import viz  # noqa: E402
from cvlab.io import outputs_dir, runs_dir  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPTS = runs_dir(rl.PROJECT, "transfer")
OUT = outputs_dir(rl.PROJECT, "solutions", create=True)


def load(name: str, mode: str) -> torch.nn.Module:
    m = rl.build_model(name, mode, pretrained=False)
    state = torch.load(CKPTS / f"{name}__{mode}.pt", map_location="cpu", weights_only=True)
    m.load_state_dict(state)
    return m.to(DEVICE).eval()


class HiResCAM(rl.GradCAM):
    """HiResCAM: element-wise gradient x activation instead of spatially averaged gradients."""

    def __call__(self, x, class_idx=None):
        self.model.eval()
        x = x.detach().clone().requires_grad_(True)
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if class_idx is None:
            idx = logits.argmax(1)
        else:
            idx = torch.as_tensor(class_idx, device=x.device).reshape(-1).expand(len(x))
        logits.gather(1, idx[:, None]).sum().backward()
        cam = F.relu((self.gradients * self.activations.detach()).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        cam = cam / cam.flatten(1).amax(1).clamp_min(1e-12)[:, None, None]
        return cam, logits.detach()


def eval_set(per_class: int = 30):
    """The same 300 validation images (30 per class, seed 0) as in notebook 03."""
    val = rl.make_dataset("val", rl.build_transform(False, 160))
    y = np.asarray(val.targets)
    rng = np.random.default_rng(0)
    idx = np.concatenate([rng.choice(np.flatnonzero(y == c), per_class, replace=False)
                          for c in range(10)])  # fmt: skip
    return torch.stack([val[int(i)][0] for i in idx])


def maps_for(cam_cls, model, layer, X, classes, bs=100) -> torch.Tensor:
    cam = cam_cls(model, layer)
    out = [cam(X[i:i + bs].to(DEVICE), classes[i:i + bs])[0].cpu() for i in range(0, len(X), bs)]
    cam.remove()
    return torch.cat(out)


def faithfulness(model, X, maps, classes, bs=50) -> tuple[float, float]:
    d, n = [], []
    for s in range(0, len(X), bs):
        dd, ii = rl.deletion_insertion(model, X[s:s + bs].to(DEVICE), maps[s:s + bs].to(DEVICE),
                                       classes[s:s + bs], steps=16)  # fmt: skip
        d.append(dd)
        n.append(ii)
    return float(rl.auc(np.concatenate(d)).mean()), float(rl.auc(np.concatenate(n)).mean())


def overlay(img_u8: np.ndarray, cam: torch.Tensor, alpha: float = 0.45) -> np.ndarray:
    heat = plt.get_cmap("jet")(cam.numpy())[..., :3]
    return ((1 - alpha) * img_u8 / 255.0 + alpha * heat).clip(0, 1)


def main() -> None:
    ft = load("resnet18.a1_in1k", "finetune")
    X = eval_set()
    with torch.no_grad():
        pred = torch.cat([ft(X[i:i + 100].to(DEVICE)).argmax(1) for i in range(0, len(X), 100)])

    print("\n=== Exercise 1: Grad-CAM at different depths (300 images) ===")
    layer_maps = {}
    for lname in ["layer2", "layer3", "layer4"]:
        layer = getattr(ft, lname)
        layer_maps[lname] = maps_for(rl.GradCAM, ft, layer, X, pred)
        d, i = faithfulness(ft, X, layer_maps[lname], pred)
        grid = rl.GradCAM(ft, layer)
        grid(X[:1].to(DEVICE))
        grid.remove()
        print(f"  {lname}: grid {tuple(grid.activations.shape[-2:])} | deletion AUC {d:.3f} | "
              f"insertion AUC {i:.3f}")  # fmt: skip
    viz.set_style()
    show = [0, 61, 122, 183, 244]
    imgs = rl.denormalize(X[show])
    fig, axes = plt.subplots(3, len(show), figsize=(2.3 * len(show), 7))
    for r, lname in enumerate(layer_maps):
        for c, i in enumerate(show):
            axes[r, c].imshow(overlay(imgs[c], layer_maps[lname][i]))
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
        axes[r, 0].set_ylabel(lname)
    fig.savefig(OUT / "ex3_1_layers.png")

    print("\n=== Exercise 2: HiResCAM vs. Grad-CAM (300 images) ===")
    for lname in ["layer4", "layer3"]:
        hires = maps_for(HiResCAM, ft, getattr(ft, lname), X, pred)
        for name, m in [("Grad-CAM", layer_maps[lname]), ("HiResCAM", hires)]:
            d, i = faithfulness(ft, X, m, pred)
            print(f"  {lname} {name:9s}: deletion AUC {d:.3f} | insertion AUC {i:.3f}")
        rho = [spearmanr(layer_maps[lname][k].flatten(), hires[k].flatten())[0]
               for k in range(len(X))]  # fmt: skip
        diff = (layer_maps[lname] - hires).abs().max().item()
        print(f"  {lname}: median Spearman rho Grad-CAM vs HiResCAM {np.nanmedian(rho):.3f}, "
              f"max |difference| {diff:.2e}")  # fmt: skip
    print(
        "  Why identical on layer4: only global average pooling and the linear head follow it,\n"
        "  so dy_c/dA_k[i, j] = W[c, k] / (h * w) is the same at every position. The element-wise\n"
        "  product then equals alpha_k * A_k, and HiResCAM reduces to Grad-CAM (and to the\n"
        "  original CAM of Zhou et al. 2016). They differ only for layers further inside (layer3)."
    )

    print("\n=== Exercise 3: Grad-CAM for ConvNeXt-Nano (stages[-1]) ===")
    cx = load("convnext_nano.in12k_ft_in1k", "finetune")
    with torch.no_grad():
        pred_cx = torch.cat([cx(X[i:i + 100].to(DEVICE)).argmax(1) for i in range(0, len(X), 100)])
    maps_cx = maps_for(rl.GradCAM, cx, cx.stages[-1], X, pred)  # explain ResNet's predicted class
    rho = np.nan_to_num([spearmanr(layer_maps["layer4"][k].flatten(), maps_cx[k].flatten())[0]
                         for k in range(len(X))])  # fmt: skip
    d, i = faithfulness(cx, X, maps_for(rl.GradCAM, cx, cx.stages[-1], X, pred_cx), pred_cx)
    agree = (pred_cx == pred).float().mean()
    print(f"  ConvNeXt agrees with ResNet-18 on {agree:.3f} of predictions")
    print(f"  Spearman rho ResNet-18 vs ConvNeXt maps: median {np.median(rho):.3f}, "
          f"mean {np.mean(rho):.3f}")  # fmt: skip
    print(f"  ConvNeXt Grad-CAM: deletion AUC {d:.3f} | insertion AUC {i:.3f}")


if __name__ == "__main__":
    main()
