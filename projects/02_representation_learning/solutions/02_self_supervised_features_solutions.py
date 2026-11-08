#!/usr/bin/env python
"""Solutions to the exercises of notebook 02 (self-supervised features).

Reuses the features cached by the notebook in ``outputs/02_representation_learning/features/``
(recomputed if missing). Run on the GPU lock::

    flock /tmp/cv-gpu.lock python \
        projects/02_representation_learning/solutions/02_self_supervised_features_solutions.py

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
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import replearn as rl  # noqa: E402

from cvlab import viz  # noqa: E402
from cvlab.io import outputs_dir  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEAT_DIR = outputs_dir(rl.PROJECT, "features", create=True)
OUT = outputs_dir(rl.PROJECT, "solutions", create=True)
DINO = ["dinov3_vits16", "dinov2_vits14"]


def get_features(key: str) -> dict:
    """Cached {split: {cls, mean_patch, labels}} from notebook 02, recomputed if missing."""
    path = FEAT_DIR / f"{key}_224.pt"
    if path.exists():
        return torch.load(path, weights_only=True)
    enc = rl.load_encoder(key, DEVICE)
    tf = rl.build_transform(False, 224)
    feats = {s: rl.extract_features(enc, rl.make_loader(rl.make_dataset(s, tf), 256), DEVICE)
             for s in ["train", "val"]}  # fmt: skip
    torch.save(feats, path)
    return feats


def fit_probe(xtr: np.ndarray, ytr: np.ndarray, xva: np.ndarray, yva: np.ndarray) -> float:
    """Logistic regression with C picked on a held-out 20 % of train (as in the notebook)."""
    xa, xb, ya, yb = train_test_split(xtr, ytr, test_size=0.2, stratify=ytr, random_state=0)

    def make(c: float):
        return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000))

    best = max([0.01, 0.1, 1.0], key=lambda c: make(c).fit(xa, ya).score(xb, yb))
    return make(best).fit(xtr, ytr).score(xva, yva)


def exercise_1_cls_plus_patch() -> None:
    print("\n=== Exercise 1: CLS vs. [CLS, mean patch] linear probe ===")
    for key in DINO:
        f = get_features(key)
        ytr, yva = f["train"]["labels"].numpy(), f["val"]["labels"].numpy()

        def build(split: str, concat: bool, f=f) -> np.ndarray:
            parts = [F.normalize(f[split]["cls"], dim=1)]
            if concat:
                parts.append(F.normalize(f[split]["mean_patch"], dim=1))
            return torch.cat(parts, dim=1).numpy()

        a_cls = fit_probe(build("train", False), ytr, build("val", False), yva)
        a_cat = fit_probe(build("train", True), ytr, build("val", True), yva)
        gain = 100 * (a_cat - a_cls)
        print(f"  {key}: CLS {a_cls:.4f} | CLS+mean-patch {a_cat:.4f} ({gain:+.2f} pp)")


def exercise_2_correspondences(n_query: int = 6) -> None:
    print("\n=== Exercise 2: dense patch correspondences (DINOv3, 448 px) ===")
    enc = rl.load_encoder("dinov3_vits16", DEVICE)
    val = rl.make_dataset("val")
    tf = rl.build_transform(False, 448)
    targets = np.asarray(val.targets)
    springer = np.flatnonzero(targets == 1)
    ia, ib = int(springer[3]), int(springer[11])
    x = torch.stack([tf(val[ia][0]), tf(val[ib][0])]).to(DEVICE)
    with torch.no_grad(), torch.autocast(DEVICE.type, dtype=torch.float16):
        _, patches = enc(x)  # (2, 28, 28, 384)
    patches = patches.float()
    h, w = patches.shape[1:3]
    _, fg = rl.pca_rgb(patches.cpu().numpy())  # foreground mask from PC1
    fa, fb = F.normalize(patches[0].reshape(-1, patches.shape[-1]), dim=1), \
        F.normalize(patches[1].reshape(-1, patches.shape[-1]), dim=1)  # fmt: skip
    sim = fa @ fb.T  # (h*w, h*w) cosine similarities
    fg_a = np.flatnonzero(fg[0].ravel())
    rng = np.random.default_rng(0)
    queries = rng.choice(fg_a, size=min(n_query, len(fg_a)), replace=False)
    best = sim[queries].argmax(1).cpu().numpy()
    scores = sim[queries, best].cpu().numpy()
    ps = 448 // h

    def centre(idx: int) -> tuple[float, float]:
        return (idx % w + 0.5) * ps, (idx // w + 0.5) * ps

    img = np.concatenate([rl.denormalize(x[0]), rl.denormalize(x[1])], axis=1)
    viz.set_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(img)
    for k, (q, b) in enumerate(zip(queries, best)):
        (xa, ya), (xb, yb) = centre(int(q)), centre(int(b))
        c = viz.color(k, unit=True)
        ax.plot([xa, xb + 448], [ya, yb], "-o", color=c, lw=1.5, ms=5)
        print(f"  query ({xa:.0f}, {ya:.0f}) px -> ({xb:.0f}, {yb:.0f}) px, cos {scores[k]:.3f}")
    ax.axis("off")
    ax.set_title("DINOv3 ViT-S/16 nearest-neighbour patch matches (left -> right)")
    fig.savefig(OUT / "ex2_2_correspondences.png")


def exercise_3_retrieval(queries=(0, 1500, 3000), top: int = 5) -> None:
    print("\n=== Exercise 3: top-5 retrieval with DINOv3 CLS features ===")
    f = get_features("dinov3_vits16")
    tr = F.normalize(f["train"]["cls"], dim=1)
    va = F.normalize(f["val"]["cls"], dim=1)
    ytr, yva = f["train"]["labels"].numpy(), f["val"]["labels"].numpy()
    sims = va[list(queries)] @ tr.T
    val_raw, train_raw = rl.make_dataset("val"), rl.make_dataset("train")
    tf = rl.build_transform(False, 160)
    viz.set_style()
    fig, axes = plt.subplots(len(queries), top + 1, figsize=(2 * (top + 1), 2.2 * len(queries)))
    for r, q in enumerate(queries):
        s, idx = sims[r].topk(top)
        labels = [rl.CLASS_NAMES[ytr[i]] for i in idx.tolist()]
        hits = sum(ytr[i] == yva[q] for i in idx.tolist())
        print(f"  query {q} ({rl.CLASS_NAMES[yva[q]]}): {hits}/{top} neighbours share the label; "
              f"cosine sims {', '.join(f'{v:.3f}' for v in s.tolist())}")  # fmt: skip
        axes[r, 0].imshow(rl.denormalize(tf(val_raw[q][0])))
        axes[r, 0].set_title(f"query: {rl.CLASS_NAMES[yva[q]]}", fontsize=8)
        for c, (i, si, lab) in enumerate(zip(idx.tolist(), s.tolist(), labels), start=1):
            axes[r, c].imshow(rl.denormalize(tf(train_raw[i][0])))
            axes[r, c].set_title(f"{lab}\ncos {si:.2f}", fontsize=8)
    for ax in axes.flat:
        ax.axis("off")
    fig.savefig(OUT / "ex2_3_retrieval.png")


if __name__ == "__main__":
    exercise_1_cls_plus_patch()
    exercise_2_correspondences()
    exercise_3_retrieval()
