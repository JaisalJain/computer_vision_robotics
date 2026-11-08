#!/usr/bin/env python
"""Data and weights for project 02 (representation learning). Idempotent.

* Imagenette-160 (shared dataset, ``data/imagenette/imagenette2-160``),
* timm backbones for transfer learning (Hugging Face cache),
* DINOv3 ViT-S/16 (official gated repo; falls back to the ungated timm mirror when the
  account has not been granted access) and DINOv2 ViT-S/14 (Hugging Face cache).

Usage::

    python scripts/download_02_representation_learning.py   # = make data-02_representation_learning
"""

from __future__ import annotations

from download_shared import imagenette

from cvlab.download import hf_download

TIMM_BACKBONES = [
    "timm/resnet18.a1_in1k",
    "timm/efficientnet_b0.ra_in1k",
    "timm/convnext_nano.in12k_ft_in1k",
]
DINOV2 = "facebook/dinov2-small"
DINOV3 = "facebook/dinov3-vits16-pretrain-lvd1689m"  # gated: accept the licence on the model page
DINOV3_TIMM_MIRROR = "timm/vit_small_patch16_dinov3.lvd1689m"  # ungated mirror (same licence)

WEIGHT_FILES = ["*.json", "model.safetensors"]


def fetch(repo: str) -> None:
    hf_download(repo, allow_patterns=WEIGHT_FILES)
    print(f"[ok] {repo}")


def main() -> None:
    root = imagenette()
    n_train = sum(1 for _ in (root / "train").rglob("*.JPEG"))
    n_val = sum(1 for _ in (root / "val").rglob("*.JPEG"))
    print(f"[ok] Imagenette-160: {n_train} train / {n_val} val images")
    for repo in [*TIMM_BACKBONES, DINOV2]:
        fetch(repo)
    try:
        fetch(DINOV3)
    except Exception as exc:  # noqa: BLE001 - gated repo: 401/403 when access is missing
        print(f"[warn] {DINOV3} not accessible ({type(exc).__name__}); request access on "
              f"https://huggingface.co/{DINOV3} - using the timm mirror instead")  # fmt: skip
        fetch(DINOV3_TIMM_MIRROR)


if __name__ == "__main__":
    main()
