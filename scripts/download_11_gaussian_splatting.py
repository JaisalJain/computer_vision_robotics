#!/usr/bin/env python
"""Data + weights for project 11 (Gaussian splatting). Idempotent: re-running is a no-op.

* Mip-NeRF 360 "bonsai" (COLMAP-processed, per-scene folder of the Hugging Face mirror
  ``nvs-bench/mipnerf360``, pinned revision; 292 photos at 1559x1039 + ``sparse/0``; ~373 MB)
  -> ``data/mipnerf360/bonsai/{images,sparse/0}``. A 2x downscaled copy (780x520, i.e. 4x
  below the 3118x2078 COLMAP camera) is written to ``data/mipnerf360/bonsai/images_4/``;
  that is the resolution the trainer uses on an 8 GB GPU.
* Depth Anything 3 (DA3-BASE, Apache-2.0) for the feed-forward initialisation
  (Hugging Face cache; shared with projects 08/10). The DA3 *code* is the pinned clone made
  by ``bash scripts/setup_third_party.sh depth-anything-3``.
* torchvision AlexNet ImageNet weights (~233 MB) used by the LPIPS metric
  -> ``weights/torch_hub/checkpoints/``.

Usage::

    python scripts/download_11_gaussian_splatting.py                # everything
    python scripts/download_11_gaussian_splatting.py --no-weights   # scene only
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cvlab.download import hf_download
from cvlab.io import data_dir, repo_root, weights_dir

HF_SCENES_REPO = "nvs-bench/mipnerf360"
HF_SCENES_REVISION = "2e0758f7f5d2bf82d5a29c795d3cbfb64af35474"  # checked 2026-11-07
DEFAULT_SCENE = "bonsai"
DOWNSCALE = 2  # mirror images (1559x1039) -> 780x520
DA3_REPO = "depth-anything/DA3-BASE"


def rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(repo_root()))
    except ValueError:
        return str(path)


def scene(name: str = DEFAULT_SCENE, downscale: int = DOWNSCALE) -> Path:
    """Fetch one Mip-NeRF 360 scene (images + sparse/0) and write a downscaled image copy."""
    import cv2

    root = data_dir("mipnerf360", create=True)
    scene_dir = root / name
    marker = scene_dir / ".done"
    out_name = f"images_{2 * downscale}"  # factor w.r.t. the full-resolution COLMAP camera
    if marker.exists() and (scene_dir / out_name / ".done").exists():
        print(f"[skip] {name} already prepared in {rel(scene_dir)}")
        return scene_dir
    if not marker.exists():
        hf_download(
            HF_SCENES_REPO,
            repo_type="dataset",
            revision=HF_SCENES_REVISION,
            allow_patterns=[f"{name}/images/*", f"{name}/sparse/0/*"],
            local_dir=str(root),
        )
        n = len(list((scene_dir / "images").glob("*")))
        marker.write_text(f"{HF_SCENES_REPO}@{HF_SCENES_REVISION}\nimages={n}\n")
        print(f"[ok] {name}: {n} images + sparse/0 -> {rel(scene_dir)}")
    out = scene_dir / out_name
    out.mkdir(exist_ok=True)
    srcs = sorted(p for p in (scene_dir / "images").iterdir() if p.suffix.lower() in
                  {".jpg", ".jpeg", ".png"})  # fmt: skip
    for src in srcs:
        dst = out / (src.stem + ".png")
        if dst.exists():
            continue
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        size = (round(w / downscale), round(h / downscale))
        cv2.imwrite(str(dst), cv2.resize(img, size, interpolation=cv2.INTER_AREA))
    (out / ".done").write_text(f"downscale={downscale} from images/\n")
    print(f"[ok] {len(srcs)} images downscaled {downscale}x -> {rel(out)}")
    return scene_dir


def weights() -> None:
    """DA3-BASE (HF cache) and the AlexNet backbone of LPIPS (weights/torch_hub)."""
    import torch
    import torchvision

    for fname in ("config.json", "model.safetensors"):
        hf_download(DA3_REPO, fname)  # returns at once when cached
    print(f"[ok] {DA3_REPO} in the Hugging Face cache")
    hub = weights_dir("torch_hub", create=True)
    torch.hub.set_dir(str(hub))
    torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.DEFAULT)
    print(f"[ok] AlexNet (LPIPS backbone) in {rel(hub / 'checkpoints')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene", default=DEFAULT_SCENE, help="Mip-NeRF 360 scene name")
    ap.add_argument("--no-weights", action="store_true", help="only download the scene")
    args = ap.parse_args()
    scene(args.scene)
    if not args.no_weights:
        weights()


if __name__ == "__main__":
    main()
