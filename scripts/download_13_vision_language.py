#!/usr/bin/env python
"""Data and weights for ``projects/13_vision_language`` (idempotent).

* Imagenette-160 (fast.ai)            -> ``data/imagenette/imagenette2-160`` (shared)
* TUM RGB-D fr1_desk                  -> ``data/tum/fr1_desk/...``             (shared)
* Ultralytics coco128 (128 COCO images with YOLO labels) -> ``data/ultralytics/coco128``
* Checkpoints at the revisions pinned in ``projects/13_vision_language/configs/models.yaml``
  (SigLIP 2 B/16-224, open_clip ViT-B/32 LAION-2B, Florence-2-base, Qwen3.5-0.8B; about
  4.0 GB in total) -> the Hugging Face cache.

Usage::

    python scripts/download_13_vision_language.py            # everything
    python scripts/download_13_vision_language.py --no-models
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import yaml
from download_shared import imagenette, tum

from cvlab.download import download_file, extract_archive, hf_download, human_size
from cvlab.io import data_dir, repo_root

COCO128_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
MODELS_YAML = Path("projects/13_vision_language/configs/models.yaml")


def coco128() -> Path:
    """Download + extract Ultralytics coco128 into ``data/ultralytics/coco128``."""
    root = data_dir("ultralytics", create=True)
    dest = root / "coco128"
    if (dest / "images" / "train2017").is_dir() and (dest / "labels" / "train2017").is_dir():
        print("[skip] data/ultralytics/coco128 already present")
        return dest
    archive = download_file(COCO128_URL, root / "coco128.zip")
    with zipfile.ZipFile(archive) as zf:
        nested = all(m.startswith("coco128/") for m in zf.namelist())
    tmp = root / ".coco128.extract"
    shutil.rmtree(tmp, ignore_errors=True)
    extract_archive(archive, tmp)
    shutil.rmtree(dest, ignore_errors=True)
    (tmp / "coco128" if nested else tmp).rename(dest)
    shutil.rmtree(tmp, ignore_errors=True)
    archive.unlink()
    n = len(list((dest / "images" / "train2017").glob("*.jpg")))
    print(f"[ok] data/ultralytics/coco128 ({n} images)")
    return dest


def models() -> None:
    """Snapshot every checkpoint listed in configs/models.yaml (pinned revision)."""
    cfg = yaml.safe_load((repo_root() / MODELS_YAML).read_text())
    for key, m in cfg.items():
        path = hf_download(m["repo"], revision=m["revision"], allow_patterns=m["allow"])
        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        print(f"[ok] {key}: {m['repo']}@{m['revision'][:8]} ({human_size(size)})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-models", action="store_true", help="datasets only")
    args = ap.parse_args()
    imagenette()
    tum("fr1_desk")
    coco128()
    if not args.no_models:
        models()


if __name__ == "__main__":
    main()
