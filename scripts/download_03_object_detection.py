#!/usr/bin/env python
"""Data and weights for ``projects/03_object_detection`` (idempotent).

* YOLO26 n/s COCO checkpoints  -> ``weights/ultralytics/`` (Ultralytics ``weights_dir``)
* african-wildlife, medical-pills and coco8 (Ultralytics YOLO-format sets)
  -> ``data/ultralytics/<name>/`` (Ultralytics ``datasets_dir``), so the built-in
  ``<name>.yaml`` dataset files resolve to the local copy without re-downloading.
* RF-DETR Nano/Small COCO checkpoints (about 0.75 GB) -> ``weights/rfdetr/``
  (point ``RF_HOME`` there so ``rfdetr`` finds them; used by notebook 04).
* TUM RGB-D ``fr1_desk`` (shared with other projects, via ``download_shared.tum``) for the
  robot-eye-view examples in notebook 01 and the notebook 04 exercises.

Usage::

    python scripts/download_03_object_detection.py              # everything
    python scripts/download_03_object_detection.py --no-rfdetr  # skip the RF-DETR checkpoints
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

from cvlab.download import download_file, extract_archive
from cvlab.io import data_dir, weights_dir

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_shared import tum  # noqa: E402

ASSETS = "https://github.com/ultralytics/assets/releases/download"
DATASETS = {
    "african-wildlife": f"{ASSETS}/v0.0.0/african-wildlife.zip",  # 1504 images, 4 classes
    "medical-pills": f"{ASSETS}/v0.0.0/medical-pills.zip",  # 115 images, 1 class
    "coco8": f"{ASSETS}/v0.0.0/coco8.zip",  # 8 images, 80 COCO classes
}
YOLO_WEIGHTS = {
    "yolo26n.pt": f"{ASSETS}/v8.4.0/yolo26n.pt",
    "yolo26s.pt": f"{ASSETS}/v8.4.0/yolo26s.pt",
}
RFDETR_WEIGHTS = {
    # Same file names, URLs and MD5s as rfdetr 1.12.0 (rfdetr/assets/model_weights.py).
    "rf-detr-nano.pth": (
        "https://repo.roboflow.com/rfdetr/nano_coco/checkpoint_best_regular.pth",
        "fb6504cce7fbdc783f7a46991f07639f",
    ),
    "rf-detr-small.pth": (
        "https://repo.roboflow.com/rfdetr/small_coco/checkpoint_best_regular.pth",
        "fb37061c1af7bace359c91b723a8d5c1",
    ),
}


def ultralytics_dataset(name: str) -> Path:
    """Download + extract one Ultralytics dataset zip into ``data/ultralytics/<name>``."""
    root = data_dir("ultralytics", create=True)
    dest = root / name
    if (dest / "images").is_dir() and (dest / "labels").is_dir():
        print(f"[skip] data/ultralytics/{name} already present")
        return dest
    archive = download_file(DATASETS[name], root / f"{name}.zip")
    with zipfile.ZipFile(archive) as zf:
        nested = all(m.startswith(f"{name}/") for m in zf.namelist())
    # Some zips contain a top-level <name>/ folder (coco8), others do not (medical-pills).
    tmp = root / f".{name}.extract"
    shutil.rmtree(tmp, ignore_errors=True)
    extract_archive(archive, tmp)
    shutil.rmtree(dest, ignore_errors=True)
    (tmp / name if nested else tmp).rename(dest)
    shutil.rmtree(tmp, ignore_errors=True)
    archive.unlink()
    n_val = len(list((dest / "images" / "val").glob("*")))
    print(f"[ok] data/ultralytics/{name} ({n_val} val images)")
    return dest


def yolo_weights() -> list[Path]:
    out = weights_dir("ultralytics", create=True)
    return [download_file(url, out / name) for name, url in YOLO_WEIGHTS.items()]


def rfdetr_weights() -> list[Path]:
    out = weights_dir("rfdetr", create=True)
    return [download_file(url, out / name, md5=md5) for name, (url, md5) in RFDETR_WEIGHTS.items()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-rfdetr", action="store_true", help="skip the RF-DETR checkpoints")
    args = ap.parse_args()
    yolo_weights()
    for name in DATASETS:
        ultralytics_dataset(name)
    tum("fr1_desk")
    if not args.no_rfdetr:
        rfdetr_weights()


if __name__ == "__main__":
    main()
