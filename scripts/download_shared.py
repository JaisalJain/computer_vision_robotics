#!/usr/bin/env python
"""Datasets shared by several projects (imported by the per-project download scripts).

Usage::

    python scripts/download_shared.py tum fr1_xyz fr1_desk
    python scripts/download_shared.py imagenette
"""

from __future__ import annotations

import sys
from pathlib import Path

from cvlab.download import download_and_extract
from cvlab.io import data_dir

TUM_URLS = {
    "fr1_xyz": "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz",
    "fr1_desk": "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz",
    "fr1_desk2": "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk2.tgz",
    "fr1_plant": "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_plant.tgz",
}
IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"


def tum(sequence: str) -> Path:
    """Download + extract a TUM RGB-D sequence; returns the sequence directory.

    Layout: ``data/tum/<sequence>/rgbd_dataset_freiburg1_xyz/{rgb,depth,rgb.txt,...}``.
    TUM RGB-D is licensed CC BY 4.0 (Sturm et al., IROS 2012).
    """
    url = TUM_URLS[sequence]
    out = download_and_extract(url, data_dir("tum", sequence, create=False))
    return out / Path(url).name.removesuffix(".tgz")


def imagenette() -> Path:
    """Download Imagenette-160 (fast.ai, ImageNet subset); returns ``.../imagenette2-160``."""
    out = download_and_extract(IMAGENETTE_URL, data_dir("imagenette", create=False))
    return out / "imagenette2-160"


if __name__ == "__main__":
    what, *rest = sys.argv[1:] or ["tum", "fr1_xyz"]
    if what == "tum":
        for seq in rest or ["fr1_xyz"]:
            print(tum(seq))
    elif what == "imagenette":
        print(imagenette())
    else:
        raise SystemExit(f"unknown dataset: {what}")
