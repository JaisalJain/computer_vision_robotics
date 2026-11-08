#!/usr/bin/env python
"""Download the data for project 01_classical_cv (idempotent, a few MB in total).

* OpenCV sample images pinned to tag 5.0.0 (opencv/opencv, Apache-2.0 repository):
  checkerboard calibration views ``left*.jpg`` / ``right*.jpg`` (9x6 inner corners; there is
  no ``*10.jpg``), the ``graf1``/``graf3`` viewpoint pair with its ground-truth homography
  ``H1to3p.xml`` (originally from the Oxford affine-covariant regions dataset, research use),
  and the ``box`` / ``box_in_scene`` pair.
* Panorama test images ``boat1-6.jpg`` (rotating handheld camera) from opencv/opencv_extra
  (no license declared): downloaded at runtime only, never committed.
* One TUM RGB-D sequence (fr1_desk, CC BY 4.0), shared with other projects.

Usage::

    python scripts/download_01_classical_cv.py        # or: make data-01_classical_cv
"""

from __future__ import annotations

import time
from pathlib import Path

from cvlab.download import download_file
from cvlab.io import data_dir

OPENCV_DATA = "https://raw.githubusercontent.com/opencv/opencv/5.0.0/samples/data"
OPENCV_EXTRA = "https://raw.githubusercontent.com/opencv/opencv_extra/5.0.0/testdata/stitching"

VIEW_IDS = [f"{i:02d}" for i in range(1, 15) if i != 10]  # left10/right10 do not exist
FILES: dict[str, list[str]] = {
    "calib": [f"left{i}.jpg" for i in VIEW_IDS] + [f"right{i}.jpg" for i in VIEW_IDS],
    "features": ["graf1.png", "graf3.png", "H1to3p.xml", "box.png", "box_in_scene.png"],
    "stitching": [f"boat{i}.jpg" for i in range(1, 7)],
}
BASE_URL = {"calib": OPENCV_DATA, "features": OPENCV_DATA, "stitching": OPENCV_EXTRA}


def fetch(url: str, dest: Path, retries: int = 4) -> Path:
    """``download_file`` with retries (raw.githubusercontent.com occasionally stalls)."""
    for attempt in range(1, retries + 1):
        try:
            return download_file(url, dest, timeout=30.0, quiet=True)
        except OSError as exc:  # URLError / timeout are OSError subclasses
            if attempt == retries:
                raise
            print(f"[retry {attempt}/{retries - 1}] {dest.name}: {exc}")
            time.sleep(2.0 * attempt)
    return dest  # pragma: no cover


def main() -> None:
    root = data_dir("01_classical_cv", create=True)
    for group, names in FILES.items():
        n_new = 0
        for name in names:
            dest = root / group / name
            existed = dest.exists()
            fetch(f"{BASE_URL[group]}/{name}", dest)
            n_new += not existed
        print(f"[ok] data/01_classical_cv/{group}: {len(names)} files ({n_new} downloaded)")

    # TUM RGB-D fr1_desk (shared with other projects): real robot-camera frames.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from download_shared import tum

    seq = tum("fr1_desk")
    print(f"[ok] data/tum/fr1_desk/{seq.name}")


if __name__ == "__main__":
    main()
