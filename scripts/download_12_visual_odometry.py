#!/usr/bin/env python
"""Data for project 12_visual_odometry (``make data-12_visual_odometry``).

Downloads two TUM RGB-D sequences (Sturm et al., IROS 2012, CC BY 4.0) into ``data/tum/``:

* ``fr1_xyz``  (~0.45 GB): slow translation along the camera axes, textured desk - the easy case.
* ``fr1_desk`` (~0.34 GB): hand-held sweep over a desk with fast rotation and motion blur.

The project uses no learned models, so nothing is written to ``weights/``. Re-running the
script is a no-op once the sequences are extracted (``.done`` markers).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from download_shared import tum  # noqa: E402

from cvlab.io import repo_root  # noqa: E402

SEQUENCES = ("fr1_xyz", "fr1_desk")


def main() -> None:
    for name in SEQUENCES:
        path = tum(name)
        n_rgb = len(list((path / "rgb").glob("*.png")))
        print(f"{name}: {path.relative_to(repo_root())} ({n_rgb} RGB frames)")


if __name__ == "__main__":
    main()
