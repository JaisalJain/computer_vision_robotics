#!/usr/bin/env python
"""Solution, notebook 02, exercise 2: forward-backward error of online vs offline CoTracker3.

Both directions use the same model (the online model never sees future frames). GPU: run
under the lock.

    flock /tmp/cv-gpu.lock python projects/06_tracking/solutions/nb02_ex2_online_fb.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from cvlab.io import read_video_frames, third_party_dir

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import pointtrack as pt  # noqa: E402


def fb_error(fwd_fn, model, frames) -> np.ndarray:
    fwd = fwd_fn(model, frames, grid_size=20)
    q = np.column_stack([np.zeros(fwd["tracks"].shape[1]), fwd["tracks"][-1]])
    back = fwd_fn(model, frames[::-1], grid_size=0, queries=q)
    err = np.linalg.norm(back["tracks"][-1] - fwd["tracks"][0], axis=1)
    err[~fwd["visible"][-1]] = np.nan
    return err


def main() -> None:
    frames = read_video_frames(third_party_dir("co-tracker", "assets", "apple.mp4"))
    for mode, fn in [("offline", pt.track_offline), ("online", pt.track_online)]:
        model = pt.load_cotracker(mode)
        e = fb_error(fn, model, frames)
        v = e[np.isfinite(e)]
        print(
            f"{mode:7s}: {len(v)} points visible at the end | "
            f"median cycle error {np.median(v):.2f} px | "
            f"< 2 px: {(v < 2).mean():.1%} | < 5 px: {(v < 5).mean():.1%}"
        )
        del model


if __name__ == "__main__":
    main()
