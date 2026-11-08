#!/usr/bin/env python
"""Solutions to the exercises of notebook 03 (object map and consistency). CPU only.

Re-uses the mask cache written by notebook 03 (runs/15_robotics_capstone/)::

    python projects/15_robotics_capstone/solutions/nb03_solutions.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import capstone as cap  # noqa: E402


def main() -> None:
    frames = cap.load_frames()
    cache = cap.load_mask_cache()
    obs, _ = cap.sequence_observations(frames, cache)

    # Exercise 1: association gate
    gates = [0.05, 0.10, 0.15, 0.30]
    rows = {f"gate {100 * g:.0f} cm": cap.consistency(obs, gate=g) for g in gates}
    print("\nExercise 1 - association gate sweep (baseline observations)")
    print(pd.DataFrame(rows).round(1).to_string())

    # Exercise 2: pose at the RGB timestamp instead of the depth timestamp
    obs_rgb, _ = cap.sequence_observations(frames, cache, pose_at="rgb")
    dt = np.array([abs(frames.depth_ts[i] - frames.seq.timestamps[i]) for i in cache])
    shift = np.array(
        [
            np.linalg.norm(frames.pose(i, "depth")[:3, 3] - frames.pose(i, "rgb")[:3, 3])
            for i in cache
        ]
    )
    print("\nExercise 2 - pose interpolated at the depth vs the RGB timestamp")
    print(
        pd.DataFrame(
            {"depth timestamp": cap.consistency(obs), "RGB timestamp": cap.consistency(obs_rgb)}
        )
        .round(1)
        .to_string()
    )
    print(
        f"|t_depth - t_rgb|: median {1000 * np.median(dt):.1f} ms, max {1000 * dt.max():.1f} ms; "
        f"camera moves {100 * np.median(shift):.1f} cm (median), "
        f"{100 * shift.max():.1f} cm (max) in between"
    )

    # Exercise 3: score threshold
    obs_hi, _ = cap.sequence_observations(frames, cache, min_score=0.5)
    print("\nExercise 3 - all detections vs score >= 0.5")
    print(
        pd.DataFrame({"all": cap.consistency(obs), "score >= 0.5": cap.consistency(obs_hi)})
        .round(1)
        .to_string()
    )


if __name__ == "__main__":
    main()
