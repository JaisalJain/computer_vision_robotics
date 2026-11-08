#!/usr/bin/env python
"""COLMAP (pycolmap, CPU) reconstructions of the project-10 scenes.

For every scene of ``configs/ff3d.yaml`` this runs SIFT feature extraction, exhaustive
matching and incremental SfM on exactly the images the feed-forward models see, and writes

* ``runs/10_feedforward_3d/colmap/<scene>/{database.db,sparse/<k>/,summary.json}`` (gitignored)
* ``projects/10_feedforward_3d/results/colmap_summary.csv`` (registered images, points,
  reprojection error, timings).

For ``kitchen`` this model *is* the evaluation reference; for ``south_building`` and
``tum_desk`` it is one more method evaluated against the dataset reference / mocap ground truth.

Usage::

    python projects/10_feedforward_3d/scripts/run_colmap.py                 # all scenes
    python projects/10_feedforward_3d/scripts/run_colmap.py --scenes kitchen
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ff3d  # noqa: E402

from cvlab.io import save_json  # noqa: E402


def run_scene(name: str) -> dict:
    cfg = ff3d.load_config()
    scene = ff3d.load_scene(name)
    image_dir = scene.image_paths[0].parent
    work = ff3d.runs_path("colmap", name)
    rec, timings = ff3d.run_colmap(
        image_dir, scene.names, work, camera_model=cfg["colmap"]["camera_model"],
        max_num_features=cfg["colmap"]["max_num_features"],
    )  # fmt: skip
    summary = {"scene": name, "n_images": len(scene), "timings_s": timings}
    if rec is None:
        summary.update(n_registered=0)
    else:
        summary.update(
            n_registered=int(rec.num_reg_images()),
            n_points3d=int(rec.num_points3D()),
            mean_reproj_error_px=float(rec.compute_mean_reprojection_error()),
            mean_track_length=float(rec.compute_mean_track_length()),
        )  # fmt: skip
    save_json(summary, work / "summary.json")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenes", nargs="*", default=["kitchen", "south_building", "tum_desk"])
    args = ap.parse_args()
    import pandas as pd

    rows = []
    for name in args.scenes:
        s = run_scene(name)
        print(f"{name}: registered {s['n_registered']}/{s['n_images']} images, "
              f"{s.get('n_points3d', 0)} points, {s['timings_s']['total_s']:.1f} s")  # fmt: skip
        rows.append({"scene": name, "n_images": s["n_images"], "n_registered": s["n_registered"],
                     "n_points3d": s.get("n_points3d", 0),
                     "mean_reproj_error_px": s.get("mean_reproj_error_px", float("nan")),
                     **{k: round(v, 2) for k, v in s["timings_s"].items()}})  # fmt: skip
    out = ff3d.results_path("colmap_summary.csv")
    old = pd.read_csv(out) if out.exists() else pd.DataFrame()
    df = pd.DataFrame(rows)
    if not old.empty:
        df = pd.concat([old[~old["scene"].isin(df["scene"])], df], ignore_index=True)
    df.sort_values("scene").to_csv(out, index=False, float_format="%.4f")
    print(f"wrote {ff3d.rel(out)}")


if __name__ == "__main__":
    main()
