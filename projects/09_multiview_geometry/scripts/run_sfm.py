#!/usr/bin/env python
"""Rebuild the South Building sparse reconstruction of project 09 from the command line.

The same pipelines as notebook 04, for reuse by later projects (feed-forward 3D, Gaussian
splatting) without opening the notebook::

    # COLMAP SIFT, CPU only (about 30 s for 32 images)
    python projects/09_multiview_geometry/scripts/run_sfm.py --features sift
    # DISK + LightGlue matches imported into COLMAP (GPU; pairs within a window of 8 images)
    flock /tmp/cv-gpu.lock python projects/09_multiview_geometry/scripts/run_sfm.py \
        --features disk_lightglue

Settings (subsets, camera model, feature budgets) come from
``projects/09_multiview_geometry/configs/sfm_south_building.yaml``.

Outputs (gitignored): ``runs/09_multiview_geometry/south_building/<features>/`` (``sift`` or
``disk_lightglue_window8``; ``wide_`` prefix for the wide subset) with
``database.db``, ``sparse/<k>/{cameras,images,points3D}.bin``, ``image_list.txt`` and
``summary.json`` (statistics + accuracy against the reference model of the dataset).
Load the model with ``pycolmap.Reconstruction(".../sparse/0")``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mvgeo as M  # noqa: E402

from cvlab.io import repo_root, save_json  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "sfm_south_building.yaml"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--features", choices=["sift", "disk_lightglue"], default="sift")
    ap.add_argument("--subset", default="half", help="a subset name from the config")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--window", type=int, default=None,
                    help="DISK+LightGlue pair window (0 = exhaustive; default from the config)")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    image_dir = repo_root() / cfg["image_dir"]
    all_names = sorted(p.name for p in image_dir.glob("*.JPG"))
    if not all_names:
        raise SystemExit("South Building not found - run `make data-09_multiview_geometry` first")
    sl = cfg["subsets"][args.subset]
    names = all_names[sl["start"] : sl["stop"] : sl["step"]]
    window = cfg["disk_lightglue"]["pair_window"] if args.window is None else args.window
    sub = args.features
    if args.features == "disk_lightglue" and window and window > 0:
        sub += f"_window{window}"  # same folder names as notebook 04
    if args.subset != "half":
        sub = f"{args.subset}_{sub}"
    out = repo_root() / cfg["output_dir"] / sub
    out.mkdir(parents=True, exist_ok=True)

    if args.features == "sift":
        rec, timings = M.colmap_sift_sfm(
            image_dir, names, out, max_num_features=cfg["colmap"]["max_num_features"],
            camera_model=cfg["colmap"]["camera_model"],
        )  # fmt: skip
        extra = {}
    else:
        import torch

        from cvlab.device import track_vram

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dlc = cfg["disk_lightglue"]
        matcher = M.DiskLightGlue(device=device, max_keypoints=dlc["max_keypoints"],
                                  fp16=dlc["fp16"])  # fmt: skip
        with track_vram() as mem:
            rec, timings = M.disk_lightglue_sfm(
                matcher, image_dir, names, out, camera_model=cfg["colmap"]["camera_model"],
                window=window,
            )
        extra = {"device": device, "peak_vram_mb": mem.peak_mb, "pair_window": window}
    if rec is None:
        raise SystemExit("mapping failed: no model was reconstructed")

    (out / "image_list.txt").write_text("\n".join(names) + "\n")
    stats = M.reconstruction_stats(rec, len(names))
    summary = {"features": args.features, "subset": args.subset, **stats.as_dict(),
               "timings_s": timings, **extra}  # fmt: skip
    ref = M.load_reference_cameras(scale_to_width=1024)
    if ref:
        cmp = M.compare_to_reference(M.rec_cameras(rec), ref)
        summary["median_rot_err_deg_vs_reference"] = float(np.median(cmp["rot_err_deg"]))
    save_json(summary, out / "summary.json")
    print(f"output: {out.relative_to(repo_root())}")
    for k, v in summary.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
