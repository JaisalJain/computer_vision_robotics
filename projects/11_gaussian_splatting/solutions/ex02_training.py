#!/usr/bin/env python
"""Solutions to the exercises of notebook 02 (COLMAP scene and training).

1. Filtering the SfM initialisation by track length and reprojection error (1500 steps).
2. SH degree 0 vs 3 (2000 steps).

Run (GPU, about 5 minutes on an RTX 5060 Laptop GPU)::

    python projects/11_gaussian_splatting/solutions/ex02_training.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import splatkit as S  # noqa: E402

from cvlab.device import setup  # noqa: E402
from cvlab.io import data_dir  # noqa: E402


def run(scene, images, pts, rgb, cfg, device, sh_eval: int) -> dict:
    splats, res = S.train(scene, images, pts, rgb, cfg, device=device, log=None)
    m = S.evaluate(splats, scene, images, scene.test_idx, sh_eval, device, lpips=False)
    out = {"psnr": m["psnr"], "ssim": m["ssim"], "n": int(splats["means"].shape[0]),
           "time_s": res.train_time_s, "n_init": res.n_init}  # fmt: skip
    del splats
    return out


def main() -> None:
    device = setup(seed=0, verbose=False)
    scene = S.load_colmap_scene(data_dir("mipnerf360", "bonsai"), images="images_4")
    images = S.images_to_tensors(scene.load_images())
    pts, rgb = S.init_points(scene, "colmap")

    print("Ex1: SfM point filter (1500 steps)")
    keep = (scene.points_track_len >= 5) & (scene.points_error < 1.0)
    cfg = S.TrainConfig(steps=1500, eval_every=0)
    for name, sel in (("all points", np.ones(len(pts), bool)), ("track>=5 & err<1px", keep)):
        r = run(scene, images, pts[sel], rgb[sel], cfg, device, sh_eval=1)
        print(f"   {name:<20} init {r['n_init']:>7,} -> {r['n']:>9,} Gaussians | test PSNR "
              f"{r['psnr']:.2f} dB | SSIM {r['ssim']:.4f} | {r['time_s']:.0f} s")  # fmt: skip

    print("Ex2: SH degree (2000 steps, degree raised every 500 steps)")
    for deg in (0, 3):
        cfg = S.TrainConfig(steps=2000, sh_degree=deg, sh_degree_interval=500, eval_every=0)
        r = run(scene, images, pts, rgb, cfg, device, sh_eval=deg)
        print(f"   sh_degree={deg}: test PSNR {r['psnr']:.2f} dB | SSIM {r['ssim']:.4f} | "
              f"{r['n']:,} Gaussians | {r['time_s']:.0f} s")  # fmt: skip


if __name__ == "__main__":
    main()
