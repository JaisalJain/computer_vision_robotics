#!/usr/bin/env python
"""Solutions to the exercises of notebook 03 (initialisation study and novel views).

1. Opacity pruning of the COLMAP-init model: test PSNR vs number of Gaussians.
2. Multi-view depth fusion: render RGB+ED from 8 test views, back-project, merge,
   voxel-downsample and write a PLY.

Needs the ``colmap_7k`` run (``scripts/train.py --init colmap --steps 7000 --name colmap_7k``).
Run (GPU)::

    python projects/11_gaussian_splatting/solutions/ex03_novel_views.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import splatkit as S  # noqa: E402

from cvlab.device import setup  # noqa: E402
from cvlab.io import data_dir, outputs_dir, repo_root, runs_dir  # noqa: E402
from cvlab.pointcloud import depth_to_points, voxel_downsample, write_ply  # noqa: E402

# x,y,z + 3 DC + 45 SH + opacity + 3 scales + 4 quats = 59 float32 (gsplat's PLY export has no
# normal fields), i.e. 236 bytes per Gaussian plus a small header.
BYTES_PER_GAUSSIAN = 59 * 4


def main() -> None:
    device = setup(seed=0, verbose=False)
    scene = S.load_colmap_scene(data_dir("mipnerf360", "bonsai"), images="images_4")
    images = S.images_to_tensors(scene.load_images())
    splats, _ = S.load_checkpoint(runs_dir(S.PROJECT, "colmap_7k") / "ckpt.pt", device=device)
    opa = torch.sigmoid(splats["opacities"])

    print("Ex1: opacity pruning")
    for tau in (0.0, 0.01, 0.05, 0.1, 0.2, 0.5):
        keep = opa > tau
        pruned = torch.nn.ParameterDict(
            {k: torch.nn.Parameter(v[keep], requires_grad=False) for k, v in splats.items()}
        )
        m = S.evaluate(pruned, scene, images, scene.test_idx, 3, device, lpips=False)
        n = int(keep.sum())
        print(f"   tau={tau:<4} {n:>9,} Gaussians ({100 * n / opa.numel():5.1f}%) | "
              f"PLY {n * BYTES_PER_GAUSSIAN / 2**20:6.1f} MiB | test PSNR {m['psnr']:.2f} dB")
        del pruned

    print("Ex2: depth fusion from 8 test views")
    clouds, colors = [], []
    for v in scene.test_idx[:8]:
        v = int(v)
        vm = torch.as_tensor(scene.w2c[v : v + 1], dtype=torch.float32, device=device)
        K = torch.as_tensor(scene.K[v : v + 1], dtype=torch.float32, device=device)
        with torch.no_grad():
            out, alpha, _ = S.render(splats, vm, K, scene.widths[v], scene.heights[v], 3,
                                     render_mode="RGB+ED")  # fmt: skip
        rgb = out[0, ..., :3].clamp(0, 1).cpu().numpy()
        depth = out[0, ..., 3].cpu().numpy()
        acc = alpha[0, ..., 0].cpu().numpy()
        p, c = depth_to_points(depth, scene.K[v], rgb=rgb, mask=acc > 0.5, stride=2)
        T = scene.c2w[v]
        clouds.append(p @ T[:3, :3].T + T[:3, 3])
        colors.append(c)
    P, C = np.concatenate(clouds), np.concatenate(colors)
    voxel = 0.01 * scene.scene_scale
    Pd, Cd = voxel_downsample(P, voxel, C)
    path = write_ply(outputs_dir(S.PROJECT, "solutions", create=True) / "fused_depth.ply", Pd, Cd)
    from scipy.spatial import cKDTree

    good = (scene.points_track_len >= 3) & (scene.points_error < 1.5)
    d, _ = cKDTree(Pd).query(scene.points[good], k=1)
    print(f"   {len(P):,} points -> {len(Pd):,} after a {voxel:.4f}-unit voxel grid "
          f"-> {path.relative_to(repo_root())}")  # fmt: skip
    print(f"   median distance COLMAP point -> fused cloud: {np.median(d):.4f} units "
          f"({100 * np.median(d) / scene.scene_scale:.2f}% of the scene scale); the fused cloud "
          f"is {len(Pd) / good.sum():.1f}x denser than the filtered SfM points")  # fmt: skip


if __name__ == "__main__":
    main()
