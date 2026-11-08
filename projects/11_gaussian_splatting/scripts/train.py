#!/usr/bin/env python
"""Train 3D Gaussian Splatting on a COLMAP scene with gsplat (one run, <= ~30 min on 8 GB).

Writes
  * ``runs/11_gaussian_splatting/<name>/ckpt.pt``        raw parameters + config (gitignored)
  * ``outputs/11_gaussian_splatting/<name>.ply``         standard 3DGS PLY (gitignored)
  * ``projects/11_gaussian_splatting/results/<name>.json`` curves + final test metrics

Examples::

    python projects/11_gaussian_splatting/scripts/train.py --init colmap --name colmap_7k
    python projects/11_gaussian_splatting/scripts/train.py --init random --name random_7k
    python projects/11_gaussian_splatting/scripts/train.py --init da3 --name da3_7k
    # your own capture (after sfm_from_images.py):
    python projects/11_gaussian_splatting/scripts/train.py --scene-dir data/my_scene \
        --images images --init colmap --steps 7000 --name my_scene_7k
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import splatkit as S  # noqa: E402

from cvlab.io import outputs_dir, project_dir, repo_root, runs_dir, save_json  # noqa: E402


def rel(path: Path | str) -> str:
    p = Path(path).resolve()
    try:
        return str(p.relative_to(repo_root()))
    except ValueError:
        return p.name


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene-dir", default="data/mipnerf360/bonsai")
    ap.add_argument("--images", default="images_4", help="image sub-folder of --scene-dir")
    ap.add_argument("--init", choices=["colmap", "random", "da3"], default="colmap")
    ap.add_argument("--da3-init", default="runs/11_gaussian_splatting/da3_init/bonsai_da3.npz")
    ap.add_argument("--n-random", type=int, default=100_000, help="points for --init random")
    ap.add_argument("--steps", type=int, default=7000)
    ap.add_argument("--name", default=None, help="run name (default: <init>_<steps>)")
    ap.add_argument("--test-every", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--sh-degree", type=int, default=3)
    ap.add_argument("--grow-grad2d", type=float, default=0.0002)
    ap.add_argument("--max-gaussians", type=int, default=3_000_000)
    ap.add_argument("--packed", action="store_true", help="packed rasterisation (less VRAM)")
    ap.add_argument("--no-lpips", action="store_true")
    ap.add_argument("--no-ply", action="store_true")
    ap.add_argument("--results-dir", default=None,
                    help="default: projects/11_gaussian_splatting/results")  # fmt: skip
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main() -> int:
    import torch

    from cvlab.bench import benchmark
    from cvlab.device import env_summary, seed_everything

    args = parse_args()
    seed_everything(args.seed)
    name = args.name or f"{args.init}_{args.steps // 1000}k"
    root = repo_root()
    scene_dir = (root / args.scene_dir) if not Path(args.scene_dir).is_absolute() else Path(
        args.scene_dir
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("gsplat needs a CUDA GPU")

    t0 = time.perf_counter()
    scene = S.load_colmap_scene(scene_dir, images=args.images, test_every=args.test_every)
    images = S.images_to_tensors(scene.load_images())
    print(
        f"[scene] {rel(scene_dir)} | {scene.n_images} images ({len(scene.train_idx)} train / "
        f"{len(scene.test_idx)} test) at {scene.widths[0]}x{scene.heights[0]} | "
        f"{len(scene.points)} SfM points | load {time.perf_counter() - t0:.1f} s"
    )
    da3 = root / args.da3_init
    pts, rgb = S.init_points(scene, args.init, n_random=args.n_random, da3_path=da3,
                             seed=args.seed)  # fmt: skip
    print(f"[init] {args.init}: {len(pts)} points")

    cfg = S.TrainConfig(
        steps=args.steps, sh_degree=args.sh_degree, grow_grad2d=args.grow_grad2d,
        max_gaussians=args.max_gaussians, packed=args.packed, eval_every=args.eval_every,
        seed=args.seed,
    )  # fmt: skip
    status = "ok"
    try:
        splats, res = S.train(scene, images, pts, rgb, cfg, device=device, eval_images=images)
    except torch.cuda.OutOfMemoryError as exc:  # documented, never silently dropped
        status = f"oom: {exc}".splitlines()[0]
        print(f"[error] {status}")
        save_json({"name": name, "status": status, "config": asdict(cfg)},
                  _results_dir(args) / f"{name}.json")  # fmt: skip
        return 1

    final = S.evaluate(splats, scene, images, scene.test_idx, cfg.sh_degree, device,
                       lpips=not args.no_lpips)  # fmt: skip
    i0 = int(scene.test_idx[0])
    vm = torch.as_tensor(scene.w2c[i0 : i0 + 1], dtype=torch.float32, device=device)
    K = torch.as_tensor(scene.K[i0 : i0 + 1], dtype=torch.float32, device=device)
    with torch.no_grad():
        fps = benchmark(lambda: S.render(splats, vm, K, scene.widths[i0], scene.heights[i0],
                                         cfg.sh_degree), warmup=5, iters=30)  # fmt: skip
    ckpt = S.save_checkpoint(runs_dir(S.PROJECT, name, create=True) / "ckpt.pt", splats, cfg,
                             {"scene_dir": rel(scene_dir), "images": args.images})  # fmt: skip
    ply_mb = None
    if not args.no_ply:
        ply = S.export_ply(splats, outputs_dir(S.PROJECT, create=True) / f"{name}.ply")
        ply_mb = ply.stat().st_size / 2**20
        print(f"[ply] {rel(ply)} ({ply_mb:.1f} MiB)")
    info = env_summary()
    out = {
        "name": name,
        "status": status,
        "scene": rel(scene_dir),
        "images": args.images,
        "resolution": [int(scene.widths[0]), int(scene.heights[0])],
        "n_train": len(scene.train_idx),
        "n_test": len(scene.test_idx),
        "init": args.init,
        "n_init": res.n_init,
        "config": asdict(cfg),
        "final": {
            "psnr": final["psnr"],
            "ssim": final["ssim"],
            "lpips": final.get("lpips"),
            "n_gaussians": int(splats["means"].shape[0]),
            "train_time_s": res.train_time_s,
            "peak_vram_mb": res.peak_vram_mb,
            "render_ms": fps["mean_ms"],
            "render_fps": fps["fps"],
            "ply_mib": ply_mb,
            "densify_capped_at_step": res.densify_capped_at_step,
        },
        "stats": S.splat_stats(splats),
        "history": res.history,
        "eval_history": res.eval_history,
        "per_image": final["per_image"],
        "test_names": [scene.names[i] for i in scene.test_idx],
        "checkpoint": rel(ckpt),
        "hardware": {k: info.get(k) for k in ("gpu", "vram_gb", "torch", "cuda")},
        "platform": platform.platform(terse=True),
        "gsplat": __import__("gsplat").__version__,
    }
    path = save_json(out, _results_dir(args) / f"{name}.json")
    f = out["final"]
    print(
        f"[done] {name}: PSNR {f['psnr']:.2f} dB | SSIM {f['ssim']:.4f} | LPIPS "
        f"{f['lpips'] if f['lpips'] is None else round(f['lpips'], 4)} | {f['n_gaussians']} "
        f"Gaussians | train {f['train_time_s']:.0f} s | peak VRAM {f['peak_vram_mb']:.0f} MiB | "
        f"render {f['render_fps']:.0f} FPS -> {rel(path)}"
    )
    return 0


def _results_dir(args: argparse.Namespace) -> Path:
    if args.results_dir:
        return repo_root() / args.results_dir
    return project_dir(S.PROJECT) / "results"


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    sys.exit(main())
