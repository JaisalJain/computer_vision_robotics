#!/usr/bin/env python
"""Feed-forward point-map initialisation with Depth Anything 3 (any-view, unposed images).

DA3 sees ``--n-views`` training images *without* any camera information and predicts, in one
forward pass, a depth map + confidence + camera pose + intrinsics per view. Back-projecting the
confident pixels gives a dense point map in DA3's own (arbitrary scale/frame) world. To use it
as the initialisation of a model trained with the COLMAP cameras, the DA3 camera centres are
aligned to the COLMAP centres of the same views with a least-squares similarity transform
(Umeyama, Sim(3)); the residuals of that fit measure how good the feed-forward poses are.

Writes
  * ``runs/11_gaussian_splatting/da3_init/<scene>_da3.npz``: aligned points + colours (+ a few
    depth maps for plotting)
  * ``projects/11_gaussian_splatting/results/da3_init.json``: timing, VRAM, alignment errors

Example::

    python projects/11_gaussian_splatting/scripts/make_da3_init.py --n-views 32
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import types
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import splatkit as S  # noqa: E402

from cvlab.io import add_third_party, project_dir, repo_root, runs_dir, save_json  # noqa: E402


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(repo_root()))
    except ValueError:
        return Path(path).name


def import_da3():
    """``DepthAnything3`` from the pinned clone (``bash scripts/setup_third_party.sh``).

    ``depth_anything_3.api`` imports ``moviepy.editor`` at import time but only uses it for
    Gaussian-splat videos; moviepy is not installed in the shared env, so a stub is inserted.
    """
    os.environ.setdefault("DA3_LOG_LEVEL", "ERROR")
    warnings.filterwarnings("ignore", message=".*register_new_resolver.*")
    add_third_party("depth-anything-3", "src")
    try:
        import moviepy.editor  # noqa: F401
    except ImportError:
        mp = types.ModuleType("moviepy")
        mp.editor = types.ModuleType("moviepy.editor")
        sys.modules.setdefault("moviepy", mp)
        sys.modules.setdefault("moviepy.editor", mp.editor)
    from depth_anything_3.api import DepthAnything3

    return DepthAnything3


def as44(T: np.ndarray) -> np.ndarray:
    out = np.tile(np.eye(4), (T.shape[0], 1, 1))
    out[:, : T.shape[1], :4] = T
    return out


def main() -> int:
    import torch
    from scipy.spatial import cKDTree

    from cvlab.device import seed_everything
    from cvlab.geometry import rotation_error_deg, umeyama_alignment
    from cvlab.pointcloud import depth_to_points

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene-dir", default="data/mipnerf360/bonsai")
    ap.add_argument("--images", default="images_4")
    ap.add_argument("--model", default="depth-anything/DA3-BASE")
    ap.add_argument("--n-views", type=int, default=32)
    ap.add_argument("--process-res", type=int, default=504)
    ap.add_argument("--conf-percentile", type=float, default=40.0,
                    help="drop pixels below this confidence percentile (DA3's GLB default)")
    ap.add_argument("--n-points", type=int, default=0,
                    help="points to keep (0 = as many as the COLMAP sparse model has)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()  # fmt: skip
    seed_everything(args.seed)

    scene_dir = repo_root() / args.scene_dir
    scene = S.load_colmap_scene(scene_dir, images=args.images)
    tr = scene.train_idx
    views = tr[np.linspace(0, len(tr) - 1, args.n_views).round().astype(int)]
    imgs = scene.load_images(views)

    DepthAnything3 = import_da3()
    t0 = time.perf_counter()
    model = DepthAnything3.from_pretrained(args.model).to("cuda").eval()
    t_load = time.perf_counter() - t0
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.inference_mode():
        pred = model.inference(imgs, process_res=args.process_res)  # no poses given
    torch.cuda.synchronize()
    t_inf = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 2**20
    del model
    torch.cuda.empty_cache()

    depth = np.asarray(pred.depth, dtype=np.float32)  # (N, h, w), DA3's arbitrary scale
    conf = np.asarray(pred.conf, dtype=np.float32)
    w2c_da3 = as44(np.asarray(pred.extrinsics, dtype=np.float64))  # (N, 4, 4) world->camera
    K_da3 = np.asarray(pred.intrinsics, dtype=np.float64)  # (N, 3, 3) at the processed size
    proc = np.asarray(pred.processed_images)  # (N, h, w, 3) uint8
    c2w_da3 = np.linalg.inv(w2c_da3)

    # --- Sim(3) alignment of the predicted camera centres to the COLMAP centres -----------
    c_da3 = c2w_da3[:, :3, 3]
    c_col = scene.centers[views]
    s, R, t = umeyama_alignment(c_da3, c_col, with_scale=True)
    c_al = (s * (R @ c_da3.T)).T + t
    center_err = np.linalg.norm(c_al - c_col, axis=1)
    rot_err = np.array(
        [rotation_error_deg(R @ c2w_da3[i, :3, :3], scene.c2w[v][:3, :3])
         for i, v in enumerate(views)]
    )  # fmt: skip

    # --- back-project confident pixels and move them into the COLMAP frame ---------------
    thr = np.percentile(conf, args.conf_percentile)
    pts, cols = [], []
    for i in range(len(views)):
        p_cam, c = depth_to_points(depth[i], K_da3[i], rgb=proc[i], mask=conf[i] >= thr)
        p_w = p_cam @ c2w_da3[i, :3, :3].T + c2w_da3[i, :3, 3]
        pts.append((s * (R @ p_w.T)).T + t)
        cols.append(c)
    pts_all = np.concatenate(pts).astype(np.float32)
    cols_all = np.concatenate(cols).astype(np.float32)
    n_keep = args.n_points or len(scene.points)
    rng = np.random.default_rng(args.seed)
    sel = rng.choice(len(pts_all), size=min(n_keep, len(pts_all)), replace=False)
    pts_keep, cols_keep = pts_all[sel], cols_all[sel]

    # --- agreement with the COLMAP sparse points (in units of the scene scale) ----------
    good = (scene.points_track_len >= 3) & (scene.points_error < 1.5)
    sfm = scene.points[good]
    d_sfm_to_da3, _ = cKDTree(pts_keep).query(sfm, k=1)
    d_da3_to_sfm, _ = cKDTree(sfm).query(pts_keep, k=1)
    ss = scene.scene_scale

    out_npz = Path(args.out) if args.out else runs_dir(S.PROJECT, "da3_init", create=True) / (
        f"{scene_dir.name}_da3.npz"
    )
    show = np.linspace(0, len(views) - 1, 4).round().astype(int)
    np.savez_compressed(
        out_npz, points=pts_keep, colors=cols_keep, views=views, sim3_s=s, sim3_R=R,
        sim3_t=t, centers_da3_aligned=c_al, show_views=views[show],
        show_depth=depth[show].astype(np.float16), show_conf=conf[show].astype(np.float16),
        show_rgb=proc[show], conf_threshold=thr,
    )  # fmt: skip
    res = {
        "model": args.model,
        "scene": rel(scene_dir),
        "n_views": int(len(views)),
        "view_names": [scene.names[v] for v in views],
        "process_res": args.process_res,
        "processed_hw": list(depth.shape[1:]),
        "load_s": t_load,
        "inference_s": t_inf,
        "peak_vram_mb": peak,
        "conf_percentile": args.conf_percentile,
        "n_points_raw": int(len(pts_all)),
        "n_points_kept": int(len(pts_keep)),
        "n_points_colmap": int(len(scene.points)),
        "sim3_scale": float(s),
        "scene_scale": ss,
        "center_rmse": float(np.sqrt((center_err**2).mean())),
        "center_rmse_rel": float(np.sqrt((center_err**2).mean()) / ss),
        "center_err_median": float(np.median(center_err)),
        "rot_err_median_deg": float(np.median(rot_err)),
        "rot_err_max_deg": float(rot_err.max()),
        "colmap_to_da3_median": float(np.median(d_sfm_to_da3)),
        "colmap_to_da3_median_rel": float(np.median(d_sfm_to_da3) / ss),
        "da3_to_colmap_median": float(np.median(d_da3_to_sfm)),
        "da3_to_colmap_median_rel": float(np.median(d_da3_to_sfm) / ss),
        "n_colmap_eval_points": int(good.sum()),
        "npz": rel(out_npz),
    }
    path = save_json(res, project_dir(S.PROJECT) / "results" / "da3_init.json")
    print(
        f"[da3] {len(views)} views at {depth.shape[2]}x{depth.shape[1]} | inference "
        f"{t_inf:.1f} s | peak VRAM {peak:.0f} MiB | Sim(3) scale {s:.3f} | centre RMSE "
        f"{res['center_rmse_rel'] * 100:.2f}% of scene scale | median rot err "
        f"{res['rot_err_median_deg']:.2f} deg | kept {len(pts_keep)} / {len(pts_all)} points"
    )
    print(f"[da3] -> {rel(out_npz)}, {rel(path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
