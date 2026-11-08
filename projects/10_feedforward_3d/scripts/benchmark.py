#!/usr/bin/env python
"""Run every feed-forward method on every scene, evaluate the cameras, and sweep the frame count.

Run it under the GPU lock (about 5-8 min on an RTX 5060 Laptop GPU, 8 GB)::

    flock /tmp/cv-gpu.lock python projects/10_feedforward_3d/scripts/benchmark.py

Requires the COLMAP models of ``run_colmap.py`` (CPU) for the kitchen reference and the
COLMAP rows. Writes

* ``runs/10_feedforward_3d/preds/<scene>/<method>.npz`` - cached predictions (cameras, depth,
  confidence, point maps) that the notebooks load instead of re-running the models;
* ``projects/10_feedforward_3d/results/pose_metrics.csv`` - one row per (scene, method);
* ``projects/10_feedforward_3d/results/scaling.csv`` - runtime, peak VRAM and pose accuracy
  vs number of input frames (South Building, frames spread over all 128 photos);
* ``projects/10_feedforward_3d/results/benchmark_meta.json`` - hardware and software versions.

Options: ``--methods vggt da3-base``, ``--scenes kitchen``, ``--no-scaling``, ``--scaling-only``.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ff3d  # noqa: E402

from cvlab.io import save_json  # noqa: E402

SCENES = ["kitchen", "south_building", "tum_desk"]
SCALING_METRICS = ("auc@5", "auc@15", "auc@30", "abs_rot_med_deg", "ate_pct")


def free(runner) -> None:
    import torch

    del runner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def metric_row(scene: ff3d.Scene, rec: ff3d.Recon) -> dict:
    m = ff3d.evaluate_poses(rec.T_cw, scene.ref_T_cw, metric_units=scene.ref_kind == "mocap")
    return {"scene": scene.name, "method": rec.method, **m, "runtime_s": rec.runtime_s,
            "total_s": rec.total_s, "peak_vram_mb": rec.peak_vram_mb,
            "weights_vram_mb": rec.weights_vram_mb,
            "proc_h": rec.proc_hw[0], "proc_w": rec.proc_hw[1]}  # fmt: skip


def merge_csv(path: Path, df, keys: list[str]) -> None:
    import pandas as pd

    if path.exists():
        old = pd.read_csv(path)
        idx = old.set_index(keys).index.isin(df.set_index(keys).index)
        df = pd.concat([old[~idx], df], ignore_index=True)
    df.sort_values(keys).to_csv(path, index=False, float_format="%.5g")
    print(f"wrote {ff3d.rel(path)}")


def run_scenes(methods: list[str], scenes: list[str]) -> None:
    import pandas as pd
    import torch

    loaded = {s: ff3d.load_scene(s) for s in scenes}
    rows = []
    by_model: dict[str, list[str]] = {}
    for m in methods:
        by_model.setdefault(ff3d.METHODS[m][0], []).append(m)
    for ms in by_model.values():
        runner = ff3d.make_runner(ms[0])
        for method in ms:
            for name, scene in loaded.items():
                # Warm-up on 2 frames so the timed run excludes CUDA/cuDNN initialisation.
                ff3d.run_method(runner, method, scene.image_paths[:2])
                try:
                    rec = ff3d.predict_cached(scene, method, runner=runner, refresh=True)
                except torch.OutOfMemoryError:
                    rec = None  # free the activations only after the traceback is released
                if rec is None:
                    rows.append({"scene": name, "method": method, "oom": True})
                    print(f"{name:15s} {method:14s} OOM")
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue
                rows.append(metric_row(scene, rec))
                r = rows[-1]
                print(f"{name:15s} {method:14s} AUC@5 {r.get('auc@5', float('nan')):.3f} "
                      f"AUC@15 {r.get('auc@15', float('nan')):.3f} fwd {rec.runtime_s:.2f}s "
                      f"peak {rec.peak_vram_mb:.0f} MiB")  # fmt: skip
        free(runner)
    for scene in loaded.values():  # COLMAP on the same images (it is the kitchen reference)
        if scene.ref_kind != "colmap":
            rows.append(metric_row(scene, ff3d.predict_cached(scene, "colmap")))
    merge_csv(ff3d.results_path("pose_metrics.csv"), pd.DataFrame(rows), ["scene", "method"])


def run_scaling(methods: list[str]) -> None:
    import pandas as pd
    import torch

    cfg = ff3d.load_config()["scaling"]
    full = ff3d.load_scene(cfg["scene"], all_images=True)
    rows = []
    for method in methods:
        runner = ff3d.make_runner(method)
        ff3d.run_method(runner, method, full.image_paths[:2])  # warm-up
        for n in cfg["n_frames"]:
            sub = full.subset(ff3d.evenly_spaced(len(full), n))
            row = {"method": method, "n_frames": n}
            try:
                rec = ff3d.run_method(runner, method, sub.image_paths)
            except torch.OutOfMemoryError:
                row.update(oom=True)
                rows.append(row)
                print(f"{method:14s} N={n:4d} OOM")
                del sub
                gc.collect()
                torch.cuda.empty_cache()
                break
            # evaluate_poses needs >= 3 cameras (Sim(3) fit), so N = 2 has timing/memory only.
            m = ff3d.evaluate_poses(rec.T_cw, sub.ref_T_cw)
            row.update(oom=False, runtime_s=rec.runtime_s, total_s=rec.total_s,
                       peak_vram_mb=rec.peak_vram_mb, weights_vram_mb=rec.weights_vram_mb,
                       **{k: m.get(k, float("nan")) for k in SCALING_METRICS})
            rows.append(row)
            print(f"{method:14s} N={n:4d} fwd {rec.runtime_s:6.2f}s "
                  f"peak {rec.peak_vram_mb:6.0f} MiB AUC@15 {row['auc@15']:.3f}")  # fmt: skip
            del rec
            torch.cuda.empty_cache()
        free(runner)
    merge_csv(ff3d.results_path("scaling.csv"), pd.DataFrame(rows), ["method", "n_frames"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--methods", nargs="*", default=list(ff3d.METHODS))
    ap.add_argument("--scenes", nargs="*", default=SCENES)
    ap.add_argument("--scaling-methods", nargs="*", default=["vggt", "da3-base", "da3-large"])
    ap.add_argument("--no-scaling", action="store_true")
    ap.add_argument("--scaling-only", action="store_true")
    args = ap.parse_args()
    import torch

    from cvlab.device import env_summary, seed_everything

    seed_everything(0)
    if not args.scaling_only:
        run_scenes(args.methods, args.scenes)
    if not args.no_scaling:
        run_scaling(args.scaling_methods)
    info = env_summary()
    save_json({k: info.get(k) for k in ("gpu", "vram_gb", "compute_capability", "torch", "cuda",
                                        "python", "numpy")}
              | {"tf32": torch.backends.cuda.matmul.allow_tf32},
              ff3d.results_path("benchmark_meta.json"))  # fmt: skip


if __name__ == "__main__":
    main()
