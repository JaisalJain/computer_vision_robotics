#!/usr/bin/env python
"""Benchmark monocular depth models against Kinect depth on TUM RGB-D (fr1_desk, fr1_xyz).

For every model (loaded one at a time) and every evenly spaced frame it:

1. predicts depth / disparity at the input resolution (640x480),
2. aligns it to the sensor depth with each protocol valid for the output type
   (raw metric, median scale, least-squares scale+shift in the native space),
3. computes AbsRel / RMSE / delta1 over valid GT pixels plus error-vs-distance bins,
4. records peak VRAM over the evaluation loop, then times ``timing_iters`` back-to-back calls
   on one frame: end-to-end latency (pre-processing + network + resize to 640x480) and the
   network forward alone (CUDA-synchronised hooks).

Outputs (small, committed): ``projects/08_monocular_depth/results/``
``tum_per_frame.csv``, ``tum_summary.csv``, ``tum_depth_bins.csv``, ``tum_eval_meta.json``.

Usage::

    flock /tmp/cv-gpu.lock python projects/08_monocular_depth/scripts/eval_tum.py
    python projects/08_monocular_depth/scripts/eval_tum.py --frames 8 --models da3-small
"""

from __future__ import annotations

import argparse
import gc
import os
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mono_depth as md  # noqa: E402

from cvlab.bench import benchmark  # noqa: E402
from cvlab.device import env_summary, seed_everything  # noqa: E402
from cvlab.io import project_dir, repo_root, save_json, third_party_dir  # noqa: E402
from cvlab.tum import TUM_INTRINSICS  # noqa: E402

PROJ = project_dir(md.PROJECT)
RESULTS = PROJ / "results"


def rel(p: Path) -> str:
    return str(Path(p).resolve().relative_to(repo_root()))


def evaluate_model(key: str, seqs: dict, frames: dict, cfg: dict) -> tuple[list, list, dict]:
    import torch

    cuda = torch.cuda.is_available()
    baseline_mib = torch.cuda.memory_allocated() / 2**20 if cuda else 0.0  # should be ~0
    t0 = time.perf_counter()
    model = md.DepthModel(key, process_res=cfg["da3_process_res"])
    load_s = time.perf_counter() - t0
    if cuda:
        torch.cuda.synchronize()
        weights_mib = torch.cuda.memory_allocated() / 2**20
        torch.cuda.reset_peak_memory_stats()
    fx, fy, cx, cy = TUM_INTRINSICS[cfg["intrinsics"]][0]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])

    first = next(iter(seqs.values())).rgb(0)
    for _ in range(cfg["warmup_iters"]):
        model.predict(first, K)

    rows, bins, latencies, net_ms = [], [], [], []
    for name, seq in seqs.items():
        n_bins = len(md.DEPTH_BINS) - 1
        acc = {p: [np.zeros(n_bins) for _ in range(3)] for p in md.PROTOCOLS[model.output]}
        for i in frames[name]:
            rgb, gt = seq.rgb(int(i)), seq.depth(int(i))
            if cuda:
                torch.cuda.synchronize()
            t = time.perf_counter()
            pred = model.predict(rgb, K)
            if cuda:
                torch.cuda.synchronize()
            latency = (time.perf_counter() - t) * 1000
            latencies.append(latency)
            net_ms.append(pred.extra["network_ms"])
            mask = md.valid_mask(gt)
            for proto, res in md.evaluate_frame(pred.values, model.output, gt).items():
                rows.append({
                    "sequence": name, "frame": int(i), "timestamp": float(seq.timestamps[i]),
                    "model": key, "label": model.spec.label, "output": model.output,
                    "protocol": proto, **{k: v for k, v in res.items() if k != "depth"},
                    "gt_median_m": float(np.median(gt[mask])), "valid_frac": float(mask.mean()),
                    "latency_ms": latency, "network_ms": pred.extra["network_ms"],
                    "focal_pred_px": pred.focal_px,
                })  # fmt: skip
                for a, b in zip(acc[proto], md.binned_errors(res["depth"], gt)):
                    a += b
        for proto, (ar, sq, n) in acc.items():
            for j in range(len(n)):
                bins.append({
                    "sequence": name, "model": key, "label": model.spec.label, "protocol": proto,
                    "bin_lo_m": md.DEPTH_BINS[j], "bin_hi_m": md.DEPTH_BINS[j + 1],
                    "abs_rel": ar[j] / n[j] if n[j] else np.nan,
                    "rmse": np.sqrt(sq[j] / n[j]) if n[j] else np.nan, "n_pixels": int(n[j]),
                })  # fmt: skip
        print(f"  {key:>24s} | {name}: {len(frames[name])} frames", flush=True)

    peak_vram = torch.cuda.max_memory_allocated() / 2**20 if cuda else 0.0

    # Dedicated timing pass: one frame, back-to-back calls with CUDA sync (cvlab.bench).
    # The machine may be shared with CPU-heavy jobs, so min and p50 are both reported.
    bench_net: list[float] = []

    def call(m: md.DepthModel = model) -> None:
        m.predict(first, K)
        bench_net.append(m.last_network_ms)

    load_avg = os.getloadavg()[0]
    e2e = benchmark(call, warmup=cfg["warmup_iters"], iters=cfg["timing_iters"])
    net = np.asarray(bench_net[-cfg["timing_iters"]:])
    lat = np.asarray(latencies)
    info = {
        "label": model.spec.label, "repo": model.spec.repo, "output": model.output,
        "license": model.spec.license, "params_m": model.n_params / 1e6,
        "proc_hw": list(pred.proc_hw), "load_s": load_s,
        "weights_mib": weights_mib if cuda else 0.0, "peak_vram_mib": peak_vram,
        "latency_ms_median": e2e["p50_ms"], "latency_ms_min": e2e["min_ms"],
        "latency_ms_p90": e2e["p90_ms"], "network_ms_median": float(np.median(net)),
        "network_ms_min": float(net.min()), "timing_iters": cfg["timing_iters"],
        "cpu_load_avg_1min": load_avg, "baseline_mib": baseline_mib,
        "eval_loop_latency_ms_median": float(np.median(lat)),
        "eval_loop_network_ms_median": float(np.median(net_ms)), "n_frames": int(lat.size),
    }  # fmt: skip
    del model, call
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()
    return rows, bins, info


def summarise(per_frame: pd.DataFrame, models: dict) -> pd.DataFrame:
    metrics = ["abs_rel", "rmse", "delta1", "delta2", "rmse_log", "scale"]
    both = per_frame.assign(sequence="all")
    df = pd.concat([per_frame, both])
    out = df.groupby(["sequence", "model", "protocol"], sort=False)[metrics].mean().reset_index()
    out["scale_std"] = (
        df.groupby(["sequence", "model", "protocol"], sort=False)["scale"].std().to_numpy()
    )
    out["n_frames"] = df.groupby(["sequence", "model", "protocol"], sort=False).size().to_numpy()
    for col in ["label", "output", "params_m", "latency_ms_median", "network_ms_median",
                "peak_vram_mib"]:
        out[col] = out["model"].map({k: v[col] for k, v in models.items()})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(PROJ / "configs" / "eval_tum.yaml"))
    ap.add_argument("--frames", type=int, help="override frames_per_sequence")
    ap.add_argument("--models", nargs="*", help="override the model list")
    ap.add_argument("--out", default=str(RESULTS), help="output directory")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.frames:
        cfg["frames_per_sequence"] = args.frames
    if args.models:
        cfg["models"] = args.models
    md.MIN_DEPTH, md.MAX_DEPTH = cfg["valid_depth_m"]
    seed_everything(0)

    seqs = {name: md.load_tum(name) for name in cfg["sequences"]}
    frames = {n: md.select_frames(len(s), cfg["frames_per_sequence"]) for n, s in seqs.items()}
    rows, bins, models = [], [], {}
    for key in cfg["models"]:
        r, b, info = evaluate_model(key, seqs, frames, cfg)
        rows += r
        bins += b
        models[key] = info
        print(f"{key}: median latency {info['latency_ms_median']:.1f} ms, "
              f"peak VRAM {info['peak_vram_mib']:.0f} MiB", flush=True)  # fmt: skip

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    per_frame = pd.DataFrame(rows)
    # Keep full timestamp precision (seconds since 1970); metrics get 6 significant digits.
    per_frame.assign(timestamp=per_frame["timestamp"].map("{:.6f}".format)).to_csv(
        out / "tum_per_frame.csv", index=False, float_format="%.6g")  # fmt: skip
    summarise(per_frame, models).to_csv(out / "tum_summary.csv", index=False, float_format="%.6g")
    pd.DataFrame(bins).to_csv(out / "tum_depth_bins.csv", index=False, float_format="%.6g")
    da3_commit = subprocess.run(
        ["git", "-C", str(third_party_dir("depth-anything-3")), "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip()  # fmt: skip
    import transformers

    save_json({
        "date": date.today().isoformat(), "env": env_summary(),
        "transformers": transformers.__version__, "depth_anything_3_commit": da3_commit,
        "config": cfg, "frames": {k: v.tolist() for k, v in frames.items()},
        "sequences": {k: rel(s.root) for k, s in seqs.items()}, "models": models,
    }, out / "tum_eval_meta.json")  # fmt: skip
    print(f"wrote {rel(out)}/tum_per_frame.csv, tum_summary.csv, tum_depth_bins.csv, "
          "tum_eval_meta.json")  # fmt: skip


if __name__ == "__main__":
    main()
