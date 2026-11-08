#!/usr/bin/env python
"""MOT17 tracking benchmark: YOLO26 n/s detections + ByteTrack / BoT-SORT / OC-SORT / SORT.

Stages (run all by default)::

    gpu         YOLO26 detections cached per sequence (detector FPS + peak VRAM),
                end-to-end ``model.track()`` with bytetrack.yaml / botsort.yaml (FPS),
                and the two BoT-SORT + appearance variants whose encoders run on the GPU:
                Ultralytics' yolo26n-reid.onnx and DINOv3 ViT-S/16 crop embeddings
    cpu         every other tracker variant on the *same* cached detections
                (tracker-only FPS) + CLEAR / Identity / HOTA metrics
    crosscheck  re-score all tracker files with TrackEval (reference implementation)

Outputs: tracker files in ``runs/06_tracking/trackers/<detector>/<variant>/data/<seq>.txt``,
detection caches in ``runs/06_tracking/dets/``, and small JSON/CSV summaries in
``projects/06_tracking/results/``.

Usage (GPU stage must hold the GPU lock)::

    flock /tmp/cv-gpu.lock python projects/06_tracking/scripts/run_mot_benchmark.py
    python projects/06_tracking/scripts/run_mot_benchmark.py --stages cpu crosscheck
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("YOLO_AUTOINSTALL", "false")  # never pip-install at runtime

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import motlib as ml  # noqa: E402

from cvlab.device import env_summary  # noqa: E402
from cvlab.io import data_dir, load_json, project_dir, runs_dir, save_json, weights_dir  # noqa: E402

PROJECT = "06_tracking"
SEQUENCES = ["MOT17-02-FRCNN", "MOT17-05-FRCNN"]
DETECTORS = ["yolo26n", "yolo26s"]
DET_CONF = 0.1  # keep low-score boxes: ByteTrack's second association needs them
WARMUP_FRAMES = 10  # excluded from FPS (CUDA init, cuDNN autotuning)
DEVICE, PRECISION = 0, 16  # GPU 0 in fp16 (the CPU smoke test overrides these)
CFG = project_dir(PROJECT) / "configs"
# variant name -> (config, needs the GPU)
VARIANTS = {
    "sort_scratch": ("sort_scratch", False),
    "bytetrack_no_low": (CFG / "bytetrack_no_low.yaml", False),
    "bytetrack": (CFG / "bytetrack.yaml", False),
    "ocsort": (CFG / "ocsort.yaml", False),
    "botsort_no_gmc": (CFG / "botsort_no_gmc.yaml", False),
    "botsort": (CFG / "botsort.yaml", False),
    "botsort_reid": (CFG / "botsort_reid.yaml", True),
    "botsort_dinov3": (CFG / "botsort_reid.yaml", True),  # same settings, DINOv3 encoder
}
GPU_VARIANTS = [v for v, (_, needs_gpu) in VARIANTS.items() if needs_gpu]
_DINO = {}  # one DINOv3 encoder shared by all runs (loaded on first use)
RESULTS = project_dir(PROJECT) / "results"
RUNS = runs_dir(PROJECT)


def imgsz_for(seq: ml.MotSequence) -> int:
    """Native resolution for VGA (640), 1280 for 1080p frames (multiple of 32)."""
    return int(min(1280, 32 * np.ceil(max(seq.width, seq.height) / 32)))


def quiet_ultralytics() -> None:
    from ultralytics.utils import LOGGER

    LOGGER.setLevel(logging.WARNING)


def detect_sequence(weights: Path, seq: ml.MotSequence) -> tuple[ml.DetectionCache, dict]:
    """Run the detector over a sequence once and cache person boxes (conf >= DET_CONF)."""
    import torch
    from ultralytics import YOLO

    from cvlab.device import peak_vram_mb, reset_peak_vram

    model = YOLO(str(weights))
    imgsz = imgsz_for(seq)
    frames, boxes, confs, wall, speed = [], [], [], [], []
    reset_peak_vram()
    t_prev = time.perf_counter()
    src = str(seq.root / seq.img_dir)
    kw = {"conf": DET_CONF, "classes": [0], "imgsz": imgsz, "quantize": PRECISION, "device": DEVICE}
    for r in model.predict(source=src, stream=True, verbose=False, **kw):
        b = r.boxes.cpu().numpy()
        t = int(Path(r.path).stem)
        frames.append(np.full(len(b), t))
        boxes.append(b.xyxy)
        confs.append(b.conf)
        now = time.perf_counter()
        wall.append(now - t_prev)
        t_prev = now
        speed.append([r.speed["preprocess"], r.speed["inference"], r.speed["postprocess"]])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall_ms = 1000 * np.asarray(wall[WARMUP_FRAMES:])
    sp = np.asarray(speed[WARMUP_FRAMES:])
    cache = ml.DetectionCache(
        np.concatenate(frames).astype(int), np.concatenate(boxes).astype(np.float32),
        np.concatenate(confs).astype(np.float32),
        {"detector": weights.stem, "seq": seq.name, "imgsz": imgsz, "conf": DET_CONF},
    )  # fmt: skip
    stats = {
        "detector": weights.stem, "seq": seq.name, "imgsz": imgsz, "frames": seq.length,
        "resolution": f"{seq.width}x{seq.height}", "dets_per_frame": len(cache.conf) / seq.length,
        "wall_ms_per_frame": float(wall_ms.mean()), "wall_fps": float(1000 / wall_ms.mean()),
        "preprocess_ms": float(sp[:, 0].mean()), "inference_ms": float(sp[:, 1].mean()),
        "postprocess_ms": float(sp[:, 2].mean()), "peak_vram_mb": peak_vram_mb(),
    }  # fmt: skip
    return cache, stats


def track_end_to_end(weights: Path, seq: ml.MotSequence, tracker: str) -> tuple[np.ndarray, dict]:
    """``model.track()`` over the image folder (decode + detect + track), timed per frame."""
    import torch
    from ultralytics import YOLO

    from cvlab.device import peak_vram_mb, reset_peak_vram

    model = YOLO(str(weights))  # fresh model = fresh tracker state
    rows, wall = [], []
    reset_peak_vram()
    t_prev = time.perf_counter()
    # persist=True is required for image folders: with persist=False Ultralytics resets the
    # tracker whenever the source path changes, i.e. on every frame of an image sequence.
    for r in model.track(source=str(seq.root / seq.img_dir), tracker=tracker, persist=True,
                         stream=True, conf=DET_CONF, classes=[0], imgsz=imgsz_for(seq),
                         quantize=PRECISION, device=DEVICE, verbose=False):  # fmt: skip
        t = int(Path(r.path).stem)
        if r.boxes is not None and r.boxes.id is not None:
            ids = r.boxes.id.cpu().numpy()
            xywh = ml.xyxy_to_xywh(r.boxes.xyxy.cpu().numpy())
            conf = r.boxes.conf.cpu().numpy()
            rows += [(t, int(i), *b, c) for i, b, c in zip(ids, xywh, conf)]
        now = time.perf_counter()
        wall.append(now - t_prev)
        t_prev = now
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall_ms = 1000 * np.asarray(wall[WARMUP_FRAMES:])
    stats = {"e2e_ms_per_frame": float(wall_ms.mean()), "e2e_fps": float(1000 / wall_ms.mean()),
             "peak_vram_mb": peak_vram_mb()}  # fmt: skip
    return np.asarray(rows, dtype=np.float64).reshape(-1, 7), stats


def tracker_file(det: str, variant: str, seq: str, e2e: bool = False) -> Path:
    sub = "trackers_e2e" if e2e else "trackers"
    return RUNS / sub / det / variant / "data" / f"{seq}.txt"


def run_variant(det: str, variant: str, seq: ml.MotSequence, gt: np.ndarray) -> dict:
    cfg, needs_gpu = VARIANTS[variant]
    dev = "cuda:0" if needs_gpu and DEVICE != "cpu" else "cpu"  # only ReID encoders use the GPU
    if variant == "botsort_dinov3":
        if dev not in _DINO:
            from dinoreid import MODEL_ID, DinoV3ReID

            _DINO[dev] = (DinoV3ReID(device=dev), MODEL_ID)
        enc, name = _DINO[dev]
        tracker = ml.make_tracker(str(cfg), device=dev, encoder=enc, encoder_name=name)
    else:
        overrides = {}
        if variant == "botsort_reid":
            overrides["model"] = str(weights_dir("ultralytics", "yolo26n-reid.onnx"))
        tracker = ml.make_tracker(str(cfg), device=dev, **overrides)
    cache = ml.DetectionCache.load(RUNS / "dets" / f"{det}_{seq.name}.npz")
    import cv2

    def load_bgr(t: int) -> np.ndarray:
        return cv2.imread(str(seq.frame_path(t)))

    rows, timing = ml.run_tracker(tracker, cache, seq.length, frame_loader=load_bgr)
    path = ml.write_mot_txt(tracker_file(det, variant, seq.name), rows)
    res = ml.evaluate(gt, ml.load_mot_txt(path), seq.length)  # score exactly what is on disk
    return {"detector": det, "tracker": variant, "seq": seq.name, **timing,
            **ml.summary_row(res), "HOTA_alpha": res["HOTA_alpha"], "_raw": res}  # fmt: skip


def stage_gpu(seqs: dict[str, ml.MotSequence]) -> None:
    from cvlab.device import peak_vram_mb, reset_peak_vram

    quiet_ultralytics()
    out = {"env": env_summary(), "det_conf": DET_CONF, "detector_runs": [], "e2e_runs": [],
           "reid_runs": []}  # fmt: skip
    for det in DETECTORS:
        weights = weights_dir("ultralytics", f"{det}.pt")
        for name, seq in seqs.items():
            cache, stats = detect_sequence(weights, seq)
            cache.save(RUNS / "dets" / f"{det}_{name}.npz")
            out["detector_runs"].append(stats)
            print(f"[det ] {det} {name}: {stats['wall_fps']:.1f} FPS, "
                  f"{stats['peak_vram_mb']:.0f} MiB peak, "
                  f"{stats['dets_per_frame']:.1f} dets/frame")
            gt = seq.gt()
            for tracker in ("bytetrack.yaml", "botsort.yaml"):  # always explicit (default changed)
                rows, st = track_end_to_end(weights, seq, tracker)
                variant = tracker.removesuffix(".yaml")
                path = ml.write_mot_txt(tracker_file(det, variant, name, e2e=True), rows)
                res = ml.evaluate(gt, ml.load_mot_txt(path), seq.length)
                out["e2e_runs"].append({"detector": det, "tracker": variant, "seq": name, **st,
                                        **ml.summary_row(res)})  # fmt: skip
                print(f"[e2e ] {det} {name} {variant}: {st['e2e_fps']:.1f} FPS, "
                      f"MOTA {100 * res['MOTA']:.1f} IDF1 {100 * res['IDF1']:.1f}")
            for variant in GPU_VARIANTS:
                reset_peak_vram()
                row = run_variant(det, variant, seq, gt)
                row.pop("_raw")
                # Peak torch memory of the DINOv3 encoder; ONNX Runtime (yolo26n-reid.onnx)
                # allocates outside torch, so its memory is not visible here.
                row["peak_vram_mb"] = peak_vram_mb() if variant == "botsort_dinov3" else None
                out["reid_runs"].append(row)
                print(f"[reid] {det} {name} {variant}: HOTA {row['HOTA']} MOTA {row['MOTA']} "
                      f"IDF1 {row['IDF1']} IDSW {row['IDSW']} "
                      f"({row['tracker_fps']:.1f} tracker FPS)")
    save_json(out, RESULTS / "mot_gpu_runs.json")


def stage_cpu(seqs: dict[str, ml.MotSequence]) -> None:
    import pandas as pd

    quiet_ultralytics()
    gpu = load_json(RESULTS / "mot_gpu_runs.json")
    rows = list(gpu["reid_runs"])
    raw: dict[tuple[str, str], list[dict]] = {}
    for det in DETECTORS:
        for name, seq in seqs.items():
            gt = seq.gt()
            for variant, (_, needs_gpu) in VARIANTS.items():
                if needs_gpu:
                    continue
                row = run_variant(det, variant, seq, gt)
                raw.setdefault((det, variant), []).append(row.pop("_raw"))
                rows.append(row)
                print(f"[trk ] {det} {name} {variant}: HOTA {row['HOTA']} MOTA {row['MOTA']} "
                      f"IDF1 {row['IDF1']} IDSW {row['IDSW']} ({row['tracker_fps']:.0f} FPS)")
    # Combined (both sequences) scores, TrackEval-style, for the CPU variants; the GPU (ReID)
    # variants are re-evaluated from their saved tracker files.
    for det in DETECTORS:
        for variant in GPU_VARIANTS:
            res = []
            for name, seq in seqs.items():
                pred = ml.load_mot_txt(tracker_file(det, variant, name))
                res.append(ml.evaluate(seq.gt(), pred, seq.length))
            raw[(det, variant)] = res
    combined = []
    for (det, variant), res in raw.items():
        c = ml.combine(res)
        fps = [r["tracker_fps"] for r in rows if r["detector"] == det and r["tracker"] == variant]
        combined.append({"detector": det, "tracker": variant, "seq": "COMBINED",
                         "tracker_fps": float(np.mean(fps)), **ml.summary_row(c)})  # fmt: skip
    # model.track() vs cached detections + the same tracker: identical up to box clipping?
    consistency = []
    for e2e in gpu["e2e_runs"]:
        seq = seqs[e2e["seq"]]
        cached = ml.load_mot_txt(tracker_file(e2e["detector"], e2e["tracker"], seq.name))[:, :7]
        clipped = ml.evaluate(seq.gt(), ml.clip_rows(cached, seq.width, seq.height), seq.length)
        consistency.append({k: e2e[k] for k in ("detector", "tracker", "seq", "MOTA", "IDF1")}
                           | {"MOTA_cached_clipped": round(100 * clipped["MOTA"], 2),
                              "IDF1_cached_clipped": round(100 * clipped["IDF1"], 2)})  # fmt: skip
    identical = all(c["MOTA"] == c["MOTA_cached_clipped"] and c["IDF1"] == c["IDF1_cached_clipped"]
                    for c in consistency)  # fmt: skip
    print(f"[e2e ] model.track() == cached detections + tracker + clipping: {identical}")
    out = {"per_sequence": rows, "combined": combined, "e2e_consistency": consistency,
           "e2e_identical_after_clipping": identical}  # fmt: skip
    save_json(out, RESULTS / "mot_benchmark.json")
    cols = ["detector", "tracker", "seq", "HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW",
            "Frag", "CLR_FP", "CLR_FN", "MT", "ML", "tracker_fps"]  # fmt: skip
    df = pd.DataFrame([{k: r[k] for k in cols} for r in rows + combined])
    df.round(2).to_csv(RESULTS / "mot_benchmark.csv", index=False)
    print(df[df.seq == "COMBINED"].round(2).to_string(index=False))


def stage_crosscheck(seqs: dict[str, ml.MotSequence]) -> None:
    """Score every tracker file with TrackEval and compare with motlib (max abs difference)."""
    import trackeval

    bench = load_json(RESULTS / "mot_benchmark.json")
    all_rows = bench["per_sequence"] + bench["combined"]
    ours = {(r["detector"], r["tracker"], r["seq"]): r for r in all_rows}
    rows = []
    for det in DETECTORS:
        eval_cfg = trackeval.Evaluator.get_default_eval_config()
        eval_cfg.update({"USE_PARALLEL": False, "PRINT_RESULTS": False, "PRINT_CONFIG": False,
                         "TIME_PROGRESS": False, "OUTPUT_SUMMARY": False, "OUTPUT_DETAILED": False,
                         "PLOT_CURVES": False, "OUTPUT_EMPTY_CLASSES": False,
                         "DISPLAY_LESS_PROGRESS": True, "LOG_ON_ERROR": None})  # fmt: skip
        ds_cfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
        ds_cfg.update({"GT_FOLDER": str(data_dir("mot17")),
                       "TRACKERS_FOLDER": str(RUNS / "trackers" / det),
                       "BENCHMARK": "MOT17", "SKIP_SPLIT_FOL": True, "PRINT_CONFIG": False,
                       "SEQ_INFO": {n: s.length for n, s in seqs.items()},
                       "TRACKERS_TO_EVAL": list(VARIANTS)})  # fmt: skip
        evaluator = trackeval.Evaluator(eval_cfg)
        dataset = trackeval.datasets.MotChallenge2DBox(ds_cfg)
        metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR({"PRINT_CONFIG": False}),
                   trackeval.metrics.Identity({"PRINT_CONFIG": False})]  # fmt: skip
        out, _ = evaluator.evaluate([dataset], metrics)
        res = out["MotChallenge2DBox"]
        for variant in VARIANTS:
            for seq_name in [*seqs, "COMBINED_SEQ"]:
                r = res[variant][seq_name]["pedestrian"]
                ref = {"HOTA": 100 * float(np.mean(r["HOTA"]["HOTA"])),
                       "MOTA": 100 * float(r["CLEAR"]["MOTA"]),
                       "IDF1": 100 * float(r["Identity"]["IDF1"]),
                       "IDSW": int(r["CLEAR"]["IDSW"])}  # fmt: skip
                key = (det, variant, "COMBINED" if seq_name == "COMBINED_SEQ" else seq_name)
                mine = ours[key]
                row = {"detector": det, "tracker": variant, "seq": key[2]}
                for k, v in ref.items():
                    row[f"{k}_trackeval"] = round(v, 3)
                    row[f"{k}_ours"] = mine[k]
                row["max_abs_diff"] = max(abs(round(v, 2) - mine[k]) for k, v in ref.items())
                rows.append(row)
    worst = max(r["max_abs_diff"] for r in rows)
    save_json({"rows": rows, "max_abs_diff_percentage_points": worst},
              RESULTS / "trackeval_crosscheck.json")  # fmt: skip
    print(f"[xchk] {len(rows)} comparisons, max |TrackEval - ours| = {worst:.4f} "
          "(percentage points)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stages", nargs="+", default=["gpu", "cpu", "crosscheck"],
                    choices=["gpu", "cpu", "crosscheck"])  # fmt: skip
    args = ap.parse_args()
    seqs = {s: ml.load_sequence(data_dir("mot17", s)) for s in SEQUENCES}
    t0 = time.perf_counter()
    for stage in args.stages:
        {"gpu": stage_gpu, "cpu": stage_cpu, "crosscheck": stage_crosscheck}[stage](seqs)
    print(f"done in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
