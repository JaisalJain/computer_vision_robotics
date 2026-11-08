#!/usr/bin/env python
"""Turn your own photos or phone video into a COLMAP scene that ``train.py`` can use.

Pipeline (official pycolmap, CPU): frames -> SIFT features -> matching -> incremental SfM
-> undistortion to a PINHOLE camera. The output folder has the same layout as the
Mip-NeRF 360 scenes::

    <out>/input/         the (resized) frames that were reconstructed
    <out>/images/        undistorted images (train on these: --images images)
    <out>/sparse/0/      cameras.bin, images.bin, points3D.bin (PINHOLE)
    <out>/sfm_summary.json

Examples::

    # a folder of photos (walk around the object, 50-150 sharp, overlapping shots)
    python projects/11_gaussian_splatting/scripts/sfm_from_images.py \
        --images path/to/photos --out data/my_scene
    # a phone video: keep every 10th frame
    python projects/11_gaussian_splatting/scripts/sfm_from_images.py \
        --video path/to/clip.mp4 --every 10 --out data/my_scene
    python projects/11_gaussian_splatting/scripts/train.py --scene-dir data/my_scene \
        --images images --init colmap --steps 7000 --name my_scene_7k
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

from cvlab.io import iter_video_frames, read_image, repo_root, save_json, write_image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(repo_root()))
    except ValueError:
        return Path(path).name


def prepare_frames(args: argparse.Namespace, input_dir: Path) -> int:
    """Copy (resized) photos or sampled video frames into ``input_dir``."""
    input_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    if args.video:
        for i, frame in enumerate(
            iter_video_frames(args.video, stride=args.every, max_side=args.max_side,
                              max_frames=args.max_frames or None)  # fmt: skip
        ):
            write_image(input_dir / f"frame_{i:05d}.jpg", frame)
            n += 1
    else:
        files = sorted(p for p in Path(args.images).iterdir()
                       if p.suffix.lower() in IMAGE_SUFFIXES)  # fmt: skip
        files = files[:: args.every][: args.max_frames or None]
        for p in files:
            write_image(input_dir / f"{p.stem}.jpg", read_image(p, max_side=args.max_side))
            n += 1
    return n


def main() -> int:
    import pycolmap

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--images", help="folder of photos")
    src.add_argument("--video", help="video file (frames are sampled with --every)")
    ap.add_argument("--out", required=True, help="output scene folder (e.g. data/my_scene)")
    ap.add_argument("--every", type=int, default=1, help="keep every N-th frame/photo")
    ap.add_argument("--max-frames", type=int, default=0, help="cap the number of frames")
    ap.add_argument("--max-side", type=int, default=1024, help="resize longest side (px)")
    ap.add_argument("--matcher", choices=["exhaustive", "sequential"], default="exhaustive",
                    help="sequential is much faster for long videos")  # fmt: skip
    ap.add_argument("--camera-model", default="OPENCV", help="COLMAP model before undistortion")
    ap.add_argument("--num-threads", type=int, default=-1)
    args = ap.parse_args()

    out = Path(args.out)
    if not out.is_absolute():
        out = repo_root() / out
    if out.exists():
        shutil.rmtree(out)
    input_dir, work = out / "input", out / "distorted"
    work.mkdir(parents=True)
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    n = prepare_frames(args, input_dir)
    timings["frames_s"] = time.perf_counter() - t0
    print(f"[frames] {n} images -> {rel(input_dir)}")
    if n < 3:
        raise SystemExit("need at least 3 images")

    db = work / "database.db"
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = args.camera_model
    extract = pycolmap.FeatureExtractionOptions()
    extract.num_threads = args.num_threads
    t0 = time.perf_counter()
    pycolmap.extract_features(
        db, input_dir, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader,
        extraction_options=extract, device=pycolmap.Device.cpu,
    )  # fmt: skip
    timings["features_s"] = time.perf_counter() - t0
    matching = pycolmap.FeatureMatchingOptions()
    matching.num_threads = args.num_threads
    t0 = time.perf_counter()
    if args.matcher == "exhaustive":
        pycolmap.match_exhaustive(db, matching_options=matching, device=pycolmap.Device.cpu)
    else:
        pycolmap.match_sequential(db, matching_options=matching, device=pycolmap.Device.cpu)
    timings["matching_s"] = time.perf_counter() - t0
    print(f"[sfm] features {timings['features_s']:.1f} s, matching {timings['matching_s']:.1f} s")

    opts = pycolmap.IncrementalPipelineOptions()
    opts.num_threads = args.num_threads
    opts.random_seed = 0
    opts.mapper.random_seed = 0
    (work / "sparse").mkdir()
    t0 = time.perf_counter()
    recs = pycolmap.incremental_mapping(db, input_dir, work / "sparse", options=opts)
    timings["mapping_s"] = time.perf_counter() - t0
    if not recs:
        raise SystemExit("SfM failed: no reconstruction (more overlap / texture needed)")
    best_id, best = max(recs.items(), key=lambda kv: kv[1].num_reg_images())
    print(f"[sfm] mapping {timings['mapping_s']:.1f} s | model {best_id}: "
          f"{best.num_reg_images()}/{n} images, {best.num_points3D()} points, "
          f"mean reproj. error {best.compute_mean_reprojection_error():.3f} px")  # fmt: skip

    # Undistort to an ideal pinhole camera: gsplat's default rasteriser assumes no distortion.
    t0 = time.perf_counter()
    pycolmap.undistort_images(out, work / "sparse" / str(best_id), input_dir)
    timings["undistort_s"] = time.perf_counter() - t0
    sparse0 = out / "sparse" / "0"
    sparse0.mkdir(parents=True, exist_ok=True)
    for f in (out / "sparse").iterdir():
        if f.is_file():
            f.replace(sparse0 / f.name)
    for extra in ("stereo", "run-colmap-geometric.sh", "run-colmap-photometric.sh"):
        p = out / extra
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()
    rec = pycolmap.Reconstruction(str(sparse0))
    cam = next(iter(rec.cameras.values()))
    summary = {
        "source": rel(args.video or args.images),
        "n_frames": n,
        "n_registered": rec.num_reg_images(),
        "n_points": rec.num_points3D(),
        "mean_reprojection_error_px": rec.compute_mean_reprojection_error(),
        "camera_model": cam.model.name,
        "image_size": [cam.width, cam.height],
        "timings_s": timings,
    }
    save_json(summary, out / "sfm_summary.json")
    print(f"[ok] {rel(out)}: {summary['n_registered']} registered images, {cam.model.name} "
          f"{cam.width}x{cam.height} -> train.py --scene-dir {rel(out)} --images images")
    return 0


if __name__ == "__main__":
    sys.exit(main())
