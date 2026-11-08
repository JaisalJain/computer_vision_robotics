#!/usr/bin/env python
"""Data + weights for project 10 (feed-forward 3D). Idempotent: re-running is a no-op.

* Kitchen: 25 PNG views from the examples folder of the VGGT repository (pinned commit)
  -> ``data/10_feedforward_3d/kitchen/images/`` (copied from ``third_party/vggt`` when the
  pinned clone exists, otherwise fetched from GitHub at the same commit).
* COLMAP "South Building" (128 photos, research use; same layout as project 09): photos are
  downscaled to 1024 px into ``data/south_building/images/`` and the reference sparse model
  shipped with the dataset is copied to ``data/south_building/reference_sparse/``.
* TUM RGB-D fr1_desk (CC BY 4.0, shared with projects 08/09/12/15) -> ``data/tum/fr1_desk/``,
  plus 24 sharp keyframes, undistorted with the published calibration, written to
  ``data/10_feedforward_3d/tum_fr1_desk/{rgb,depth}/`` with ``groundtruth.txt`` (TUM format,
  camera-to-world) and ``camera.json`` (pinhole K of the undistorted images).
* Weights into the Hugging Face cache: ``facebook/VGGT-1B`` (5.0 GB fp32, CC-BY-NC-4.0),
  ``depth-anything/DA3-BASE`` (0.54 GB, Apache-2.0), ``depth-anything/DA3-LARGE-1.1``
  (1.6 GB, CC BY-NC 4.0 per the DA3 model table).

Usage::

    python scripts/download_10_feedforward_3d.py                # everything (~7.2 GB of weights)
    python scripts/download_10_feedforward_3d.py --no-weights   # scenes only
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from download_shared import tum  # noqa: E402

from cvlab.download import download_file, hf_download  # noqa: E402
from cvlab.io import data_dir, repo_root, third_party_dir  # noqa: E402

VGGT_COMMIT = "a288dd0f14786c93483e45524328726ab7b1b4ce"
KITCHEN_RAW = (
    "https://raw.githubusercontent.com/facebookresearch/vggt/"
    f"{VGGT_COMMIT}/examples/kitchen/images/{{:02d}}.png"
)
SOUTH_BUILDING = "https://github.com/colmap/colmap/releases/download/3.11.1/south-building.zip"
SOUTH_BUILDING_MAX_SIDE = 1024
TUM_KEYFRAMES, TUM_WINDOW = 24, 10  # must match configs/ff3d.yaml (tum_desk)

HF_MODELS = {
    "depth-anything/DA3-BASE": ["config.json", "model.safetensors"],  # 0.54 GB
    "depth-anything/DA3-LARGE-1.1": ["config.json", "model.safetensors"],  # 1.64 GB
    "facebook/VGGT-1B": ["config.json", "model.safetensors"],  # 5.03 GB
}


def rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(repo_root()))
    except ValueError:
        return str(path)


def fetch(url: str, dest: Path) -> Path:
    """Download with urllib (2 tries), falling back to ``curl`` (handles flaky hosts)."""
    if dest.exists():
        return dest
    for attempt in range(2):
        try:
            return download_file(url, dest, timeout=30.0, quiet=True)
        except Exception as exc:  # noqa: BLE001 - any network error -> retry / fallback
            print(f"[warn] attempt {attempt + 1} failed for {url}: {exc}")
    if shutil.which("curl") is None:
        raise RuntimeError(f"could not download {url}")
    part = dest.with_name(dest.name + ".part")
    subprocess.run(
        ["curl", "-fL", "--retry", "5", "--retry-delay", "3", "-C", "-", "-o", str(part), url],
        check=True,
    )
    part.replace(dest)
    return dest


# ------------------------------------------------------------------------------ kitchen
def kitchen() -> Path:
    """25 views of the VGGT 'kitchen' example (779x520 PNG)."""
    out = data_dir("10_feedforward_3d", "kitchen", "images", create=True)
    marker = out.parent / ".done"
    if marker.exists():
        print(f"[skip] kitchen already in {rel(out)}")
        return out
    src = third_party_dir("vggt", "examples", "kitchen", "images")
    for i in range(25):
        dest = out / f"{i:02d}.png"
        if (src / dest.name).is_file():
            shutil.copy2(src / dest.name, dest)
        else:
            fetch(KITCHEN_RAW.format(i), dest)
    marker.write_text(f"facebookresearch/vggt@{VGGT_COMMIT} examples/kitchen/images\n")
    print(f"[ok] kitchen: 25 images -> {rel(out)}")
    return out


# ----------------------------------------------------------------------- south building
def south_building() -> Path:
    """South Building photos (downscaled) + the reference sparse model shipped with them."""
    import cv2

    root = data_dir("south_building", create=True)
    marker = root / ".done"
    if marker.exists():
        print(f"[skip] South Building already prepared in {rel(root)}")
        return root
    archive = fetch(SOUTH_BUILDING, root / "south-building.zip")
    img_dir = root / "images"
    img_dir.mkdir(exist_ok=True)
    n_img = 0
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        for name in names:
            p = Path(name)
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and p.parent.name == "images":
                buf = np.frombuffer(zf.read(name), np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                h, w = img.shape[:2]
                s = SOUTH_BUILDING_MAX_SIDE / max(h, w)
                if s < 1.0:
                    size = (round(w * s), round(h * s))
                    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(img_dir / p.name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                n_img += 1
        model_files = [n for n in names if Path(n).name in {"cameras.bin", "cameras.txt"}]
        if model_files:
            src = str(Path(sorted(model_files, key=len)[0]).parent)
            ref = root / "reference_sparse"
            ref.mkdir(exist_ok=True)
            for n in names:
                if str(Path(n).parent) == src and not n.endswith("/"):
                    (ref / Path(n).name).write_bytes(zf.read(n))
    marker.write_text(f"{SOUTH_BUILDING}\nimages={n_img} max_side={SOUTH_BUILDING_MAX_SIDE}\n")
    archive.unlink()
    print(f"[ok] South Building: {n_img} images -> {rel(img_dir)}")
    return root


# ----------------------------------------------------------------------------- TUM desk
def tum_desk_keyframes() -> Path:
    """Undistorted sharp keyframes of TUM fr1_desk with mocap poses and Kinect depth."""
    import cv2

    from cvlab.tum import load_sequence, write_trajectory

    out = data_dir("10_feedforward_3d", "tum_fr1_desk", create=True)
    marker = out / ".done"
    if marker.exists():
        print(f"[skip] TUM fr1_desk keyframes already in {rel(out)}")
        return out
    seq = load_sequence(tum("fr1_desk"))
    K = seq.camera.K
    w, h = seq.camera.width, seq.camera.height
    # alpha=0: crop to the region where every pixel is valid after undistortion.
    K_new, _ = cv2.getOptimalNewCameraMatrix(K, seq.dist, (w, h), 0.0, (w, h))
    m1, m2 = cv2.initUndistortRectifyMap(K, seq.dist, None, K_new, (w, h), cv2.CV_32FC1)
    (out / "rgb").mkdir(exist_ok=True)
    (out / "depth").mkdir(exist_ok=True)
    picked, stamps, poses = [], [], []
    for k in range(TUM_KEYFRAMES):
        window = range(k * TUM_WINDOW, (k + 1) * TUM_WINDOW)
        # Sharpest frame in the window = largest variance of the Laplacian (least motion blur).
        sharp = [cv2.Laplacian(cv2.cvtColor(seq.rgb(i), cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
                 for i in window]  # fmt: skip
        i = window[int(np.argmax(sharp))]
        rgb = cv2.remap(seq.rgb(i), m1, m2, cv2.INTER_LINEAR)
        raw = cv2.imread(str(seq.depth_paths[i]), cv2.IMREAD_UNCHANGED)  # uint16, 5000 = 1 m
        depth = cv2.remap(raw, m1, m2, cv2.INTER_NEAREST)  # never blend depth across edges
        cv2.imwrite(str(out / "rgb" / f"{k:03d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / "depth" / f"{k:03d}.png"), depth)
        picked.append(int(i))
        stamps.append(float(seq.timestamps[i]))
        poses.append(seq.gt_poses[i])
    write_trajectory(out / "groundtruth.txt", np.array(stamps), np.stack(poses))
    meta = {
        "source": "TUM RGB-D rgbd_dataset_freiburg1_desk (CC BY 4.0)",
        "K": K_new.tolist(), "width": w, "height": h, "depth_scale": 5000.0,
        "source_frame_index": picked, "timestamps": stamps,
        "note": "undistorted with the published fr1 calibration (alpha=0, same image size)",
    }  # fmt: skip
    (out / "camera.json").write_text(json.dumps(meta, indent=2) + "\n")
    marker.write_text("ok\n")
    print(f"[ok] TUM fr1_desk: {len(picked)} undistorted keyframes -> {rel(out)}")
    return out


def weights() -> None:
    for repo, files in HF_MODELS.items():
        for name in files:  # hf_hub_download returns at once when the file is cached
            hf_download(repo, name)
        print(f"[ok] {repo} ({', '.join(files)}) in the Hugging Face cache")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-weights", action="store_true", help="only prepare the scenes")
    args = ap.parse_args()
    kitchen()
    south_building()
    tum_desk_keyframes()
    if not args.no_weights:
        weights()
    for name in ("vggt", "depth-anything-3"):
        if not third_party_dir(name).is_dir():
            print(f"[note] {name} code missing: run `bash scripts/setup_third_party.sh {name}`")
        else:
            print(f"[ok] third_party/{name} (pinned clone) present")


if __name__ == "__main__":
    main()
