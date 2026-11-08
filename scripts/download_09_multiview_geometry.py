#!/usr/bin/env python
"""Data + weights for project 09 (multi-view geometry). Idempotent: re-running is a no-op.

* Middlebury MiddEval3 quarter-resolution stereo pairs + left GT disparity
  (``MiddEval3-data-Q.zip`` 31 MB, ``MiddEval3-GT0-Q.zip`` 14 MB)
  -> ``data/middlebury/MiddEval3/{trainingQ,testQ}/<scene>/``
* COLMAP "South Building" (128 photos, 419 MB zip, research use) -> the photos are
  downscaled to 1024 px (longest side) into ``data/south_building/images/`` and the
  reference sparse model shipped with the dataset (if any) is copied to
  ``data/south_building/reference_sparse/``; the archive is deleted afterwards.
* TUM RGB-D fr1_desk (CC BY 4.0, shared with projects 08/12/15) -> ``data/tum/fr1_desk/``
* DISK (``kornia/disk``, depth checkpoint) and LightGlue-for-DISK (``kornia/lightglue``)
  weights -> ``weights/torch_hub/checkpoints/`` (where the notebooks load them from).

Usage::

    python scripts/download_09_multiview_geometry.py               # everything (~0.5 GB transfer)
    python scripts/download_09_multiview_geometry.py --no-south-building --no-weights
"""

from __future__ import annotations

import argparse
import io
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from download_shared import tum  # noqa: E402

from cvlab.download import download_file, extract_archive  # noqa: E402
from cvlab.io import data_dir, repo_root  # noqa: E402

MIDDLEBURY = {
    "data_Q": "https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-data-Q.zip",
    "gt0_Q": "https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-GT0-Q.zip",
}
SOUTH_BUILDING = "https://github.com/colmap/colmap/releases/download/3.11.1/south-building.zip"
SOUTH_BUILDING_MAX_SIDE = 1024


def rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(repo_root()))
    except ValueError:
        return str(path)


def fetch(url: str, dest: Path) -> Path:
    """Download with urllib (2 tries), falling back to ``curl`` (handles flaky CDN hosts)."""
    if dest.exists():
        return dest
    for attempt in range(2):
        try:
            return download_file(url, dest, timeout=30.0)
        except Exception as exc:  # noqa: BLE001 - any network error -> retry / fallback
            print(f"[warn] urllib attempt {attempt + 1} failed for {url}: {exc}")
    if shutil.which("curl") is None:
        raise RuntimeError(f"could not download {url}")
    part = dest.with_name(dest.name + ".part")
    subprocess.run(
        ["curl", "-fL", "--retry", "5", "--retry-delay", "3", "-C", "-", "-o", str(part), url],
        check=True,
    )
    part.replace(dest)
    return dest


def middlebury() -> Path:
    """MiddEval3 quarter-resolution training/test pairs + training GT."""
    root = data_dir("middlebury", create=True)
    for key, url in MIDDLEBURY.items():
        marker = root / f".done_{key}"
        if marker.exists():
            print(f"[skip] Middlebury {key} already extracted")
            continue
        archive = fetch(url, root / url.rsplit("/", 1)[-1])
        extract_archive(archive, root)
        marker.write_text(url + "\n")
        archive.unlink()
        print(f"[ok] Middlebury {key} -> {rel(root / 'MiddEval3')}")
    return root / "MiddEval3"


def south_building() -> Path:
    """South Building photos (downscaled) + the reference sparse model, if shipped."""
    import cv2
    import numpy as np

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
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # BGR, EXIF orientation applied
                h, w = img.shape[:2]
                s = SOUTH_BUILDING_MAX_SIDE / max(h, w)
                if s < 1.0:
                    size = (round(w * s), round(h * s))
                    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(img_dir / p.name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                n_img += 1
        # Reference model: the first folder holding cameras/images/points3D (.bin or .txt).
        model_files = [n for n in names if Path(n).name in {"cameras.bin", "cameras.txt"}]
        if model_files:
            src = str(Path(sorted(model_files, key=len)[0]).parent)
            ref = root / "reference_sparse"
            ref.mkdir(exist_ok=True)
            for n in names:
                if str(Path(n).parent) == src and not n.endswith("/"):
                    (ref / Path(n).name).write_bytes(io.BytesIO(zf.read(n)).getvalue())
            print(f"[ok] reference model {src}/ -> {rel(ref)}")
        else:
            print("[info] the archive ships no reference sparse model")
    marker.write_text(f"{SOUTH_BUILDING}\nimages={n_img} max_side={SOUTH_BUILDING_MAX_SIDE}\n")
    archive.unlink()
    print(f"[ok] South Building: {n_img} images -> {rel(img_dir)}")
    return root


def weights() -> None:
    """DISK + LightGlue checkpoints via kornia (CPU; cached in weights/torch_hub)."""
    sys.path.insert(0, str(repo_root() / "projects" / "09_multiview_geometry" / "scripts"))
    import kornia.feature as KF
    from mvgeo import set_torch_hub_dir

    hub = set_torch_hub_dir()
    KF.DISK.from_pretrained("depth")
    KF.LightGlue("disk")
    print(f"[ok] DISK + LightGlue weights in {rel(hub / 'checkpoints')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-south-building", action="store_true")
    ap.add_argument("--no-weights", action="store_true")
    args = ap.parse_args()

    middlebury()
    seq = tum("fr1_desk")
    print(f"[ok] TUM fr1_desk: {rel(seq)}")
    if not args.no_south_building:
        south_building()
    if not args.no_weights:
        weights()


if __name__ == "__main__":
    main()
