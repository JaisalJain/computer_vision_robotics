#!/usr/bin/env python
"""Data and weights for ``projects/05_segmentation`` (idempotent).

* coco8-seg and coco128-seg (Ultralytics; 8 / 128 COCO train2017 images with instance
  polygons) -> ``data/ultralytics/coco8-seg`` and ``data/ultralytics/coco128-seg``
* YOLO26 n/s segmentation checkpoints -> ``weights/ultralytics/``
* DAVIS 2017 TrainVal 480p, only the sequences listed in ``DAVIS_SEQUENCES``
  -> ``data/davis/DAVIS/{JPEGImages,Annotations}/480p/<seq>/``. The 833 MB zip is never
  downloaded in full: its central directory and the wanted members are read with HTTP
  range requests (the server supports ``Accept-Ranges: bytes``). ``--full-zip`` falls back
  to downloading the whole archive once, extracting the sequences and deleting it.
* TUM RGB-D fr1_desk (shared with projects 08/12/15) for a robotics desk scene.
* Hugging Face checkpoints (HF cache): SAM 3 (gated: accept the licence on the model page
  and ``hf auth login``; only ``model.safetensors`` + configs, ~3.4 GB), SAM 2.1
  hiera tiny/small/base-plus and Grounding DINO tiny (safetensors only).

Usage::

    python scripts/download_05_segmentation.py              # everything
    python scripts/download_05_segmentation.py --no-hf      # datasets + YOLO weights only
"""

from __future__ import annotations

import argparse
import io
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

from cvlab.download import download_file, extract_archive, hf_download
from cvlab.io import data_dir, weights_dir

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_shared import tum  # noqa: E402

ASSETS = "https://github.com/ultralytics/assets/releases/download"
COCO_SEG_SETS = ["coco8-seg", "coco128-seg"]
YOLO_WEIGHTS = {
    "yolo26n-seg.pt": f"{ASSETS}/v8.4.0/yolo26n-seg.pt",
    "yolo26s-seg.pt": f"{ASSETS}/v8.4.0/yolo26s-seg.pt",
}
DAVIS_URL = "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip"
# Both from the DAVIS 2017 *val* split. car-roundabout: 1 object, 75 frames (a car turning
# through a roundabout: scale and viewpoint change); dogs-jump: 3 objects (a person and two
# dogs), 66 frames (fast motion, occlusion, similar-looking instances).
DAVIS_SEQUENCES = ["car-roundabout", "dogs-jump"]

HF_MODELS = {
    # repo id: allow_patterns (skip duplicate .pt/.bin checkpoints)
    "facebook/sam3": ["*.json", "*.txt", "model.safetensors", "LICENSE", "README.md"],
    "facebook/sam2.1-hiera-tiny": ["*.json", "model.safetensors"],
    "facebook/sam2.1-hiera-small": ["*.json", "model.safetensors"],
    "facebook/sam2.1-hiera-base-plus": ["*.json", "model.safetensors"],
    "IDEA-Research/grounding-dino-tiny": ["*.json", "*.txt", "model.safetensors"],
}


# --------------------------------------------------------------------------- COCO-seg
def coco_seg(name: str) -> Path:
    """Download + extract an Ultralytics COCO-seg subset into ``data/ultralytics/<name>``."""
    root = data_dir("ultralytics", create=True)
    dest = root / name
    if (dest / "images").is_dir() and (dest / "labels").is_dir():
        print(f"[skip] data/ultralytics/{name} already present")
        return dest
    archive = download_file(f"{ASSETS}/v0.0.0/{name}.zip", root / f"{name}.zip")
    tmp = root / f".{name}.extract"
    shutil.rmtree(tmp, ignore_errors=True)
    extract_archive(archive, tmp)
    shutil.rmtree(dest, ignore_errors=True)
    (tmp / name).rename(dest)
    shutil.rmtree(tmp, ignore_errors=True)
    archive.unlink()
    print(f"[ok] data/ultralytics/{name} ({len(list(dest.glob('images/*/*')))} images)")
    return dest


def yolo_weights() -> list[Path]:
    out = weights_dir("ultralytics", create=True)
    return [download_file(url, out / name) for name, url in YOLO_WEIGHTS.items()]


# ------------------------------------------------------------------------------ DAVIS
class HttpRangeFile(io.RawIOBase):
    """Read-only, seekable file object over HTTP range requests (with a block cache).

    ``zipfile.ZipFile`` only needs ``seek``/``tell``/``read``, so this lets us list the
    archive and extract single members without downloading the whole file.
    """

    def __init__(self, url: str, block: int = 4 << 20, timeout: float = 60.0) -> None:
        self.url, self.block, self.timeout = url, block, timeout
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.headers.get("Accept-Ranges") != "bytes":
                raise OSError("server does not support range requests")
            self.size = int(resp.headers["Content-Length"])
        self.pos = 0
        self._cache: dict[int, bytes] = {}
        self.fetched = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = max(0, base + offset)
        return self.pos

    def _get_block(self, idx: int) -> bytes:
        if idx not in self._cache:
            start = idx * self.block
            end = min(self.size, start + self.block) - 1
            req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
            for attempt in range(4):
                try:
                    with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                        data = resp.read()
                    break
                except OSError:
                    if attempt == 3:
                        raise
            if len(self._cache) > 16:  # keep memory bounded (~64 MB)
                self._cache.pop(next(iter(self._cache)))
            self._cache[idx] = data
            self.fetched += len(data)
        return self._cache[idx]

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        out = bytearray()
        while n > 0:
            idx, off = divmod(self.pos, self.block)
            chunk = self._get_block(idx)[off : off + n]
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, b) -> int:  # pragma: no cover - used by some zipfile code paths
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def _davis_members(names: list[str], seq: str) -> list[str]:
    prefixes = (f"DAVIS/JPEGImages/480p/{seq}/", f"DAVIS/Annotations/480p/{seq}/")
    return [n for n in names if n.startswith(prefixes) and not n.endswith("/")]


def _extract_members(zf: zipfile.ZipFile, members: list[str], dest: Path) -> None:
    for name in members:
        target = dest / name
        if not target.resolve().is_relative_to(dest.resolve()):
            raise ValueError(f"unsafe path in archive: {name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".part")
        with zf.open(name) as src, open(tmp, "wb") as f:
            shutil.copyfileobj(src, f)
        tmp.replace(target)


def davis(sequences: list[str] = DAVIS_SEQUENCES, full_zip: bool = False) -> Path:
    """Fetch selected DAVIS 2017 480p sequences into ``data/davis/DAVIS``; returns that dir."""
    root = data_dir("davis", create=True)
    todo = [s for s in sequences if not (root / "DAVIS" / ".done" / s).exists()]
    if not todo:
        print(f"[skip] DAVIS sequences already present: {', '.join(sequences)}")
        return root / "DAVIS"
    if full_zip:
        archive = download_file(DAVIS_URL, root / "DAVIS-2017-trainval-480p.zip")
        zf = zipfile.ZipFile(archive)
    else:
        remote = HttpRangeFile(DAVIS_URL)
        zf = zipfile.ZipFile(io.BufferedReader(remote, buffer_size=1 << 16))
    with zf:
        names = zf.namelist()
        for seq in todo:
            members = _davis_members(names, seq)
            if not members:
                raise KeyError(f"sequence {seq!r} not found in the DAVIS archive")
            _extract_members(zf, members, root)
            (root / "DAVIS" / ".done").mkdir(parents=True, exist_ok=True)
            (root / "DAVIS" / ".done" / seq).write_text(f"{len(members)} files\n")
            n_jpg = sum(m.endswith(".jpg") for m in members)
            print(f"[ok] data/davis/DAVIS/*/480p/{seq} ({n_jpg} frames)")
    if full_zip:
        archive.unlink()
    else:
        got, total = remote.fetched / 2**20, remote.size / 2**20
        print(f"[info] fetched {got:.1f} MiB of the {total:.0f} MiB zip")
    return root / "DAVIS"


# ------------------------------------------------------------------------- HF weights
def hf_models() -> None:
    for repo, patterns in HF_MODELS.items():
        try:
            path = hf_download(repo, None, allow_patterns=patterns)
        except Exception as exc:  # noqa: BLE001 - gated repos raise various HTTP errors
            print(f"[warn] {repo}: {type(exc).__name__}: {exc}".splitlines()[0])
            if repo == "facebook/sam3":
                print(
                    "       SAM 3 is gated: request access on its model page, then `hf auth login`."
                )
            continue
        size = sum(f.stat().st_size for f in path.iterdir() if f.is_file())
        print(f"[ok] {repo} ({size / 2**20:.0f} MiB in the HF cache)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-hf", action="store_true", help="skip Hugging Face checkpoints")
    ap.add_argument("--full-zip", action="store_true", help="download the full 833 MB DAVIS zip")
    args = ap.parse_args()
    for name in COCO_SEG_SETS:
        coco_seg(name)
    yolo_weights()
    davis(full_zip=args.full_zip)
    tum("fr1_desk")
    if not args.no_hf:
        hf_models()


if __name__ == "__main__":
    main()
