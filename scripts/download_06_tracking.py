#!/usr/bin/env python
"""Data and weights for ``projects/06_tracking`` (idempotent: re-running is a no-op).

* MOT17 train sequences MOT17-02 and MOT17-05 (FRCNN folder: images, ``gt/gt.txt``,
  ``det/det.txt``, ``seqinfo.ini``) -> ``data/mot17/<seq>/``. Licence: CC BY-NC-SA 3.0.
  The official ``MOT17.zip`` is 5.9 GB because every train sequence is stored three times
  (DPM/FRCNN/SDP) next to the test split. The server supports HTTP range requests, so we
  read only the zip's central directory, then fetch the contiguous byte span that holds
  one sequence (about 100 MB for MOT17-02, 31 MB for MOT17-05) and decompress it locally.
* YOLO26 n/s COCO detectors and the yolo26n-reid.onnx appearance encoder
  -> ``weights/ultralytics/`` (AGPL-3.0).
* CoTracker3 ``scaled_offline.pth`` / ``scaled_online.pth`` -> Hugging Face cache
  (``facebook/cotracker3``, CC-BY-NC-4.0). The CoTracker *code* is the pinned clone made by
  ``scripts/setup_third_party.sh`` (its ``assets/apple.mp4`` demo clip is used as well).
* DINOv3 ViT-S/16 (``facebook/dinov3-vits16-pretrain-lvd1689m``, 83 MB, gated: accept the
  DINOv3 licence on the model page first) -> Hugging Face cache; the appearance (ReID)
  encoder of the BoT-SORT + DINOv3 variant. Skipped with a note when access is missing.
* SAM 3 (``facebook/sam3``, gated: request access on the model page first) -> Hugging Face
  cache; only needed by notebook 03. Skipped with a note when access is missing.

Usage::

    python scripts/download_06_tracking.py              # everything
    python scripts/download_06_tracking.py --no-sam3    # skip the 3.3 GB SAM 3 checkpoint
"""

from __future__ import annotations

import argparse
import configparser
import io
import shutil
import urllib.request
import zipfile
import zlib
from pathlib import Path

from cvlab.download import download_file, hf_download, human_size
from cvlab.io import data_dir, third_party_dir, weights_dir

MOT17_URL = "https://motchallenge.net/data/MOT17.zip"
SEQUENCES = ["MOT17-02-FRCNN", "MOT17-05-FRCNN"]
ASSETS = "https://github.com/ultralytics/assets/releases/download/v8.4.0"
YOLO_WEIGHTS = ["yolo26n.pt", "yolo26s.pt", "yolo26n-reid.onnx"]  # ReID: BoT-SORT ablation
COTRACKER_FILES = ["scaled_offline.pth", "scaled_online.pth"]
DINOV3_REPO = "facebook/dinov3-vits16-pretrain-lvd1689m"
DINOV3_FILES = ["config.json", "preprocessor_config.json", "model.safetensors"]
SAM3_FILES = [
    "config.json", "processor_config.json", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "vocab.json", "merges.txt", "model.safetensors",
]  # fmt: skip
_UA = {"User-Agent": "cvlab-downloader/0.1"}


class HttpRangeFile(io.RawIOBase):
    """Read-only, seekable file object over HTTP ``Range`` requests (for ``zipfile``)."""

    def __init__(self, url: str, timeout: float = 60.0) -> None:
        self.url, self.timeout, self.pos = url, timeout, 0
        req = urllib.request.Request(url, method="HEAD", headers=_UA)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if r.headers.get("Accept-Ranges") != "bytes":
                raise OSError(f"{url} does not support HTTP range requests")
            self.size = int(r.headers["Content-Length"])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        base = {0: 0, 1: self.pos, 2: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def fetch(self, start: int, end: int) -> bytes:
        """Bytes ``[start, end)`` of the remote file."""
        hdr = {**_UA, "Range": f"bytes={start}-{end - 1}"}
        req = urllib.request.Request(self.url, headers=hdr)
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = r.read()
        if len(data) != end - start:
            raise OSError(f"short read: wanted {end - start} bytes, got {len(data)}")
        return data

    def readinto(self, b) -> int:
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        data = self.fetch(self.pos, self.pos + n)
        b[:n] = data
        self.pos += n
        return n


def _member_bytes(span: bytes, span_start: int, info: zipfile.ZipInfo) -> bytes:
    """Decompress one zip member from a locally downloaded byte span of the archive."""
    off = info.header_offset - span_start
    header = span[off : off + 30]
    if header[:4] != b"PK\x03\x04":
        raise OSError(f"bad local header for {info.filename}")
    name_len = int.from_bytes(header[26:28], "little")
    extra_len = int.from_bytes(header[28:30], "little")
    start = off + 30 + name_len + extra_len
    raw = span[start : start + info.compress_size]
    if info.compress_type == zipfile.ZIP_STORED:
        data = raw
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(raw, -15)
    else:
        raise OSError(f"unsupported compression {info.compress_type} for {info.filename}")
    if zlib.crc32(data) != info.CRC:
        raise OSError(f"CRC mismatch for {info.filename}")
    return data


def mot17_sequence(seq: str, root: Path) -> Path:
    """Extract ``MOT17/train/<seq>/`` from the remote MOT17.zip into ``root/<seq>``."""
    dest = root / seq
    marker = dest / ".done"
    if marker.exists():
        print(f"[skip] data/mot17/{seq} already extracted")
        return dest
    remote = HttpRangeFile(MOT17_URL)
    with zipfile.ZipFile(io.BufferedReader(remote, buffer_size=1 << 16)) as zf:
        prefix = f"MOT17/train/{seq}/"
        members = [i for i in zf.infolist() if i.filename.startswith(prefix) and not i.is_dir()]
    if not members:
        raise OSError(f"{seq} not found in {MOT17_URL}")
    lo = min(i.header_offset for i in members)
    hi = max(i.header_offset + 30 + len(i.orig_filename) + 1024 + i.compress_size for i in members)
    hi = min(hi, remote.size)
    print(f"[get ] {seq}: {len(members)} files, one ranged request of {human_size(hi - lo)}")
    span = remote.fetch(lo, hi)
    shutil.rmtree(dest, ignore_errors=True)
    for info in members:
        rel = Path(info.filename[len(prefix) :])
        out = (dest / rel).resolve()
        if dest.resolve() not in out.parents:
            raise ValueError(f"unsafe path in archive: {info.filename}")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(_member_bytes(span, lo, info))
    # Sanity check: the number of frames must match seqinfo.ini.
    ini = configparser.ConfigParser()
    ini.read(dest / "seqinfo.ini")
    n_expected = int(ini["Sequence"]["seqLength"])
    n_found = len(list((dest / "img1").glob("*.jpg")))
    if n_found != n_expected or not (dest / "gt" / "gt.txt").is_file():
        raise OSError(f"{seq}: found {n_found} frames, expected {n_expected}")
    marker.write_text(f"{MOT17_URL} {seq}\n")
    print(f"[ok  ] data/mot17/{seq}: {n_found} frames + gt/gt.txt")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-sam3", action="store_true", help="skip the SAM 3 checkpoint")
    args = ap.parse_args()

    root = data_dir("mot17", create=True)
    for seq in SEQUENCES:
        mot17_sequence(seq, root)

    for name in YOLO_WEIGHTS:
        download_file(f"{ASSETS}/{name}", weights_dir("ultralytics", create=True) / name)

    for name in COTRACKER_FILES:
        hf_download("facebook/cotracker3", name)
    print(f"[ok  ] facebook/cotracker3 ({', '.join(COTRACKER_FILES)}) in the Hugging Face cache")
    if (third_party_dir("co-tracker") / "cotracker").is_dir():
        print("[ok  ] third_party/co-tracker (pinned clone) present")
    else:
        print("[note] CoTracker code missing: run `bash scripts/setup_third_party.sh co-tracker`")

    gated = [(DINOV3_REPO, DINOV3_FILES, "the BoT-SORT + DINOv3 variant")]
    if not args.no_sam3:
        gated.append(("facebook/sam3", SAM3_FILES, "notebook 03"))
    for repo, files, used_by in gated:
        try:
            for name in files:
                hf_download(repo, name)
            print(f"[ok  ] {repo} in the Hugging Face cache")
        except Exception as exc:  # noqa: BLE001 - gated repo: report and continue
            print(f"[note] {repo} not downloaded ({type(exc).__name__}); {used_by} needs")
            print(f"       access: accept the licence on huggingface.co/{repo} and log in.")


if __name__ == "__main__":
    main()
