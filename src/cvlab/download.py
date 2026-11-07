"""Idempotent downloads with checksum verification and safe archive extraction."""

from __future__ import annotations

import hashlib
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

_USER_AGENT = "cvlab-downloader/0.1"


def human_size(n: float) -> str:
    """``1536 -> '1.5 KB'``."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} GB"  # pragma: no cover


def file_hash(path: str | Path, algo: str = "sha256", chunk: int = 1 << 20) -> str:
    """Hex digest of a file."""
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def download_file(
    url: str,
    dest: str | Path,
    sha256: str | None = None,
    md5: str | None = None,
    overwrite: bool = False,
    timeout: float = 60.0,
    quiet: bool = False,
) -> Path:
    """Download ``url`` to ``dest`` unless it already exists (and matches the checksum).

    The file is written to ``dest.part`` and atomically renamed when complete, so
    interrupted downloads never leave a truncated file behind.
    """
    dest = Path(dest)
    algo, expected = ("sha256", sha256) if sha256 else ("md5", md5) if md5 else (None, None)
    if dest.exists() and not overwrite:
        if expected is None or file_hash(dest, algo) == expected:
            if not quiet:
                print(f"[skip] {dest.name} already present ({human_size(dest.stat().st_size)})")
            return dest
        print(f"[redo] checksum mismatch for {dest.name}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as f:
        total = int(resp.headers.get("Content-Length") or 0)
        bar = None
        if not quiet:
            try:
                from tqdm import tqdm

                bar = tqdm(total=total or None, unit="B", unit_scale=True, desc=dest.name)
            except ImportError:  # pragma: no cover
                bar = None
        while chunk := resp.read(1 << 20):
            f.write(chunk)
            if bar is not None:
                bar.update(len(chunk))
        if bar is not None:
            bar.close()
    if expected is not None and file_hash(tmp, algo) != expected:
        tmp.unlink()
        raise OSError(f"Checksum mismatch for {url}")
    tmp.replace(dest)
    return dest


def _is_within(base: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def extract_archive(archive: str | Path, dest_dir: str | Path) -> Path:
    """Extract ``.zip``, ``.tar``, ``.tar.gz``/``.tgz``, ``.tar.bz2``, ``.tar.xz`` safely.

    Members that would escape ``dest_dir`` (absolute paths, ``..``) are rejected.
    """
    archive, dest_dir = Path(archive), Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                if not _is_within(dest_dir, dest_dir / member):
                    raise ValueError(f"Unsafe path in archive: {member}")
            zf.extractall(dest_dir)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            tf.extractall(dest_dir, filter="data")
    else:
        raise ValueError(f"Unsupported archive format: {archive}")
    return dest_dir


def download_and_extract(
    url: str,
    dest_dir: str | Path,
    archive_name: str | None = None,
    sha256: str | None = None,
    md5: str | None = None,
    keep_archive: bool = False,
) -> Path:
    """Download an archive and extract it into ``dest_dir`` once (``.done`` marker)."""
    dest_dir = Path(dest_dir)
    marker = dest_dir / ".done"
    if marker.exists():
        print(f"[skip] {dest_dir.name} already extracted")
        return dest_dir
    archive = dest_dir.parent / (archive_name or url.rstrip("/").split("/")[-1].split("?")[0])
    download_file(url, archive, sha256=sha256, md5=md5)
    extract_archive(archive, dest_dir)
    marker.write_text(url + "\n")
    if not keep_archive:
        archive.unlink(missing_ok=True)
    return dest_dir


def hf_download(repo_id: str, filename: str | None = None, **kwargs) -> Path:
    """Fetch a file (or, with ``filename=None``, a snapshot) from the Hugging Face Hub cache."""
    from huggingface_hub import hf_hub_download, snapshot_download

    if filename is None:
        return Path(snapshot_download(repo_id, **kwargs))
    return Path(hf_hub_download(repo_id, filename, **kwargs))


def copy_if_missing(src: str | Path, dest: str | Path) -> Path:
    """Copy a local file unless ``dest`` exists (used by tests and offline mirrors)."""
    dest = Path(dest)
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    return dest
