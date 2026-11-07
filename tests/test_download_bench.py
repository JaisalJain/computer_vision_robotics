import hashlib
import tarfile
import zipfile

import pytest

from cvlab import bench, download


def test_download_file_idempotent(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"hello world")
    digest = hashlib.sha256(b"hello world").hexdigest()
    dest = tmp_path / "out" / "file.bin"
    download.download_file(src.as_uri(), dest, sha256=digest, quiet=True)
    assert dest.read_bytes() == b"hello world"
    mtime = dest.stat().st_mtime_ns
    download.download_file(src.as_uri(), dest, sha256=digest, quiet=True)
    assert dest.stat().st_mtime_ns == mtime  # second call is a no-op
    with pytest.raises(OSError):
        download.download_file(src.as_uri(), tmp_path / "bad.bin", sha256="0" * 64, quiet=True)
    assert not (tmp_path / "bad.bin").exists()


def test_extract_zip_and_tar(tmp_path):
    zpath = tmp_path / "a.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("dir/x.txt", "x")
    download.extract_archive(zpath, tmp_path / "z")
    assert (tmp_path / "z" / "dir" / "x.txt").read_text() == "x"
    member = tmp_path / "y.txt"
    member.write_text("y")
    tpath = tmp_path / "a.tar.gz"
    with tarfile.open(tpath, "w:gz") as tf:
        tf.add(member, arcname="sub/y.txt")
    download.extract_archive(tpath, tmp_path / "t")
    assert (tmp_path / "t" / "sub" / "y.txt").read_text() == "y"


def test_zip_slip_rejected(tmp_path):
    zpath = tmp_path / "evil.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("../evil.txt", "x")
    with pytest.raises(ValueError):
        download.extract_archive(zpath, tmp_path / "out")


def test_download_and_extract_marker(tmp_path):
    zpath = tmp_path / "pkg.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("a.txt", "a")
    out = download.download_and_extract(zpath.as_uri(), tmp_path / "data" / "pkg")
    assert (out / "a.txt").exists() and (out / ".done").exists()
    assert download.download_and_extract(zpath.as_uri(), out) == out


def test_human_size():
    assert download.human_size(512) == "512 B"
    assert download.human_size(1536) == "1.5 KB"


def test_benchmark_cpu():
    res = bench.benchmark(lambda: sum(range(1000)), warmup=2, iters=5, cuda=False)
    assert res["mean_ms"] > 0 and res["fps"] > 0
    with bench.Timer() as t:
        sum(range(1000))
    assert t.ms >= 0


def test_device_helpers():
    torch = pytest.importorskip("torch")
    from cvlab import device

    dev = device.setup(seed=1, verbose=False)
    assert isinstance(dev, torch.device)
    a = torch.rand(3)
    device.seed_everything(1)
    b = torch.rand(3)
    device.seed_everything(1)
    assert torch.equal(b, torch.rand(3)) and a.shape == b.shape
    info = device.env_summary()
    assert "torch" in info and "numpy" in info
    with device.track_vram() as mem:
        _ = torch.ones(4, device=dev) * 2
    assert mem.peak_mb >= 0
