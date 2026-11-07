#!/usr/bin/env python
"""Sanity-check the Python environment after every install.

Checks: interpreter/env, torch is the pinned CUDA 12.8 build with ``sm_120`` in its
arch list, a real CUDA matmul, ``pip check``, NumPy/OpenCV majors, and key imports.

Usage::

    python scripts/check_env.py            # full check (GPU required)
    python scripts/check_env.py --cpu-ok   # allow machines without CUDA
"""

from __future__ import annotations

import argparse
import importlib
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE_IMPORTS = ["numpy", "cv2", "torch", "torchvision", "matplotlib", "pandas", "scipy",
                "PIL", "cvlab"]  # fmt: skip
OPTIONAL_IMPORTS = ["sklearn", "timm", "transformers", "ultralytics", "huggingface_hub",
                    "onnx", "onnxruntime", "tensorrt", "pycolmap", "gsplat", "kornia", "evo",
                    "rerun", "lightglue", "open_clip"]  # fmt: skip


def pinned(name: str) -> str | None:
    """Version pinned with ``==`` in requirements/constraints.txt (if any)."""
    path = ROOT / "requirements" / "constraints.txt"
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        m = re.match(rf"^\s*{re.escape(name)}\s*==\s*([^\s#;]+)", line)
        if m:
            return m.group(1)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-ok", action="store_true", help="do not fail without CUDA")
    args = ap.parse_args()
    fails: list[str] = []

    def report(ok: bool, name: str, detail: str = "", warn: bool = False) -> None:
        tag = "OK  " if ok else ("WARN" if warn else "FAIL")
        print(f"[{tag}] {name}{': ' + detail if detail else ''}")
        if not ok and not warn:
            fails.append(name)

    report(True, "python", f"{sys.version.split()[0]} @ {Path(sys.prefix).name}")
    report(sys.version_info[:2] >= (3, 12), "python >= 3.12")

    try:
        import torch

        want = pinned("torch")
        report(want is None or torch.__version__ == want, "torch version",
               f"{torch.__version__} (pinned {want})")  # fmt: skip
        arch = torch.cuda.get_arch_list() if torch.cuda.is_available() else []
        if torch.cuda.is_available():
            report("sm_120" in arch, "sm_120 in torch arch list", ", ".join(arch))
            name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            report(True, "GPU", f"{name} (sm_{cap[0]}{cap[1]}), CUDA {torch.version.cuda}")
            a = torch.randn(1024, 1024, device="cuda")
            b = torch.randn(1024, 1024, device="cuda")
            c = (a @ b).float().cpu()
            ref = a.cpu() @ b.cpu()
            err = (c - ref).abs().max().item() / ref.abs().max().item()
            report(err < 1e-2, "CUDA matmul", f"relative max error {err:.2e}")
            h = torch.randn(256, 256, device="cuda", dtype=torch.float16)
            report(bool(torch.isfinite(h @ h).all()), "CUDA fp16 matmul")
        else:
            report(args.cpu_ok, "CUDA available", "no CUDA device", warn=args.cpu_ok)
        tv = pinned("torchvision")
        import torchvision

        report(tv is None or torchvision.__version__ == tv, "torchvision version",
               f"{torchvision.__version__} (pinned {tv})")  # fmt: skip
    except Exception as exc:  # noqa: BLE001
        report(False, "torch import / CUDA", repr(exc))

    for mod in CORE_IMPORTS:
        try:
            m = importlib.import_module(mod)
            report(True, f"import {mod}", getattr(m, "__version__", ""))
        except Exception as exc:  # noqa: BLE001
            report(False, f"import {mod}", repr(exc))
    try:
        import cv2
        import numpy as np

        report(int(np.__version__.split(".")[0]) >= 2, "numpy >= 2", np.__version__)
        report(int(cv2.__version__.split(".")[0]) >= 5, "opencv >= 5", cv2.__version__, warn=True)
    except Exception:  # noqa: BLE001
        pass
    for mod in OPTIONAL_IMPORTS:
        try:
            m = importlib.import_module(mod)
            print(f"[opt ] {mod} {getattr(m, '__version__', '')}")
        except Exception:  # noqa: BLE001
            pass

    out = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    report(
        out.returncode == 0, "pip check", out.stdout.strip().splitlines()[-1] if out.stdout else ""
    )

    print()
    if fails:
        print(f"ENVIRONMENT CHECK FAILED: {', '.join(fails)}")
        return 1
    print("ENVIRONMENT OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
