#!/usr/bin/env python
"""Data + weights for project 08 (monocular depth). Idempotent: re-running is a no-op.

* TUM RGB-D fr1_desk and fr1_xyz (CC BY 4.0) -> ``data/tum/<seq>/...`` (shared with 09/12/15)
* Depth Anything 3 checkpoints (DA3-SMALL, DA3-BASE, DA3METRIC-LARGE; Apache-2.0)
* Depth Anything V2 Small (relative) and Metric-Indoor-Small (metric) in transformers format

Weights go to the Hugging Face cache (``~/.cache/huggingface``), where both
``DepthAnything3.from_pretrained`` and ``transformers`` look for them.
The Depth Anything 3 *code* is a pinned clone made by ``scripts/setup_third_party.sh``.

Usage::

    python scripts/download_08_monocular_depth.py            # everything (~2.2 GB of weights)
    python scripts/download_08_monocular_depth.py --no-weights
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from download_shared import tum  # noqa: E402

from cvlab.download import hf_download  # noqa: E402
from cvlab.io import third_party_dir  # noqa: E402

# repo id -> files needed for inference (README/.gitattributes are skipped); smallest first.
HF_MODELS = {
    "depth-anything/Depth-Anything-V2-Small-hf": [
        "config.json", "model.safetensors", "preprocessor_config.json",
    ],  # 99 MB
    "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf": [
        "config.json", "model.safetensors", "preprocessor_config.json",
    ],  # 99 MB
    "depth-anything/DA3-SMALL": ["config.json", "model.safetensors"],  # 137 MB
    "depth-anything/DA3-BASE": ["config.json", "model.safetensors"],  # 542 MB
    "depth-anything/DA3METRIC-LARGE": ["config.json", "model.safetensors"],  # 1.34 GB
}  # fmt: skip
TUM_SEQUENCES = ["fr1_desk", "fr1_xyz"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-weights", action="store_true", help="only download the TUM data")
    args = ap.parse_args()

    for seq in TUM_SEQUENCES:
        path = tum(seq)
        print(f"[ok] TUM {seq}: data/tum/{seq}/{path.name}")

    if not args.no_weights:
        for repo, files in HF_MODELS.items():
            # hf_hub_download returns the cached file immediately when it is present.
            for name in files:
                hf_download(repo, name)
            print(f"[ok] {repo} ({', '.join(files)}) in the Hugging Face cache")

    if not third_party_dir("depth-anything-3", "src", "depth_anything_3").is_dir():
        print("[note] Depth Anything 3 code missing: run `bash scripts/setup_third_party.sh`")
    else:
        print("[ok] third_party/depth-anything-3 (pinned clone) present")


if __name__ == "__main__":
    main()
