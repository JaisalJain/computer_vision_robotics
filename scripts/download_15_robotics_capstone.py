#!/usr/bin/env python
"""Data + weights for project 15 (robotics capstone). Idempotent: re-running is a no-op.

* TUM RGB-D fr1/desk (CC BY 4.0) -> ``data/tum/fr1_desk/...`` (shared with projects 08/09/12)
* Hugging Face checkpoints (pinned revisions) into the Hugging Face cache:
  Grounding DINO-T and OWLv2-B/16 (Apache-2.0), SAM 2.1-S (Apache-2.0),
  Depth Anything V2 Metric-Indoor-Small (Apache-2.0) and SAM 3 (SAM License, **gated**:
  request access on https://huggingface.co/facebook/sam3 and run ``hf auth login`` first).

Usage::

    python scripts/download_15_robotics_capstone.py              # everything (~4.4 GB of weights)
    python scripts/download_15_robotics_capstone.py --no-sam3    # skip the 3.3 GB gated SAM 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from download_shared import tum  # noqa: E402

from cvlab.download import hf_download  # noqa: E402

# repo id -> (revision, files needed for inference). Only model.safetensors is fetched
# (never the duplicate .bin / sam3.pt checkpoints).
HF_MODELS = {
    "IDEA-Research/grounding-dino-tiny": (
        "a2bb814dd30d776dcf7e30523b00659f4f141c71",
        [
            "config.json",
            "model.safetensors",
            "preprocessor_config.json",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "vocab.txt",
            "added_tokens.json",
        ],
    ),  # 689 MB
    "google/owlv2-base-patch16-ensemble": (
        "cfd3195ba4ea9592eec887ded089f4c08eff231d",
        [
            "config.json",
            "model.safetensors",
            "preprocessor_config.json",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "added_tokens.json",
        ],
    ),  # 620 MB
    "facebook/sam2.1-hiera-small": (
        "ee5bba1d82bb8749febdf90f45e84b687142ba03",
        ["config.json", "model.safetensors", "preprocessor_config.json", "processor_config.json"],
    ),  # 184 MB
    "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf": (
        "8078d68a9c75a972131914f6afd0c1723be0da7f",
        ["config.json", "model.safetensors", "preprocessor_config.json"],
    ),  # 99 MB
    "facebook/sam3": (
        "3c879f39826c281e95690f02c7821c4de09afae7",
        [
            "config.json",
            "model.safetensors",
            "processor_config.json",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "vocab.json",
            "merges.txt",
            "LICENSE",
        ],
    ),  # 3.3 GB, gated
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-weights", action="store_true", help="only download the TUM data")
    ap.add_argument("--no-sam3", action="store_true", help="skip the gated SAM 3 checkpoint")
    args = ap.parse_args()

    path = tum("fr1_desk")
    print(f"[ok] TUM fr1_desk: data/tum/fr1_desk/{path.name}")

    if args.no_weights:
        return
    for repo, (rev, files) in HF_MODELS.items():
        if args.no_sam3 and repo == "facebook/sam3":
            print("[skip] facebook/sam3 (--no-sam3): notebooks fall back to SAM 2.1-S")
            continue
        try:
            for name in files:  # returns immediately when the file is already cached
                hf_download(repo, name, revision=rev)
        except Exception as exc:  # noqa: BLE001 - gated repo without access, offline, ...
            print(f"[warn] {repo}: {type(exc).__name__}: {exc}")
            continue
        print(f"[ok] {repo}@{rev[:7]} ({len(files)} files) in the Hugging Face cache")


if __name__ == "__main__":
    main()
