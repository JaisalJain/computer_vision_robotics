#!/usr/bin/env python
"""Data and weights for ``projects/04_open_vocab_detection`` (idempotent).

* COCO val2017 subset: ``instances_val2017.json`` (extracted from the official
  annotation zip, which is deleted afterwards) plus a fixed-seed subset of
  ``N`` images fetched one by one -> ``data/coco/``.
  A subset annotation file ``data/coco/annotations/instances_val2017_ovd<N>.json``
  is written for pycocotools.
* african-wildlife (Ultralytics YOLO-format, 4 classes; two of them, buffalo and
  rhino, are not COCO classes) -> ``data/ultralytics/african-wildlife/``.
* TUM RGB-D ``fr1_desk`` (shared dataset, a robot-eye view of an office desk).
* Ultralytics weights -> ``weights/ultralytics/``: ``yolo26s.pt`` (closed-set COCO),
  ``yoloe-26s-seg.pt`` (open-vocabulary, text/visual prompts) and its text encoder
  ``mobileclip2_b.ts`` (Ultralytics looks it up in its ``weights_dir``).
* Hugging Face checkpoints (safetensors only, pinned revisions) -> HF cache:
  Grounding DINO-T, LLMDet-T, OWLv2-B/16, OmDet-Turbo-T, SAM 3 and (for the exercise
  solutions) MM-Grounding-DINO-T. SAM 3 is gated: request
  access on https://huggingface.co/facebook/sam3 and run ``hf auth login`` first (or skip it
  with ``--no-sam3``; the notebooks then fail at the SAM 3 cells).

Usage::

    python scripts/download_04_open_vocab_detection.py              # everything
    python scripts/download_04_open_vocab_detection.py --n-coco 200 # subset size
    python scripts/download_04_open_vocab_detection.py --no-sam3    # skip SAM 3 (~3.3 GB, gated)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from download_shared import tum  # noqa: E402

from cvlab.download import download_file, extract_archive  # noqa: E402
from cvlab.io import data_dir, weights_dir  # noqa: E402

ASSETS = "https://github.com/ultralytics/assets/releases/download"
COCO_ANN_ZIP = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_IMG_URL = "http://images.cocodataset.org/val2017/{file_name}"
WILDLIFE_URL = f"{ASSETS}/v0.0.0/african-wildlife.zip"
# Extra val2017 images used in the notebooks' qualitative figures (two cats + two remotes).
COCO_SHOWCASE = ["000000039769.jpg"]

ULTRALYTICS_WEIGHTS = {
    "yolo26s.pt": f"{ASSETS}/v8.4.0/yolo26s.pt",
    "yoloe-26s-seg.pt": f"{ASSETS}/v8.4.0/yoloe-26s-seg.pt",
    "mobileclip2_b.ts": f"{ASSETS}/v8.4.0/mobileclip2_b.ts",
}
# repo id -> pinned commit (checked 2026-11-07)
HF_MODELS = {
    "IDEA-Research/grounding-dino-tiny": "a2bb814dd30d776dcf7e30523b00659f4f141c71",
    "iSEE-Laboratory/llmdet_tiny": "d05199165a19320a9236396e20fca0a5e065189c",
    "google/owlv2-base-patch16-ensemble": "cfd3195ba4ea9592eec887ded089f4c08eff231d",
    "omlab/omdet-turbo-swin-tiny-hf": "7fe93cecfb770c4d76cf71163956221249cab566",
    # used only by solutions/solutions_02.py (exercise 1)
    "openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det": (
        "ef2053aa1e0cc9d9cee1df31ca919ee9aae4fc8e"
    ),
}
SAM3 = ("facebook/sam3", "3c879f39826c281e95690f02c7821c4de09afae7")
HF_PATTERNS = ["*.json", "*.safetensors", "*.txt", "LICENSE*"]  # skips duplicate .bin weights


def coco_annotations() -> Path:
    """``data/coco/annotations/instances_val2017.json`` (only this member is extracted)."""
    ann_dir = data_dir("coco", "annotations", create=True)
    out = ann_dir / "instances_val2017.json"
    if out.exists():
        print("[skip] data/coco/annotations/instances_val2017.json already present")
        return out
    archive = download_file(COCO_ANN_ZIP, data_dir("coco") / "annotations_trainval2017.zip")
    with zipfile.ZipFile(archive) as zf, zf.open("annotations/instances_val2017.json") as src:
        tmp = out.with_suffix(".json.part")
        with open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst)
        tmp.replace(out)
    archive.unlink()  # 253 MB zip; only the 25 MB val json is kept
    print("[ok] data/coco/annotations/instances_val2017.json")
    return out


def coco_subset(n_images: int = 200, seed: int = 0) -> Path:
    """Fixed-seed subset of val2017 images that contain at least one non-crowd object."""
    sub_json = data_dir("coco", "annotations") / f"instances_val2017_ovd{n_images}.json"
    img_dir = data_dir("coco", "val2017", create=True)
    if sub_json.exists():
        subset = json.loads(sub_json.read_text())
    else:
        full = json.loads(coco_annotations().read_text())
        has_obj = sorted({a["image_id"] for a in full["annotations"] if not a["iscrowd"]})
        rng = np.random.default_rng(seed)
        keep = set(rng.choice(has_obj, size=n_images, replace=False).tolist())
        subset = {
            "info": {**full.get("info", {}), "subset": f"{n_images} images, seed {seed}"},
            "licenses": full.get("licenses", []),
            "categories": full["categories"],
            "images": [im for im in full["images"] if im["id"] in keep],
            "annotations": [a for a in full["annotations"] if a["image_id"] in keep],
        }
    wanted = [im["file_name"] for im in subset["images"]] + COCO_SHOWCASE
    missing = [f for f in wanted if not (img_dir / f).exists()]

    def fetch(file_name: str) -> None:
        download_file(COCO_IMG_URL.format(file_name=file_name), img_dir / file_name, quiet=True)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(fetch, missing))
    if not sub_json.exists():
        sub_json.write_text(json.dumps(subset))
    n_ann = len(subset["annotations"])
    print(f"[ok] COCO val2017 subset: {len(subset['images'])} images, {n_ann} annotations "
          f"({len(missing)} fetched now) -> data/coco/val2017/")  # fmt: skip
    return sub_json


def african_wildlife() -> Path:
    """Ultralytics african-wildlife (same location/layout as project 03 uses)."""
    root = data_dir("ultralytics", create=True)
    dest = root / "african-wildlife"
    if (dest / "images").is_dir() and (dest / "labels").is_dir():
        print("[skip] data/ultralytics/african-wildlife already present")
        return dest
    archive = download_file(WILDLIFE_URL, root / "african-wildlife.zip")
    tmp = root / ".african-wildlife.extract"
    shutil.rmtree(tmp, ignore_errors=True)
    extract_archive(archive, tmp)
    nested = tmp / "african-wildlife"
    (nested if nested.is_dir() else tmp).rename(dest)
    shutil.rmtree(tmp, ignore_errors=True)
    archive.unlink()
    print("[ok] data/ultralytics/african-wildlife")
    return dest


def ultralytics_weights() -> list[Path]:
    out = weights_dir("ultralytics", create=True)
    return [download_file(url, out / name) for name, url in ULTRALYTICS_WEIGHTS.items()]


def hf_models(include_sam3: bool = True) -> None:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError

    models = dict(HF_MODELS)
    if include_sam3:
        models[SAM3[0]] = SAM3[1]
    for repo, rev in models.items():
        # SAM 3 ships a duplicate sam3.pt; only the transformers safetensors are fetched.
        try:
            snapshot_download(repo, revision=rev, allow_patterns=HF_PATTERNS)
        except GatedRepoError:
            print(f"[warn] {repo} is gated and this HF account has no access: request it on "
                  f"https://huggingface.co/{repo}, run `hf auth login`, then re-run")  # fmt: skip
            continue
        print(f"[ok] HF cache: {repo}@{rev[:7]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-coco", type=int, default=200, help="COCO val2017 subset size")
    ap.add_argument("--no-sam3", action="store_true", help="skip gated facebook/sam3 (~3.3 GB)")
    args = ap.parse_args()
    ultralytics_weights()
    hf_models(include_sam3=not args.no_sam3)
    coco_subset(args.n_coco)
    african_wildlife()
    print(f"[ok] {tum('fr1_desk')}".replace(str(data_dir()), "data"))


if __name__ == "__main__":
    main()
