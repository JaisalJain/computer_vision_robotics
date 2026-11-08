#!/usr/bin/env python
"""Ground-truth VQA probes for the small VLM on coco128 (presence, counting, spatial).

Every question is generated from the coco128 YOLO labels (see ``vl_common``):

* **presence** (POPE-style): "Is there a <class> in the image?" for up to 2 classes that are
  present and 2 random classes that are absent per image (yes/no answers, close to balanced:
  an image with a single class contributes one "yes" question);
* **count**: "How many <class plural> are in the image?" for every class with 1-6 instances;
* **spatial**: "Is the A to the left or to the right of the B?" / "above or below" for the
  most clearly separated pair of single-instance objects per image and axis.

The VLM answers each question greedily; answers are parsed (first yes/no word, first number,
first relation word) and compared with the truth. Results go to ``results/`` as one CSV row per
question plus a JSON summary with the run's settings, timing and peak VRAM.

Usage::

    python projects/13_vision_language/scripts/eval_vqa_probes.py              # ~2 min
    python projects/13_vision_language/scripts/eval_vqa_probes.py --max-visual-tokens 75 \\
        --tag lowres                                                         # exercise 2
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
import vl_common as vl  # noqa: E402

from cvlab.device import seed_everything  # noqa: E402
from cvlab.io import data_dir, project_dir, save_json  # noqa: E402


def build_probes(items: list[vl.CocoItem], seed: int = 0) -> list[dict]:
    """All three probe kinds, in a fixed order (deterministic for a given seed)."""
    return (
        vl.presence_probes(items, n_pos=2, n_neg=2, seed=seed)
        + vl.count_probes(items, max_count=6)
        + vl.spatial_probes(items, min_gap=0.05, seed=seed)
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--max-visual-tokens",
        type=int,
        default=None,
        help="cap the image at this many 32x32 visual tokens (default: native size)",
    )
    ap.add_argument("--kinds", nargs="*", default=["presence", "count", "spatial"])
    ap.add_argument("--limit", type=int, default=None, help="only the first N probes (debug)")
    ap.add_argument("--tag", default="", help="suffix of the output files (vqa_probes<_tag>.csv)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    import transformers

    transformers.logging.set_verbosity_error()
    seed_everything(args.seed)
    coco = data_dir("ultralytics", "coco128")
    items = vl.load_yolo_split(coco / "images" / "train2017", coco / "labels" / "train2017")
    probes = [p for p in build_probes(items, args.seed) if p["kind"] in args.kinds]
    probes = probes[: args.limit] if args.limit else probes
    print(
        f"{len(probes)} probes on {len(items)} coco128 images:",
        pd.Series([p["kind"] for p in probes]).value_counts().to_dict(),
    )

    with vl.vram_meter() as mem_load:
        vlm = vl.SmallVLM("cuda")
    if args.max_visual_tokens:
        vlm.set_pixel_budget(args.max_visual_tokens * 32 * 32)
    vlm.ask(vl.load_rgb(items[0].path), "Warm-up. Answer yes or no.", max_new_tokens=2)

    rows, images = [], {}
    t0 = time.perf_counter()
    with vl.vram_meter() as mem_run:
        for k, p in enumerate(probes):
            it = items[p["item"]]
            if it.path not in images:
                images[it.path] = vl.load_rgb(it.path)
            a = vlm.ask(images[it.path], p["question"], max_new_tokens=16)
            pred, ok = vl.score_probe(p["kind"], p["truth"], a.text)
            rows.append({"image": it.path.name, "kind": p["kind"], "object": p["object"],
                         "question": p["question"], "truth": p["truth"], "answer": a.text,
                         "pred": pred, "correct": ok, "visual_tokens": a.image_tokens,
                         "prompt_tokens": a.prompt_tokens, "new_tokens": a.new_tokens,
                         "latency_s": round(a.latency_s, 4)})  # fmt: skip
            if (k + 1) % 100 == 0:
                print(f"  {k + 1}/{len(probes)} ({time.perf_counter() - t0:.0f} s)")
    wall = time.perf_counter() - t0

    out_dir = project_dir(vl.PROJECT) / "results"
    out_dir.mkdir(exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"vqa_probes{suffix}.csv", index=False)
    cfg = vl.model_cfg("vlm")
    summary = {
        "model": cfg["repo"],
        "revision": cfg["revision"],
        "dtype": "bfloat16",
        "decoding": "greedy, max_new_tokens=16, enable_thinking=False",
        "max_visual_tokens": args.max_visual_tokens,
        "pixel_budget": list(vlm.pixel_budget),
        "n_probes": len(df),
        "accuracy": df.groupby("kind")["correct"].mean().round(4).to_dict(),
        "n": df.groupby("kind").size().to_dict(),
        "mean_visual_tokens": float(df["visual_tokens"].mean()),
        "mean_latency_s": float(df["latency_s"].mean()),
        "wall_time_s": round(wall, 1),
        "weights_mib": round(mem_load.delta_mb, 1),
        "peak_vram_mib": round(mem_run.peak_mb, 1),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    save_json(summary, out_dir / f"vqa_probes{suffix}_summary.json")
    print(df.groupby("kind")["correct"].agg(["mean", "size"]))
    print(
        f"wall {wall:.0f} s | mean latency {summary['mean_latency_s']:.3f} s | "
        f"peak VRAM {mem_run.peak_mb:.0f} MiB | -> {vl.rel(out_dir)}/vqa_probes{suffix}.csv"
    )


if __name__ == "__main__":
    main()
