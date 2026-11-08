#!/usr/bin/env python
"""Solutions to the exercises of notebooks/03_small_vlm_vqa.ipynb.

1. Direct vs step-by-step counting prompts (count probes with a true count >= 2).
2. Resolution vs accuracy: the probe benchmark at 75 visual tokens vs the native resolution
   (runs ``scripts/eval_vqa_probes.py --max-visual-tokens 75 --tag lowres`` if needed).
3. Adversarial (co-occurrence) absent-object questions vs random ones.

Run (GPU, about 1.5 minutes; the first run adds about 1 minute for the low-resolution benchmark)::

    python projects/13_vision_language/solutions/ex03_small_vlm.py
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from collections import Counter
from itertools import permutations
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import transformers  # noqa: E402
import vl_common as vl  # noqa: E402

from cvlab.device import setup  # noqa: E402
from cvlab.io import data_dir, project_dir  # noqa: E402

transformers.logging.set_verbosity_error()
RESULTS = project_dir(vl.PROJECT) / "results"
N_COUNT = 40  # step-by-step answers are long: limit exercise 1 to the first 40 probes


def total_from_cot(answer: str) -> int | None:
    """Number after 'Total:' (falls back to the last number in the answer)."""
    m = re.search(r"total\s*[:=]?\s*(\d+)", answer.lower())
    if m:
        return int(m.group(1))
    nums = re.findall(r"\b\d+\b", answer)
    return int(nums[-1]) if nums else None


def main() -> None:
    setup(seed=0)
    # ---- Exercise 2 first (it needs the GPU for a few minutes in a subprocess)
    low = RESULTS / "vqa_probes_lowres.csv"
    if not low.exists():
        subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "eval_vqa_probes.py"),
                "--max-visual-tokens",
                "75",
                "--tag",
                "lowres",
            ],
            check=True,
        )
    rows = []
    for tag, path in [
        ("native (<= 640 px)", RESULTS / "vqa_probes.csv"),
        ("75 visual tokens", low),
    ]:
        df = pd.read_csv(path, dtype={"truth": str, "pred": str})
        acc = df.groupby("kind")["correct"].mean()
        rows.append(
            {
                "resolution": tag,
                "mean visual tokens": df.visual_tokens.mean(),
                **{f"acc {k}": v for k, v in acc.items()},
                "mean latency (s)": df.latency_s.mean(),
            }
        )
    print("Exercise 2 - resolution vs accuracy")
    print(pd.DataFrame(rows).to_string(index=False, float_format="{:.3f}".format))

    coco = data_dir("ultralytics", "coco128")
    items = vl.load_yolo_split(coco / "images" / "train2017", coco / "labels" / "train2017")
    vlm = vl.SmallVLM("cuda")

    # ---- Exercise 1: direct vs step-by-step counting
    probes = [p for p in vl.count_probes(items) if p["truth"] >= 2][:N_COUNT]
    res = []
    for p in probes:
        img = vl.load_rgb(items[p["item"]].path)
        direct = vlm.ask(img, p["question"], max_new_tokens=16)
        cot_q = (
            f"Count the {vl.plural(p['object'])} in the image one by one, "
            "then give the total as 'Total: N'."
        )
        cot = vlm.ask(img, cot_q, max_new_tokens=160)
        res.append(
            {
                "truth": p["truth"],
                "direct": vl.parse_count(direct.text),
                "step-by-step": total_from_cot(cot.text),
                "cot_tokens": cot.new_tokens,
                "cot_s": cot.latency_s,
            }
        )
    r = pd.DataFrame(res)
    print(f"\nExercise 1 - counting prompts on {len(r)} probes with true count >= 2")
    print(f"  direct accuracy       {np.mean(r.direct == r.truth):.3f}")
    print(
        f"  step-by-step accuracy {np.mean(r['step-by-step'] == r.truth):.3f} "
        f"(mean {r.cot_tokens.mean():.0f} tokens, {r.cot_s.mean():.2f} s per answer)"
    )

    # ---- Exercise 3: adversarial (co-occurrence) negatives vs random negatives
    co = Counter()
    for it in items:
        for a, b in permutations(set(it.names()), 2):
            co[a, b] += 1
    adv = []
    for k, it in enumerate(items):
        present = set(it.names())
        absent = [c for c in vl.COCO_NAMES if c not in present]
        score = {c: sum(co[p, c] for p in present) for c in absent}
        for c in sorted(absent, key=lambda c: (-score[c], c))[:2]:
            art = "an" if c[0] in "aeiou" else "a"
            adv.append(
                {
                    "item": k,
                    "object": c,
                    "co-occurrence": score[c],
                    "question": f"Is there {art} {c} in the image? Answer yes or no.",
                }
            )
    said_yes = []
    for p in adv:
        a = vlm.ask(vl.load_rgb(items[p["item"]].path), p["question"], max_new_tokens=8)
        said_yes.append(vl.parse_yes_no(a.text) is True)
    rnd = pd.read_csv(RESULTS / "vqa_probes.csv", dtype={"truth": str, "pred": str})
    rnd = rnd[(rnd.kind == "presence") & (rnd.truth == "False")]
    print("\nExercise 3 - hallucination rate (answers 'yes' for an absent object)")
    print(f"  random negatives      {np.mean(rnd.pred == 'True'):.3f} (n={len(rnd)})")
    print(f"  adversarial negatives {np.mean(said_yes):.3f} (n={len(said_yes)})")
    top = Counter(p["object"] for p, y in zip(adv, said_yes) if y).most_common(8)
    print("  most hallucinated adversarial objects:", top)


if __name__ == "__main__":
    main()
