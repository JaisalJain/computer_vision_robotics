#!/usr/bin/env python
"""Solutions to the exercises of notebook 03 (video object segmentation on DAVIS 2017).

Run from the repository root (GPU; hold the lock)::

    flock /tmp/cv-gpu.lock python projects/05_segmentation/solutions/solutions_03.py

Figures go to ``outputs/05_segmentation/solutions/`` (gitignored).
"""

from __future__ import annotations

import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from cvlab.device import setup  # noqa: E402
from cvlab.io import outputs_dir, project_dir  # noqa: E402

sys.path.insert(0, str(project_dir("05_segmentation") / "scripts"))
import segkit as sk  # noqa: E402

OUT = outputs_dir("05_segmentation", "solutions", create=True)
SEQS = ["car-roundabout", "dogs-jump"]


def propagate(vt: sk.VideoTracker, session, frames, ids) -> np.ndarray:
    """Run propagation on a prepared session and return T x H x W label maps."""
    h, w = frames[0].shape[:2]
    logits = np.full((len(frames), len(ids), h, w), -np.inf, dtype=np.float32)
    with torch.inference_mode():
        for out in vt.model.propagate_in_video_iterator(session, start_frame_idx=0):
            m = vt.processor.post_process_masks(
                [out.pred_masks.float()], original_sizes=[[h, w]], binarize=False
            )[0]
            for k, oid in enumerate(out.object_ids):
                logits[out.frame_idx, ids.index(int(oid))] = m[k, 0].cpu().numpy()
    return sk.logits_to_labels(logits, ids)


def exercise_1() -> None:
    """Correction clicks on each object's worst frame, then re-propagate (SAM 2.1-S, dogs-jump)."""
    frames, labels = sk.load_davis("dogs-jump")
    ids = sk.object_ids(labels)
    vt = sk.VideoTracker(sk.SAM2_IDS["small"])
    pred0, _ = vt.track(frames, labels[0], prompt="click")
    per0 = sk.evaluate_vos(pred0, labels)
    session = vt.processor.init_video_session(
        video=frames,
        inference_device="cuda",
        video_storage_device="cpu",
        processing_device="cpu",
        dtype=vt.dtype,
    )
    vt.processor.add_inputs_to_inference_session(
        inference_session=session,
        frame_idx=0,
        obj_ids=list(ids),
        input_points=[[[list(sk.interior_point(labels[0] == i))] for i in ids]],
        input_labels=[[[1] for _ in ids]],
    )
    with torch.inference_mode():
        vt.model(inference_session=session, frame_idx=0)
    for oid in ids:
        t = int(np.argmin(per0[oid]["J"])) + 1  # +1: frame 0 is not in the per-frame arrays
        gt_t, pred_t = labels[t] == oid, pred0[t] == oid
        pts, labs = [list(sk.interior_point(gt_t))], [1]
        click = sk.next_click(pred_t, gt_t)
        if click is not None and click[1] == 0:  # largest error is a false positive
            pts.append(list(click[0]))
            labs.append(0)
        vt.processor.add_inputs_to_inference_session(
            inference_session=session,
            frame_idx=t,
            obj_ids=[oid],
            input_points=[[pts]],
            input_labels=[[labs]],
        )
        with torch.inference_mode():
            vt.model(inference_session=session, frame_idx=t)
        print(
            f"[ex1] object {oid}: worst frame {t} (J {per0[oid]['J'][t - 1]:.3f}), "
            f"correction labels {labs}"
        )
    pred1 = propagate(vt, session, frames, ids)
    before = sk.summarize_vos(per0)
    after = sk.summarize_vos(sk.evaluate_vos(pred1, labels))
    print(
        f"[ex1] SAM 2.1-S click prompt on dogs-jump: J&F {before['JF_mean']:.3f} -> "
        f"{after['JF_mean']:.3f} after correction clicks "
        f"(J {before['J_mean']:.3f} -> {after['J_mean']:.3f})"
    )
    del vt
    torch.cuda.empty_cache()


def exercise_2() -> None:
    """Temporal subsampling: SAM 3 tracker with the mask prompt on every k-th frame."""
    frames, labels = sk.load_davis("dogs-jump")
    vt = sk.VideoTracker(sk.SAM3_ID)
    rows = []
    for k in (1, 2, 3):
        pred, info = vt.track(frames[::k], labels[0], prompt="mask")
        st = sk.summarize_vos(sk.evaluate_vos(pred, labels[::k]))
        rows.append(
            {
                "every k-th frame": k,
                "frames": len(frames[::k]),
                **st,
                "ms/frame": info["ms_per_frame"],
            }
        )
    df = pd.DataFrame(rows).set_index("every k-th frame")
    cols = ["frames", "J_mean", "F_mean", "JF_mean", "J_decay", "ms/frame"]
    print(f"[ex2] SAM 3 tracker, mask prompt, dogs-jump:\n{df[cols].round(3).to_string()}")
    del vt
    torch.cuda.empty_cache()


def exercise_3() -> None:
    """All SAM 2.1 sizes on video (mask prompt): J&F vs ms/frame, plus notebook 03's rows."""
    davis = {s: sk.load_davis(s) for s in SEQS}
    rows = []
    for mid in [sk.SAM2_IDS["tiny"], sk.SAM2_IDS["base-plus"]]:
        vt = sk.VideoTracker(mid)
        for seq, (frames, labels) in davis.items():
            pred, info = vt.track(frames, labels[0], prompt="mask")
            st = sk.summarize_vos(sk.evaluate_vos(pred, labels))
            rows.append(
                {
                    "model": vt.label,
                    "sequence": seq,
                    "JF_mean": st["JF_mean"],
                    "ms/frame": info["ms_per_frame"],
                }
            )
        del vt
        torch.cuda.empty_cache()
    nb3 = pd.read_csv(sk.results_dir() / "nb03_davis_summary.csv")
    nb3 = nb3[(nb3.prompt == "mask")][["model", "sequence", "JF_mean", "ms/frame"]]
    df = pd.concat([pd.DataFrame(rows), nb3]).groupby("model")[["JF_mean", "ms/frame"]].mean()
    print(f"[ex3] mask prompt, mean over both sequences:\n{df.round(3).to_string()}")
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for m, r in df.iterrows():
        ax.scatter(r["ms/frame"], r["JF_mean"], s=60, label=m)
    ax.set_xlabel("ms / frame")
    ax.set_ylabel("J&F (2 DAVIS sequences)")
    ax.set_title("Video trackers: accuracy vs speed")
    ax.legend()
    fig.savefig(OUT / "ex03_3_video_sizes.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup(seed=0)
    warnings.filterwarnings("ignore")
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    exercise_1()
    exercise_2()
    exercise_3()


if __name__ == "__main__":
    main()
