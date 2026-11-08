#!/usr/bin/env python
"""Solutions to the exercises of notebook 02 (pose evaluation vs COLMAP and ground truth).

CPU only: everything is computed from the cached predictions of ``benchmark.py``::

    python projects/10_feedforward_3d/solutions/solutions_02.py

Writes the RRA/RTA plot of exercise 1 to ``outputs/10_feedforward_3d/solutions/`` (gitignored).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ff3d  # noqa: E402

from cvlab import viz  # noqa: E402
from cvlab.geometry import umeyama_alignment  # noqa: E402
from cvlab.io import outputs_dir  # noqa: E402

METHODS = ["vggt", "da3-base", "da3-base-ray", "da3-large", "da3-large-ray", "colmap"]


def pair_errors(scene: ff3d.Scene, method: str) -> dict[str, np.ndarray]:
    """Pairwise errors on the images both the method and the reference have."""
    rec = ff3d.predict_cached(scene, method)
    both = np.isfinite(rec.T_cw).all(axis=(1, 2)) & scene.has_ref
    return ff3d.relative_pose_errors(rec.T_cw[both], scene.ref_T_cw[both])


def exercise_1() -> None:
    """RRA@tau and RTA@tau curves on South Building."""
    print("\n=== Exercise 1: RRA / RTA vs threshold (South Building) ===")
    scene = ff3d.load_scene("south_building")
    taus = np.linspace(0.5, 30, 60)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    rows = []
    for k, m in enumerate(["vggt", "da3-large-ray"]):
        pe = pair_errors(scene, m)
        rra = [(pe["rot_deg"] < t).mean() for t in taus]
        rta = [(pe["trans_deg"] < t).mean() for t in taus]
        axes[0].plot(taus, rra, label=ff3d.METHOD_LABELS[m], color=viz.color(k, unit=True))
        axes[1].plot(taus, rta, label=ff3d.METHOD_LABELS[m], color=viz.color(k, unit=True))
        for t in (1, 2, 5, 15):
            rows.append({"method": ff3d.METHOD_LABELS[m], "tau (deg)": t,
                         "RRA": (pe["rot_deg"] < t).mean(),
                         "RTA": (pe["trans_deg"] < t).mean()})  # fmt: skip
    for ax, name in zip(axes, ["RRA (relative rotation)", "RTA (relative translation direction)"]):
        ax.set_xlabel("threshold tau [deg]")
        ax.set_title(name)
        ax.legend()
    axes[0].set_ylabel("fraction of pairs below tau")
    out = outputs_dir("10_feedforward_3d", "solutions", create=True) / "ex02_1_rra_rta.png"
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    df = pd.DataFrame(rows)
    print(df.pivot(index="tau (deg)", columns="method").round(3).to_string())
    print("plot:", ff3d.rel(out))
    at1 = df[df["tau (deg)"] == 1].set_index("method")
    for name, r in at1.iterrows():
        better = "rotation" if r.RRA > r.RTA else "translation direction"
        print(f"{name}: at tau = 1 deg RRA {r.RRA:.3f} vs RTA {r.RTA:.3f} -> the {better} is "
              "the more accurate part of its relative poses")  # fmt: skip
    print("Which part limits the AUC differs between models, so report both RRA and RTA rather")
    print("than only their max; all curves reach ~1 by 5-15 deg (no gross failures).")


def exercise_2() -> None:
    """Fold the translation-direction error as min(e, 180 - e) (Co3D evaluation convention)."""
    print("\n=== Exercise 2: translation sign ambiguity (TUM fr1_desk) ===")
    scene = ff3d.load_scene("tum_desk")
    rows = []
    for m in METHODS:
        pe = pair_errors(scene, m)
        folded = np.minimum(pe["trans_deg"], 180.0 - pe["trans_deg"])
        rows.append({"method": ff3d.METHOD_LABELS[m],
                     "AUC@15 signed": ff3d.pose_auc(pe["rot_deg"], pe["trans_deg"], 15),
                     "AUC@15 folded": ff3d.pose_auc(pe["rot_deg"], folded, 15),
                     "pairs with e_t > 90 deg": int((pe["trans_deg"] > 90).sum()),
                     "pairs": len(folded)})  # fmt: skip
    df = pd.DataFrame(rows).set_index("method")
    df["gain"] = df["AUC@15 folded"] - df["AUC@15 signed"]
    print(df.round(4).to_string())
    print("Folding only helps pairs whose predicted translation points backwards (e_t > 90 deg).")
    print(f"On TUM fr1_desk {int(df['pairs with e_t > 90 deg'].sum())} pairs of any method do, so")
    print("the gain is zero here. Backwards translations happen for nearly pure rotations or very")
    print("short baselines, where the direction of t is poorly constrained; folding would hide")
    print("such errors, so this project keeps the signed (stricter) version.")


def exercise_3() -> None:
    """SE(3) vs Sim(3) alignment of VGGT's TUM trajectory to mocap."""
    print("\n=== Exercise 3: SE(3) vs Sim(3) alignment (VGGT on TUM fr1_desk) ===")
    scene = ff3d.load_scene("tum_desk")
    rec = ff3d.predict_cached(scene, "vggt")
    C_p, C_r = ff3d.camera_centers(rec.T_cw), ff3d.camera_centers(scene.ref_T_cw)
    rows = []
    for name, with_scale in [("Sim(3)", True), ("SE(3)", False)]:
        s, R, t = umeyama_alignment(C_p, C_r, with_scale=with_scale)
        res = C_r - (s * C_p @ R.T + t)
        rows.append({"alignment": name, "scale": s,
                     "ATE RMSE (m)": float(np.sqrt(np.mean(np.sum(res**2, axis=1))))})  # fmt: skip
    print(pd.DataFrame(rows).set_index("alignment").round(4).to_string())
    print("VGGT's unit is arbitrary (its scene is normalised), so without the scale factor the")
    print("trajectory has the wrong size and SE(3) can only centre and rotate it: the residual is")
    print("dominated by the scale mismatch, not by the shape of the trajectory.")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
