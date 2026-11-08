"""Notebook 02, exercise 2: sweep the keyframe threshold of the RGB-D PnP tracker (ORB)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from _common import load, vo  # noqa: E402

from cvlab.io import runs_dir  # noqa: E402

fe = vo.FeatureFrontend("orb", 2000)
rows = []
for name in ("fr1_xyz", "fr1_desk"):
    seq = load(name)
    for thr in (50, 100, 150, 250, 400):
        res = vo.run_pnp_vo(seq, fe, keyframes=True, kf_min_inliers=thr)
        s = vo.evaluate(res, seq, align="se3")
        rows.append(
            {
                "sequence": name,
                "kf_min_inliers": thr,
                "keyframes": s["keyframes"],
                "ATE [cm]": 100 * s["ate_rmse_m"],
                "RPE 1s [cm]": 100 * s["rpe_trans_rmse_m"],
                "RPE 1s [deg]": s["rpe_rot_rmse_deg"],
            }
        )
df = pd.DataFrame(rows)
print(df.round(2).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for name, g in df.groupby("sequence"):
    axes[0].plot(g["keyframes"], g["ATE [cm]"], "o-", label=name)
    axes[1].plot(g["keyframes"], g["RPE 1s [cm]"], "o-", label=name)
for ax, lab in zip(axes, ["ATE [cm]", "RPE over 1 s [cm]"]):
    ax.set_xlabel("number of keyframes")
    ax.set_ylabel(lab)
    ax.legend()
fig.suptitle("RGB-D keyframe VO (ORB): keyframe threshold sweep")
out = runs_dir("12_visual_odometry", create=True) / "ex2_2_keyframe_sweep.png"
fig.savefig(out, dpi=100, bbox_inches="tight")
print("figure written to runs/12_visual_odometry/" + out.name)
