"""Notebook 02, exercise 3: number of ORB features vs accuracy and runtime (fr1/desk)."""

import pandas as pd
from _common import load, vo

seq = load("fr1_desk")
rows = []
for n in (250, 500, 1000, 2000, 4000):
    fe = vo.FeatureFrontend("orb", n)
    res = vo.run_pnp_vo(seq, fe, keyframes=True)
    s = vo.evaluate(res, seq, align="se3")
    rows.append(
        {
            "ORB features": n,
            "ATE [cm]": 100 * s["ate_rmse_m"],
            "RPE 1s [cm]": 100 * s["rpe_trans_rmse_m"],
            "lost": s["lost"],
            "keyframes": s["keyframes"],
            "ms/frame": s["ms_per_frame"],
        }
    )
print(pd.DataFrame(rows).round(2).to_string(index=False))
print("\nA 30 Hz camera leaves 33 ms per frame for the whole perception stack.")
