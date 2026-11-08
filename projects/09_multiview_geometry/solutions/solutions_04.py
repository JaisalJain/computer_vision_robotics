#!/usr/bin/env python
"""Solutions to the exercises of notebook 04 (incremental SfM with pycolmap).

Run (CPU only, about 2 minutes; needs ``make data-09_multiview_geometry``)::

    python projects/09_multiview_geometry/solutions/solutions_04.py

Writes COLMAP databases/models to ``runs/09_multiview_geometry/solutions/`` (gitignored).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pycolmap

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import mvgeo as M  # noqa: E402

from cvlab.io import runs_dir  # noqa: E402

pycolmap.logging.minloglevel = 2
IMAGE_DIR = M.south_building_dir() / "images"
NAMES = sorted(p.name for p in IMAGE_DIR.glob("*.JPG"))[0:64:2]  # same subset as the notebook
OUT = runs_dir(M.PROJECT, "solutions", create=True)
REF = M.load_reference_cameras(scale_to_width=1024)


def summarise(name, rec, extra=None):
    st = M.reconstruction_stats(rec, len(NAMES))
    cmp = M.compare_to_reference(M.rec_cameras(rec), REF)
    rot = float(np.median(cmp["rot_err_deg"]))
    return {"setting": name, "registered": st.num_reg_images, "points": st.num_points3D,
            "reproj err [px]": st.mean_reprojection_error, "median rot err [deg]": rot,
            **(extra or {})}  # fmt: skip


# ----------------------------------------------------------------------------- exercise 1
def exercise_1() -> None:
    rows = []
    for model in ["PINHOLE", "SIMPLE_RADIAL", "OPENCV"]:
        rec, _ = M.colmap_sift_sfm(IMAGE_DIR, NAMES, OUT / f"ex1_{model}", camera_model=model)
        cam = next(iter(rec.cameras.values()))
        rows.append(summarise(model, rec, {"params": np.round(cam.params, 4).tolist()}))
    df = pd.DataFrame(rows).set_index("setting")
    print("Exercise 1 - camera models (SIFT, 32 images)")
    print(df.round(4).to_string())
    print(f"PINHOLE cannot model the lens' radial distortion: reprojection error "
          f"{df.loc['PINHOLE', 'reproj err [px]']:.3f} px and median rotation error "
          f"{df.loc['PINHOLE', 'median rot err [deg]']:.2f} deg vs "
          f"{df.loc['SIMPLE_RADIAL', 'median rot err [deg]']:.2f} deg with SIMPLE_RADIAL.")
    print(f"OPENCV fits the images slightly better ({df.loc['OPENCV', 'reproj err [px]']:.3f} px) "
          f"but its extra parameters are weakly constrained by 32 images (rotation error "
          f"{df.loc['OPENCV', 'median rot err [deg]']:.2f} deg): lower reprojection error does "
          "not mean more accurate poses.\n")


# ----------------------------------------------------------------------------- exercise 2
def exercise_2() -> None:
    rows = []
    for mode in ["exhaustive", "sequential (overlap 5)"]:
        work = OUT / ("ex2_exhaustive" if mode == "exhaustive" else "ex2_sequential")
        work.mkdir(parents=True, exist_ok=True)
        db = work / "database.db"
        db.unlink(missing_ok=True)
        reader = pycolmap.ImageReaderOptions()
        reader.camera_model = "SIMPLE_RADIAL"
        pycolmap.extract_features(db, IMAGE_DIR, image_names=NAMES,
                                  camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader,
                                  device=pycolmap.Device.cpu)  # fmt: skip
        t0 = time.perf_counter()
        if mode == "exhaustive":
            pycolmap.match_exhaustive(db, device=pycolmap.Device.cpu)
        else:
            pairing = pycolmap.SequentialPairingOptions()
            pairing.overlap = 5
            pycolmap.match_sequential(db, pairing_options=pairing, device=pycolmap.Device.cpu)
        t_match = time.perf_counter() - t0
        dbs = M.database_stats(db)
        rec, _ = M._map(db, IMAGE_DIR, work, NAMES, -1)
        rows.append(summarise(mode, rec, {"matched pairs": dbs["num_matched_pairs"],
                                          "matching [s]": t_match}))  # fmt: skip
    df = pd.DataFrame(rows).set_index("setting")
    seq = df.loc["sequential (overlap 5)"]
    print("Exercise 2 - exhaustive vs sequential matching")
    print(df.round(3).to_string())
    print(f"Sequential matching produced matches for {int(seq['matched pairs'])} pairs instead "
          f"of {int(df.loc['exhaustive', 'matched pairs'])} and registered "
          f"{int(seq['registered'])} of {len(NAMES)} images.")
    print("With N images exhaustive matching costs O(N^2) pairs, sequential O(N * overlap).")
    print("Without loop detection, sequential matching cannot close loops, so drift is not")
    print("corrected on long trajectories. (Matching times depend on CPU load.)\n")


# ----------------------------------------------------------------------------- exercise 3
def exercise_3() -> None:
    model = runs_dir(M.PROJECT, "south_building", "sift", "sparse", "0")
    if not (model / "points3D.bin").exists():
        M.colmap_sift_sfm(IMAGE_DIR, NAMES, runs_dir(M.PROJECT, "south_building", "sift"))
    rec = pycolmap.Reconstruction(model)
    pid = max(rec.points3D, key=lambda i: rec.points3D[i].track.length())
    pt = rec.points3D[pid]
    A, views = [], []
    for el in pt.track.elements:
        img = rec.images[el.image_id]
        cam = img.camera
        xy = np.array(img.points2D[el.point2D_idx].xy, dtype=np.float64)
        xn = cam.cam_from_img(xy[None])[0]  # undistorted, normalised image coordinates
        P = img.cam_from_world().matrix()  # 3x4 world -> camera
        A += [xn[0] * P[2] - P[0], xn[1] * P[2] - P[1]]
        views.append((img, cam, xy, P))
    X = np.linalg.svd(np.array(A))[2][-1]
    X = X[:3] / X[3]

    def reproj(Xw):
        errs = []
        for _img, cam, xy, P in views:
            Xc = P[:, :3] @ Xw + P[:, 3]
            errs.append(np.linalg.norm(cam.img_from_cam(Xc[None])[0] - xy))
        return np.array(errs)

    print(f"Exercise 3 - N-view DLT for point {pid} (track length {pt.track.length()})")
    e_dlt, e_ba = reproj(X).mean(), reproj(np.asarray(pt.xyz)).mean()
    print(f"  DLT xyz    = {X.round(5)}  mean reprojection error {e_dlt:.3f} px")
    print(f"  COLMAP xyz = {np.asarray(pt.xyz).round(5)}  mean reprojection error "
          f"{e_ba:.3f} px (point error field: {pt.error:.3f} px)")
    print(f"  distance between the two estimates: {np.linalg.norm(X - np.asarray(pt.xyz)):.2e} "
          "model units")
    print("  The linear DLT minimises an algebraic error; COLMAP's bundle-adjusted point")
    print("  minimises the robustified reprojection error (jointly with the cameras).")
    lower = "COLMAP's" if e_ba < e_dlt else "the DLT"
    print(f"  Here {lower} point has the lower mean pixel error ({min(e_ba, e_dlt):.3f} vs "
          f"{max(e_ba, e_dlt):.3f} px).")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
