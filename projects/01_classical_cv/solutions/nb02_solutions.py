#!/usr/bin/env python
"""Solutions to the exercises of notebook 02 (features, RANSAC homographies, panoramas).

Run from the repository root (CPU only, under a minute)::

    python projects/01_classical_cv/solutions/nb02_solutions.py
"""

from __future__ import annotations

import textwrap
import time

import cv2
import numpy as np

from cvlab import io

FEAT = io.data_dir("01_classical_cv", "features")


def warp_pts(pts: np.ndarray, H: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, np.float64).reshape(-1, 1, 2)  # OpenCV 5 needs (N, 1, 2) here
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def load():
    g1 = cv2.imread(str(FEAT / "graf1.png"), cv2.IMREAD_GRAYSCALE)
    g3 = cv2.imread(str(FEAT / "graf3.png"), cv2.IMREAD_GRAYSCALE)
    fs = cv2.FileStorage(str(FEAT / "H1to3p.xml"), cv2.FILE_STORAGE_READ)
    H_gt = fs.getNode("H13").mat()
    fs.release()
    sift = cv2.SIFT_create(nfeatures=4000)
    k1, d1 = sift.detectAndCompute(g1, None)
    k3, d3 = sift.detectAndCompute(g3, None)
    return g1, H_gt, k1, d1, k3, d3


def ratio_matches(knn, k1, k3, r=0.8):
    good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < r * n.distance]
    x1 = np.float64([k1[m.queryIdx].pt for m in good])
    x3 = np.float64([k3[m.trainIdx].pt for m in good])
    return x1, x3


def note(text: str) -> None:
    """Print an explanatory paragraph, wrapped and indented."""
    print(textwrap.fill(text, 96, initial_indent="  ", subsequent_indent="  "))


# ----------------------------------------------------------------------------- exercise 1
def exercise1(H_gt, k1, d1, k3, d3) -> None:
    print("Exercise 1: FLANN (k-d tree) vs brute-force matching of SIFT descriptors (ratio 0.8)")
    matchers = {"BFMatcher (exact)": lambda: cv2.BFMatcher(cv2.NORM_L2)}
    for trees, checks in [(1, 16), (5, 50), (5, 128), (8, 256)]:
        matchers[f"FLANN trees={trees} checks={checks}"] = lambda t=trees, c=checks: (
            cv2.FlannBasedMatcher(dict(algorithm=1, trees=t), dict(checks=c))
        )
    for name, make in matchers.items():
        matcher = make()
        matcher.knnMatch(d1, d3, k=2)  # warm-up (FLANN builds its index here)
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            knn = matcher.knnMatch(d1, d3, k=2)
            times.append(1e3 * (time.perf_counter() - t0))
        x1, x3 = ratio_matches(knn, k1, k3)
        correct = np.linalg.norm(warp_pts(x1, H_gt) - x3, axis=1) < 3
        print(f"  {name:28s} {np.mean(times):6.2f} ms | {len(x1):4d} matches, "
              f"{correct.sum():4d} correct, precision {correct.mean():.3f}")  # fmt: skip
    note(
        "More trees/checks approach the exact result. Here FLANN is not faster: OpenCV's "
        "brute-force matcher is SIMD-optimised and ~3k x 3.5k descriptors is a small "
        "problem. Approximate search pays off for much larger databases (e.g. place "
        "recognition over many keyframes)."
    )


# ----------------------------------------------------------------------------- exercise 2
def exercise2(g1, H_gt, k1, d1, k3, d3) -> None:
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d3, k=2)
    x1, x3 = ratio_matches(knn, k1, k3)
    h, w = g1.shape
    corners = np.float64([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])

    def corner_err(H):
        return np.linalg.norm(warp_pts(corners, H) - warp_pts(corners, H_gt), axis=1).mean()

    H, _ = cv2.findHomography(x1, x3, cv2.USAC_MAGSAC, 3.0)
    A, inl = cv2.estimateAffine2D(x1, x3, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    A3 = np.vstack([A, [0, 0, 1]])
    print("\nExercise 2: affine (6 DoF) vs homography (8 DoF) on graf1 -> graf3")
    print(f"  homography (MAGSAC++): corner error {corner_err(H):7.2f} px")
    print(
        f"  affine (RANSAC)      : corner error {corner_err(A3):7.2f} px, {int(inl.sum())} inliers"
    )
    centre = np.linalg.norm(x1 - [w / 2, h / 2], axis=1) < 120  # matches near the image centre only
    A_c, _ = cv2.estimateAffine2D(
        x1[centre], x3[centre], method=cv2.RANSAC, ransacReprojThreshold=3.0
    )
    A_c3 = np.vstack([A_c, [0, 0, 1]])
    gx, gy = np.meshgrid(
        np.linspace(w / 2 - 100, w / 2 + 100, 9), np.linspace(h / 2 - 100, h / 2 + 100, 9)
    )
    grid = np.stack([gx, gy], -1).reshape(-1, 2)  # 9x9 points in the central 200x200 px
    local = np.linalg.norm(warp_pts(grid, A_c3) - warp_pts(grid, H_gt), axis=1).mean()
    print(f"  affine fitted on {centre.sum()} central matches: error {local:.2f} px inside the")
    print(f"  central 200x200 px, but {corner_err(A_c3):.1f} px at the image corners")
    note(
        "The perspective terms (h31, h32) matter for a 40-degree viewpoint change; "
        "locally, or for distant/flat scenes, a homography is close to affine and the "
        "cheaper model is enough."
    )


# ----------------------------------------------------------------------------- exercise 3
def dlt(x1, x2):
    def norm(x):
        c = x.mean(0)
        s = np.sqrt(2) / np.mean(np.linalg.norm(x - c, axis=1))
        return (x - c) * s, np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])

    n1, T1 = norm(x1)
    n2, T2 = norm(x2)
    x, y = n1.T
    u, v = n2.T
    z, o = np.zeros_like(x), np.ones_like(x)
    A = np.empty((2 * len(x), 9))
    A[0::2] = np.stack([x, y, o, z, z, z, -u * x, -u * y, -u], 1)
    A[1::2] = np.stack([z, z, z, x, y, o, -v * x, -v * y, -v], 1)
    H = np.linalg.inv(T2) @ np.linalg.svd(A)[2][-1].reshape(3, 3) @ T1
    return H / H[2, 2]


def transfer(H, x1, x2):
    p = np.c_[x1, np.ones(len(x1))] @ H.T
    with np.errstate(all="ignore"):
        return np.linalg.norm(p[:, :2] / p[:, 2:3] - x2, axis=1)


def ransac_fixed(x1, x2, n_iters, rng, thresh=3.0):
    """RANSAC with a fixed number of iterations; also reports whether the best sample was clean."""
    best, best_idx = np.zeros(len(x1), bool), None
    for _ in range(n_iters):
        idx = rng.choice(len(x1), 4, replace=False)
        with np.errstate(all="ignore"):
            try:
                inl = transfer(dlt(x1[idx], x2[idx]), x1, x2) < thresh
            except np.linalg.LinAlgError:
                continue
        if inl.sum() > best.sum():
            best, best_idx = inl, idx
    if best.sum() < 4:
        return None, best_idx
    return dlt(x1[best], x2[best]), best_idx


def exercise3(g1, H_gt, k1, d1, k3, d3) -> None:
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d3, k=2)
    x1, x3 = ratio_matches(knn, k1, k3)
    ok = np.linalg.norm(warp_pts(x1, H_gt) - x3, axis=1) < 3
    x1, x3 = x1[ok], x3[ok]  # ground-truth-correct matches only
    h, w = g1.shape
    corners = np.float64([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    rng = np.random.default_rng(0)
    print(
        f"\nExercise 3: RANSAC success vs theory ({len(x1)} correct matches + synthetic outliers,"
    )
    print("  100 trials each)")
    print(
        f"  {'eps':>4} {'N':>4} {'theory':>7} {'clean best sample':>18} {'corner err < 5 px':>18}"
    )
    for eps in [0.3, 0.5, 0.7]:
        w_in = 1 - eps
        for n_iters in [1, 3, 10, 30, 100, 300]:
            clean = success = 0
            for _ in range(100):
                x3c = x3.copy()
                out = rng.random(len(x1)) < eps  # replace a fraction eps by random points
                x3c[out] = rng.uniform([0, 0], [w, h], (out.sum(), 2))
                H, idx = ransac_fixed(x1, x3c, n_iters, rng)
                clean += idx is not None and not out[idx].any()
                if H is not None:
                    e = np.linalg.norm(
                        warp_pts(corners, H) - warp_pts(corners, H_gt), axis=1
                    ).mean()
                    success += e < 5
            theory = 1 - (1 - w_in**4) ** n_iters
            print(
                f"  {eps:4.1f} {n_iters:4d} {theory:7.3f} {clean / 100:18.2f} {success / 100:18.2f}"
            )
    note(
        "The measured rate of a clean winning sample follows 1 - (1 - w^4)^N; final "
        "accuracy can be a bit lower because a clean but badly conditioned sample can "
        "still win."
    )


if __name__ == "__main__":
    g1, H_gt, k1, d1, k3, d3 = load()
    exercise1(H_gt, k1, d1, k3, d3)
    exercise2(g1, H_gt, k1, d1, k3, d3)
    exercise3(g1, H_gt, k1, d1, k3, d3)
