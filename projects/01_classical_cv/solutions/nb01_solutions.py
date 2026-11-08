#!/usr/bin/env python
"""Solutions to the exercises of notebook 01 (image formation, filtering and edges).

Run from the repository root (CPU only, a few seconds)::

    python projects/01_classical_cv/solutions/nb01_solutions.py
"""

from __future__ import annotations

import textwrap
import time

import cv2
import numpy as np

from cvlab import io, metrics, tum

TUM_DIR = io.data_dir("tum", "fr1_desk", "rgbd_dataset_freiburg1_desk")


def note(text: str) -> None:
    """Print an explanatory paragraph, wrapped and indented."""
    print(textwrap.fill(text, 96, initial_indent="  ", subsequent_indent="  "))


# ----------------------------------------------------------------------------- exercise 1
def bilateral_scratch(im: np.ndarray, radius: int = 4, sigma_s: float = 5.0, sigma_r: float = 50.0):
    """Bilateral filter on a grey float image with a circular (2r+1)^2 window, like OpenCV's."""
    H, W = im.shape
    p = np.pad(im, radius, mode="reflect")  # == OpenCV's BORDER_REFLECT_101
    num = np.zeros_like(im, dtype=np.float64)
    den = np.zeros_like(im, dtype=np.float64)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            r2 = dx * dx + dy * dy
            if r2 > radius * radius:  # OpenCV only uses offsets inside the circle of radius d/2
                continue
            shifted = p[radius + dy : radius + dy + H, radius + dx : radius + dx + W]
            w = np.exp(-r2 / (2 * sigma_s**2)) * np.exp(-((shifted - im) ** 2) / (2 * sigma_r**2))
            num += w * shifted
            den += w
    return (num / den).astype(np.float32)


def exercise1(img: np.ndarray) -> None:
    rng = np.random.default_rng(0)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    noisy = np.clip(gray + rng.normal(0, 20, gray.shape), 0, 255).astype(np.float32)
    x = noisy[300:460, 180:380]  # same crop as in the notebook
    t0 = time.perf_counter()
    ours = bilateral_scratch(x, radius=4, sigma_s=5.0, sigma_r=50.0)
    t1 = time.perf_counter()
    ref = cv2.bilateralFilter(x, 9, 50, 5)  # d = 2r + 1, sigmaColor, sigmaSpace
    t2 = time.perf_counter()
    clean = gray[300:460, 180:380]
    print("Exercise 1: bilateral filter from scratch (160x200 crop, sigma_s=5, sigma_r=50)")
    agree, max_diff = metrics.psnr(ours, ref, 255), np.abs(ours - ref).max()
    print(f"  PSNR(ours, OpenCV) = {agree:.1f} dB, max |diff| = {max_diff:.2f} grey levels")
    p_noisy, p_ours = metrics.psnr(x, clean, 255), metrics.psnr(ours, clean, 255)
    p_ref = metrics.psnr(ref, clean, 255)
    print(f"  PSNR vs clean: noisy {p_noisy:.2f} dB, ours {p_ours:.2f} dB, OpenCV {p_ref:.2f} dB")
    print(f"  time: ours {1e3 * (t1 - t0):.1f} ms, OpenCV {1e3 * (t2 - t1):.2f} ms")
    note("With a circular window and reflect-101 borders the result matches OpenCV's float path.")


# ----------------------------------------------------------------------------- exercise 2
def lap_var(g: np.ndarray) -> float:
    # OpenCV 5 rejects float32 -> CV_64F in Laplacian, so float input is promoted to float64 first
    g = g if g.dtype == np.uint8 else np.asarray(g, np.float64)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def exercise2(seq) -> None:
    ids = np.arange(0, len(seq), 3)
    scores = [lap_var(cv2.cvtColor(seq.rgb(i), cv2.COLOR_RGB2GRAY)) for i in ids]
    best, worst = int(ids[np.argmax(scores)]), int(ids[np.argmin(scores)])
    g = cv2.cvtColor(seq.rgb(worst), cv2.COLOR_RGB2GRAY).astype(np.float32)
    print(f"\nExercise 2: unsharp masking of the blurriest frame #{worst} (sharpest is #{best})")
    print(f"  sharpest frame score S = {max(scores):.1f}")
    print(f"  blurriest frame:  alpha=0.0 -> S = {lap_var(g):.1f}")
    flat = g[400:470, 400:600]  # low-texture desk region: its Laplacian is mostly noise
    for alpha in [0.5, 1.0, 2.0, 4.0]:
        sharp = np.clip(g + alpha * (g - cv2.GaussianBlur(g, (0, 0), 2.0)), 0, 255)
        f = sharp[400:470, 400:600]
        noise, noise0 = np.sqrt(lap_var(f)), np.sqrt(lap_var(flat))
        print(
            f"{'':20s}alpha={alpha:.1f} -> S = {lap_var(sharp):7.1f} | "
            f"Laplacian std in a flat desk patch {noise:.2f} (before: {noise0:.2f})"
        )
    note(
        "Sharpening raises S and amplifies noise in flat regions, but it cannot restore "
        "detail lost to motion blur: the measure is fooled, so select frames on S before "
        "any sharpening."
    )


# ----------------------------------------------------------------------------- exercise 3
def harris(gray_f: np.ndarray, k: float = 0.04, window: str = "box") -> np.ndarray:
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    if window == "box":  # what cv2.cornerHarris does (blockSize x blockSize sum)
        smooth = lambda a: cv2.boxFilter(a, -1, (3, 3), normalize=False)  # noqa: E731
    else:
        smooth = lambda a: cv2.GaussianBlur(a, (0, 0), 1.0)  # noqa: E731
    sxx, sxy, syy = smooth(gx * gx), smooth(gx * gy), smooth(gy * gy)
    return sxx * syy - sxy**2 - k * (sxx + syy) ** 2


def top_corners(R: np.ndarray, n: int = 200) -> np.ndarray:
    peaks = (R == cv2.dilate(R, np.ones((5, 5), np.uint8))) & (R > 0)  # 5x5 non-max suppression
    ys, xs = np.nonzero(peaks)
    order = np.argsort(R[ys, xs])[::-1][:n]
    return np.stack([xs[order], ys[order]], axis=1).astype(float)


def exercise3(img: np.ndarray) -> None:
    gray_f = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    ref = cv2.cornerHarris(gray_f, 3, 3, 0.04)
    c_ref = top_corners(ref)
    print("\nExercise 3: Harris response from the structure tensor (top-200 corners, 5x5 NMS)")
    for window in ["box", "gaussian"]:
        R = harris(gray_f, window=window)
        c = top_corners(R)
        d = np.min(np.linalg.norm(c[:, None] - c_ref[None], axis=2), axis=1)
        corr = np.corrcoef(R.ravel(), ref.ravel())[0, 1]
        print(f"  {window:8s} window: corr(R, cv2.cornerHarris) = {corr:.4f}, "
              f"{(d <= 1).mean() * 100:.0f}% of our corners within 1 px of OpenCV's")  # fmt: skip
    note(
        "The box window reproduces OpenCV up to a constant scale; a Gaussian window "
        "changes the ranking of the corners somewhat."
    )


if __name__ == "__main__":
    seq = tum.load_sequence(TUM_DIR)
    ids = np.arange(0, len(seq), 3)
    best = int(ids[np.argmax([lap_var(cv2.cvtColor(seq.rgb(i), cv2.COLOR_RGB2GRAY)) for i in ids])])
    frame = seq.rgb(best)
    exercise1(frame)
    exercise2(seq)
    exercise3(frame)
