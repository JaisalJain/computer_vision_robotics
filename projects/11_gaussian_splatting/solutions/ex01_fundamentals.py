#!/usr/bin/env python
"""Solutions to the exercises of notebook 01 (Gaussian splatting fundamentals).

1. Anisotropy of the projected covariance of an isotropic Gaussian across the image.
2. Image fitting with L1 vs 0.8 L1 + 0.2 D-SSIM.
3. Rate-distortion: final PSNR vs number of Gaussians.

Run (GPU)::

    python projects/11_gaussian_splatting/solutions/ex01_fundamentals.py

Figures go to ``outputs/11_gaussian_splatting/solutions/`` (gitignored).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import splatkit as S  # noqa: E402

from cvlab import viz  # noqa: E402
from cvlab.device import setup  # noqa: E402
from cvlab.io import data_dir, outputs_dir, read_image, repo_root  # noqa: E402

OUT = outputs_dir(S.PROJECT, "solutions", create=True)


def exercise1() -> None:
    """Perspective makes an isotropic Gaussian look elongated towards the image border."""
    K = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])
    cov = S.covariance_from_scale_rot(np.full(3, 0.05), np.array([1.0, 0, 0, 0]))
    us, ratios = [], []
    for x in np.linspace(0.0, 1.5, 16):
        m2d, c2d = S.project_covariance(np.array([x, 0.0, 2.0]), cov, np.eye(4), K)
        ev = np.linalg.eigvalsh(c2d)
        us.append(m2d[0])
        ratios.append(float(np.sqrt(ev[1] / ev[0])))
    print("Ex1: u [px] -> major/minor axis ratio")
    for u, r in zip(us[::3], ratios[::3], strict=True):
        print(f"   u = {u:6.1f}  ratio = {r:.3f}")
    # Analytic check: for an isotropic Gaussian the ratio is sqrt(1 + (x/z)^2) along u.
    x_last = 1.5
    print(f"   analytic at x=1.5 m: {np.sqrt(1 + (x_last / 2.0) ** 2):.3f} "
          f"(the -fx*x/z^2 column of J adds the depth spread to the u direction)")  # fmt: skip
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(us, ratios, "o-")
    ax.set_xlabel("u of the projected mean [px]")
    ax.set_ylabel("major / minor axis ratio")
    ax.set_title("Isotropic 3D Gaussian (s = 5 cm, z = 2 m) projected across the image")
    fig.savefig(OUT / "ex01_anisotropy.png", dpi=100)
    plt.close(fig)


def fit_image(target, n, device, steps=1000, ssim_lambda=0.0, seed=0):
    """The notebook's image-fitting loop with an optional D-SSIM term."""
    from gsplat import rasterization

    H, W = target.shape[:2]
    f = float(0.5 * W / np.tan(np.radians(30)))  # python float: keep every tensor float32
    K = torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], dtype=torch.float32,
                     device=device)[None]  # fmt: skip
    view = torch.eye(4, device=device)[None]
    z0 = 2.0
    g = torch.Generator(device=device).manual_seed(seed)
    half = torch.tensor([z0 * W / (2 * f), z0 * H / (2 * f), 0.05], device=device,
                        dtype=torch.float32)  # gsplat kernels reject float64
    p = {
        "means": (torch.rand(n, 3, generator=g, device=device) * 2 - 1) * half
        + torch.tensor([0, 0, z0], device=device),
        "scales": torch.full((n, 3), float(np.log(2 * half[0].item() / np.sqrt(n))),
                             device=device),  # fmt: skip
        "quats": torch.rand(n, 4, generator=g, device=device),
        "opacities": torch.zeros(n, device=device),
        "colors": torch.zeros(n, 3, device=device),
    }
    for v in p.values():
        v.requires_grad_(True)
    lrs = {"means": 2e-3, "scales": 1e-2, "quats": 1e-2, "opacities": 5e-2, "colors": 5e-2}
    opt = torch.optim.Adam([{"params": [p[k]], "lr": lr} for k, lr in lrs.items()])
    tgt_chw = target.permute(2, 0, 1)[None]
    t0 = time.perf_counter()
    for step in range(steps + 1):
        out, _, _ = rasterization(p["means"], p["quats"], p["scales"].exp(),
                                  p["opacities"].sigmoid(), p["colors"].sigmoid(), view, K, W,
                                  H)  # fmt: skip
        pred = out[0]
        if step == steps:
            break
        loss = (pred - target).abs().mean()
        if ssim_lambda > 0:
            d_ssim = 1 - S.ssim(pred.permute(2, 0, 1)[None], tgt_chw)
            loss = (1 - ssim_lambda) * loss + ssim_lambda * d_ssim
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    with torch.no_grad():
        pred = pred.clamp(0, 1)
        psnr = -10 * torch.log10(((pred - target) ** 2).mean()).item()
        ssim = S.ssim(pred.permute(2, 0, 1)[None], tgt_chw).item()
    return {"n": n, "psnr": psnr, "ssim": ssim, "time_s": time.perf_counter() - t0}


def main() -> None:
    device = setup(seed=0, verbose=False)
    viz.set_style()
    exercise1()

    img_path = sorted((data_dir("mipnerf360", "bonsai") / "images_4").glob("*.png"))[1]
    print(f"target: {img_path.relative_to(repo_root())}")
    target = torch.from_numpy(read_image(img_path, max_side=400)).float().div(255).to(device)

    print("Ex2: 20k Gaussians, 1000 steps")
    for lam in (0.0, 0.2):
        r = fit_image(target, 20_000, device, ssim_lambda=lam)
        name = "L1" if lam == 0 else "0.8 L1 + 0.2 D-SSIM"
        print(f"   {name:<20} PSNR {r['psnr']:.2f} dB | SSIM {r['ssim']:.4f} | "
              f"{r['time_s']:.1f} s")  # fmt: skip

    print("Ex3: rate-distortion (L1, 1000 steps)")
    ns, ps = [], []
    for n in (500, 2_000, 8_000, 32_000):
        r = fit_image(target, n, device)
        ns.append(n)
        ps.append(r["psnr"])
        print(f"   {n:>6} Gaussians: PSNR {r['psnr']:.2f} dB ({r['time_s']:.1f} s)")
    slope = np.polyfit(np.log2(ns), ps, 1)[0]
    print(f"   least-squares slope: {slope:.2f} dB per doubling of the Gaussian count")
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.semilogx(ns, ps, "o-", base=2)
    ax.set_xlabel("number of Gaussians")
    ax.set_ylabel("PSNR after 1000 steps [dB]")
    ax.set_title("Image fitting: rate vs distortion")
    fig.savefig(OUT / "ex01_rate_distortion.png", dpi=100)
    plt.close(fig)
    print(f"figures -> {OUT.relative_to(repo_root())}")


if __name__ == "__main__":
    main()
