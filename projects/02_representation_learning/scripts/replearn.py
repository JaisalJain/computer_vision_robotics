"""Shared code for project 02 (representation learning on Imagenette-160).

The training script, the three notebooks, the exercise solutions and the unit tests
all import from this module so that data handling, model construction and the
evaluation helpers are identical everywhere.

Contents
--------
* Imagenette class metadata and torchvision data pipelines,
* timm model construction for *linear probing* (frozen backbone) and *full fine-tuning*,
* a mixed-precision train / evaluate loop,
* frozen-feature tools: weighted k-NN classifier, nearest-class-mean classifier and
  the PCA-of-patch-tokens RGB visualisation,
* Grad-CAM (forward/backward hooks) and the deletion / insertion faithfulness curves.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

PROJECT = "02_representation_learning"

# WordNet id -> readable name. Sorted by wnid, which is also torchvision ImageFolder order.
IMAGENETTE_CLASSES: dict[str, str] = {
    "n01440764": "tench",
    "n02102040": "English springer",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute",
}
CLASS_NAMES: list[str] = list(IMAGENETTE_CLASSES.values())
# Index of each Imagenette class inside the 1000 ImageNet-1k logits (checked with
# timm.data.ImageNetInfo). Imagenette *is* an ImageNet subset, which matters when we
# interpret transfer-learning numbers (see the notebooks).
IMAGENET1K_INDEX: list[int] = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

BACKBONES: list[str] = [
    "resnet18.a1_in1k",
    "efficientnet_b0.ra_in1k",
    "convnext_nano.in12k_ft_in1k",
]
MODES: list[str] = ["linear_probe", "finetune"]


# ------------------------------------------------------------------------------- data
def imagenette_root() -> Path:
    """``data/imagenette/imagenette2-160`` (``make data-02_representation_learning``)."""
    from cvlab.io import data_dir

    return data_dir("imagenette", "imagenette2-160")


def build_transform(
    train: bool,
    size: int = 160,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
    resize: int | None = None,
):
    """torchvision v2 pipeline: PIL image -> normalised float tensor (3, size, size).

    Train: random-resized crop (35-100 % of the area) + horizontal flip.
    Eval: resize the short side to ``resize`` (default ``size``) and centre-crop ``size``.
    """
    from torchvision.transforms import v2

    if train:
        geo = [v2.RandomResizedCrop(size, scale=(0.35, 1.0), antialias=True),
               v2.RandomHorizontalFlip()]  # fmt: skip
    else:
        geo = [v2.Resize(resize or size, antialias=True), v2.CenterCrop(size)]
    return v2.Compose(
        [v2.ToImage(), *geo, v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean, std)]
    )


def make_dataset(split: str, transform=None, root: Path | None = None):
    """``torchvision.datasets.ImageFolder`` for ``split`` in {"train", "val"}."""
    from torchvision.datasets import ImageFolder

    ds = ImageFolder(str((root or imagenette_root()) / split), transform=transform)
    assert [IMAGENETTE_CLASSES[c] for c in ds.classes] == CLASS_NAMES, ds.classes
    return ds


def make_loader(
    dataset, batch_size: int = 128, train: bool = False, workers: int = 8, seed: int = 0
) -> torch.utils.data.DataLoader:
    """DataLoader with pinned memory and persistent workers (shuffled when ``train``)."""
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        drop_last=train,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        generator=g,
    )


def denormalize(x: torch.Tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> np.ndarray:
    """Normalised (3, H, W) or (B, 3, H, W) tensor -> uint8 RGB (H, W, 3) / (B, H, W, 3)."""
    m = torch.tensor(mean, device=x.device).view(-1, 1, 1)
    s = torch.tensor(std, device=x.device).view(-1, 1, 1)
    img = (x.detach().float() * s + m).clamp(0, 1)
    img = img.movedim(-3, -1).cpu().numpy()
    return (img * 255).round().astype(np.uint8)


# ----------------------------------------------------------------------------- models
def build_model(
    name: str, mode: str = "finetune", num_classes: int = 10, pretrained: bool = True
) -> nn.Module:
    """timm backbone with a fresh ``num_classes`` head.

    ``mode="linear_probe"`` freezes every parameter except the final classifier
    (``model.get_classifier()``), so only ``C x (D + 1)`` weights are learned.
    ``mode="finetune"`` leaves everything trainable.
    """
    import timm

    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
    if mode == "linear_probe":
        model.requires_grad_(False)
        model.get_classifier().requires_grad_(True)
    return model


def count_params(model: nn.Module) -> tuple[int, int]:
    """(total, trainable) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def forward_logits(model: nn.Module, x: torch.Tensor, mode: str) -> torch.Tensor:
    """Forward pass; for a linear probe the frozen trunk runs without autograd.

    Skipping the backbone graph is what makes linear probing cheap: no activations
    are stored for backprop, so memory and time drop sharply.
    """
    if mode == "linear_probe":
        with torch.no_grad():
            feats = model.forward_features(x)
        return model.forward_head(feats)
    return model(x)


def cosine_warmup(total_steps: int, warmup_steps: int) -> Callable[[int], float]:
    """LR multiplier: linear warm-up then cosine decay to 0 (for ``LambdaLR``)."""

    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))

    return fn


def make_optimizer(
    model: nn.Module, mode: str, lr_head: float, lr_backbone: float, weight_decay: float
) -> torch.optim.Optimizer:
    """AdamW; in fine-tuning the pretrained trunk gets a smaller LR than the new head."""
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    head = [p for p in model.parameters() if p.requires_grad and id(p) in head_ids]
    trunk = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    groups = [{"params": head, "lr": lr_head}]
    if mode == "finetune" and trunk:
        groups.append({"params": trunk, "lr": lr_backbone})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    mode: str,
    amp_dtype: torch.dtype | None = torch.bfloat16,
    label_smoothing: float = 0.1,
    max_steps: int | None = None,
) -> dict[str, float]:
    """One pass over ``loader``; returns mean loss, accuracy, wall time and images/s."""
    # A linear probe keeps the trunk in eval mode: BatchNorm statistics stay frozen.
    model.train(mode == "finetune")
    loss_sum, correct, n = 0.0, 0, 0
    t0 = time.perf_counter()
    for step, (x, y) in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = forward_logits(model, x, mode)
            loss = F.cross_entropy(logits.float(), y, label_smoothing=label_smoothing)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        loss_sum += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return {"loss": loss_sum / n, "acc": correct / n, "seconds": dt, "imgs_per_s": n / dt}


@torch.no_grad()
def predict(
    model: nn.Module,
    loader,
    device: torch.device,
    amp_dtype: torch.dtype | None = torch.bfloat16,
) -> tuple[np.ndarray, np.ndarray]:
    """Logits (N, C) float32 and labels (N,) for a whole loader."""
    model.eval()
    all_logits, all_y = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(x)
        all_logits.append(logits.float().cpu())
        all_y.append(y)
    return torch.cat(all_logits).numpy(), torch.cat(all_y).numpy()


def evaluate(
    model: nn.Module, loader, device: torch.device, amp_dtype: torch.dtype | None = torch.bfloat16
) -> dict[str, float]:
    """Top-1 accuracy and cross-entropy (no label smoothing) over a loader."""
    logits, y = predict(model, loader, device, amp_dtype)
    loss = F.cross_entropy(torch.from_numpy(logits), torch.from_numpy(y)).item()
    return {"loss": loss, "acc": float((logits.argmax(1) == y).mean())}


# --------------------------------------------------------------------- frozen features
def knn_predict(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    test_feats: torch.Tensor,
    k: int = 20,
    temperature: float | None = 0.07,
    num_classes: int | None = None,
    chunk: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cosine-similarity k-NN classifier (the DINO evaluation protocol).

    Each of the ``k`` nearest training samples votes for its label with weight
    ``exp(s / T)`` where ``s`` is the cosine similarity (``temperature=None`` gives
    a plain majority vote).

    Returns:
        ``(pred, votes)``: predicted labels (M,) and class vote totals (M, C).
    """
    c = int(num_classes or int(train_labels.max()) + 1)
    tr = F.normalize(train_feats.float(), dim=1)
    te = F.normalize(test_feats.float().to(tr.device), dim=1)
    labels = train_labels.to(tr.device).long()
    out = []
    for i in range(0, len(te), chunk):
        sims = te[i : i + chunk] @ tr.T  # (b, N) cosine similarities
        top_s, top_i = sims.topk(k, dim=1)  # (b, k)
        w = torch.ones_like(top_s) if temperature is None else (top_s / temperature).exp()
        votes = torch.zeros(len(top_s), c, device=tr.device)
        votes.scatter_add_(1, labels[top_i], w)
        out.append(votes)
    votes = torch.cat(out)
    return votes.argmax(1), votes


def nearest_class_mean(
    train_feats: torch.Tensor, train_labels: torch.Tensor, test_feats: torch.Tensor
) -> torch.Tensor:
    """Prototype classifier: assign each test feature to the closest (cosine) class mean."""
    tr = F.normalize(train_feats.float(), dim=1)
    classes = torch.unique(train_labels)
    protos = torch.stack([tr[train_labels == c].mean(0) for c in classes])
    sims = F.normalize(test_feats.float(), dim=1) @ F.normalize(protos, dim=1).T
    return classes[sims.argmax(1)]


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Otsu's threshold of a 1-D sample (maximises the between-class variance)."""
    v = np.asarray(values, dtype=np.float64).ravel()
    hist, edges = np.histogram(v, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers) / np.maximum(w0, 1)
    m1 = (np.sum(hist * centers) - np.cumsum(hist * centers)) / np.maximum(w1, 1)
    between = w0 * w1 * (m0 - m1) ** 2
    # Empty bins between two modes give a plateau of equal scores: take its middle.
    best = np.flatnonzero(between >= between.max() * (1 - 1e-9))
    return float(centers[best].mean())


def _pca(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-``k`` principal directions (k, D) and the projected data (N, k)."""
    mu = x.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(x - mu, full_matrices=False)
    comps = vt[:k]
    return comps, (x - mu) @ comps.T


def pca_rgb(
    patch_feats: np.ndarray, foreground: bool = True, clip_pct: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """DINO-style visualisation of patch tokens as colours.

    Args:
        patch_feats: (B, h, w, D) patch tokens of B images (fit jointly so colours are
            comparable across images).
        foreground: Threshold the first principal component (Otsu) to separate object
            from background, then run a second 3-component PCA on foreground patches only.
            The sign of PC1 is chosen so that border patches count as background.
        clip_pct: Percentile clipping before min-max scaling each component to [0, 1].

    Returns:
        ``(rgb, fg)``: (B, h, w, 3) floats in [0, 1] (background black) and the (B, h, w)
        boolean foreground mask (all True when ``foreground=False``).
    """
    b, h, w, d = patch_feats.shape
    x = patch_feats.reshape(-1, d).astype(np.float64)
    fg = np.ones(len(x), dtype=bool)
    if foreground:
        _, proj = _pca(x, 1)
        pc1 = proj[:, 0].reshape(b, h, w)
        border = np.concatenate(
            [pc1[:, 0].ravel(), pc1[:, -1].ravel(), pc1[:, :, 0].ravel(), pc1[:, :, -1].ravel()]
        )
        if border.mean() > pc1.mean():  # make "high PC1" mean "object"
            pc1 = -pc1
        fg = (pc1 > otsu_threshold(pc1)).ravel()
        if fg.sum() < 3:
            fg[:] = True
    _, proj = _pca(x[fg], 3)
    lo = np.percentile(proj, clip_pct, axis=0)
    hi = np.percentile(proj, 100 - clip_pct, axis=0)
    proj = np.clip((proj - lo) / np.maximum(hi - lo, 1e-12), 0, 1)
    rgb = np.zeros((len(x), 3))
    rgb[fg] = proj
    return rgb.reshape(b, h, w, 3), fg.reshape(b, h, w)


# --------------------------------------------------------------------------- Grad-CAM
class GradCAM:
    """Grad-CAM (Selvaraju et al., 2017) with forward/backward hooks.

    For a target layer with activations ``A`` (B, K, h, w) and a class score ``y_c``::

        alpha_k = mean_{i,j} dy_c / dA_k[i, j]          # importance of channel k
        L_c     = ReLU( sum_k alpha_k * A_k )            # (h, w) class activation map

    The map is upsampled to the input size and scaled to [0, 1] per image.

    Example::

        cam = GradCAM(model, model.layer4)
        heat, logits = cam(x)          # explain the top-1 class of each image
        cam.remove()
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handle = target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module: nn.Module, inputs, output: torch.Tensor) -> None:
        self.activations = output
        if output.requires_grad:
            # A tensor hook receives dL/dA for exactly this tensor during backward().
            output.register_hook(self._store_grad)

    def _store_grad(self, grad: torch.Tensor) -> None:
        self.gradients = grad

    def __call__(
        self, x: torch.Tensor, class_idx: int | Sequence[int] | torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(cam, logits)``: cam (B, H, W) in [0, 1] for the requested classes."""
        self.model.eval()
        # requires_grad on the input keeps the graph alive even when the trunk is frozen.
        x = x.detach().clone().requires_grad_(True)
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if class_idx is None:
            idx = logits.argmax(1)
        else:
            idx = torch.as_tensor(class_idx, device=logits.device).long().reshape(-1)
            idx = idx.expand(len(logits)) if idx.numel() == 1 else idx
        score = logits.gather(1, idx[:, None]).sum()  # images are independent -> sum is fine
        score.backward()
        acts, grads = self.activations.detach(), self.gradients.detach()  # (B, K, h, w)
        alpha = grads.mean(dim=(2, 3), keepdim=True)  # global-average-pooled gradients
        cam = F.relu((alpha * acts).sum(1, keepdim=True))  # (B, 1, h, w)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        cam = cam / cam.flatten(1).amax(1).clamp_min(1e-12).view(-1, 1, 1)
        return cam, logits.detach()

    def remove(self) -> None:
        self._handle.remove()


def blur_baseline(x: torch.Tensor, kernel: int = 31, sigma: float = 10.0) -> torch.Tensor:
    """Heavily Gaussian-blurred copy of a (B, 3, H, W) batch (the 'removed' state)."""
    from torchvision.transforms.v2 import functional as TF

    k = min(kernel, 2 * min(x.shape[-2:]) - 1)  # reflect padding needs k // 2 < H, W
    k = k if k % 2 == 1 else k - 1
    return TF.gaussian_blur(x, kernel_size=[k, k], sigma=[sigma, sigma])


@torch.no_grad()
def deletion_insertion(
    model: nn.Module,
    x: torch.Tensor,
    saliency: torch.Tensor,
    class_idx: torch.Tensor,
    steps: int = 16,
    baseline: torch.Tensor | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Deletion and insertion curves (Petsiuk et al., RISE, 2018).

    Pixels are ranked by ``saliency`` (B, H, W). *Deletion* replaces the top fraction
    ``f`` of pixels by the blurred baseline and tracks the softmax probability of
    ``class_idx``; a faithful map makes it drop fast (low area under the curve).
    *Insertion* starts from the baseline and pastes the top pixels back; a faithful
    map makes the probability rise fast (high AUC).

    Returns:
        ``(deletion, insertion)`` arrays of shape (B, steps + 1) for fractions
        ``0, 1/steps, ..., 1``.
    """
    model.eval()
    b, _, h, w = x.shape
    base = blur_baseline(x) if baseline is None else baseline
    order = saliency.flatten(1).argsort(dim=1, descending=True)  # (B, H*W)
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(h * w, device=x.device).expand(b, -1))
    rank = rank.view(b, 1, h, w)
    dele, ins = [], []
    for s in range(steps + 1):
        top = (rank < round(s / steps * h * w)).to(x.dtype)  # 1 = among the top pixels
        x_del = x * (1 - top) + base * top
        x_ins = base * (1 - top) + x * top
        p = model(torch.cat([x_del, x_ins])).float().softmax(1)
        p = p.gather(1, torch.cat([class_idx, class_idx])[:, None])[:, 0]
        dele.append(p[:b].cpu())
        ins.append(p[b:].cpu())
    return torch.stack(dele, 1).numpy(), torch.stack(ins, 1).numpy()


def auc(curve: np.ndarray) -> np.ndarray:
    """Trapezoidal area under curves sampled uniformly on [0, 1] (last axis)."""
    y = np.asarray(curve, dtype=np.float64)
    return (y[..., 1:] + y[..., :-1]).sum(-1) / (2 * (y.shape[-1] - 1))


# --------------------------------------------------------------------- frozen encoders
DINOV3_REPO = "facebook/dinov3-vits16-pretrain-lvd1689m"  # gated (DINOv3 License)
DINOV3_TIMM_MIRROR = "vit_small_patch16_dinov3.lvd1689m"  # ungated timm mirror, same weights
DINOV2_REPO = "facebook/dinov2-small"  # Apache-2.0
SUPERVISED_CNN = "convnext_nano.in12k_ft_in1k"  # Apache-2.0, supervised ImageNet labels


class Encoder(nn.Module):
    """Frozen image encoder -> (global descriptor (B, D), patch-feature grid (B, h, w, D)).

    Wraps three backends behind one interface:

    * ``"hf_vit"``: a transformers ViT (DINOv2 / DINOv3). Token layout of
      ``last_hidden_state`` is ``[CLS, registers..., patches...]``; the global
      descriptor is the (layer-normed) CLS token.
    * ``"timm_vit"``: a timm ViT (``forward_features`` returns the same layout).
    * ``"timm_cnn"``: a timm CNN; the global descriptor is the pooled pre-logits
      feature and the "patches" are the cells of the last feature map.
    """

    def __init__(self, key: str, backend: str, model: nn.Module, patch_size: int,
                 n_prefix: int, dim: int, source: str) -> None:  # fmt: skip
        super().__init__()
        self.key, self.backend, self.model = key, backend, model.eval().requires_grad_(False)
        self.patch_size, self.n_prefix, self.dim, self.source = patch_size, n_prefix, dim, source

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, _, hh, ww = x.shape
        if self.backend == "timm_cnn":
            fmap = self.model.forward_features(x)  # (B, C, h, w)
            glob = self.model.forward_head(fmap, pre_logits=True)  # (B, C)
            return glob.float(), fmap.permute(0, 2, 3, 1).float()
        if self.backend == "hf_vit":
            tokens = self.model(pixel_values=x).last_hidden_state
        else:
            tokens = self.model.forward_features(x)
        h, w = hh // self.patch_size, ww // self.patch_size
        patches = tokens[:, self.n_prefix :].reshape(b, h, w, -1)
        return tokens[:, 0].float(), patches.float()


def load_encoder(key: str, device: torch.device | str = "cpu") -> Encoder:
    """Load one of ``"dinov3_vits16"``, ``"dinov2_vits14"``, ``"convnext_nano_sup"``.

    DINOv3 is loaded from the official gated repository; if the account has no access
    the ungated timm mirror of the same checkpoint is used instead (``Encoder.source``
    records which one was loaded).
    """
    import timm

    if key == "dinov3_vits16":
        try:
            from transformers import AutoModel

            m = AutoModel.from_pretrained(DINOV3_REPO)
            enc = Encoder(key, "hf_vit", m, 16, 1 + m.config.num_register_tokens,
                          m.config.hidden_size, DINOV3_REPO)  # fmt: skip
        except OSError:  # gated repo without access -> timm mirror
            m = timm.create_model(DINOV3_TIMM_MIRROR, pretrained=True, num_classes=0)
            enc = Encoder(key, "timm_vit", m, 16, m.num_prefix_tokens, m.num_features,
                          "timm/" + DINOV3_TIMM_MIRROR)  # fmt: skip
    elif key == "dinov2_vits14":
        from transformers import AutoModel

        m = AutoModel.from_pretrained(DINOV2_REPO)
        enc = Encoder(key, "hf_vit", m, 14, 1, m.config.hidden_size, DINOV2_REPO)
    elif key == "convnext_nano_sup":
        m = timm.create_model(SUPERVISED_CNN, pretrained=True)
        enc = Encoder(key, "timm_cnn", m, 32, 0, m.num_features, "timm/" + SUPERVISED_CNN)
    else:
        raise KeyError(key)
    return enc.to(device)


@torch.no_grad()
def extract_features(
    encoder: Encoder,
    loader,
    device: torch.device,
    amp_dtype: torch.dtype | None = torch.float16,
) -> dict[str, torch.Tensor]:
    """Run a frozen encoder over a loader.

    Returns CPU float32 tensors ``cls`` (N, D), ``mean_patch`` (N, D) and ``labels`` (N,).
    """
    cls, mean_patch, labels = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            g, p = encoder(x)
        cls.append(g.cpu())
        mean_patch.append(p.mean(dim=(1, 2)).cpu())
        labels.append(y)
    return {"cls": torch.cat(cls), "mean_patch": torch.cat(mean_patch), "labels": torch.cat(labels)}
