"""CPU tests for project 02's shared module (k-NN, PCA colouring, Grad-CAM, model setup)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

SCRIPTS = Path(__file__).resolve().parents[1] / "projects/02_representation_learning/scripts"
sys.path.insert(0, str(SCRIPTS))
import replearn as rl  # noqa: E402


def _clusters(rng: np.random.Generator, centers: np.ndarray, n_per: int = 30):
    x = np.concatenate([c + rng.normal(size=(n_per, centers.shape[1])) for c in centers])
    y = np.repeat(np.arange(len(centers)), n_per)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y)


def test_knn_and_prototypes_separate_clusters(rng):
    centers = rng.normal(size=(4, 16)) * 5
    x, y = _clusters(rng, centers)
    xt, yt = _clusters(rng, centers)
    for temp in (0.07, None):
        pred, votes = rl.knn_predict(x, y, xt, k=5, temperature=temp, chunk=7)
        assert votes.shape == (len(xt), 4)
        assert (pred == yt).float().mean() > 0.95
    assert (rl.nearest_class_mean(x, y, xt) == yt).float().mean() > 0.95


def test_knn_majority_vote_counts():
    train = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    labels = torch.tensor([0, 0, 1])
    pred, votes = rl.knn_predict(train, labels, torch.tensor([[1.0, 0.05]]), k=3, temperature=None)
    assert pred.item() == 0
    assert votes.tolist() == [[2.0, 1.0]]


def test_otsu_threshold_bimodal(rng):
    v = np.concatenate([rng.normal(0, 0.3, 500), rng.normal(5, 0.3, 500)])
    t = rl.otsu_threshold(v)
    assert 1.0 < t < 4.0


def test_pca_rgb_finds_central_object(rng):
    b, h, w, d = 2, 12, 12, 8
    feats = rng.normal(0, 0.1, size=(b, h, w, d))
    feats[:, 3:9, 3:9, 0] += 5.0  # an "object" in the centre with a distinct feature
    feats[:, 3:9, 3:6, 1] += 2.0  # two "parts" inside the object
    rgb, fg = rl.pca_rgb(feats)
    assert rgb.shape == (b, h, w, 3) and fg.shape == (b, h, w)
    assert rgb.min() >= 0 and rgb.max() <= 1
    assert fg[:, 3:9, 3:9].all() and not fg[:, 0, :].any()
    assert np.all(rgb[~fg] == 0)


class _SpotNet(nn.Module):
    """Score of class c = mean activation of feature channel c (an identity 'conv')."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Conv2d(3, 3, 3, padding=1)
        nn.init.dirac_(self.features.weight)
        nn.init.zeros_(self.features.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = torch.relu(self.features(x))  # (B, 3, H, W) == x for x >= 0
        return a[:, :2].mean(dim=(2, 3))  # (B, 2) class scores


def test_gradcam_is_class_discriminative():
    # Evidence for class 0 sits top-left (channel 0), for class 1 bottom-right (channel 1).
    x = torch.zeros(1, 3, 16, 16)
    x[:, 0, 2:6, 2:6] = 1.0
    x[:, 1, 10:14, 10:14] = 1.0
    net = _SpotNet()
    cam = rl.GradCAM(net, net.features)
    heat0, logits = cam(x, class_idx=0)
    heat1, _ = cam(x, class_idx=1)
    cam.remove()
    assert heat0.shape == (1, 16, 16) and logits.shape == (1, 2)
    assert float(heat0.min()) >= 0 and float(heat0.max()) == pytest.approx(1.0)
    assert heat0[0, 2:6, 2:6].mean() > 0.5 and heat0[0, 10:14, 10:14].max() < 0.05
    assert heat1[0, 10:14, 10:14].mean() > 0.5 and heat1[0, 2:6, 2:6].max() < 0.05


def test_gradcam_works_with_frozen_weights():
    net = _SpotNet().requires_grad_(False)
    cam = rl.GradCAM(net, net.features)
    heat, _ = cam(torch.rand(1, 3, 16, 16))  # top-1 class
    cam.remove()
    assert torch.isfinite(heat).all() and heat.max() > 0


def test_deletion_insertion_endpoints():
    torch.manual_seed(0)
    net = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                        nn.Linear(4, 3)).eval()  # fmt: skip
    x = torch.rand(2, 3, 8, 8)
    sal = torch.rand(2, 8, 8)
    cls = torch.tensor([0, 2])
    dele, ins = rl.deletion_insertion(net, x, sal, cls, steps=4)
    p = net(x).softmax(1).gather(1, cls[:, None])[:, 0].detach().numpy()
    assert dele.shape == ins.shape == (2, 5)
    np.testing.assert_allclose(dele[:, 0], p, rtol=1e-5)  # nothing deleted yet
    np.testing.assert_allclose(ins[:, -1], p, rtol=1e-5)  # everything re-inserted
    np.testing.assert_allclose(dele[:, -1], ins[:, 0], rtol=1e-5)  # both = blurred image


def test_auc_and_schedule():
    assert rl.auc(np.full((3, 11), 0.5)).tolist() == pytest.approx([0.5] * 3)
    assert rl.auc(np.linspace(0, 1, 101)) == pytest.approx(0.5)
    f = rl.cosine_warmup(total_steps=100, warmup_steps=10)
    assert f(0) == pytest.approx(0.1) and f(9) == pytest.approx(1.0)
    assert f(10) == pytest.approx(1.0) and f(100) == pytest.approx(0.0, abs=1e-12)


def test_linear_probe_freezes_all_but_head():
    pytest.importorskip("timm")
    model = rl.build_model("resnet18", "linear_probe", num_classes=10, pretrained=False)
    total, trainable = rl.count_params(model)
    assert trainable == 512 * 10 + 10 < total
    opt = rl.make_optimizer(model, "linear_probe", 1e-3, 1e-4, 0.05)
    assert len(opt.param_groups) == 1
    ft = rl.build_model("resnet18", "finetune", num_classes=10, pretrained=False)
    assert rl.count_params(ft)[1] == rl.count_params(ft)[0]
    assert len(rl.make_optimizer(ft, "finetune", 1e-3, 1e-4, 0.05).param_groups) == 2
    with pytest.raises(ValueError):
        rl.build_model("resnet18", "lora", pretrained=False)


def test_transforms_and_denormalize_roundtrip(rng):
    from PIL import Image

    img = Image.fromarray(rng.integers(0, 255, size=(160, 200, 3), dtype=np.uint8))
    x = rl.build_transform(train=False, size=160)(img)
    assert x.shape == (3, 160, 160)
    back = rl.denormalize(x)
    assert back.shape == (160, 160, 3) and back.dtype == np.uint8
    xt = rl.build_transform(train=True, size=128)(img)
    assert xt.shape == (3, 128, 128)


def test_class_metadata():
    assert len(rl.CLASS_NAMES) == len(rl.IMAGENET1K_INDEX) == 10
    assert list(rl.IMAGENETTE_CLASSES) == sorted(rl.IMAGENETTE_CLASSES)
