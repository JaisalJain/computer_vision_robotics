"""Feed-forward multi-view 3D reconstruction (VGGT, Depth Anything 3) evaluated against COLMAP.

The notebooks, ``run_colmap.py``, ``benchmark.py`` and the exercise solutions of project 10 all
import this module, so every number in the README comes from the same code path.

Conventions
-----------
* Every pose is a 4x4 **camera-from-world** matrix ``T_cw`` in the OpenCV camera frame
  (x right, y down, z forward). This is what COLMAP stores in ``images.txt``, and what VGGT
  and Depth Anything 3 (DA3) return as "extrinsics". Camera centre ``C = -R^T t``.
* Feed-forward models and COLMAP each reconstruct in their own gauge (arbitrary world frame
  and scale), so absolute comparisons need a Sim(3) alignment first. Pairwise relative-pose
  errors (the AUC metric) are gauge-free and need no alignment.

Usage::

    import sys; sys.path.insert(0, "projects/10_feedforward_3d/scripts")
    import ff3d
    scene = ff3d.load_scene("kitchen")
    rec = ff3d.predict_cached(scene, "vggt")
    print(ff3d.evaluate_poses(rec.T_cw, scene.ref_T_cw))
"""

from __future__ import annotations

import os
import sys
import time
import types
import warnings
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np

from cvlab.geometry import invert_T, umeyama_alignment
from cvlab.io import add_third_party, repo_root

# The CUDA caching allocator otherwise fragments badly when models with different activation
# shapes run one after another (DA3-Large then OOMs with ~1.8 GiB reserved but unused). Read at
# the first CUDA allocation, so import this module before running any model.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

PROJECT = "10_feedforward_3d"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "ff3d.yaml"
AUC_THRESHOLDS = (5, 10, 15, 30)

# Method key -> (model key in the config, DA3 ray-pose head?) for the feed-forward methods.
METHODS: dict[str, tuple[str, bool]] = {
    "vggt": ("vggt", False),
    "da3-base": ("da3-base", False),
    "da3-base-ray": ("da3-base", True),
    "da3-large": ("da3-large", False),
    "da3-large-ray": ("da3-large", True),
}
METHOD_LABELS = {
    "vggt": "VGGT-1B",
    "da3-base": "DA3-Base (cam head)",
    "da3-base-ray": "DA3-Base (ray head)",
    "da3-large": "DA3-Large-1.1 (cam head)",
    "da3-large-ray": "DA3-Large-1.1 (ray head)",
    "colmap": "COLMAP (SIFT + SfM)",
}


def rel(path: str | Path) -> str:
    """Repository-relative path string (never print absolute paths in notebooks)."""
    try:
        return str(Path(path).resolve().relative_to(repo_root()))
    except ValueError:
        return str(path)


def load_config(path: str | Path = CONFIG_PATH) -> dict:
    import yaml

    return yaml.safe_load(Path(path).read_text())


def runs_path(*parts: str) -> Path:
    """``runs/10_feedforward_3d/...`` (gitignored; caches and COLMAP workspaces)."""
    return repo_root() / load_config()["runs_dir"] / Path(*parts)


def results_path(*parts: str) -> Path:
    """``projects/10_feedforward_3d/results/...`` (small committed metric files)."""
    return repo_root() / "projects" / PROJECT / "results" / Path(*parts)


# ------------------------------------------------------------------------------- scenes
@dataclass
class Scene:
    """An ordered set of images plus reference cameras for evaluation."""

    name: str
    label: str
    image_paths: list[Path]
    ref_kind: str  # "colmap" | "dataset_colmap" | "mocap"
    ref_T_cw: np.ndarray | None = None  # Nx4x4, NaN-filled rows for images without reference
    ref_K: np.ndarray | None = None  # Nx3x3 intrinsics of the reference (its own resolution)
    ref_size: tuple[int, int] | None = None  # (width, height) the reference K refers to
    depth_paths: list[Path] | None = None  # metric depth PNGs (TUM only)
    depth_scale: float = 5000.0
    extra: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.image_paths)

    @property
    def names(self) -> list[str]:
        return [p.name for p in self.image_paths]

    @property
    def has_ref(self) -> np.ndarray:
        """Boolean mask of images with a reference pose."""
        if self.ref_T_cw is None:
            return np.zeros(len(self), bool)
        return np.isfinite(self.ref_T_cw).all(axis=(1, 2))

    def subset(self, idx: np.ndarray | list[int]) -> Scene:
        """A new scene with the images ``idx`` (in that order)."""
        idx = [int(i) for i in idx]
        pick = (lambda a: None if a is None else a[idx])  # noqa: E731
        return Scene(
            name=self.name, label=self.label,
            image_paths=[self.image_paths[i] for i in idx], ref_kind=self.ref_kind,
            ref_T_cw=pick(self.ref_T_cw), ref_K=pick(self.ref_K), ref_size=self.ref_size,
            depth_paths=None if self.depth_paths is None else [self.depth_paths[i] for i in idx],
            depth_scale=self.depth_scale, extra=dict(self.extra),
        )  # fmt: skip

    def read_depth(self, i: int) -> np.ndarray:
        """Metric depth (metres, 0 = missing) of image ``i`` (TUM only)."""
        import cv2

        raw = cv2.imread(str(self.depth_paths[i]), cv2.IMREAD_UNCHANGED)
        return raw.astype(np.float32) / self.depth_scale


def evenly_spaced(n_total: int, n: int) -> np.ndarray:
    """``n`` indices spread evenly over ``range(n_total)`` (first and last included)."""
    if n >= n_total:
        return np.arange(n_total)
    return np.unique(np.round(np.linspace(0, n_total - 1, n)).astype(int))


def load_scene(name: str, all_images: bool = False) -> Scene:
    """Load one scene of ``configs/ff3d.yaml`` with its reference cameras.

    Args:
        name: ``kitchen`` | ``south_building`` | ``tum_desk``.
        all_images: South Building only: return all 128 photos instead of the 24-view subset.
    """
    cfg = load_config()
    sc = cfg["scenes"][name]
    img_dir = repo_root() / sc["images"]
    paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not paths:
        raise FileNotFoundError(f"no images in {rel(img_dir)} - run `make data-{PROJECT}`")
    if "subset" in sc and not all_images:
        s = sc["subset"]
        paths = paths[s["start"] : s["stop"] : s["step"]]
    scene = Scene(name=name, label=sc["label"], image_paths=paths, ref_kind=sc["reference"])
    if sc["reference"] == "colmap":
        work = runs_path("colmap", name)
        if (work / "sparse").is_dir():
            rec = load_colmap(work)
            _attach_colmap_reference(scene, rec)
    elif sc["reference"] == "dataset_colmap":
        import pycolmap

        rec = pycolmap.Reconstruction(str(repo_root() / "data/south_building/reference_sparse"))
        _attach_colmap_reference(scene, rec)
    elif sc["reference"] == "mocap":
        import json

        from cvlab.tum import read_trajectory

        root = img_dir.parent
        _, T_wc = read_trajectory(root / "groundtruth.txt")
        meta = json.loads((root / "camera.json").read_text())
        scene.ref_T_cw = np.stack([invert_T(T) for T in T_wc])
        scene.ref_K = np.repeat(np.asarray(meta["K"])[None], len(paths), axis=0)
        scene.ref_size = (meta["width"], meta["height"])
        scene.depth_paths = [root / "depth" / p.name for p in paths]
        scene.depth_scale = meta["depth_scale"]
        scene.extra["timestamps"] = np.asarray(meta["timestamps"])
    return scene


def _attach_colmap_reference(scene: Scene, rec) -> None:
    poses = colmap_cameras(rec)
    n = len(scene)
    T = np.full((n, 4, 4), np.nan)
    K = np.full((n, 3, 3), np.nan)
    for i, name in enumerate(scene.names):
        if name in poses:
            T[i] = poses[name]["T_cw"]
            K[i] = poses[name]["K"]
    scene.ref_T_cw, scene.ref_K = T, K
    any_cam = next(iter(poses.values()), None)
    if any_cam is not None:
        scene.ref_size = (any_cam["width"], any_cam["height"])


# ------------------------------------------------------------------------------- COLMAP
def colmap_cameras(rec) -> dict[str, dict]:
    """``{image name: {"T_cw" 4x4, "K" 3x3, "width", "height", "params"}}`` of a pycolmap model.

    ``K`` is converted to the OpenCV pixel convention (pixel centres at integer coordinates):
    COLMAP puts the top-left pixel centre at (0.5, 0.5), so ``cx_cv = cx_colmap - 0.5``.
    """
    out = {}
    for img in rec.images.values():
        if not img.has_pose:
            continue
        T = np.eye(4)
        T[:3, :] = img.cam_from_world().matrix()
        cam = img.camera
        K = np.array(cam.calibration_matrix(), dtype=np.float64)
        K[0, 2] -= 0.5
        K[1, 2] -= 0.5
        out[img.name] = {"T_cw": T, "K": K, "width": int(cam.width), "height": int(cam.height),
                         "params": np.array(cam.params, dtype=np.float64)}  # fmt: skip
    return out


def run_colmap(
    image_dir: Path,
    image_names: list[str],
    work_dir: Path,
    camera_model: str = "SIMPLE_RADIAL",
    max_num_features: int = 8192,
    num_threads: int = -1,
) -> tuple[object | None, dict]:
    """COLMAP SIFT + exhaustive matching + incremental SfM on the CPU (pycolmap).

    Writes ``work_dir/database.db`` and ``work_dir/sparse/<k>/``. Returns the reconstruction
    with the most registered images and per-stage wall-clock timings in seconds.
    """
    import shutil

    import pycolmap

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    db = work_dir / "database.db"
    db.unlink(missing_ok=True)
    sparse = work_dir / "sparse"
    if sparse.exists():
        shutil.rmtree(sparse)
    sparse.mkdir(parents=True)
    timings = {}
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = camera_model
    extract = pycolmap.FeatureExtractionOptions()
    extract.num_threads = num_threads
    extract.sift.max_num_features = max_num_features
    t0 = time.perf_counter()
    pycolmap.extract_features(
        db, image_dir, image_names=image_names, camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader, extraction_options=extract, device=pycolmap.Device.cpu,
    )  # fmt: skip
    timings["features_s"] = time.perf_counter() - t0
    matching = pycolmap.FeatureMatchingOptions()
    matching.num_threads = num_threads
    t0 = time.perf_counter()
    pycolmap.match_exhaustive(db, matching_options=matching, device=pycolmap.Device.cpu)
    timings["matching_s"] = time.perf_counter() - t0
    opts = pycolmap.IncrementalPipelineOptions()
    opts.num_threads = num_threads
    opts.random_seed = 0
    opts.mapper.random_seed = 0
    t0 = time.perf_counter()
    recs = pycolmap.incremental_mapping(db, image_dir, sparse, options=opts)
    timings["mapping_s"] = time.perf_counter() - t0
    timings["total_s"] = sum(timings.values())
    if not recs:
        return None, timings
    return max(recs.values(), key=lambda r: r.num_reg_images()), timings


def load_colmap(work_dir: Path):
    """Largest reconstruction (most registered images) in ``work_dir/sparse/<k>/``."""
    import pycolmap

    models = [pycolmap.Reconstruction(str(d)) for d in sorted((Path(work_dir) / "sparse").iterdir())
              if d.is_dir() and any(d.iterdir())]  # fmt: skip
    if not models:
        raise FileNotFoundError(f"no COLMAP model in {rel(work_dir)}")
    return max(models, key=lambda r: r.num_reg_images())


def colmap_points(rec) -> tuple[np.ndarray, np.ndarray]:
    """``(xyz Nx3, rgb Nx3 uint8)`` of the sparse COLMAP points."""
    pts = list(rec.points3D.values())
    xyz = np.array([np.asarray(p.xyz, dtype=np.float64) for p in pts]).reshape(-1, 3)
    rgb = np.array([p.color for p in pts], dtype=np.uint8).reshape(-1, 3)
    return xyz, rgb


def colmap_as_recon(scene: Scene, rec, timings: dict | None = None) -> Recon:
    """Wrap a COLMAP model as a :class:`Recon` over ``scene`` (NaN poses = not registered)."""
    poses = colmap_cameras(rec)
    n = len(scene)
    T = np.full((n, 4, 4), np.nan)
    K = np.full((n, 3, 3), np.nan)
    for i, name in enumerate(scene.names):
        if name in poses:
            T[i], K[i] = poses[name]["T_cw"], poses[name]["K"]
    cam = next(iter(poses.values()))
    return Recon(method="colmap", T_cw=T, K=K, proc_hw=(cam["height"], cam["width"]),
                 runtime_s=(timings or {}).get("total_s", float("nan")),
                 extra={"timings": timings or {}})  # fmt: skip


# ------------------------------------------------------------------------------- models
@dataclass
class Recon:
    """Output of one reconstruction method for an ordered list of images."""

    method: str
    T_cw: np.ndarray  # Nx4x4 camera-from-world (NaN rows = not reconstructed)
    K: np.ndarray  # Nx3x3 intrinsics at the processed resolution
    proc_hw: tuple[int, int]  # (H, W) of the network input
    depth: np.ndarray | None = None  # NxHxW depth (z), same gauge/scale as T_cw
    conf: np.ndarray | None = None  # NxHxW depth confidence (higher = better)
    images: np.ndarray | None = None  # NxHxWx3 uint8 network input (for colouring points)
    world_points: np.ndarray | None = None  # NxHxWx3 VGGT point-map branch (world frame)
    world_conf: np.ndarray | None = None  # NxHxW confidence of the point-map branch
    runtime_s: float = float("nan")  # network forward (feed-forward) or SfM wall-clock
    total_s: float = float("nan")  # incl. image loading/pre-processing and copy to host
    peak_vram_mb: float = float("nan")  # peak allocated incl. the weights
    weights_vram_mb: float = float("nan")  # allocated by the weights alone
    extra: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.T_cw)

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {k: getattr(self, k) for k in ("T_cw", "K", "depth", "conf", "images",
                                                 "world_points", "world_conf")}  # fmt: skip
        arrays = {k: v for k, v in arrays.items() if v is not None}
        if "world_points" in arrays:
            arrays["world_points"] = arrays["world_points"].astype(np.float16)
        meta = {"method": self.method, "proc_hw": list(self.proc_hw), "runtime_s": self.runtime_s,
                "total_s": self.total_s, "peak_vram_mb": self.peak_vram_mb,
                "weights_vram_mb": self.weights_vram_mb, "extra": self.extra}  # fmt: skip
        import json

        np.savez(path, meta=np.array(json.dumps(meta, default=float)), **arrays)
        return path

    @classmethod
    def load(cls, path: Path) -> Recon:
        import json

        with np.load(path) as z:
            meta = json.loads(str(z["meta"]))
            arr = {k: z[k] for k in z.files if k != "meta"}
        if "world_points" in arr:
            arr["world_points"] = arr["world_points"].astype(np.float32)
        return cls(method=meta["method"], proc_hw=tuple(meta["proc_hw"]),
                   runtime_s=meta["runtime_s"], total_s=meta["total_s"],
                   peak_vram_mb=meta["peak_vram_mb"], weights_vram_mb=meta["weights_vram_mb"],
                   extra=meta.get("extra", {}), **arr)  # fmt: skip


def _sync() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _mb(x: float) -> float:
    return x / 2**20


class VGGTRunner:
    """VGGT-1B with a memory-aware precision recipe for an 8 GB GPU.

    The checkpoint is 5.0 GB in fp32 (1.26 B parameters). It is loaded on the CPU and then:

    * ``precision="mixed"`` (default): the 0.91 B-parameter aggregator (the transformer
      backbone) is cast to bf16; the camera, depth and point heads stay fp32 and receive the
      aggregator tokens upcast to fp32. This mirrors the official recipe (fp32 weights +
      bf16 autocast for the backbone, heads with autocast disabled) at about half the memory.
    * ``precision="bf16"``: every module in bf16 (smallest footprint, heads lose precision).

    The tracking head is dropped (66 M parameters) because it is not used here.
    """

    def __init__(self, precision: str = "mixed", device: str = "cuda",
                 point_head: bool = True) -> None:  # fmt: skip
        import torch

        add_third_party("vggt")
        from vggt.models.vggt import VGGT

        if precision not in ("mixed", "bf16"):
            raise ValueError(f"unknown precision {precision!r}")
        self.precision = precision
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        model = VGGT.from_pretrained(load_config()["models"]["vggt"]["repo"])  # fp32, on CPU
        model.track_head = None
        if not point_head:
            model.point_head = None
        model.aggregator.to(torch.bfloat16)
        if precision == "bf16":
            model.to(torch.bfloat16)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
        self.model = model.eval().to(self.device)
        self.weights_vram_mb = (
            _mb(torch.cuda.memory_allocated() - base) if self.device.type == "cuda" else 0.0
        )
        self.n_params = sum(p.numel() for p in self.model.parameters())

    def preprocess(self, image_paths: list[Path]):
        """VGGT 'crop' pre-processing: width 518 px, height rounded to a multiple of 14."""
        from vggt.utils.load_fn import load_and_preprocess_images

        return load_and_preprocess_images([str(p) for p in image_paths], mode="crop")

    def predict(self, image_paths: list[Path], method: str = "vggt") -> Recon:
        import torch
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        t_total = time.perf_counter()
        images = self.preprocess(image_paths).to(self.device)  # (N, 3, H, W) in [0, 1]
        head_dtype = next(self.model.camera_head.parameters()).dtype
        cuda = self.device.type == "cuda"
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        _sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            with torch.autocast(self.device.type, dtype=torch.bfloat16):
                tokens, ps_idx = self.model.aggregator(images[None])  # list of (1, N, P, 2C)
            for i, tok in enumerate(tokens):  # upcast one layer at a time (frees the bf16 copy)
                if tok is not None:
                    tokens[i] = tok.to(head_dtype)
            imgs_h = images[None].to(head_dtype)
            # All-bf16 heads add fp32 positional embeddings inside the DPT heads, so they need
            # autocast to keep conv inputs and weights in one dtype; fp32 heads run without it.
            bf16_heads = head_dtype == torch.bfloat16
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=bf16_heads):
                pose_enc = self.model.camera_head(tokens)[-1]  # (1, N, 9): t, quaternion, FoV
                depth, depth_conf = self.model.depth_head(tokens, images=imgs_h,
                                                          patch_start_idx=ps_idx)  # fmt: skip
                wp = wp_conf = None
                if self.model.point_head is not None:
                    wp, wp_conf = self.model.point_head(tokens, images=imgs_h,
                                                        patch_start_idx=ps_idx)  # fmt: skip
            extr, intr = pose_encoding_to_extri_intri(pose_enc.float(), images.shape[-2:])
        _sync()
        runtime = time.perf_counter() - t0
        peak = _mb(torch.cuda.max_memory_allocated()) if cuda else float("nan")
        del tokens
        n = images.shape[0]
        T = np.tile(np.eye(4), (n, 1, 1))
        T[:, :3, :] = extr[0].float().cpu().numpy()
        rgb = (images.permute(0, 2, 3, 1).float().cpu().numpy() * 255).round().astype(np.uint8)
        out = Recon(
            method=method, T_cw=T, K=intr[0].float().cpu().numpy().astype(np.float64),
            proc_hw=tuple(images.shape[-2:]),
            depth=depth[0, ..., 0].float().cpu().numpy(), conf=depth_conf[0].float().cpu().numpy(),
            images=rgb,
            world_points=None if wp is None else wp[0].float().cpu().numpy(),
            world_conf=None if wp_conf is None else wp_conf[0].float().cpu().numpy(),
            runtime_s=runtime, peak_vram_mb=peak, weights_vram_mb=self.weights_vram_mb,
            extra={"precision": self.precision},
        )  # fmt: skip
        out.total_s = time.perf_counter() - t_total
        return out


def stub_moviepy() -> bool:
    """Register an empty ``moviepy.editor`` module if moviepy is not importable.

    ``depth_anything_3.api`` imports ``moviepy.editor`` at import time, but only uses it to
    render Gaussian-splatting videos (not used here). moviepy 1.0.3 (the last release with
    ``moviepy.editor``) would downgrade packages of the shared environment, so it is not
    installed. Returns ``True`` when the stub was inserted.
    """
    try:
        import moviepy.editor  # noqa: F401

        return False
    except ImportError:
        pass
    moviepy = types.ModuleType("moviepy")
    editor = types.ModuleType("moviepy.editor")
    moviepy.editor = editor
    sys.modules.setdefault("moviepy", moviepy)
    sys.modules.setdefault("moviepy.editor", editor)
    return True


def import_da3():
    """``DepthAnything3`` class from the pinned clone (quiet logger, moviepy stub)."""
    os.environ.setdefault("DA3_LOG_LEVEL", "ERROR")  # DA3's logger prints to stdout otherwise
    warnings.filterwarnings("ignore", message=".*register_new_resolver.*")
    add_third_party("depth-anything-3", "src")
    stub_moviepy()
    from depth_anything_3.api import DepthAnything3

    return DepthAnything3


class DA3Runner:
    """Depth Anything 3 any-view model (multi-view depth + cameras).

    ``predict(paths, use_ray_pose=False)`` reads the camera from the dedicated camera head;
    ``use_ray_pose=True`` instead fits each camera to the predicted per-pixel ray map, which
    the DA3 authors report as more accurate but slower. Both come from the same forward pass
    weights, so one runner serves both methods.
    """

    def __init__(self, model_key: str = "da3-base", device: str = "cuda") -> None:
        import torch

        spec = load_config()["models"][model_key]
        self.model_key = model_key
        self.process_res = int(spec["process_res"])
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            base = torch.cuda.memory_allocated()
        self.model = import_da3().from_pretrained(spec["repo"]).to(self.device).eval()
        self.weights_vram_mb = (
            _mb(torch.cuda.memory_allocated() - base) if self.device.type == "cuda" else 0.0
        )
        self.n_params = sum(p.numel() for p in self.model.parameters())
        # Time the network alone (no CPU pre/post-processing) with hooks on the core module.
        self._t0, self.last_forward_s = 0.0, float("nan")
        self.model.model.register_forward_pre_hook(lambda *_: self._tic())
        self.model.model.register_forward_hook(lambda *_: self._toc())

    def _tic(self) -> None:
        _sync()
        self._t0 = time.perf_counter()

    def _toc(self) -> None:
        _sync()
        self.last_forward_s = time.perf_counter() - self._t0

    def predict(self, image_paths: list[Path], use_ray_pose: bool = False,
                method: str | None = None) -> Recon:  # fmt: skip
        import torch

        cuda = self.device.type == "cuda"
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        pred = self.model.inference(
            [str(p) for p in image_paths], process_res=self.process_res,
            use_ray_pose=use_ray_pose, ref_view_strategy="saddle_balanced",
        )  # fmt: skip
        total = time.perf_counter() - t0
        n = len(image_paths)
        T = np.tile(np.eye(4), (n, 1, 1))
        T[:, :3, :] = np.asarray(pred.extrinsics, dtype=np.float64)[:, :3, :]
        return Recon(
            method=method or (self.model_key + ("-ray" if use_ray_pose else "")),
            T_cw=T, K=np.asarray(pred.intrinsics, dtype=np.float64),
            proc_hw=tuple(pred.depth.shape[-2:]), depth=np.asarray(pred.depth, np.float32),
            conf=None if pred.conf is None else np.asarray(pred.conf, np.float32),
            images=np.asarray(pred.processed_images),
            runtime_s=self.last_forward_s, total_s=total,
            peak_vram_mb=_mb(torch.cuda.max_memory_allocated()) if cuda else float("nan"),
            weights_vram_mb=self.weights_vram_mb,
            extra={"process_res": self.process_res, "use_ray_pose": use_ray_pose},
        )  # fmt: skip


def make_runner(method: str, device: str = "cuda"):
    """Runner for a feed-forward method key of :data:`METHODS`."""
    model_key, _ = METHODS[method]
    return VGGTRunner(device=device) if model_key == "vggt" else DA3Runner(model_key, device)


def run_method(runner, method: str, image_paths: list[Path]) -> Recon:
    """Run ``runner`` (from :func:`make_runner`) as feed-forward method ``method``."""
    _, ray = METHODS[method]
    if isinstance(runner, VGGTRunner):
        return runner.predict(image_paths, method=method)
    return runner.predict(image_paths, use_ray_pose=ray, method=method)


def cache_path(scene: Scene, method: str, tag: str = "") -> Path:
    return runs_path("preds", scene.name, f"{method}{'_' + tag if tag else ''}.npz")


def predict_cached(scene: Scene, method: str, runner=None, tag: str = "",
                   refresh: bool = False) -> Recon:  # fmt: skip
    """Load a cached prediction (written by ``benchmark.py``) or run the model and cache it.

    ``method="colmap"`` returns the subset COLMAP model computed by ``run_colmap.py``.
    """
    if method == "colmap":
        import json

        work = runs_path("colmap", scene.name)
        summary = json.loads((work / "summary.json").read_text())
        return colmap_as_recon(scene, load_colmap(work), summary.get("timings_s"))
    path = cache_path(scene, method, tag)
    if path.exists() and not refresh:
        return Recon.load(path)
    runner = runner or make_runner(method)
    rec = run_method(runner, method, scene.image_paths)
    rec.save(path)
    return rec


# ------------------------------------------------------------------------------ metrics
def camera_centers(T_cw: np.ndarray) -> np.ndarray:
    """Nx3 camera centres ``C = -R^T t`` of camera-from-world poses."""
    T = np.asarray(T_cw, dtype=np.float64)
    return -np.einsum("nji,nj->ni", T[:, :3, :3], T[:, :3, 3])


def relative_pose_errors(T_pred: np.ndarray, T_ref: np.ndarray) -> dict[str, np.ndarray]:
    """Rotation and translation-direction errors (degrees) of every image pair ``i < j``.

    The relative pose maps camera i to camera j: ``T_ji = T_j @ inv(T_i)``. Its rotation is
    compared by geodesic angle; its translation only by **direction** (angle between the
    two vectors), because each reconstruction has its own unknown scale.
    """
    rot, trans, pairs = [], [], []
    for i, j in combinations(range(len(T_pred)), 2):
        Tp = T_pred[j] @ invert_T(T_pred[i])
        Tr = T_ref[j] @ invert_T(T_ref[i])
        cos = (np.trace(Tr[:3, :3].T @ Tp[:3, :3]) - 1.0) / 2.0
        rot.append(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
        a, b = Tp[:3, 3], Tr[:3, 3]
        cos_t = a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
        trans.append(np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0))))
        pairs.append((i, j))
    return {"rot_deg": np.array(rot), "trans_deg": np.array(trans), "pairs": np.array(pairs)}


def pose_auc(rot_deg: np.ndarray, trans_deg: np.ndarray, max_threshold: int) -> float:
    """Area under the pose-accuracy curve up to ``max_threshold`` degrees (VGGT protocol).

    Each pair's error is ``max(rotation error, translation-direction error)``. The accuracy
    curve ``acc(tau) = fraction of pairs with error < tau`` is integrated on 1-degree bins
    and normalised by ``max_threshold`` (same as ``calculate_auc_np`` of the VGGT evaluation
    code), so 1.0 means every pair is within 1 degree.
    """
    err = np.maximum(rot_deg, trans_deg)
    hist, _ = np.histogram(err, bins=np.arange(max_threshold + 1))
    return float(np.mean(np.cumsum(hist / len(err))))


def sim3_align(T_pred: np.ndarray, T_ref: np.ndarray) -> dict:
    """Umeyama Sim(3) fit of predicted to reference camera centres: ``C_ref ~ s R C_pred + t``.

    Returns the transform, the aligned camera-to-world poses and per-camera errors: the
    rotation error after applying ``R`` and the centre error in reference units.
    """
    C_p, C_r = camera_centers(T_pred), camera_centers(T_ref)
    s, R, t = umeyama_alignment(C_p, C_r, with_scale=True)
    T_wc_aligned = np.empty_like(np.asarray(T_pred, dtype=np.float64))
    rot_err = np.empty(len(T_pred))
    for k, T in enumerate(T_pred):
        T_wc = invert_T(T)
        T_wc_aligned[k] = np.eye(4)
        T_wc_aligned[k, :3, :3] = R @ T_wc[:3, :3]
        T_wc_aligned[k, :3, 3] = s * R @ T_wc[:3, 3] + t
        R_ref_wc = invert_T(T_ref[k])[:3, :3]
        cos = (np.trace(R_ref_wc.T @ T_wc_aligned[k, :3, :3]) - 1.0) / 2.0
        rot_err[k] = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    centre_err = np.linalg.norm(T_wc_aligned[:, :3, 3] - C_r, axis=1)
    return {"s": s, "R": R, "t": t, "T_wc_aligned": T_wc_aligned, "rot_err_deg": rot_err,
            "centre_err": centre_err}  # fmt: skip


def trajectory_extent(C: np.ndarray) -> float:
    """Diameter of a set of camera centres (largest pairwise distance)."""
    C = np.asarray(C, dtype=np.float64)
    d = np.linalg.norm(C[:, None] - C[None], axis=-1)
    return float(d.max())


def evaluate_poses(T_pred: np.ndarray, T_ref: np.ndarray, metric_units: bool = False) -> dict:
    """All pose metrics of one method on the images that both sides reconstructed.

    Returns ``n_eval`` (images evaluated), ``auc@k`` (k in :data:`AUC_THRESHOLDS`),
    ``rra@5``/``rta@5`` (fraction of pairs with rotation / translation-direction error < 5
    deg), median pairwise errors, the median absolute rotation error after Sim(3) alignment,
    and the ATE RMSE of camera centres as % of the reference trajectory diameter
    (plus in metres when ``metric_units``).
    """
    T_pred, T_ref = np.asarray(T_pred, float), np.asarray(T_ref, float)
    ok = np.isfinite(T_pred).all(axis=(1, 2)) & np.isfinite(T_ref).all(axis=(1, 2))
    Tp, Tr = T_pred[ok], T_ref[ok]
    out: dict = {"n_eval": int(ok.sum()), "n_total": len(T_pred)}
    if ok.sum() < 3:
        return out
    rel_err = relative_pose_errors(Tp, Tr)
    for k in AUC_THRESHOLDS:
        out[f"auc@{k}"] = pose_auc(rel_err["rot_deg"], rel_err["trans_deg"], k)
    out["rra@5"] = float(np.mean(rel_err["rot_deg"] < 5))
    out["rta@5"] = float(np.mean(rel_err["trans_deg"] < 5))
    out["rel_rot_med_deg"] = float(np.median(rel_err["rot_deg"]))
    out["rel_trans_med_deg"] = float(np.median(rel_err["trans_deg"]))
    al = sim3_align(Tp, Tr)
    ate = float(np.sqrt(np.mean(al["centre_err"] ** 2)))
    out["abs_rot_med_deg"] = float(np.median(al["rot_err_deg"]))
    out["ate_pct"] = 100.0 * ate / trajectory_extent(camera_centers(Tr))
    if metric_units:
        out["ate_m"] = ate
    out["sim3_scale"] = float(al["s"])
    return out


def fov_deg(K: np.ndarray, width: float) -> float:
    """Horizontal field of view in degrees of intrinsics ``K`` for an image ``width`` wide."""
    return float(np.degrees(2 * np.arctan(0.5 * width / K[0, 0])))


# --------------------------------------------------------------------------- point clouds
def recon_points(
    rec: Recon,
    conf_quantile: float = 0.3,
    stride: int = 2,
    use_point_map: bool = False,
    frames: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """World-frame point cloud of a feed-forward reconstruction.

    Points come from back-projecting each depth map with its predicted K and pose
    (``X_w = R^T (z K^-1 [u, v, 1]^T - t)``), or directly from the VGGT point-map branch with
    ``use_point_map``. Pixels below the ``conf_quantile`` of the confidence are dropped.

    Returns ``(points Nx3, colours Nx3 in [0, 1])``.
    """
    from cvlab.pointcloud import depth_to_points

    conf = rec.world_conf if use_point_map else rec.conf
    thr = np.quantile(conf, conf_quantile) if conf is not None else -np.inf
    pts, cols = [], []
    for i in frames if frames is not None else range(len(rec)):
        if not np.isfinite(rec.T_cw[i]).all():
            continue
        mask = conf[i] >= thr if conf is not None else None
        if use_point_map:
            m = np.ones(rec.world_points[i].shape[:2], bool) if mask is None else mask
            sub = np.zeros_like(m)
            sub[::stride, ::stride] = True
            P = rec.world_points[i][m & sub]
            c = rec.images[i][m & sub] / 255.0
        else:
            P_cam, c = depth_to_points(rec.depth[i], rec.K[i], rgb=rec.images[i], mask=mask,
                                       stride=stride)  # fmt: skip
            T_wc = invert_T(rec.T_cw[i])
            P = P_cam @ T_wc[:3, :3].T + T_wc[:3, 3]
        pts.append(P)
        cols.append(c)
    return np.concatenate(pts), np.concatenate(cols)


def sim3_apply(points: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """``s R p + t`` for Nx3 points."""
    return s * np.asarray(points) @ np.asarray(R).T + np.asarray(t)


def sim3_apply_poses(T_cw: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Re-express camera-from-world poses after the world is mapped by ``X' = s R X + t``.

    Camera coordinates are scaled by ``s`` as well (the whole reconstruction is rescaled):
    ``R' = R_cw R^T`` and ``t' = s t_cw - R_cw R^T t``.
    """
    T = np.asarray(T_cw, dtype=np.float64)
    out = np.tile(np.eye(4), (len(T), 1, 1))
    Rc = T[:, :3, :3] @ np.asarray(R).T
    out[:, :3, :3] = Rc
    out[:, :3, 3] = s * T[:, :3, 3] - Rc @ np.asarray(t)
    return out


def upright_rotation(T_cw: np.ndarray) -> np.ndarray:
    """Rotation mapping the average camera 'up' direction to +z (for intuitive 3D plots).

    Hand-held and robot cameras are mostly upright, so the mean of the cameras' up vectors
    (``-y`` of the OpenCV camera frame, expressed in the world) approximates the true up.
    """
    from cvlab.geometry import so3_exp

    ok = np.isfinite(T_cw).all(axis=(1, 2))
    up = -np.mean([invert_T(T)[:3, 1] for T in T_cw[ok]], axis=0)
    up /= np.linalg.norm(up)
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(up, z)
    angle = np.arctan2(np.linalg.norm(axis), up @ z)
    if np.linalg.norm(axis) < 1e-9:
        return np.eye(3) if up @ z > 0 else np.diag([1.0, -1.0, -1.0])
    return so3_exp(axis / np.linalg.norm(axis) * angle)


def crop_to_core(points: np.ndarray, colors: np.ndarray, quantile: float = 0.95):
    """Drop the farthest ``1 - quantile`` of points from the median (background/outliers)."""
    d = np.linalg.norm(points - np.median(points, axis=0), axis=1)
    keep = d <= np.quantile(d, quantile)
    return points[keep], colors[keep]


def chamfer(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Accuracy (a -> b) and completeness (b -> a) nearest-neighbour distances.

    With ``a`` = reconstruction and ``b`` = ground truth, *accuracy* is how far reconstructed
    points are from the true surface and *completeness* how much of the true surface is
    covered. Returns medians and means of both plus their mean (Chamfer distance).
    """
    from scipy.spatial import cKDTree

    d_ab, _ = cKDTree(b).query(a, workers=-1)
    d_ba, _ = cKDTree(a).query(b, workers=-1)
    return {"acc_median": float(np.median(d_ab)), "comp_median": float(np.median(d_ba)),
            "acc_mean": float(d_ab.mean()), "comp_mean": float(d_ba.mean()),
            "chamfer_mean": float(0.5 * (d_ab.mean() + d_ba.mean()))}  # fmt: skip


def resize_depth_nearest(depth: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize of a depth map to ``(H, W)`` (never blends across edges)."""
    import cv2

    return cv2.resize(depth, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)


# ---------------------------------------------------------------------------- plotting
def plot_cameras_topdown(
    ax, centres: dict[str, np.ndarray], dims: tuple[int, int] = (0, 2),
    units: str = "", title: str | None = None, connect: bool = True,
) -> None:  # fmt: skip
    """2D scatter (+ path if ``connect``) of several sets of (already aligned) camera centres.

    Use ``connect=False`` for unordered photo collections, where a path through the images
    in file order means nothing; the first set is then linked to each other set by thin
    grey residual segments (one per camera).
    """
    from cvlab.viz import color

    names = "xyz"
    first = None
    for k, (label, C) in enumerate(centres.items()):
        C = np.asarray(C)
        style = {"marker": "o", "ms": 3.5, "lw": 1.0, "color": color(k, unit=True)}
        if k == 0:
            style.update(marker="s", ms=6, lw=2.0, alpha=0.6)
            first = C
        if not connect:
            style["ls"] = "none"
            if k > 0 and first is not None and len(first) == len(C):
                for a, b in zip(first, C):
                    ax.plot([a[dims[0]], b[dims[0]]], [a[dims[1]], b[dims[1]]], color="0.5", lw=0.8)
        ax.plot(C[:, dims[0]], C[:, dims[1]], label=label, **style)
    unit = f" [{units}]" if units else ""
    ax.set_xlabel(f"{names[dims[0]]}{unit}")
    ax.set_ylabel(f"{names[dims[1]]}{unit}")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best")
    if title:
        ax.set_title(title)


def plot_scene(
    ax, points: np.ndarray, colors: np.ndarray, T_cw: np.ndarray, K: np.ndarray,
    hw: tuple[int, int], frustum_scale: float, max_points: int = 25000,
    elev: float = -60.0, azim: float = -90.0, title: str | None = None,
    cam_color: tuple[float, float, float] = (0.84, 0.37, 0.0), units: str = "scene units",
) -> None:  # fmt: skip
    """Coloured point cloud + camera frustums on a 3D matplotlib axis (static render)."""
    from cvlab.viz import plot_camera_frustums, plot_point_cloud

    plot_point_cloud(points, colors, ax=ax, max_points=max_points, size=0.6, elev=elev,
                     azim=azim, title=title)  # fmt: skip
    ok = np.isfinite(T_cw).all(axis=(1, 2))
    T_wc = np.stack([invert_T(T) for T in T_cw[ok]])
    plot_camera_frustums(ax, T_wc, K, hw[1], hw[0], scale=frustum_scale, color_rgb=cam_color)
    ax.set_xlabel(f"x [{units}]")
    ax.set_ylabel(f"y [{units}]")
    ax.set_zlabel(f"z [{units}]")
