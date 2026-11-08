#!/usr/bin/env python
"""Solutions to the exercises of notebook 03 (ONNX export and runtimes).

Run from the repository root under the GPU lock::

    flock /tmp/cv-gpu.lock python projects/03_object_detection/solutions/sol_03_onnx.py

Exported graphs are written to ``outputs/03_object_detection/solutions/`` (gitignored).
"""

from __future__ import annotations

import logging
import os
import shutil

os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402,F401  (load the CUDA libraries before onnxruntime)
from ultralytics import YOLO  # noqa: E402
from ultralytics.utils import LOGGER  # noqa: E402

# isort: split
import onnxruntime as ort  # noqa: E402

from cvlab import io as cvio  # noqa: E402
from cvlab.bench import benchmark  # noqa: E402
from cvlab.metrics import box_iou  # noqa: E402

LOGGER.setLevel(logging.ERROR)
PROJ = cvio.project_dir("03_object_detection")
ROOT = cvio.repo_root()
CKPT = ROOT / cvio.load_json(PROJ / "results" / "yolo26n_wildlife_summary.json")["best_checkpoint"]
OUT = cvio.outputs_dir("03_object_detection", "solutions", create=True)
PROVIDERS = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]


def export(name: str, **kwargs) -> str:
    f = YOLO(CKPT).export(format="onnx", simplify=True, device=0, **kwargs)
    dst = OUT / f"{name}.onnx"
    shutil.move(f, dst)
    return str(dst)


def cuda_session(path: str) -> ort.InferenceSession:
    sess = ort.InferenceSession(path, providers=PROVIDERS)
    assert sess.get_providers()[0] == "CUDAExecutionProvider"
    return sess


def main() -> None:
    rng = np.random.default_rng(0)

    # Exercise 1: dynamic batch, throughput vs batch size.
    sess = cuda_session(export("ex1_dynamic", imgsz=640, nms=False, dynamic=True))
    name = sess.get_inputs()[0].name
    print("Exercise 1 - dynamic batch input shape:", sess.get_inputs()[0].shape)
    rows = []
    for b in (1, 2, 4, 8):
        x = rng.random((b, 3, 640, 640), dtype=np.float32)
        r = benchmark(lambda x=x: sess.run(None, {name: x}), warmup=5, iters=30)
        rows.append({"batch": b, "latency [ms]": r["p50_ms"], "images/s": 1000 * b / r["p50_ms"]})
    print(pd.DataFrame(rows).round(1).to_string(index=False))

    # Exercise 2: NMS embedded in the graph (one-to-many head) vs the one-to-one graph.
    e2e = cuda_session(export("ex2_one_to_one", imgsz=640, nms=False))
    emb = cuda_session(export("ex2_embedded_nms", imgsz=640, nms=True))
    print(
        "\nExercise 2 - output shapes: one-to-one",
        e2e.get_outputs()[0].shape,
        "| embedded NMS",
        emb.get_outputs()[0].shape,
    )
    from PIL import Image

    img = Image.open(
        sorted(cvio.data_dir("ultralytics", "african-wildlife", "images", "val").glob("*.jpg"))[10]
    )
    x = (
        np.asarray(img.convert("RGB").resize((640, 640)), np.float32).transpose(2, 0, 1)[None]
        / 255.0
    )
    a = e2e.run(None, {"images": x})[0][0]
    b = emb.run(None, {"images": x})[0][0]
    a, b = a[a[:, 4] > 0.25], b[b[:, 4] > 0.25]
    ious = box_iou(a[:, :4], b[:, :4])
    print(
        f"  detections > 0.25: one-to-one {len(a)}, embedded NMS {len(b)}; "
        f"best-match IoUs {np.round(ious.max(1), 3).tolist() if ious.size else []}"
    )
    for tag, s in [("one-to-one", e2e), ("embedded NMS", emb)]:
        r = benchmark(lambda s=s: s.run(None, {"images": x}), warmup=10, iters=100)
        print(f"  {tag:13s} ORT CUDA p50 {r['p50_ms']:.2f} ms")

    # Exercise 3: input resolution trade-off (latency + val mAP).
    rows = []
    for size in (320, 480, 640):
        path = export(f"ex3_{size}", imgsz=size, nms=False)
        s = cuda_session(path)
        x = rng.random((1, 3, size, size), dtype=np.float32)
        lat = benchmark(lambda s=s, x=x: s.run(None, {"images": x}), warmup=10, iters=100)["p50_ms"]
        m = YOLO(path, task="detect").val(
            data="african-wildlife.yaml",
            imgsz=size,
            batch=1,
            plots=False,
            project=str(cvio.runs_dir("03_object_detection")),
            name="sol03_val",
            exist_ok=True,
            verbose=False,
        )
        rows.append(
            {"imgsz": size, "ORT CUDA p50 [ms]": lat, "mAP50": m.box.map50, "mAP50-95": m.box.map}
        )
    print("\nExercise 3 - resolution trade-off (fine-tuned YOLO26n, one-to-one ONNX)")
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
