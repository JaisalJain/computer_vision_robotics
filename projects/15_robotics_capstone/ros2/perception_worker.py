#!/usr/bin/env python
"""Model side of the ROS 2 node: runs in the ``cv`` conda env, talks over stdin/stdout.

Protocol (little-endian): request = ``uint64 length`` + ``.npz`` bytes with ``rgb`` (HxWx3
uint8), ``depth`` (HxW float32 metres), ``K`` (3x3), ``D`` (distortion, may be zeros) and
``queries`` (array of str); response = ``uint64 length`` + UTF-8 JSON
``{"objects": [...], "timings": {...}}``. Logs go to stderr so stdout stays a clean channel.

Started by ``capstone_node.py`` (backend ``models``); it can also be smoke-tested alone::

    python projects/15_robotics_capstone/ros2/perception_worker.py --selftest
"""

from __future__ import annotations

import argparse
import io
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))

import capstone as cap  # noqa: E402
import grasp3d as g3  # noqa: E402


def log(msg: str) -> None:
    print(f"[perception_worker] {msg}", file=sys.stderr, flush=True)


class Worker:
    def __init__(self, detector: str, segmenter: str, device: str = "cuda") -> None:
        det_cls = cap.GroundingDinoDetector if detector == "gdino" else cap.Owlv2Detector
        self.detector = det_cls(device)
        self.segmenter = cap.BoxSegmenter(segmenter, device)
        self._rays: dict = {}

    def rays(self, K: np.ndarray, D: np.ndarray, w: int, h: int) -> np.ndarray:
        key = (K.tobytes(), D.tobytes(), w, h)
        if key not in self._rays:
            self._rays[key] = g3.pixel_rays(K, D if np.any(D) else None, w, h)
        return self._rays[key]

    def __call__(self, rgb, depth, K, D, queries) -> dict:
        t0 = time.perf_counter()
        dets = self.detector.detect(rgb, list(queries))
        self.segmenter.set_image(rgb)
        masks, _ = self.segmenter.segment(dets.boxes)
        cap._sync()
        t1 = time.perf_counter()
        results = cap.estimate_objects(rgb, depth, dets, masks, self.rays(K, D, *depth.shape[::-1]))
        objects = []
        for r in results:
            if r.box is None:
                continue
            g = r.best_grasp
            objects.append(
                {
                    "label": r.label,
                    "score": r.score,
                    "center": r.box.center.tolist(),
                    "quat_xyzw": r.box.quat_xyzw.tolist(),
                    "extents": r.box.extents.tolist(),
                    "grasp": None
                    if g is None
                    else {
                        "position": g.position.tolist(),
                        "quat_xyzw": g.quat_xyzw.tolist(),
                        "width": g.width,
                        "opening": g.opening,
                        "score": g.score,
                    },
                }
            )
        t2 = time.perf_counter()
        return {
            "objects": objects,
            "timings": {"models_ms": (t1 - t0) * 1e3, "geometry_ms": (t2 - t1) * 1e3},
        }


def _read_exact(stream, n: int) -> bytes | None:
    out = b""
    while len(out) < n:
        chunk = stream.read(n - len(out))
        if not chunk:
            return None
        out += chunk
    return out


def serve(worker: Worker) -> None:
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    n_done = 0
    while True:
        head = _read_exact(stdin, 8)
        if head is None:
            break
        req = np.load(io.BytesIO(_read_exact(stdin, struct.unpack("<Q", head)[0])))
        res = worker(req["rgb"], req["depth"], req["K"], req["D"], [str(q) for q in req["queries"]])
        data = json.dumps(res).encode()
        stdout.write(struct.pack("<Q", len(data)) + data)
        stdout.flush()
        n_done += 1
    log(f"input closed after {n_done} frames")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--detector", default="gdino", choices=["gdino", "owlv2"])
    ap.add_argument("--segmenter", default="sam3", choices=["sam3", "sam2.1-s"])
    ap.add_argument("--selftest", action="store_true", help="run one TUM frame and print JSON")
    args = ap.parse_args()
    worker = Worker(args.detector, args.segmenter)
    log("models loaded")
    if args.selftest:
        frames = cap.load_frames()
        K, D = cap.CAMERAS[cap.DEFAULT_CAMERA]
        res = worker(frames.rgb(540), frames.depth(540), K, D, cap.QUERIES)
        print(
            json.dumps(
                {"n_objects": len(res["objects"]), "labels": [o["label"] for o in res["objects"]]}
            )
        )
        return
    serve(worker)


if __name__ == "__main__":
    main()
