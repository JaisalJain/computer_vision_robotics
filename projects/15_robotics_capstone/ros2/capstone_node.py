#!/usr/bin/env python3
"""ROS 2 Jazzy node: RGB-D frames in, 3D object boxes and a grasp pose out.

Subscribes (synchronised colour + aligned depth, cached intrinsics)::

    ~/rgb          sensor_msgs/Image       rgb8 or bgr8
    ~/depth        sensor_msgs/Image       16UC1 (millimetres, REP-118) or 32FC1 (metres)
    ~/camera_info  sensor_msgs/CameraInfo  K (and plumb_bob D) of the colour camera

Publishes (in the colour camera's optical frame, i.e. ``header.frame_id`` of the image)::

    ~/detections   vision_msgs/Detection3DArray   one oriented box per object
    ~/grasp        geometry_msgs/PoseStamped      best top-down grasp (z = approach, x = closing)

Run it with the **system Python of ROS 2 Jazzy** (never inside the ``cv`` conda env)::

    source /opt/ros/jazzy/setup.bash
    python3 -s projects/15_robotics_capstone/ros2/capstone_node.py --ros-args \
        -r "~/rgb:=/camera/color/image_raw" \
        -r "~/depth:=/camera/aligned_depth_to_color/image_raw" \
        -r "~/camera_info:=/camera/color/camera_info" -p backend:=tabletop

(quote the ``~/...`` remap rules, otherwise the shell expands ``~`` to the home directory).

Backends:

* ``tabletop`` (default, NumPy/SciPy only): class-agnostic table-top segmentation (RANSAC
  table plane + connected components of the pixels above it), then the capstone geometry.
* ``models``: open-vocabulary detection + SAM masks. The models need PyTorch, which lives in
  the ``cv`` env, so the node starts ``perception_worker.py`` with the env's interpreter
  (parameter ``cv_python``) in a *sanitised* environment (no ROS paths) and exchanges frames
  over a pipe. ROS never enters the ``cv`` env and torch never enters the ROS interpreter.

``cv_bridge`` is deliberately not used: the apt build is compiled against NumPy 1.x. Images
are decoded with ``numpy.frombuffer`` instead.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
# The shared geometry (grasp3d + cvlab) is pure NumPy, so the ROS interpreter can import it.
for p in (REPO / "src", HERE.parent / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import grasp3d as g3  # noqa: E402

# ------------------------------------------------------------------- message conversion


def image_to_numpy(msg) -> np.ndarray:
    """sensor_msgs/Image -> NumPy (RGB uint8 HxWx3, or depth float32 metres HxW)."""
    h, w, step = msg.height, msg.width, msg.step
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    if msg.encoding in ("rgb8", "bgr8"):
        img = buf.reshape(h, step)[:, : w * 3].reshape(h, w, 3)
        return img[..., ::-1].copy() if msg.encoding == "bgr8" else img.copy()
    if msg.encoding in ("16UC1", "mono16"):
        dt = np.dtype(">u2" if msg.is_bigendian else "<u2")
        d = np.frombuffer(bytes(msg.data), dtype=dt).reshape(h, step // 2)[:, :w]
        return d.astype(np.float32) / 1000.0  # REP-118: millimetres
    if msg.encoding == "32FC1":
        dt = np.dtype(">f4" if msg.is_bigendian else "<f4")
        d = np.frombuffer(bytes(msg.data), dtype=dt).reshape(h, step // 4)[:, :w]
        return np.nan_to_num(d.astype(np.float32), nan=0.0, posinf=0.0)
    raise ValueError(f"unsupported image encoding {msg.encoding!r}")


def pose_msg(position, quat_xyzw):
    from geometry_msgs.msg import Pose

    p = Pose()
    p.position.x, p.position.y, p.position.z = (float(v) for v in position)
    o = p.orientation
    o.x, o.y, o.z, o.w = (float(v) for v in quat_xyzw)
    return p


# ------------------------------------------------------------------------------ backends
class TabletopBackend:
    """Class-agnostic objects = connected blobs of pixels above the dominant (table) plane."""

    def __init__(self, min_height: float = 0.015, min_pixels: int = 400, max_objects: int = 8):
        self.min_height, self.min_pixels, self.max_objects = min_height, min_pixels, max_objects
        self._rays_key = None
        self._rays = None

    def rays(self, K, D, w, h):
        key = (tuple(np.round(K.ravel(), 6)), tuple(np.round(D, 6)), w, h)
        if key != self._rays_key:
            self._rays, self._rays_key = g3.pixel_rays(K, D if np.any(D) else None, w, h), key
        return self._rays

    def __call__(self, rgb, depth, K, D, queries=None) -> list[dict]:
        from scipy import ndimage

        h, w = depth.shape
        rays = self.rays(K, D, w, h)
        pts, pix = g3.backproject(depth, rays, stride=4)
        fit = g3.fit_plane_ransac(pts, threshold=0.012)
        if fit is None:
            return []
        plane = fit[0]
        all_pts, all_pix = g3.backproject(depth, rays)
        above = np.zeros((h, w), bool)
        hgt = plane.height(all_pts)
        sel = (hgt > self.min_height) & (hgt < 0.5)
        above[all_pix[sel, 0], all_pix[sel, 1]] = True
        above = ndimage.binary_opening(above, iterations=2)
        lab, n = ndimage.label(above)
        sizes = ndimage.sum(above, lab, index=np.arange(1, n + 1))
        order = np.argsort(-sizes)[: self.max_objects]
        objects = []
        for k in order:
            if sizes[k] < self.min_pixels:
                break
            mask = lab == (k + 1)
            cloud = g3.object_cloud(depth, mask, rays, erode_px=1)
            if cloud is None:
                continue
            box = g3.table_box(cloud.points, plane)
            scene = all_pts[~mask[all_pix[:, 0], all_pix[:, 1]]]
            scene = scene[np.linalg.norm(scene - box.center, axis=1) < 0.3]
            grasps = g3.top_down_grasps(cloud.points, box, plane, scene)
            best = grasps[0] if grasps and grasps[0].feasible else None
            objects.append(_object_dict("object", float(sizes[k]) / (h * w), box, best))
        return objects


def _object_dict(label, score, box, grasp) -> dict:
    return {
        "label": label,
        "score": float(score),
        "center": box.center.tolist(),
        "quat_xyzw": box.quat_xyzw.tolist(),
        "extents": box.extents.tolist(),
        "grasp": None
        if grasp is None
        else {
            "position": grasp.position.tolist(),
            "quat_xyzw": grasp.quat_xyzw.tolist(),
            "width": grasp.width,
            "opening": grasp.opening,
            "score": grasp.score,
        },
    }


class WorkerBackend:
    """Open-vocabulary models in a separate process running the ``cv`` env's Python."""

    def __init__(self, cv_python: str, detector: str = "gdino", segmenter: str = "sam3"):
        # Strip every ROS / user-site variable so the cv env stays isolated from /opt/ros.
        drop = (
            "PYTHONPATH",
            "LD_LIBRARY_PATH",
            "AMENT_PREFIX_PATH",
            "CMAKE_PREFIX_PATH",
            "COLCON_PREFIX_PATH",
            "ROS_DISTRO",
            "ROS_VERSION",
            "ROS_PYTHON_VERSION",
        )
        env = {k: v for k, v in os.environ.items() if k not in drop}
        env["PYTHONNOUSERSITE"] = "1"
        cmd = [
            cv_python,
            "-s",
            str(HERE / "perception_worker.py"),
            "--detector",
            detector,
            "--segmenter",
            segmenter,
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)

    def __call__(self, rgb, depth, K, D, queries) -> list[dict]:
        import io

        buf = io.BytesIO()
        np.savez(
            buf,
            rgb=rgb,
            depth=depth.astype(np.float32),
            K=K,
            D=np.asarray(D, float),
            queries=np.array(queries),
        )
        data = buf.getvalue()
        self.proc.stdin.write(struct.pack("<Q", len(data)) + data)
        self.proc.stdin.flush()
        n = struct.unpack("<Q", _read_exact(self.proc.stdout, 8))[0]
        return json.loads(_read_exact(self.proc.stdout, n).decode())["objects"]

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.stdin.close()
            self.proc.wait(timeout=30)


def _read_exact(stream, n: int) -> bytes:
    out = b""
    while len(out) < n:
        chunk = stream.read(n - len(out))
        if not chunk:
            raise EOFError("perception worker closed the pipe")
        out += chunk
    return out


# ---------------------------------------------------------------------------------- node
def make_node(name: str = "capstone_perception", **node_kwargs):
    """Build the node (imports rclpy lazily so the module imports without ROS).

    ``node_kwargs`` go to ``rclpy.node.Node`` (e.g. ``parameter_overrides``, ``cli_args``).
    """
    import message_filters
    from geometry_msgs.msg import PoseStamped
    from rclpy.node import Node
    from sensor_msgs.msg import CameraInfo, Image
    from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose

    class CapstoneNode(Node):
        def __init__(self) -> None:
            super().__init__(name, **node_kwargs)
            self.declare_parameter("backend", "tabletop")
            self.declare_parameter("queries", ["game controller", "telephone", "cereal box"])
            self.declare_parameter("cv_python", "")
            self.declare_parameter("detector", "gdino")
            self.declare_parameter("segmenter", "sam3")
            backend = self.get_parameter("backend").value
            if backend == "models":
                cv_python = self.get_parameter("cv_python").value
                if not cv_python:
                    raise ValueError(
                        "backend=models needs -p cv_python:=<path to the cv env python>"
                    )
                self.backend = WorkerBackend(
                    cv_python,
                    self.get_parameter("detector").value,
                    self.get_parameter("segmenter").value,
                )
            else:
                self.backend = TabletopBackend()
            self.K = self.D = None
            self.create_subscription(CameraInfo, "~/camera_info", self.on_info, 1)
            rgb_sub = message_filters.Subscriber(self, Image, "~/rgb")
            depth_sub = message_filters.Subscriber(self, Image, "~/depth")
            self.sync = message_filters.ApproximateTimeSynchronizer(
                [rgb_sub, depth_sub], queue_size=5, slop=0.03
            )
            self.sync.registerCallback(self.on_frame)
            self.pub_det = self.create_publisher(Detection3DArray, "~/detections", 10)
            self.pub_grasp = self.create_publisher(PoseStamped, "~/grasp", 10)
            self.n_frames = 0
            self.get_logger().info(f"capstone perception node ready (backend={backend})")

        def on_info(self, msg) -> None:
            self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self.D = np.array(msg.d, dtype=np.float64)[:5] if len(msg.d) else np.zeros(5)

        def on_frame(self, rgb_msg, depth_msg) -> None:
            if self.K is None:
                return
            rgb, depth = image_to_numpy(rgb_msg), image_to_numpy(depth_msg)
            queries = list(self.get_parameter("queries").value)
            objects = self.backend(rgb, depth, self.K, self.D, queries)
            arr = Detection3DArray()
            arr.header = rgb_msg.header
            best = None
            for i, o in enumerate(objects):
                det = Detection3D()
                det.header = rgb_msg.header
                det.id = str(i)
                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = o["label"]
                hyp.hypothesis.score = float(o["score"])
                hyp.pose.pose = pose_msg(o["center"], o["quat_xyzw"])
                det.results.append(hyp)
                det.bbox.center = pose_msg(o["center"], o["quat_xyzw"])
                det.bbox.size.x, det.bbox.size.y, det.bbox.size.z = (float(v) for v in o["extents"])
                arr.detections.append(det)
                g = o["grasp"]
                if g is not None and (best is None or g["score"] > best["score"]):
                    best = g
            self.pub_det.publish(arr)
            if best is not None:
                ps = PoseStamped()
                ps.header = rgb_msg.header
                ps.pose = pose_msg(best["position"], best["quat_xyzw"])
                self.pub_grasp.publish(ps)
            self.n_frames += 1

        def destroy_node(self) -> None:
            if isinstance(self.backend, WorkerBackend):
                self.backend.close()
            super().destroy_node()

    return CapstoneNode()


def main() -> None:
    import rclpy
    from rclpy.executors import ExternalShutdownException

    rclpy.init()
    node = make_node()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
