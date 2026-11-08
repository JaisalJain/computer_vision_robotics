#!/usr/bin/env python3
"""Publish TUM RGB-D frames as ROS 2 messages (stand-in for a live RGB-D camera driver).

Topics: ``/tum/rgb`` (sensor_msgs/Image rgb8), ``/tum/depth`` (16UC1, millimetres per
REP-118; TUM stores 5000 counts per metre), ``/tum/camera_info`` (fr1 intrinsics + plumb_bob
distortion). Stamps are the original TUM timestamps, so colour and depth differ by up to
20 ms exactly like the recording. Pure NumPy + Pillow: runs on the ROS 2 system Python.

    source /opt/ros/jazzy/setup.bash
    python3 -s projects/15_robotics_capstone/ros2/replay_tum.py --rate 2 --stride 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))

from cvlab.tum import TUM_INTRINSICS, associate, read_file_list  # noqa: E402

SEQ = REPO / "data/tum/fr1_desk/rgbd_dataset_freiburg1_desk"
FRAME_ID = "tum_rgb_optical_frame"


def load_pairs(root: Path = SEQ) -> list[tuple[float, Path, float, Path]]:
    """Associated (t_rgb, rgb_png, t_depth, depth_png) tuples (TUM associate.py rule)."""
    rgb, depth = read_file_list(root / "rgb.txt"), read_file_list(root / "depth.txt")
    return [(ta, root / rgb[ta][0], tb, root / depth[tb][0]) for ta, tb in associate(rgb, depth)]


def stamp(t: float):
    from builtin_interfaces.msg import Time

    sec = int(t)
    return Time(sec=sec, nanosec=int(round((t - sec) * 1e9)))


def rgb_msg(path: Path, t: float):
    from PIL import Image as PILImage
    from sensor_msgs.msg import Image

    arr = np.asarray(PILImage.open(path).convert("RGB"), dtype=np.uint8)
    msg = Image(
        height=arr.shape[0],
        width=arr.shape[1],
        encoding="rgb8",
        is_bigendian=0,
        step=arr.shape[1] * 3,
        data=arr.tobytes(),
    )
    msg.header.stamp, msg.header.frame_id = stamp(t), FRAME_ID
    return msg


def depth_msg(path: Path, t: float):
    from PIL import Image as PILImage
    from sensor_msgs.msg import Image

    raw = np.asarray(PILImage.open(path), dtype=np.uint16)  # 5000 counts per metre
    mm = np.round(raw.astype(np.float64) / 5.0).astype("<u2")
    msg = Image(
        height=mm.shape[0],
        width=mm.shape[1],
        encoding="16UC1",
        is_bigendian=0,
        step=mm.shape[1] * 2,
        data=mm.tobytes(),
    )
    msg.header.stamp, msg.header.frame_id = stamp(t), FRAME_ID
    return msg


def info_msg(t: float, width: int = 640, height: int = 480):
    from sensor_msgs.msg import CameraInfo

    (fx, fy, cx, cy), dist = TUM_INTRINSICS["fr1"]
    msg = CameraInfo(width=width, height=height, distortion_model="plumb_bob")
    msg.d = [float(x) for x in dist]
    msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    msg.header.stamp, msg.header.frame_id = stamp(t), FRAME_ID
    return msg


def make_replay_node(indices: list[int], rate: float | None):
    from rclpy.node import Node
    from sensor_msgs.msg import CameraInfo, Image

    class Replay(Node):
        def __init__(self) -> None:
            super().__init__("tum_replay")
            self.pairs = load_pairs()
            self.indices = list(indices)
            self.pub_rgb = self.create_publisher(Image, "/tum/rgb", 10)
            self.pub_depth = self.create_publisher(Image, "/tum/depth", 10)
            self.pub_info = self.create_publisher(CameraInfo, "/tum/camera_info", 10)
            self.sent = 0
            if rate:
                self.create_timer(1.0 / rate, self.publish_next)

        def publish(self, i: int) -> None:
            t_rgb, p_rgb, t_d, p_d = self.pairs[i]
            self.pub_info.publish(info_msg(t_rgb))
            self.pub_rgb.publish(rgb_msg(p_rgb, t_rgb))
            self.pub_depth.publish(depth_msg(p_d, t_d))

        def publish_next(self) -> None:
            if self.sent < len(self.indices):
                self.publish(self.indices[self.sent])
                self.sent += 1

    return Replay()


def main() -> None:
    import rclpy
    from rclpy.executors import ExternalShutdownException

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rate", type=float, default=2.0, help="frames per second")
    ap.add_argument("--stride", type=int, default=10)
    args = ap.parse_args()
    rclpy.init()
    node = make_replay_node(list(range(0, len(load_pairs()), args.stride)), args.rate)
    try:
        while rclpy.ok() and node.sent < len(node.indices):
            rclpy.spin_once(node, timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
