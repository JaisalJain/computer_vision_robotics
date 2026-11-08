#!/usr/bin/env python3
"""End-to-end ROS 2 smoke test: replay TUM frames -> capstone node -> collect outputs.

All three nodes share one process and one executor, so the test needs no launch file and no
network setup. Frames are sent in lock-step (the next frame is published once the node has
answered the previous one). Run with the ROS 2 Jazzy system Python::

    source /opt/ros/jazzy/setup.bash
    python3 -s projects/15_robotics_capstone/ros2/smoke_test.py --backend tabletop
    # open-vocabulary models (GPU) via the cv env's interpreter:
    python3 -s projects/15_robotics_capstone/ros2/smoke_test.py --backend models \
        --cv-python "$(conda run -n cv which python)"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import capstone_node  # noqa: E402
import replay_tum  # noqa: E402


def main() -> int:
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from vision_msgs.msg import Detection3DArray

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backend", default="tabletop", choices=["tabletop", "models"])
    ap.add_argument("--cv-python", default="")
    ap.add_argument("--frames", type=int, nargs="+", default=[100, 300, 540])
    ap.add_argument("--timeout", type=float, default=180.0, help="seconds per frame")
    args = ap.parse_args()

    import numpy

    print(f"python {sys.version.split()[0]} | numpy {numpy.__version__} | backend {args.backend}")
    rclpy.init()
    replay = replay_tum.make_replay_node(args.frames, rate=None)
    overrides = [
        Parameter("backend", Parameter.Type.STRING, args.backend),
        Parameter("cv_python", Parameter.Type.STRING, args.cv_python),
    ]
    remaps = [
        "--ros-args",
        "-r",
        "~/rgb:=/tum/rgb",
        "-r",
        "~/depth:=/tum/depth",
        "-r",
        "~/camera_info:=/tum/camera_info",
    ]
    node = capstone_node.make_node(parameter_overrides=overrides, cli_args=remaps)

    got: dict[str, list] = {"det": [], "grasp": []}
    sink = Node("smoke_sink")
    sink.create_subscription(
        Detection3DArray, "/capstone_perception/detections", lambda m: got["det"].append(m), 10
    )
    sink.create_subscription(
        PoseStamped, "/capstone_perception/grasp", lambda m: got["grasp"].append(m), 10
    )
    ex = SingleThreadedExecutor()
    for n in (replay, node, sink):
        ex.add_node(n)

    t_end = time.time() + 10  # wait for discovery of the in-process endpoints
    while time.time() < t_end and (
        replay.pub_depth.get_subscription_count() == 0
        or sink.count_publishers("/capstone_perception/detections") == 0
    ):
        ex.spin_once(timeout_sec=0.05)

    ok = True
    for i in args.frames:
        n_before = len(got["det"])
        t0 = time.time()
        replay.publish(i)
        while len(got["det"]) == n_before and time.time() - t0 < args.timeout:
            ex.spin_once(timeout_sec=0.05)
        dt = time.time() - t0
        if len(got["det"]) == n_before:
            print(f"frame {i}: no Detection3DArray within {args.timeout:.0f} s")
            ok = False
            continue
        msg = got["det"][-1]
        labels = [d.results[0].hypothesis.class_id for d in msg.detections]
        sizes = [
            f"{d.bbox.size.x * 100:.0f}x{d.bbox.size.y * 100:.0f}x{d.bbox.size.z * 100:.0f}"
            for d in msg.detections
        ]
        print(
            f"frame {i}: {len(msg.detections)} boxes in {msg.header.frame_id} after "
            f"{dt:.2f} s: " + ", ".join(f"{a} {b} cm" for a, b in zip(labels, sizes))
        )
    for g in got["grasp"]:
        p = g.pose.position
        print(
            f"grasp @ stamp {g.header.stamp.sec}.{g.header.stamp.nanosec:09d}: "
            f"position [{p.x:.3f}, {p.y:.3f}, {p.z:.3f}] m in {g.header.frame_id}"
        )
    print(
        f"received {len(got['det'])} Detection3DArray and {len(got['grasp'])} PoseStamped "
        f"for {len(args.frames)} frames"
    )
    for n in (replay, node, sink):
        n.destroy_node()
    rclpy.shutdown()
    return 0 if ok and np.all([len(got["det"]) == len(args.frames)]) else 1


if __name__ == "__main__":
    sys.exit(main())
