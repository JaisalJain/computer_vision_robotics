# Optional: the capstone as a ROS 2 Jazzy node

This folder wraps the capstone pipeline in a ROS 2 node so it can sit behind a real RGB-D
camera driver. It is optional and is **not** part of `make verify` or CI.

```
/camera rgb + aligned depth + camera_info ──► capstone_perception ──► ~/detections  vision_msgs/Detection3DArray
                                                                 └──► ~/grasp       geometry_msgs/PoseStamped
```

| file | runs under | what it does |
|---|---|---|
| `capstone_node.py` | ROS 2 system Python | subscribes to `~/rgb`, `~/depth` (16UC1 mm or 32FC1 m) and `~/camera_info`, publishes one oriented 3D box per object and the best top-down grasp, both in the colour optical frame |
| `perception_worker.py` | `cv` conda env | the GPU models (Grounding DINO-T + SAM 3 by default) behind a stdin/stdout pipe |
| `replay_tum.py` | ROS 2 system Python | publishes TUM fr1/desk frames as `/tum/rgb`, `/tum/depth`, `/tum/camera_info` (stand-in for a camera driver) |
| `smoke_test.py` | ROS 2 system Python | replay + node + output collector in one process, frames sent in lock-step |

## Design: two interpreters, no `cv_bridge`

- ROS 2 Jazzy uses the system Python 3.12 with NumPy 1.26; the `cv` env has NumPy 2.5 and
  PyTorch. Mixing them in one interpreter is fragile, and this repository never adds
  `/opt/ros` to the `cv` env's `sys.path`.
- The geometry (`../scripts/grasp3d.py` and the NumPy parts of `cvlab`) works on both NumPy
  versions, so the node imports it directly.
- Two backends:
  - `tabletop` (default, CPU only) does class-agnostic table-top segmentation: a RANSAC desk
    plane, then connected components of the pixels above it. The capstone geometry follows.
  - `models` starts `perception_worker.py` with the `cv` env's interpreter, in an environment
    with the ROS paths removed, and exchanges `.npz` frames and JSON results over a pipe.
    Torch never enters the ROS process.
- `cv_bridge` from apt is built against NumPy 1.x, and the node does not need it: images are
  decoded with `numpy.frombuffer`.

## What was run (Ubuntu 24.04, ROS 2 Jazzy from apt, RTX 5060 Laptop GPU)

Start from a shell where the `cv` env is **not** active, and use the system interpreter
explicitly (`/usr/bin/python3 -s`). If the conda base env is active, `python3` would resolve to
conda's Python, which has no `rclpy`.

```bash
source /opt/ros/jazzy/setup.bash
# 1. CPU backend: replay frames 100, 300, 540 -> node -> collected outputs
/usr/bin/python3 -s projects/15_robotics_capstone/ros2/smoke_test.py --backend tabletop
# 2. GPU backend: the node starts the model worker with the cv env's interpreter
flock /tmp/cv-gpu.lock /usr/bin/python3 -s projects/15_robotics_capstone/ros2/smoke_test.py \
    --backend models --cv-python "$(conda run -n cv python -c 'import sys; print(sys.executable)')"
```

Output of run 1:

```
python 3.12.3 | numpy 1.26.4 | backend tabletop
frame 100: 8 boxes in tum_rgb_optical_frame after 0.52 s: object 40x34x12 cm, object 25x17x4 cm, ...
frame 300: 8 boxes in tum_rgb_optical_frame after 0.31 s: object 54x23x7 cm, object 74x44x26 cm, ...
frame 540: 6 boxes in tum_rgb_optical_frame after 0.28 s: object 38x18x12 cm, object 24x17x4 cm, ...
received 3 Detection3DArray and 2 PoseStamped for 3 frames
```

Output of run 2, with the node's default queries `game controller`, `telephone` and
`cereal box`. The first frame includes model loading:

```
python 3.12.3 | numpy 1.26.4 | backend models
frame 100: 3 boxes in tum_rgb_optical_frame after 5.74 s: cereal box 24x17x4 cm, cereal box 11x8x18 cm, telephone 22x18x12 cm
frame 300: 3 boxes in tum_rgb_optical_frame after 0.94 s: cereal box 25x18x7 cm, game controller 26x17x12 cm, cereal box 23x16x4 cm
frame 540: 4 boxes in tum_rgb_optical_frame after 1.04 s: telephone 25x14x8 cm, game controller 14x8x7 cm, cereal box 29x14x4 cm, cereal box 12x6x15 cm
grasp @ stamp 1305031464.127681017: position [0.371, -0.014, 1.119] m in tum_rgb_optical_frame
received 3 Detection3DArray and 1 PoseStamped for 3 frames
```

The class-agnostic `tabletop` backend merges touching objects (e.g. 74x44x26 cm in frame 300),
which is why the open-vocabulary `models` backend exists. As in notebook 01, Grounding DINO
labels books as *cereal box* when that phrase is in the caption.

The node was also run as separate processes, with the replay standing in for a camera driver
and the output checked with the ROS 2 CLI:

```bash
source /opt/ros/jazzy/setup.bash
/usr/bin/python3 -s projects/15_robotics_capstone/ros2/capstone_node.py --ros-args \
    -r "~/rgb:=/tum/rgb" -r "~/depth:=/tum/depth" -r "~/camera_info:=/tum/camera_info" \
    -p backend:=tabletop &
/usr/bin/python3 -s projects/15_robotics_capstone/ros2/replay_tum.py --rate 2 --stride 10 &
ros2 topic echo --once /capstone_perception/grasp
```

It printed a `PoseStamped` in `tum_rgb_optical_frame`. **Quote the `~/...` remap rules**,
otherwise the shell expands `~` to your home directory and the remapping silently does
nothing.

For a real camera, remap the three inputs to your driver's topics, e.g. a RealSense
`/camera/color/image_raw`, `/camera/aligned_depth_to_color/image_raw` and
`/camera/color/camera_info`. Pass `-p backend:=models -p cv_python:=<cv env python>` and
optionally `-p queries:="['cup', 'book']"`. This was not tested with a live camera.

## Limitations

- Outputs are in the camera optical frame. A robot needs TF (`base_link → camera`) to move
  them into its own frame. The notebooks do this step with the TUM motion-capture pose instead.
- One frame is processed at a time, and frames that arrive while the worker is busy are dropped
  by the approximate-time synchroniser queue. In the smoke test the `models` backend answered in
  0.94-1.04 s per frame after the first, model-loading frame.
- Not packaged as a colcon package or launch file: the scripts are run directly.
