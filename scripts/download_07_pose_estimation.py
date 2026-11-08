#!/usr/bin/env python
"""Data + weights for project 07 (pose estimation). Idempotent: re-running is a no-op.

* YOLO26n-pose / YOLO26s-pose checkpoints (Ultralytics, AGPL-3.0) -> ``weights/ultralytics/``
* COCO8-Pose (8 COCO images with 17 keypoints, Ultralytics packaging) -> ``data/ultralytics/``
* TUM RGB-D fr1_desk (CC BY 4.0, shared): real desk images used as backgrounds for the
  synthetic fiducial renders -> ``data/tum/fr1_desk/...``
* One real photo of ArUco markers from the OpenCV tutorials and the camera calibration file
  shipped with that tutorial (Apache-2.0) -> ``data/07_pose_estimation/``

The fiducial markers themselves (AprilTag 36h11, ArUco, ChArUco) are generated locally
with ``cv2.aruco``, so they need no download.

Usage::

    python scripts/download_07_pose_estimation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from download_shared import tum  # noqa: E402

from cvlab.download import download_file, extract_archive  # noqa: E402
from cvlab.io import data_dir, weights_dir  # noqa: E402

ULTRA_RELEASE = "https://github.com/ultralytics/assets/releases/download"
WEIGHTS = {
    "yolo26n-pose.pt": f"{ULTRA_RELEASE}/v8.4.0/yolo26n-pose.pt",  # 7.9 MB
    "yolo26s-pose.pt": f"{ULTRA_RELEASE}/v8.4.0/yolo26s-pose.pt",  # 24.2 MB
}
COCO8_POSE_URL = f"{ULTRA_RELEASE}/v0.0.0/coco8-pose.zip"  # 0.34 MB
OPENCV_RAW = "https://raw.githubusercontent.com/opencv/opencv/5.0.0"
OPENCV_FILES = [
    f"{OPENCV_RAW}/doc/tutorials/objdetect/aruco_detection/images/singlemarkersoriginal.jpg",
    f"{OPENCV_RAW}/samples/cpp/tutorial_code/objectDetection/tutorial_camera_params.yml",
]


def coco8_pose() -> Path:
    """Extract COCO8-Pose into the Ultralytics datasets dir (``data/ultralytics/coco8-pose``)."""
    root = data_dir("ultralytics", create=True)
    out = root / "coco8-pose"
    if (out / "images" / "val").is_dir():
        print("[skip] coco8-pose already extracted")
        return out
    archive = download_file(COCO8_POSE_URL, root / "coco8-pose.zip")
    extract_archive(archive, root)  # the zip contains a top-level coco8-pose/ folder
    archive.unlink(missing_ok=True)
    return out


def main() -> None:
    for name, url in WEIGHTS.items():
        download_file(url, weights_dir("ultralytics", create=True) / name)
    print("[ok] YOLO26 pose weights in weights/ultralytics/")

    out = coco8_pose()
    n_val = len(list((out / "images" / "val").glob("*.jpg")))
    print(f"[ok] coco8-pose: data/ultralytics/coco8-pose ({n_val} val images)")

    seq = tum("fr1_desk")
    print(f"[ok] TUM fr1_desk: data/tum/fr1_desk/{seq.name}")

    for url in OPENCV_FILES:
        download_file(url, data_dir("07_pose_estimation", create=True) / Path(url).name)
    print("[ok] OpenCV ArUco tutorial photo + calibration in data/07_pose_estimation/")


if __name__ == "__main__":
    main()
