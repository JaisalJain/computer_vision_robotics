# Verified software stack

Versions below were installed and verified in the `cv` environment (Ubuntu 24.04, RTX 5060 Laptop GPU 8 GB, sm_120) on the date shown. Every install used `-c requirements/constraints.txt`, so torch/torchvision were never replaced. `python scripts/check_env.py` passed after each install.

## Core (checked 2026-11-07)

| Package | Version | Install method | Source |
|---|---|---|---|
| Python | 3.12 | conda | https://www.python.org |
| torch | 2.11.0+cu128 | pip, https://download.pytorch.org/whl/cu128 | https://pytorch.org/get-started/locally/ |
| torchvision | 0.26.0+cu128 | pip, https://download.pytorch.org/whl/cu128 | https://pytorch.org/vision |
| numpy | 2.5.4 | pip | https://pypi.org/project/numpy/ |
| opencv-python | 5.0.0.93 | pip | https://pypi.org/project/opencv-python/ |
| scipy | 1.18.1 | pip | https://pypi.org/project/scipy/ |
| scikit-learn | 1.9.1 | pip | https://pypi.org/project/scikit-learn/ |
| matplotlib | 3.11.2 | pip | https://pypi.org/project/matplotlib/ |
| pandas | 3.1.1 | pip | https://pypi.org/project/pandas/ |
| pillow | 13.0.0 | pip | https://pypi.org/project/pillow/ |
| huggingface_hub | 2.4.0 | pip | https://pypi.org/project/huggingface-hub/ |
| jupyterlab / nbconvert / ipykernel | 4.6.6 / 7.17.2 / 7.4.0 | pip | https://jupyter.org |
| ruff | 0.17.4 | pip | https://pypi.org/project/ruff/ |
| pytest | 9.2.0 | pip | https://pypi.org/project/pytest/ |
| pre-commit | 4.7.0 | pip | https://pypi.org/project/pre-commit/ |

`torch.cuda.get_arch_list()` includes `sm_120`, so the Blackwell GPU runs native kernels without PTX JIT.

## 07_pose_estimation (checked 2026-11-08)

| Package / model | Version | Install / source |
|---|---|---|
| ultralytics | 8.4.204 | pip install -c requirements/constraints.txt ultralytics==8.4.204 (via make setup-07_pose_estimation; YOLO_AUTOINSTALL=false at runtime). Source: https://pypi.org/project/ultralytics/8.4.204/ (checked 2026-11-07) |
| YOLO26n-pose / YOLO26s-pose weights | assets v8.4.0 (yolo26n-pose.pt 7.5 MB, yolo26s-pose.pt 23.0 MB) | python scripts/download_07_pose_estimation.py (make data-07_pose_estimation) -> weights/ultralytics/. Source: https://github.com/ultralytics/assets/releases/tag/v8.4.0 (checked 2026-11-07) |
| pupil-apriltags | 1.0.4.post11 | pip install -c requirements/constraints.txt pupil-apriltags==1.0.4.post11 (cp312 manylinux wheel; dry-run changed no installed package). Source: https://pypi.org/project/pupil-apriltags/1.0.4.post11/ (checked 2026-11-07) |
| opencv-python cv2.aruco + solvePnP (IPPE_SQUARE, IPPE, SQPNP, ITERATIVE, RefineLM) | 5.0.0.93 | core stack (requirements/base.txt). Source: https://github.com/opencv/opencv/tree/5.0.0/modules/objdetect/include/opencv2/objdetect (checked 2026-11-07) |
| COCO8-Pose | assets v0.0.0 (0.34 MB zip, 4 train / 4 val images) | make data-07_pose_estimation -> data/ultralytics/coco8-pose. Source: https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8-pose.zip (checked 2026-11-07) |
| TUM RGB-D fr1_desk | rgbd_dataset_freiburg1_desk | scripts/download_shared.py tum('fr1_desk') (called from download_07_pose_estimation.py). Source: https://cvg.cit.tum.de/data/datasets/rgbd-dataset (checked 2026-11-07) |
| OpenCV ArUco tutorial photo + calibration | OpenCV 5.0.0 tag | make data-07_pose_estimation -> data/07_pose_estimation/. Source: https://raw.githubusercontent.com/opencv/opencv/5.0.0/doc/tutorials/objdetect/aruco_detection/images/singlemarkersoriginal.jpg (checked 2026-11-07) |

## 12_visual_odometry (checked 2026-11-08)

| Package / model | Version | Install / source |
|---|---|---|
| evo | 1.38.1 | pip install -c requirements/constraints.txt evo==1.38.1 (make setup-12_visual_odometry; already installed, verified no-op, pip check clean). Source: https://pypi.org/project/evo/ (checked 2026-11-07) |
| opencv-python (core stack) | 5.0.0.93 | requirements/base.txt. Source: https://pypi.org/project/opencv-python/ (checked 2026-11-07) |
| TUM RGB-D fr1/xyz + fr1/desk | freiburg1 sequences (~0.45 GB + ~0.34 GB tgz) | make data-12_visual_odometry (scripts/download_12_visual_odometry.py -> scripts/download_shared.tum()). Source: https://cvg.cit.tum.de/rgbd/dataset/freiburg1/ (checked 2026-11-07) |
