# Third-party licenses

The MIT license of this repository covers only its own code. Every model, dataset and library used by the projects keeps its own license, listed below per project. Entries marked **non-commercial** or **copyleft** carry restrictions on use beyond this educational portfolio.

## Core stack

| Component | License | Source |
|---|---|---|
| PyTorch, torchvision | BSD-3-Clause | https://github.com/pytorch/pytorch |
| NumPy | BSD-3-Clause | https://numpy.org |
| OpenCV | Apache-2.0 | https://opencv.org |
| SciPy | BSD-3-Clause | https://scipy.org |
| scikit-learn | BSD-3-Clause | https://scikit-learn.org |
| matplotlib | Matplotlib License (BSD-compatible) | https://matplotlib.org |
| pandas | BSD-3-Clause | https://pandas.pydata.org |
| Pillow | MIT-CMU | https://python-pillow.org |
| Jupyter (nbconvert, ipykernel) | BSD-3-Clause | https://jupyter.org |
| Hugging Face Hub client | Apache-2.0 | https://github.com/huggingface/huggingface_hub |

## 07_pose_estimation

| Component | Kind | License | Source | Notes |
|---|---|---|---|---|
| ultralytics 8.4.204 (library) | library | **AGPL-3.0** | https://github.com/ultralytics/ultralytics | Copyleft. Commercial use needs an Ultralytics enterprise license. |
| YOLO26n-pose / YOLO26s-pose weights (assets v8.4.0) | model | **AGPL-3.0** | https://github.com/ultralytics/assets/releases/tag/v8.4.0 | Copyleft (AGPL-3.0). |
| COCO8-Pose (8 COCO train2017 images, Ultralytics packaging) | dataset | **AGPL-3.0 packaging; COCO annotations CC BY 4.0; images under their Flickr licenses** | https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8-pose.zip | Mixed license; the image licenses vary per Flickr image. |
| Ultralytics sample images bus.jpg, zidane.jpg | dataset | **AGPL-3.0 (shipped inside the ultralytics package)** | https://github.com/ultralytics/ultralytics/tree/main/ultralytics/assets | Copyleft. |
| TUM RGB-D fr1_desk (backgrounds, table texture, fr1 intrinsics) | dataset | CC BY 4.0 | https://cvg.cit.tum.de/data/datasets/rgbd-dataset | Shared with projects 08, 09 and 12. |
| OpenCV tutorial photo singlemarkersoriginal.jpg + tutorial_camera_params.yml (OpenCV 5.0.0 repo) | dataset | Apache-2.0 | https://github.com/opencv/opencv/tree/5.0.0/doc/tutorials/objdetect/aruco_detection/images | The marker size (5 cm) is assumed; the photo has no ground truth. |
| opencv-python 5.0.0.93 (cv2.aruco ArUco/AprilTag/ChArUco, solvePnP) | library | Apache-2.0 | https://github.com/opencv/opencv | Core stack. The AprilTag 36h11 / ArUco / ChArUco patterns are generated locally. |
| pupil-apriltags 1.0.4.post11 (bindings) + bundled AprilTag 3 C library | library | MIT (bindings) / BSD-2-Clause (AprilTag C library) | https://github.com/pupil-labs/apriltags | Optional comparison detector; the notebooks fall back to cv2.aruco if it is missing. |
| scipy 1.18.1 (Rotation, linear_sum_assignment) | library | BSD-3-Clause | https://github.com/scipy/scipy | Core stack. |

## 12_visual_odometry

| Component | Kind | License | Source | Notes |
|---|---|---|---|---|
| TUM RGB-D benchmark (fr1/xyz, fr1/desk) | dataset | CC BY 4.0 | https://cvg.cit.tum.de/data/datasets/rgbd-dataset | Attribution required (Sturm et al., IROS 2012). |
| evo 1.38.1 | library | **GPL-3.0** | https://github.com/MichaelGrupp/evo | COPYLEFT. Used only as an evaluation tool / test dependency (lazy import in vo.evo_metrics, evo_ape CLI); not part of the VO code path. |
| OpenCV 5.0 (ORB, SIFT, findEssentialMat/USAC_MAGSAC, recoverPose, solvePnPRansac/RefineLM, triangulatePoints, calcOpticalFlowPyrLK) | library | Apache-2.0 | https://github.com/opencv/opencv | SIFT patent expired in 2020; it is in the main package. |
| NumPy / SciPy / pandas / matplotlib | library | BSD-3-Clause (matplotlib: Matplotlib license, BSD-style) | https://numpy.org, https://scipy.org, https://pandas.pydata.org, https://matplotlib.org | Core stack. |
| seaborn, rosbags (evo transitive dependencies) | library | BSD-3-Clause / Apache-2.0 | https://pypi.org/project/seaborn/, https://pypi.org/project/rosbags/ | Installed by evo; not imported by project code. |
