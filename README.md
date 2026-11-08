# Computer Vision for Robotics

Hands-on, tested tutorials covering modern computer vision for robot perception: classical geometry, self-supervised features, (open-vocabulary) detection, segmentation, tracking, pose, depth, multi-view geometry, feed-forward 3D reconstruction, Gaussian splatting, visual odometry, vision-language models, deployment, and an end-to-end perception capstone.

[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.11](https://img.shields.io/badge/PyTorch-2.11%20%2B%20CUDA%2012.8-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![CI](https://github.com/JaisalJain/computer-vision/actions/workflows/ci.yml/badge.svg)](https://github.com/JaisalJain/computer-vision/actions/workflows/ci.yml)

Every notebook is a tutorial: an explanation precedes each code cell, every code cell starts with a comment block, and each notebook ends with a **Results** section (figures + summary table from a real run on the hardware below) and **Exercises** with solutions.

## Projects

| # | Project | Preview | What you'll build | Key models / tools |
|---|---|---|---|---|
| 02 | [Representation learning](projects/02_representation_learning/) | <img src="assets/02_representation_learning/thumb.png" width="160" alt="Representation learning"> | Linear probe vs fine-tuning of timm CNNs, frozen DINOv3/DINOv2 k-NN and few-shot features, Grad-CAM from scratch | ResNet-18, EfficientNet-B0, ConvNeXt-Nano (timm), DINOv3 ViT-S/16, DINOv2 ViT-S/14 |
| 05 | [Segmentation](projects/05_segmentation/) | <img src="assets/05_segmentation/thumb.png" width="160" alt="Segmentation"> | SAM 3 vs SAM 2.1 click/box/text segmentation, DAVIS video propagation with J&F, and YOLO26-seg real-time masks | SAM 3, SAM 2.1, Grounding DINO-T, YOLO26-seg |
| 07 | [Pose estimation](projects/07_pose_estimation/) | <img src="assets/07_pose_estimation/thumb.png" width="160" alt="Pose estimation"> | YOLO26 human keypoints with OKS/mAP, an AprilTag/ArUco 6-DoF PnP accuracy benchmark, and robot frame chains | YOLO26n/s-pose, cv2.aruco (AprilTag 36h11, ChArUco), AprilTag 3, solvePnP IPPE |
| 08 | [Monocular depth](projects/08_monocular_depth/) | <img src="assets/08_monocular_depth/thumb.png" width="160" alt="Monocular depth"> | Depth Anything 3 vs V2 on TUM RGB-D: relative vs metric depth, alignment, benchmark, point clouds | DA3-Small/Base, DA3Metric-Large, Depth Anything V2 (Small, Metric-Indoor) |
| 09 | [Multi-view geometry](projects/09_multiview_geometry/) | <img src="assets/09_multiview_geometry/thumb.png" width="160" alt="Multi-view geometry"> | Epipolar geometry, stereo rectification + SGBM, DISK+LightGlue matching and COLMAP SfM checked against reference models | OpenCV 5 (5-pt/MAGSAC++, SGBM), DISK + LightGlue (kornia), COLMAP/pycolmap 4.2 |
| 10 | [Feed-forward 3D reconstruction](projects/10_feedforward_3d/) | <img src="assets/10_feedforward_3d/thumb.png" width="160" alt="Feed-forward 3D reconstruction"> | Unposed photos to cameras, depth and point clouds with VGGT and Depth Anything 3, scored against COLMAP and mocap | VGGT-1B, DA3-Base, DA3-Large-1.1, pycolmap |
| 12 | [Visual odometry](projects/12_visual_odometry/) | <img src="assets/12_visual_odometry/thumb.png" width="160" alt="Visual odometry"> | Monocular and RGB-D feature-based VO on TUM fr1, scored with ATE/RPE, with drift and failure analysis | ORB, SIFT, 5-point E + MAGSAC++, EPnP-RANSAC, evo |
| 13 | [Vision-language models](projects/13_vision_language/) | <img src="assets/13_vision_language/thumb.png" width="160" alt="Vision-language models"> | Zero-shot SigLIP 2/CLIP classification, Florence-2 captions and grounding, and a benchmarked small VLM for robot-desk VQA | SigLIP 2, CLIP ViT-B/32, Florence-2-base, Qwen3.5-0.8B |
| 15 | [Robotics capstone](projects/15_robotics_capstone/) | <img src="assets/15_robotics_capstone/thumb.png" width="160" alt="Robotics capstone"> | Text-queried RGB-D perception: open-vocab detection, SAM masks, object clouds, 3D boxes, grasp poses, world object map | Grounding DINO-T, OWLv2, SAM 3, SAM 2.1-S, Depth Anything V2 Metric |

*Projects are added as they are completed.*

## Tested on

| Component | Version |
|---|---|
| OS | Ubuntu 24.04 |
| CPU / RAM | Intel Core Ultra 9 275HX, 32 GB |
| GPU | NVIDIA RTX 5060 Laptop GPU, 8 GB VRAM (Blackwell, sm_120) |
| Python | 3.12 |
| PyTorch | 2.11.0 + CUDA 12.8, torchvision 0.26.0 |
| NumPy / OpenCV | 2.5 / 5.0 |

All numbers in this repository were measured on this machine. Every model and setting was chosen to fit in 8 GB of VRAM.

## Installation

A CUDA 12.8+ PyTorch build is required for Blackwell (sm_120) GPUs. Every install goes through `requirements/constraints.txt`, which pins `torch`/`torchvision` so no package can silently replace them.

### Option A: conda

```bash
conda env create -f environment.yml
conda activate cv
python scripts/check_env.py
```

### Option B: venv

```bash
python3.12 -m venv .venv
source .venv/bin/activate
make setup            # core stack + editable cvlab package + environment check
```

### Project extras and data

```bash
make setup-<project>  # e.g. make setup-03_object_detection
make data-<project>   # idempotent dataset / weight download into data/ and weights/
```

Register the environment as a Jupyter kernel named `cv` to execute notebooks: `python -m ipykernel install --user --name cv`.

## Quickstart

```bash
make check-env        # torch sees the GPU, sm_120 kernels, pip check is clean
make test             # CPU unit tests of the shared cvlab package
make verify           # lint + tests + notebook / README / size / link checks
make nb-<project>     # re-execute a project's notebooks headless
```

## Repository structure

```
computer-vision/
├── src/cvlab/            # shared, tested code: device, io, viz, geometry, pointcloud, metrics, download, bench
├── tests/                # CPU-only pytest suite
├── scripts/              # check_env.py, verify_repo.py, nb_clean.py, download_<project>.py
├── projects/<NN>_<name>/ # README, notebooks/, scripts/, configs/, solutions/
├── assets/<NN>_<name>/   # small figures used by the READMEs
├── requirements/         # constraints.txt, base.txt, <project>.txt
├── docs/stack.md         # verified versions, install methods and sources
└── data/ weights/ runs/ outputs/ third_party/   # created on demand, gitignored
```

## Datasets

Only small public datasets are used. They are downloaded on demand by `make data-<project>`, and each project README lists its sources and licenses.

## License

The code in this repository is released under the [MIT License](LICENSE). Models, datasets and libraries keep their own licenses. Some are copyleft (e.g. Ultralytics, AGPL-3.0) or non-commercial; see [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

## Citation

If this repository is useful for your work, please cite the original papers listed in each project README, and optionally this repository:

```bibtex
@misc{jain2026cvrobotics,
  author       = {Jaisal Jain},
  title        = {Computer Vision for Robotics: tested tutorials and benchmarks},
  year         = {2026},
  howpublished = {\url{https://github.com/JaisalJain/computer-vision}}
}
```
