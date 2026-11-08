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

## 02_representation_learning (checked 2026-11-08)

| Package / model | Version | Install / source |
|---|---|---|
| timm | 1.0.31 | pip install -c requirements/constraints.txt timm==1.0.31 (requirements/02_representation_learning.txt). Source: https://pypi.org/project/timm/1.0.31/ |
| transformers | 5.22.2 | pip install -c requirements/constraints.txt transformers==5.22.2 (requirements/02_representation_learning.txt). Source: https://pypi.org/project/transformers/5.22.2/ |
| safetensors | 0.8.0 | pip install -c requirements/constraints.txt safetensors==0.8.0 (requirements/02_representation_learning.txt). Source: https://pypi.org/project/safetensors/0.8.0/ |
| Jinja2 | 3.1.6 | pip install -c requirements/constraints.txt Jinja2==3.1.6 (requirements/02_representation_learning.txt). Source: https://pypi.org/project/Jinja2/3.1.6/ |
| scikit-learn | 1.9.1 | core stack (requirements/base.txt). Source: https://pypi.org/project/scikit-learn/1.9.1/ |
| timm/resnet18.a1_in1k | HF rev 491b427b45c9 (45 MiB) | make data-02_representation_learning (HF cache). Source: https://huggingface.co/timm/resnet18.a1_in1k |
| timm/efficientnet_b0.ra_in1k | HF rev 1b5383e5f79c (21 MiB) | make data-02_representation_learning (HF cache). Source: https://huggingface.co/timm/efficientnet_b0.ra_in1k |
| timm/convnext_nano.in12k_ft_in1k | HF rev f185680dbe45 (60 MiB) | make data-02_representation_learning (HF cache). Source: https://huggingface.co/timm/convnext_nano.in12k_ft_in1k |
| facebook/dinov3-vits16-pretrain-lvd1689m | HF rev 114c13799502 (83 MiB), transformers DINOv3ViTModel, 4 register tokens | accept the license on the HF model page + hf auth login; make data-02_representation_learning. Source: https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m (gated; access confirmed 2026-11-07) |
| timm/vit_small_patch16_dinov3.lvd1689m (fallback) | HF rev 3bf4720a82ec (83 MiB) | auto-downloaded by the download script / replearn.load_encoder only when the gated repo is inaccessible. Source: https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m |
| facebook/dinov2-small | HF rev ed25f3a31f01 (85 MiB) | make data-02_representation_learning (HF cache). Source: https://huggingface.co/facebook/dinov2-small |
| Imagenette-160 | imagenette2-160 (99 MB tgz, 9469 train / 3925 val) | make data-02_representation_learning (scripts/download_shared.py imagenette()). Source: https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz |

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

## 08_monocular_depth (checked 2026-11-08)

| Package / model | Version | Install / source |
|---|---|---|
| depth-anything-3 (source clone) | commit 3d835ec1a5802d64a8b8b15f817a1ab54809bfe4 | bash scripts/setup_third_party.sh depth-anything-3; imported via cvlab.io.add_third_party('depth-anything-3', 'src') (not pip-installed: metadata pins numpy<2). Source: https://github.com/ByteDance-Seed/Depth-Anything-3 |
| transformers | 5.22.2 | pip install -c requirements/constraints.txt transformers==5.22.2 (requirements/08_monocular_depth.txt). Source: https://pypi.org/project/transformers/ |
| safetensors | 0.8.0 | pip (requirements/08_monocular_depth.txt, constraints). Source: https://pypi.org/project/safetensors/ |
| einops | 0.8.2 | pip (requirements/08_monocular_depth.txt, constraints). Source: https://pypi.org/project/einops/ |
| omegaconf | 2.4.0 | pip (requirements/08_monocular_depth.txt, constraints). Source: https://pypi.org/project/omegaconf/ |
| addict | 2.4.0 | pip (requirements/08_monocular_depth.txt, constraints). Source: https://pypi.org/project/addict/ |
| imageio | 2.38.1 | pip (requirements/08_monocular_depth.txt, constraints). Source: https://pypi.org/project/imageio/ |
| trimesh | 5.2.0 | pip (requirements/08_monocular_depth.txt, constraints). Source: https://pypi.org/project/trimesh/ |
| plyfile | 1.1.5 | pip (requirements/08_monocular_depth.txt, constraints). Source: https://pypi.org/project/plyfile/ |
| pycolmap | 4.2.1 | pip (requirements/08_monocular_depth.txt, constraints). Source: https://pypi.org/project/pycolmap/ |
| evo | 1.38.1 | pip (requirements/08_monocular_depth.txt, constraints). Source: https://pypi.org/project/evo/ |
| DA3-SMALL | HF main (34.3 M params) | make data-08_monocular_depth (HF cache). Source: https://huggingface.co/depth-anything/DA3-SMALL |
| DA3-BASE | HF main (135.4 M params) | make data-08_monocular_depth (HF cache). Source: https://huggingface.co/depth-anything/DA3-BASE |
| DA3METRIC-LARGE | HF main (334.2 M params) | make data-08_monocular_depth (HF cache). Source: https://huggingface.co/depth-anything/DA3METRIC-LARGE |
| Depth-Anything-V2-Small-hf | HF main (24.8 M params) | make data-08_monocular_depth (HF cache). Source: https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf |
| Depth-Anything-V2-Metric-Indoor-Small-hf | HF main (24.8 M params) | make data-08_monocular_depth (HF cache). Source: https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf |
| TUM RGB-D fr1_desk + fr1_xyz | freiburg1 tgz | make data-08_monocular_depth (scripts/download_shared.py tum()). Source: https://cvg.cit.tum.de/data/datasets/rgbd-dataset |

## 09_multiview_geometry (checked 2026-11-08)

| Package / model | Version | Install / source |
|---|---|---|
| pycolmap | 4.2.1 (COLMAP 4.2.1, CPU build) | pip install -c requirements/constraints.txt pycolmap==4.2.1 (via make setup-09_multiview_geometry); never together with pycolmap-cuda12 or the [panorama] extra. Source: https://pypi.org/project/pycolmap/ (checked 2026-11-08) |
| kornia | 0.9.0 | pip install -c requirements/constraints.txt kornia==0.9.0 (via make setup-09_multiview_geometry). Source: https://pypi.org/project/kornia/ (checked 2026-11-08) |
| kornia_rs | 0.2.0 | Installed automatically as a kornia dependency. Source: https://pypi.org/project/kornia-rs/ (checked 2026-11-08) |
| DISK depth checkpoint | depth-save.pth (4.4 MB) | python scripts/download_09_multiview_geometry.py -> weights/torch_hub/checkpoints (kornia DISK.from_pretrained('depth')). Source: https://github.com/cvlab-epfl/disk (checked 2026-11-08) |
| LightGlue for DISK | v0.1_arxiv release (disk_lightglue.pth, 47.6 MB) | python scripts/download_09_multiview_geometry.py -> weights/torch_hub/checkpoints (kornia LightGlue('disk')). Source: https://github.com/cvg/LightGlue/releases/tag/v0.1_arxiv (checked 2026-11-08) |
| Middlebury MiddEval3 quarter resolution | MiddEval3-data-Q.zip + MiddEval3-GT0-Q.zip (45 MB) | make data-09_multiview_geometry. Source: https://vision.middlebury.edu/stereo/submit3/ (checked 2026-11-08) |
| COLMAP South Building | colmap release 3.11.1 asset south-building.zip (419 MB; photos downscaled to 1024 px) | make data-09_multiview_geometry. Source: https://github.com/colmap/colmap/releases/download/3.11.1/south-building.zip (checked 2026-11-08) |
| TUM RGB-D fr1_desk | rgbd_dataset_freiburg1_desk | make data-09_multiview_geometry (scripts/download_shared.py tum('fr1_desk')). Source: https://cvg.cit.tum.de/data/datasets/rgbd-dataset (checked 2026-11-08) |

## 12_visual_odometry (checked 2026-11-08)

| Package / model | Version | Install / source |
|---|---|---|
| evo | 1.38.1 | pip install -c requirements/constraints.txt evo==1.38.1 (make setup-12_visual_odometry; already installed, verified no-op, pip check clean). Source: https://pypi.org/project/evo/ (checked 2026-11-07) |
| opencv-python (core stack) | 5.0.0.93 | requirements/base.txt. Source: https://pypi.org/project/opencv-python/ (checked 2026-11-07) |
| TUM RGB-D fr1/xyz + fr1/desk | freiburg1 sequences (~0.45 GB + ~0.34 GB tgz) | make data-12_visual_odometry (scripts/download_12_visual_odometry.py -> scripts/download_shared.tum()). Source: https://cvg.cit.tum.de/rgbd/dataset/freiburg1/ (checked 2026-11-07) |

## 13_vision_language (checked 2026-11-08)

| Package / model | Version | Install / source |
|---|---|---|
| transformers | 5.22.2 | pip install -c requirements/constraints.txt transformers==5.22.2 (make setup-13_vision_language). Source: https://pypi.org/project/transformers/ (checked 2026-11-08) |
| open_clip_torch | 4.2.1 | pip install -c requirements/constraints.txt open_clip_torch==4.2.1 (make setup-13_vision_language). Source: https://pypi.org/project/open-clip-torch/ (checked 2026-11-08) |
| timm | 1.0.31 | pip install -c requirements/constraints.txt timm==1.0.31 (make setup-13_vision_language). Source: https://pypi.org/project/timm/ (checked 2026-11-08) |
| google/siglip2-base-patch16-224 | rev 75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2 (1.4 GB snapshot) | make data-13_vision_language (HF hub, pinned in projects/13_vision_language/configs/models.yaml). Source: https://huggingface.co/google/siglip2-base-patch16-224 (checked 2026-11-08) |
| laion/CLIP-ViT-B-32-laion2B-s34B-b79K | rev 1a25a446712ba5ee05982a381eed697ef9b435cf (577 MB, open_clip_model.safetensors) | make data-13_vision_language (HF hub, pinned). Source: https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K (checked 2026-11-08) |
| florence-community/Florence-2-base | rev 00921df66db728a9ceb750f5eca43e5c203a2051 (447 MB) | make data-13_vision_language (HF hub, pinned); Florence2ForConditionalGeneration, no trust_remote_code. Source: https://huggingface.co/florence-community/Florence-2-base (checked 2026-11-08) |
| Qwen/Qwen3.5-0.8B | rev 2fc06364715b967f1860aea9cf38778875588b17 (1.6 GB bf16) | make data-13_vision_language (HF hub, pinned); Qwen3_5ForConditionalGeneration, native in transformers 5.22. Source: https://huggingface.co/Qwen/Qwen3.5-0.8B (checked 2026-11-08) |
| Ultralytics coco128 | assets v0.0.0 (128 images, 929 boxes) | make data-13_vision_language -> data/ultralytics/coco128. Source: https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip (checked 2026-11-08) |
| Imagenette-160 + TUM RGB-D fr1_desk | imagenette2-160; rgbd_dataset_freiburg1_desk | make data-13_vision_language (calls scripts/download_shared.py imagenette() and tum('fr1_desk')). Source: https://github.com/fastai/imagenette, https://cvg.cit.tum.de/data/datasets/rgbd-dataset (checked 2026-11-08) |
