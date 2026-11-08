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

## 01_classical_cv (checked 2026-11-08)

| Package / model | Version | Install / source |
|---|---|---|
| opencv-python | 5.0.0.93 (cv2 5.0.0) | core stack (requirements/base.txt); no extra packages for this project. Source: https://pypi.org/project/opencv-python/ (checked 2026-11-08; GitHub has OpenCV 5.1.0 since 2026-11-06 but PyPI wheel is still 5.0.0.93) |
| numpy | 2.5.4 | core stack. Source: https://pypi.org/project/numpy/ |
| scipy | 1.18.1 | core stack. Source: https://pypi.org/project/scipy/ |
| pandas | 3.1.1 | core stack. Source: https://pypi.org/project/pandas/ |
| matplotlib | 3.11.2 | core stack. Source: https://pypi.org/project/matplotlib/ |
| PyYAML | 6.0.3 | already installed in cv env (dependency of the core stack). Source: https://pypi.org/project/PyYAML/ |
| OpenCV sample data | tag 5.0.0 | python scripts/download_01_classical_cv.py (make data-01_classical_cv). Source: https://raw.githubusercontent.com/opencv/opencv/5.0.0/samples/data/ and https://raw.githubusercontent.com/opencv/opencv_extra/5.0.0/testdata/stitching/ |

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

## 03_object_detection (checked 2026-11-08)

| Package / model | Version | Install / source |
|---|---|---|
| ultralytics | 8.4.204 | pip install -c requirements/constraints.txt ultralytics==8.4.204 (make setup-03_object_detection). Source: https://pypi.org/project/ultralytics/ |
| onnx | 1.23.2 | pip install -c requirements/constraints.txt onnx==1.23.2. Source: https://pypi.org/project/onnx/ |
| onnxslim | 0.1.99 | pip install -c requirements/constraints.txt onnxslim==0.1.99. Source: https://pypi.org/project/onnxslim/ |
| onnxruntime-gpu | 1.26.0 | pip install -c requirements/constraints.txt onnxruntime-gpu==1.26.0 (no [cuda,cudnn] extras; import torch first). Source: https://pypi.org/project/onnxruntime-gpu/1.26.0/ and https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html |
| nvidia-ml-py | 13.615.71 | pip install -c requirements/constraints.txt nvidia-ml-py==13.615.71. Source: https://pypi.org/project/nvidia-ml-py/ |
| rfdetr | 1.12.0 | pip install -c requirements/constraints.txt rfdetr==1.12.0 (no [train] extra; pulled in pydantic 2.14.3, av 19.0.2, pyDeprecate 0.9.0). Source: https://pypi.org/project/rfdetr/ and https://github.com/roboflow/rf-detr |
| supervision | 0.31.5 | pip install -c requirements/constraints.txt supervision==0.31.5. Source: https://pypi.org/project/supervision/ |
| YOLO26n / YOLO26s weights | assets v8.4.0 | make data-03_object_detection (scripts/download_03_object_detection.py -> weights/ultralytics/). Source: https://github.com/ultralytics/assets/releases/tag/v8.4.0 |
| RF-DETR Nano / Small COCO weights | rfdetr 1.12.0 checkpoints (md5 fb6504cc..., fb37061c...) | make data-03_object_detection -> weights/rfdetr/ (set RF_HOME there). Source: https://raw.githubusercontent.com/roboflow/rf-detr/develop/src/rfdetr/assets/model_weights.py |
| african-wildlife / medical-pills / coco8 datasets | Ultralytics assets v0.0.0 | make data-03_object_detection -> data/ultralytics/<name>/. Source: https://github.com/ultralytics/assets/releases/tag/v0.0.0 |

## 05_segmentation (checked 2026-11-08)

| Package / model | Version | Install / source |
|---|---|---|
| transformers (Sam3Model, Sam3TrackerModel, Sam3TrackerVideoModel, Sam3VideoModel, Sam2Model, Sam2VideoModel, GroundingDino) | 5.22.2 | python -m pip install -c requirements/constraints.txt transformers==5.22.2 (via make setup-05_segmentation). Source: https://pypi.org/project/transformers/ (checked 2026-11-08) |
| tokenizers | 0.23.3 | make setup-05_segmentation (pinned in requirements/05_segmentation.txt). Source: https://pypi.org/project/tokenizers/ (checked 2026-11-08) |
| safetensors | 0.8.0 | make setup-05_segmentation. Source: https://pypi.org/project/safetensors/ (checked 2026-11-08) |
| ultralytics | 8.4.204 | python -m pip install -c requirements/constraints.txt ultralytics==8.4.204; YOLO_AUTOINSTALL=false at runtime. Source: https://pypi.org/project/ultralytics/ (checked 2026-11-08) |
| SAM 3 checkpoint facebook/sam3 | HF revision 3c879f3982 (model.safetensors + configs only, 3285 MiB) | python scripts/download_05_segmentation.py (huggingface_hub snapshot_download with allow_patterns; gated). Source: https://huggingface.co/facebook/sam3 (checked 2026-11-08) |
| SAM 2.1 facebook/sam2.1-hiera-tiny / -small / -base-plus | HF revisions de431c4043 / ee5bba1d82 / b7320756a1 | python scripts/download_05_segmentation.py. Source: https://huggingface.co/facebook/sam2.1-hiera-small (checked 2026-11-08) |
| Grounding DINO IDEA-Research/grounding-dino-tiny | HF revision a2bb814dd3 | python scripts/download_05_segmentation.py. Source: https://huggingface.co/IDEA-Research/grounding-dino-tiny (checked 2026-11-08) |
| YOLO26n-seg / YOLO26s-seg weights | ultralytics/assets v8.4.0 | python scripts/download_05_segmentation.py -> weights/ultralytics/. Source: https://github.com/ultralytics/assets/releases/tag/v8.4.0 (checked 2026-11-08) |
| DAVIS 2017 TrainVal 480p (2 sequences, 17 MB extracted) | 2017 release | python scripts/download_05_segmentation.py (HTTP range requests into the 833 MB zip; --full-zip fallback). Source: https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip (checked 2026-11-08) |
| coco8-seg / coco128-seg | Ultralytics assets v0.0.0 | python scripts/download_05_segmentation.py -> data/ultralytics/. Source: https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128-seg.zip (checked 2026-11-08) |

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

## 10_feedforward_3d (checked 2026-11-08)

| Package / model | Version | Install / source |
|---|---|---|
| VGGT (code) | git a288dd0f14786c93483e45524328726ab7b1b4ce | bash scripts/setup_third_party.sh vggt (pinned clone, imported via cvlab.io.add_third_party('vggt'); not pip-installed because pyproject pins numpy<2). Source: https://github.com/facebookresearch/vggt (checked 2026-11-08) |
| Depth Anything 3 (code) | git 3d835ec1a5802d64a8b8b15f817a1ab54809bfe4 | bash scripts/setup_third_party.sh depth-anything-3 (imported via add_third_party('depth-anything-3','src') with a moviepy.editor stub; not pip-installed because of numpy<2 / moviepy==1.0.3 pins). Source: https://github.com/ByteDance-Seed/Depth-Anything-3 (checked 2026-11-08) |
| facebook/VGGT-1B | HF main (model.safetensors 5.03 GB fp32, 1.26 B params) | make data-10_feedforward_3d (hf_hub_download into HF cache). Source: https://huggingface.co/facebook/VGGT-1B (checked 2026-11-08) |
| depth-anything/DA3-BASE | HF main (0.54 GB, 135 M params) | make data-10_feedforward_3d (HF cache). Source: https://huggingface.co/depth-anything/DA3-BASE (checked 2026-11-08) |
| depth-anything/DA3-LARGE-1.1 | HF main (1.64 GB) | make data-10_feedforward_3d (HF cache). Source: https://huggingface.co/depth-anything/DA3-LARGE-1.1 (checked 2026-11-08) |
| pycolmap | 4.2.1 | pip install -c requirements/constraints.txt pycolmap==4.2.1 (requirements/10_feedforward_3d.txt). Source: https://pypi.org/project/pycolmap/ (checked 2026-11-08) |
| evo | 1.38.1 | pip install -c requirements/constraints.txt evo==1.38.1. Source: https://pypi.org/project/evo/ (checked 2026-11-08) |
| einops / safetensors / omegaconf / addict / imageio / trimesh / plyfile | 0.8.2 / 0.8.0 / 2.4.0 / 2.4.0 / 2.38.1 / 5.2.0 / 1.1.5 | make setup-10_feedforward_3d (pip -c requirements/constraints.txt -r requirements/10_feedforward_3d.txt; already installed, no-op). Source: https://pypi.org/ (checked 2026-11-08) |
| South Building dataset | COLMAP release 3.11.1 zip (128 photos -> 1024 px + reference sparse model) | make data-10_feedforward_3d. Source: https://github.com/colmap/colmap/releases/download/3.11.1/south-building.zip (checked 2026-11-08) |
| VGGT kitchen example images | 25 PNG at vggt@a288dd0 | make data-10_feedforward_3d (copied from third_party/vggt or raw.githubusercontent.com). Source: https://github.com/facebookresearch/vggt/tree/a288dd0f14786c93483e45524328726ab7b1b4ce/examples/kitchen (checked 2026-11-08) |
| TUM RGB-D fr1_desk | rgbd_dataset_freiburg1_desk (24 undistorted sharp keyframes derived) | make data-10_feedforward_3d (download_shared.tum). Source: https://cvg.cit.tum.de/data/datasets/rgbd-dataset (checked 2026-11-08) |

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

## 15_robotics_capstone (checked 2026-11-08)

| Package / model | Version | Install / source |
|---|---|---|
| transformers | 5.22.2 | pip install -c requirements/constraints.txt transformers==5.22.2 (make setup-15_robotics_capstone). Source: https://pypi.org/project/transformers/5.22.2/ (checked 2026-11-08) |
| tokenizers | 0.23.3 | pip (dependency of transformers, pinned in requirements/15_robotics_capstone.txt). Source: https://pypi.org/project/tokenizers/ (checked 2026-11-08) |
| safetensors | 0.8.0 | pip (pinned in requirements/15_robotics_capstone.txt). Source: https://pypi.org/project/safetensors/ (checked 2026-11-08) |
| rerun-sdk | 0.42.0 | pip install -c requirements/constraints.txt rerun-sdk==0.42.0. Source: https://pypi.org/project/rerun-sdk/0.42.0/ (checked 2026-11-08) |
| pyarrow | 26.0.1 | pip (dependency of rerun-sdk, pinned). Source: https://pypi.org/project/pyarrow/ (checked 2026-11-08) |
| Grounding DINO-T | IDEA-Research/grounding-dino-tiny @ a2bb814dd30d776dcf7e30523b00659f4f141c71 | make data-15_robotics_capstone (HF cache, safetensors only). Source: https://huggingface.co/IDEA-Research/grounding-dino-tiny (checked 2026-11-08) |
| OWLv2-B/16 ensemble | google/owlv2-base-patch16-ensemble @ cfd3195ba4ea9592eec887ded089f4c08eff231d | make data-15_robotics_capstone. Source: https://huggingface.co/google/owlv2-base-patch16-ensemble (checked 2026-11-08) |
| SAM 3 (Sam3TrackerModel box prompts, Sam3Model text prompts) | facebook/sam3 @ 3c879f39826c281e95690f02c7821c4de09afae7 | make data-15_robotics_capstone (gated: accept license + hf auth login). Source: https://huggingface.co/facebook/sam3 (checked 2026-11-08) |
| SAM 2.1 Hiera-S | facebook/sam2.1-hiera-small @ ee5bba1d82bb8749febdf90f45e84b687142ba03 | make data-15_robotics_capstone. Source: https://huggingface.co/facebook/sam2.1-hiera-small (checked 2026-11-08) |
| Depth Anything V2 Metric-Indoor-Small | depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf @ 8078d68a9c75a972131914f6afd0c1723be0da7f | make data-15_robotics_capstone. Source: https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf (checked 2026-11-08) |
| TUM RGB-D fr1/desk | rgbd_dataset_freiburg1_desk (344 MB tgz) | make data-15_robotics_capstone (scripts/download_shared.py tum('fr1_desk')). Source: https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz (checked 2026-11-08) |
| ROS 2 Jazzy (rclpy 7.1.12, vision_msgs 4.1.1, message_filters), optional node only | jazzy (apt) | apt: ros-jazzy-rclpy ros-jazzy-vision-msgs ros-jazzy-sensor-msgs ros-jazzy-geometry-msgs (system Python 3.12, NOT the cv env). Source: https://docs.ros.org/en/jazzy/ (checked 2026-11-08) |
