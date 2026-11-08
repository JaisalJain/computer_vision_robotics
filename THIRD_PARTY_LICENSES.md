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

## 01_classical_cv

| Component | Kind | License | Source | Notes |
|---|---|---|---|---|
| OpenCV 5.0 (opencv-python 5.0.0.93) | library | Apache-2.0 | https://github.com/opencv/opencv |  |
| OpenCV sample images left*/right*.jpg, box.png, box_in_scene.png (samples/data @5.0.0) | dataset | Apache-2.0 (repository license; per-image provenance not documented) | https://github.com/opencv/opencv/tree/5.0.0/samples/data | downloaded at runtime into data/, not committed |
| graf1.png, graf3.png, H1to3p.xml (Oxford affine-covariant regions dataset, via OpenCV samples/data) | dataset | **research dataset, terms not stated explicitly** | https://www.robots.ox.ac.uk/~vgg/data/affine/ | UNCLEAR license: downloaded at runtime only, not committed, not shown in README figures (summary plots only) |
| boat1-6.jpg (opencv_extra testdata/stitching @5.0.0) | dataset | **unclear (repository declares no license)** | https://github.com/opencv/opencv_extra/tree/5.0.0/testdata/stitching | UNCLEAR license: downloaded at runtime only, never committed as files; downscaled panoramas built from them appear only as notebook outputs, not in assets/ |
| TUM RGB-D fr1_desk | dataset | CC BY 4.0 | https://cvg.cit.tum.de/data/datasets/rgbd-dataset | shared dataset (scripts/download_shared.py); attribution: Sturm et al., IROS 2012 |
| NumPy | library | BSD-3-Clause | https://github.com/numpy/numpy |  |
| SciPy | library | BSD-3-Clause | https://github.com/scipy/scipy |  |
| pandas | library | BSD-3-Clause | https://github.com/pandas-dev/pandas |  |
| matplotlib | library | Matplotlib License (PSF-based, BSD-compatible) | https://github.com/matplotlib/matplotlib |  |
| PyYAML | library | MIT | https://github.com/yaml/pyyaml |  |

## 02_representation_learning

| Component | Kind | License | Source | Notes |
|---|---|---|---|---|
| Imagenette-160 (fast.ai) | dataset | **Apache-2.0 (repository); images are an ImageNet subset, ImageNet terms of access apply (non-commercial research/education)** | https://github.com/fastai/imagenette | ImageNet terms of access apply to the images (non-commercial research/education). |
| timm/resnet18.a1_in1k | model | Apache-2.0 | https://huggingface.co/timm/resnet18.a1_in1k | Trained on ImageNet-1k. |
| timm/efficientnet_b0.ra_in1k | model | Apache-2.0 | https://huggingface.co/timm/efficientnet_b0.ra_in1k | Trained on ImageNet-1k. |
| timm/convnext_nano.in12k_ft_in1k | model | Apache-2.0 | https://huggingface.co/timm/convnext_nano.in12k_ft_in1k | Trained on ImageNet-12k, then fine-tuned on ImageNet-1k. |
| facebook/dinov3-vits16-pretrain-lvd1689m (DINOv3 ViT-S/16) | model | **DINOv3 License (custom, Meta)** | https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m | Gated (manual access, granted to the owner, used for committed outputs). Commercial use allowed but requires 'Built with DINOv3' attribution when redistributing; acceptable-use restrictions (e.g. military/ITAR). |
| timm/vit_small_patch16_dinov3.lvd1689m (ungated mirror, fallback only) | model | **DINOv3 License (custom, Meta)** | https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m | Only downloaded/used when the gated repo is inaccessible; same weights and same license terms. |
| facebook/dinov2-small (DINOv2 ViT-S/14) | model | Apache-2.0 | https://huggingface.co/facebook/dinov2-small |  |
| timm 1.0.31 | library | Apache-2.0 | https://github.com/huggingface/pytorch-image-models |  |
| transformers 5.22.2 | library | Apache-2.0 | https://github.com/huggingface/transformers |  |
| safetensors 0.8.0 | library | Apache-2.0 | https://github.com/huggingface/safetensors |  |
| scikit-learn 1.9.1 | library | BSD-3-Clause | https://scikit-learn.org | Logistic regression, t-SNE, StandardScaler. |
| Jinja2 3.1.6 | library | BSD-3-Clause | https://github.com/pallets/jinja | Needed by the pandas Styler for the notebook tables. |
| Grad-CAM / HiResCAM / deletion-insertion implementation | code | MIT (this repository) | projects/02_representation_learning/scripts/replearn.py | Own implementation; the grad-cam PyPI package (MIT) is not used because it hard-depends on opencv-python-headless. |

## 03_object_detection

| Component | Kind | License | Source | Notes |
|---|---|---|---|---|
| ultralytics 8.4.204 | library | **AGPL-3.0** | https://github.com/ultralytics/ultralytics | COPYLEFT (AGPL-3.0, also covers network use); Enterprise license available. |
| YOLO26n / YOLO26s COCO weights (assets v8.4.0) and models fine-tuned from them | model | **AGPL-3.0** | https://github.com/ultralytics/assets/releases/tag/v8.4.0 | COPYLEFT: fine-tuned derivatives are also AGPL-3.0; no weights are committed. |
| rfdetr 1.12.0 (RF-DETR code) | library | Apache-2.0 | https://github.com/roboflow/rf-detr | Inference-only install (no [train] extra). The [plus] models use PML 1.0 and are not used. |
| RF-DETR Nano / Small COCO checkpoints | model | Apache-2.0 | https://repo.roboflow.com/rfdetr/nano_coco/checkpoint_best_regular.pth ; https://repo.roboflow.com/rfdetr/small_coco/checkpoint_best_regular.pth | Downloaded to weights/rfdetr/ (about 0.7 GB in total). |
| supervision 0.31.5 | library | MIT | https://github.com/roboflow/supervision | Installed as an rfdetr dependency. |
| onnx 1.23.2 | library | Apache-2.0 | https://github.com/onnx/onnx |  |
| onnxslim 0.1.99 | library | MIT | https://github.com/inisis/OnnxSlim |  |
| onnxruntime-gpu 1.26.0 | library | MIT | https://github.com/microsoft/onnxruntime |  |
| nvidia-ml-py 13.615.71 | library | BSD | https://pypi.org/project/nvidia-ml-py/ | NVML bindings used to read process GPU memory (also an ultralytics dependency). |
| african-wildlife (Ultralytics) | dataset | **unclear (source images: no license stated; Ultralytics packaging AGPL-3.0)** | https://github.com/ultralytics/assets/releases/download/v0.0.0/african-wildlife.zip | UNCLEAR: the docs page says 'No license' for the source data. Not redistributed; downscaled images appear only in figures. |
| medical-pills (Ultralytics) | dataset | **AGPL-3.0** | https://github.com/ultralytics/assets/releases/download/v0.0.0/medical-pills.zip | COPYLEFT packaging; used only in the exercise solution. |
| coco8 (Ultralytics) | dataset | **COCO annotations CC BY 4.0; images under their Flickr licenses; Ultralytics packaging AGPL-3.0** | https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip | Mixed license. |
| TUM RGB-D fr1_desk | dataset | CC BY 4.0 | https://cvg.cit.tum.de/data/datasets/rgbd-dataset | Shared dataset; attribution: Sturm et al., IROS 2012. |

## 04_open_vocab_detection

| Component | Kind | License | Source | Notes |
|---|---|---|---|---|
| ultralytics package (YOLO26 / YOLOE-26 code) | library | **AGPL-3.0** | https://pypi.org/project/ultralytics/8.4.204/ | Copyleft; closed-source commercial use needs an Ultralytics Enterprise license |
| YOLO26-s weights (yolo26s.pt) | model | **AGPL-3.0** | https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt | Copyleft |
| YOLOE-26-s-seg weights (yoloe-26s-seg.pt) | model | **AGPL-3.0** | https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26s-seg.pt | Copyleft |
| mobileclip2_b.ts (YOLOE-26 text encoder, from Apple MobileCLIP2-B) | model | **Apple AMLR (apple-amlr on the HF model card)** | https://github.com/ultralytics/assets/releases/download/v8.4.0/mobileclip2_b.ts | Research-only terms, so NON-COMMERCIAL; Ultralytics redistribution terms also apply |
| clip (Ultralytics fork of OpenAI CLIP, YOLOE tokenizer) @7ffa84b | library | **MIT (OpenAI CLIP); the fork's pyproject header mentions AGPL-3.0** | https://github.com/ultralytics/CLIP | License is unclear: MIT code inside an Ultralytics fork whose header says AGPL; treat as AGPL to be safe |
| Grounding DINO-T (IDEA-Research/grounding-dino-tiny @a2bb814) | model | Apache-2.0 | https://huggingface.co/IDEA-Research/grounding-dino-tiny |  |
| LLMDet-T (iSEE-Laboratory/llmdet_tiny @d051991) | model | Apache-2.0 | https://huggingface.co/iSEE-Laboratory/llmdet_tiny |  |
| MM-Grounding-DINO-T (openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det @ef2053a, solutions only) | model | Apache-2.0 | https://huggingface.co/openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det |  |
| OWLv2-B/16 (google/owlv2-base-patch16-ensemble @cfd3195) | model | Apache-2.0 | https://huggingface.co/google/owlv2-base-patch16-ensemble |  |
| OmDet-Turbo-T (omlab/omdet-turbo-swin-tiny-hf @7fe93ce) | model | Apache-2.0 | https://huggingface.co/omlab/omdet-turbo-swin-tiny-hf |  |
| SAM 3 (facebook/sam3 @3c879f3) | model | **SAM License (Meta, custom)** | https://huggingface.co/facebook/sam3 | Gated on HF; custom non-OSI license (royalty-free use and redistribution under Meta's terms; trade-control restrictions) |
| transformers 5.22.2 | library | Apache-2.0 | https://pypi.org/project/transformers/5.22.2/ |  |
| pycocotools 2.0.11 | library | BSD-2-Clause (FreeBSD) | https://pypi.org/project/pycocotools/2.0.11/ |  |
| COCO val2017 (200-image fixed-seed subset + instances_val2017 annotations) | dataset | Annotations CC BY 4.0; images under Flickr terms of use | https://cocodataset.org/#download | Images fetched one by one from images.cocodataset.org |
| Ultralytics african-wildlife | dataset | **AGPL-3.0 (Ultralytics release)** | https://github.com/ultralytics/assets/releases/download/v0.0.0/african-wildlife.zip | Copyleft packaging; original image sources not documented (unclear provenance) |
| TUM RGB-D freiburg1_desk | dataset | CC BY 4.0 | https://cvg.cit.tum.de/data/datasets/rgbd-dataset |  |

## 05_segmentation

| Component | Kind | License | Source | Notes |
|---|---|---|---|---|
| SAM 3 (facebook/sam3, rev 3c879f3982) | model | **SAM License (custom Meta license, Nov 2025)** | https://huggingface.co/facebook/sam3 | GATED (manual approval). Custom license: commercial use permitted, but military/ITAR/weapons uses are banned, the license must ship with redistributions, and publications must acknowledge SAM. Flag as custom/restricted. |
| SAM 2.1 hiera tiny/small/base-plus (facebook/sam2.1-hiera-*) | model | Apache-2.0 | https://huggingface.co/facebook/sam2.1-hiera-small |  |
| Grounding DINO tiny (IDEA-Research/grounding-dino-tiny) | model | Apache-2.0 | https://huggingface.co/IDEA-Research/grounding-dino-tiny |  |
| YOLO26n-seg / YOLO26s-seg weights | model | **AGPL-3.0** | https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-seg.pt | Copyleft: networked/distributed use requires releasing source or an Ultralytics enterprise license. |
| ultralytics 8.4.204 | library | **AGPL-3.0** | https://github.com/ultralytics/ultralytics | Copyleft (AGPL). |
| transformers 5.22.2 (+ tokenizers 0.23.3, safetensors 0.8.0) | library | Apache-2.0 | https://github.com/huggingface/transformers |  |
| coco8-seg / coco128-seg (Ultralytics packaging of COCO train2017) | dataset | **COCO annotations CC BY 4.0; images under their Flickr licenses; the Ultralytics zip ships a GPL-3.0 LICENSE file** | https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128-seg.zip | Copyleft packaging license (GPL-3.0 per the LICENSE file in the zip); not redistributed here. |
| DAVIS 2017 TrainVal 480p (car-roundabout, dogs-jump only) | dataset | **Annotations described as CC BY 4.0 on the DAVIS challenge pages; frames from the original videos (research use); evaluation code BSD-3-Clause** | https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip | UNCLEAR: the current site states no explicit license. Not redistributed; only a few downscaled frames appear in figures and the GIF. |
| TUM RGB-D fr1_desk (one frame) | dataset | CC BY 4.0 | https://cvg.cit.tum.de/data/datasets/rgbd-dataset | Shared dataset (projects 08, 12, 15). |

## 06_tracking

| Component | Kind | License | Source | Notes |
|---|---|---|---|---|
| MOT17 (MOT17-02-FRCNN, MOT17-05-FRCNN images + GT) | dataset | **CC BY-NC-SA 3.0 (non-commercial, share-alike)** | https://motchallenge.net/data/MOT17/ | NON-COMMERCIAL, share-alike; fetched on demand via HTTP range reads of MOT17.zip, never redistributed |
| YOLO26n / YOLO26s detectors (yolo26n.pt, yolo26s.pt) | model | **AGPL-3.0** | https://github.com/ultralytics/assets/releases/tag/v8.4.0 | COPYLEFT (AGPL) |
| yolo26n-reid.onnx appearance encoder | model | **AGPL-3.0** | https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-reid.onnx | COPYLEFT (AGPL) |
| ultralytics 8.4.204 (ByteTrack, BoT-SORT, OC-SORT trackers) | library | **AGPL-3.0** | https://github.com/ultralytics/ultralytics | COPYLEFT (AGPL) |
| CoTracker3 code (pinned clone 82e02e8) incl. assets/apple.mp4 demo clip | code | **CC BY-NC 4.0 (non-commercial)** | https://github.com/facebookresearch/co-tracker | NON-COMMERCIAL |
| CoTracker3 weights scaled_offline.pth / scaled_online.pth (facebook/cotracker3) | model | **CC BY-NC 4.0 (non-commercial)** | https://huggingface.co/facebook/cotracker3 | NON-COMMERCIAL |
| SAM 3 (facebook/sam3) | model | **SAM License (Meta, custom)** | https://huggingface.co/facebook/sam3 | Gated (accept licence on HF); custom licence, check terms before commercial use |
| DINOv3 ViT-S/16 (facebook/dinov3-vits16-pretrain-lvd1689m) | model | **DINOv3 License (Meta, custom)** | https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m | Gated (accept licence on HF); custom licence https://ai.meta.com/resources/models-and-libraries/dinov3-license |
| transformers 5.22.2 (Sam3VideoModel, DINOv3) | library | Apache-2.0 | https://github.com/huggingface/transformers |  |
| trackeval 1.3.0 | library | MIT | https://pypi.org/project/trackeval/ | PyPI package from the kovalp/TrackEval fork of JonathonLuiten/TrackEval |
| lap 0.5.14 | library | BSD-2-Clause | https://github.com/gatagat/lap |  |
| onnxruntime-gpu 1.26.0 | library | MIT | https://github.com/microsoft/onnxruntime |  |

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

## 08_monocular_depth

| Component | Kind | License | Source | Notes |
|---|---|---|---|---|
| Depth Anything 3 code (ByteDance-Seed/Depth-Anything-3 @3d835ec) | code | Apache-2.0 | https://github.com/ByteDance-Seed/Depth-Anything-3 | pinned clone in gitignored third_party/, imported from source, not redistributed |
| DA3-SMALL weights | model | Apache-2.0 | https://huggingface.co/depth-anything/DA3-SMALL |  |
| DA3-BASE weights | model | Apache-2.0 | https://huggingface.co/depth-anything/DA3-BASE |  |
| DA3METRIC-LARGE weights | model | Apache-2.0 | https://huggingface.co/depth-anything/DA3METRIC-LARGE |  |
| Depth-Anything-V2-Small-hf | model | Apache-2.0 | https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf |  |
| Depth-Anything-V2-Metric-Indoor-Small-hf | model | **unclear** | https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf | UNCLEAR: the -hf repo has no license tag; the original Depth-Anything-V2-Metric-Hypersim-Small checkpoint repo is tagged Apache-2.0 |
| TUM RGB-D (fr1_desk, fr1_xyz) | dataset | CC BY 4.0 | https://cvg.cit.tum.de/data/datasets/rgbd-dataset | attribution required (Sturm et al., IROS 2012); small figures in assets/ show frames |
| transformers 5.22.2 | library | Apache-2.0 | https://github.com/huggingface/transformers |  |
| safetensors 0.8.0 | library | Apache-2.0 | https://github.com/huggingface/safetensors |  |
| evo 1.38.1 | library | **GPL-3.0** | https://github.com/MichaelGrupp/evo | COPYLEFT; imported by depth_anything_3.api at import time and used unmodified as a library |
| plyfile 1.1.5 | library | **GPL-3.0** | https://github.com/dranjan/python-plyfile | COPYLEFT; DA3 import-time dependency, used unmodified |
| pycolmap 4.2.1 | library | BSD-3-Clause | https://github.com/colmap/colmap |  |
| omegaconf 2.4.0 | library | BSD-3-Clause | https://github.com/omry/omegaconf |  |
| trimesh 5.2.0 | library | MIT | https://github.com/mikedh/trimesh |  |
| einops 0.8.2 | library | MIT | https://github.com/arogozhnikov/einops |  |
| addict 2.4.0 | library | MIT | https://github.com/mewwts/addict |  |
| imageio 2.38.1 | library | BSD-2-Clause | https://github.com/imageio/imageio |  |

## 09_multiview_geometry

| Component | Kind | License | Source | Notes |
|---|---|---|---|---|
| COLMAP / pycolmap 4.2.1 | library | BSD-3-Clause | https://github.com/colmap/colmap | CPU wheel from PyPI |
| kornia 0.9.0 (+ kornia_rs 0.2.0) | library | Apache-2.0 | https://github.com/kornia/kornia | DISK and LightGlue implementations |
| DISK weights (depth-save.pth) | model | Apache-2.0 | https://github.com/cvlab-epfl/disk | Downloaded by kornia DISK.from_pretrained('depth') from raw.githubusercontent.com/cvlab-epfl/disk; the HF kornia/disk card says MIT |
| LightGlue weights for DISK (disk_lightglue.pth, release v0.1_arxiv) | model | Apache-2.0 | https://github.com/cvg/LightGlue | Downloaded by kornia LightGlue('disk'); SuperPoint (non-commercial) deliberately not used |
| OpenCV 5.0 | library | Apache-2.0 | https://github.com/opencv/opencv | Core stack |
| Middlebury MiddEval3 quarter-resolution (data-Q + GT0-Q) | dataset | **Permission to use and publish images and disparity maps, with citation (Scharstein et al., GCPR 2014); no formal SPDX license** | https://vision.middlebury.edu/stereo/submit3/ | Citation required. |
| COLMAP South Building (128 photos + reference sparse model) | dataset | **UNCLEAR: none stated (research use; photos by C. Zach, distributed with COLMAP)** | https://github.com/colmap/colmap/releases/download/3.11.1/south-building.zip | Only downscaled figures are committed. |
| TUM RGB-D fr1_desk | dataset | CC BY 4.0 | https://cvg.cit.tum.de/data/datasets/rgbd-dataset | Shared with projects 08/12/15. |

## 10_feedforward_3d

| Component | Kind | License | Source | Notes |
|---|---|---|---|---|
| VGGT-1B weights (facebook/VGGT-1B) | model | **CC-BY-NC-4.0** | https://huggingface.co/facebook/VGGT-1B | NON-COMMERCIAL |
| VGGT code (facebookresearch/vggt @a288dd0) | code | VGGT License v1 (custom; commercial use allowed except military) | https://github.com/facebookresearch/vggt | pinned clone in gitignored third_party/, imported from source, not redistributed; custom license |
| DA3-BASE weights (depth-anything/DA3-BASE) | model | Apache-2.0 | https://huggingface.co/depth-anything/DA3-BASE |  |
| DA3-LARGE-1.1 weights (depth-anything/DA3-LARGE-1.1) | model | **CC-BY-NC-4.0 (treated as such)** | https://huggingface.co/depth-anything/DA3-LARGE-1.1 | NON-COMMERCIAL / UNCLEAR: HF card metadata says apache-2.0 but the DA3 GitHub README model table says CC BY-NC 4.0; treated as non-commercial |
| Depth Anything 3 code (ByteDance-Seed/Depth-Anything-3 @3d835ec) | code | Apache-2.0 | https://github.com/ByteDance-Seed/Depth-Anything-3 | pinned clone in gitignored third_party/ |
| pycolmap / COLMAP 4.2.1 | library | BSD-3-Clause | https://github.com/colmap/colmap |  |
| evo 1.38.1 | library | **GPL-3.0** | https://github.com/MichaelGrupp/evo | COPYLEFT; import-time dependency of depth_anything_3.api, used unmodified |
| plyfile 1.1.5 | library | **GPL-3.0** | https://github.com/dranjan/python-plyfile | COPYLEFT; import-time dependency of DA3, used unmodified |
| einops 0.8.2, safetensors 0.8.0, omegaconf 2.4.0, addict 2.4.0, trimesh 5.2.0, imageio 2.38.1 | library | MIT / Apache-2.0 / BSD | https://pypi.org/ | import-time deps of VGGT / DA3; permissive |
| Kitchen scene images (VGGT repo examples/kitchen, 25 PNG) | dataset | **UNCLEAR (shipped in the VGGT repository; scene matches Mip-NeRF 360 kitchen)** | https://github.com/facebookresearch/vggt/tree/a288dd0f14786c93483e45524328726ab7b1b4ce/examples/kitchen | UNCLEAR provenance/license; downloaded locally only, figures in assets/ |
| South Building (COLMAP dataset) | dataset | **UNCLEAR (research use)** | https://colmap.github.io/datasets.html (zip: https://github.com/colmap/colmap/releases/download/3.11.1/south-building.zip) | UNCLEAR; shared with project 09; downloaded locally only |
| TUM RGB-D fr1_desk | dataset | CC BY 4.0 | https://cvg.cit.tum.de/data/datasets/rgbd-dataset | attribution Sturm et al., IROS 2012; shared with projects 08/09/12/15 |

## 12_visual_odometry

| Component | Kind | License | Source | Notes |
|---|---|---|---|---|
| TUM RGB-D benchmark (fr1/xyz, fr1/desk) | dataset | CC BY 4.0 | https://cvg.cit.tum.de/data/datasets/rgbd-dataset | Attribution required (Sturm et al., IROS 2012). |
| evo 1.38.1 | library | **GPL-3.0** | https://github.com/MichaelGrupp/evo | COPYLEFT. Used only as an evaluation tool / test dependency (lazy import in vo.evo_metrics, evo_ape CLI); not part of the VO code path. |
| OpenCV 5.0 (ORB, SIFT, findEssentialMat/USAC_MAGSAC, recoverPose, solvePnPRansac/RefineLM, triangulatePoints, calcOpticalFlowPyrLK) | library | Apache-2.0 | https://github.com/opencv/opencv | SIFT patent expired in 2020; it is in the main package. |
| NumPy / SciPy / pandas / matplotlib | library | BSD-3-Clause (matplotlib: Matplotlib license, BSD-style) | https://numpy.org, https://scipy.org, https://pandas.pydata.org, https://matplotlib.org | Core stack. |
| seaborn, rosbags (evo transitive dependencies) | library | BSD-3-Clause / Apache-2.0 | https://pypi.org/project/seaborn/, https://pypi.org/project/rosbags/ | Installed by evo; not imported by project code. |

## 13_vision_language

| Component | Kind | License | Source | Notes |
|---|---|---|---|---|
| google/siglip2-base-patch16-224 | model | Apache-2.0 | https://huggingface.co/google/siglip2-base-patch16-224 | revision 75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2 |
| laion/CLIP-ViT-B-32-laion2B-s34B-b79K (open_clip ViT-B-32 laion2b_s34b_b79k) | model | MIT | https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K | trained on LAION-2B web data; revision 1a25a446712ba5ee05982a381eed697ef9b435cf |
| florence-community/Florence-2-base | model | MIT | https://huggingface.co/florence-community/Florence-2-base | native transformers port of microsoft/Florence-2-base; revision 00921df66db728a9ceb750f5eca43e5c203a2051 |
| Qwen/Qwen3.5-0.8B | model | Apache-2.0 | https://huggingface.co/Qwen/Qwen3.5-0.8B | revision 2fc06364715b967f1860aea9cf38778875588b17 |
| Imagenette-160 (fast.ai) | dataset | **Apache-2.0 (repository); images are an ImageNet subset under the ImageNet terms of access** | https://github.com/fastai/imagenette | NON-COMMERCIAL: ImageNet terms (research/education) apply to the images; shared with project 02 |
| TUM RGB-D fr1_desk | dataset | CC BY 4.0 | https://cvg.cit.tum.de/data/datasets/rgbd-dataset | attribution required (Sturm et al., IROS 2012); shared dataset |
| Ultralytics coco128 | dataset | **GPL-3.0 (LICENSE file shipped in the Ultralytics zip); COCO annotations CC BY 4.0; images under their Flickr licenses** | https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip | COPYLEFT packaging; mixed per-image Flickr licenses; COCO train2017 subset (part of Florence-2 training data) |
| transformers | library | Apache-2.0 | https://github.com/huggingface/transformers |  |
| open_clip_torch | library | MIT | https://github.com/mlfoundations/open_clip |  |
| timm | library | Apache-2.0 | https://github.com/huggingface/pytorch-image-models | dependency of open_clip |
| scikit-learn | library | BSD-3-Clause | https://github.com/scikit-learn/scikit-learn | core stack; used for the logistic-regression probe in the ex01 solution |

## 15_robotics_capstone

| Component | Kind | License | Source | Notes |
|---|---|---|---|---|
| IDEA-Research/grounding-dino-tiny (rev a2bb814) | model | Apache-2.0 | https://huggingface.co/IDEA-Research/grounding-dino-tiny |  |
| google/owlv2-base-patch16-ensemble (rev cfd3195) | model | Apache-2.0 | https://huggingface.co/google/owlv2-base-patch16-ensemble |  |
| facebook/sam3 (rev 3c879f3) | model | **SAM License (Meta custom license)** | https://huggingface.co/facebook/sam3 | Gated, custom non-OSI license with use restrictions. |
| facebook/sam2.1-hiera-small (rev ee5bba1) | model | Apache-2.0 | https://huggingface.co/facebook/sam2.1-hiera-small |  |
| depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf (rev 8078d68) | model | **Apache-2.0 (Small variant, per upstream README; HF card has no license tag)** | https://github.com/DepthAnything/Depth-Anything-V2 | The Base/Large metric variants are CC BY-NC 4.0 and are NOT used. |
| TUM RGB-D fr1/desk | dataset | CC BY 4.0 | https://cvg.cit.tum.de/data/datasets/rgbd-dataset | Cite Sturm et al., IROS 2012. |
| transformers 5.22.2 | library | Apache-2.0 | https://github.com/huggingface/transformers |  |
| rerun-sdk 0.42.0 | library | MIT OR Apache-2.0 | https://github.com/rerun-io/rerun | The PyPI license field is empty; the repo is dual-licensed. |
| pyarrow 26.0.1 (rerun dependency) | library | Apache-2.0 | https://pypi.org/project/pyarrow/ |  |
| scipy (KD-tree, convex hull, morphology) | library | BSD-3-Clause | https://scipy.org |  |
| ROS 2 Jazzy rclpy / sensor_msgs / geometry_msgs / message_filters (optional node) | library | Apache-2.0 | https://docs.ros.org/en/jazzy/ | System apt packages, used only by the optional ros2/ node. |
| vision_msgs 4.1.1 (optional node) | library | Apache-2.0 | https://github.com/ros-perception/vision_msgs |  |
