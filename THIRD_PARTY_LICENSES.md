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

## 12_visual_odometry

| Component | Kind | License | Source | Notes |
|---|---|---|---|---|
| TUM RGB-D benchmark (fr1/xyz, fr1/desk) | dataset | CC BY 4.0 | https://cvg.cit.tum.de/data/datasets/rgbd-dataset | Attribution required (Sturm et al., IROS 2012). |
| evo 1.38.1 | library | **GPL-3.0** | https://github.com/MichaelGrupp/evo | COPYLEFT. Used only as an evaluation tool / test dependency (lazy import in vo.evo_metrics, evo_ape CLI); not part of the VO code path. |
| OpenCV 5.0 (ORB, SIFT, findEssentialMat/USAC_MAGSAC, recoverPose, solvePnPRansac/RefineLM, triangulatePoints, calcOpticalFlowPyrLK) | library | Apache-2.0 | https://github.com/opencv/opencv | SIFT patent expired in 2020; it is in the main package. |
| NumPy / SciPy / pandas / matplotlib | library | BSD-3-Clause (matplotlib: Matplotlib license, BSD-style) | https://numpy.org, https://scipy.org, https://pandas.pydata.org, https://matplotlib.org | Core stack. |
| seaborn, rosbags (evo transitive dependencies) | library | BSD-3-Clause / Apache-2.0 | https://pypi.org/project/seaborn/, https://pypi.org/project/rosbags/ | Installed by evo; not imported by project code. |
