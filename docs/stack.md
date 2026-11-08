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

## 12_visual_odometry (checked 2026-11-08)

| Package / model | Version | Install / source |
|---|---|---|
| evo | 1.38.1 | pip install -c requirements/constraints.txt evo==1.38.1 (make setup-12_visual_odometry; already installed, verified no-op, pip check clean). Source: https://pypi.org/project/evo/ (checked 2026-11-07) |
| opencv-python (core stack) | 5.0.0.93 | requirements/base.txt. Source: https://pypi.org/project/opencv-python/ (checked 2026-11-07) |
| TUM RGB-D fr1/xyz + fr1/desk | freiburg1 sequences (~0.45 GB + ~0.34 GB tgz) | make data-12_visual_odometry (scripts/download_12_visual_odometry.py -> scripts/download_shared.tum()). Source: https://cvg.cit.tum.de/rgbd/dataset/freiburg1/ (checked 2026-11-07) |
