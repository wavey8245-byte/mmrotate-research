#!/usr/bin/env bash
# Reproduce the tested MMRotate 0.3.4 environment on an Ubuntu 22.04
# Miniconda image. Run from the transferred mmrotate034-src directory.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

if ! command -v conda >/dev/null 2>&1; then
    echo 'Conda is unavailable; select a Miniconda image or install Conda first.' >&2
    exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo 'nvidia-smi is unavailable; enable the server GPU before installing.' >&2
    exit 1
fi
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader

conda_base="$(conda info --base)"
# shellcheck source=/dev/null
source "$conda_base/etc/profile.d/conda.sh"

if conda env list --json | python3 -c \
    'import json,sys; print(any(p.rsplit("/",1)[-1] == "mmrotate034" for p in json.load(sys.stdin)["envs"]))' \
    | grep -qx True; then
    echo 'Conda environment mmrotate034 already exists; refusing to overwrite it.' >&2
    exit 1
fi

conda create -y -n mmrotate034 python=3.9
conda activate mmrotate034
conda install -y pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.7 \
    -c pytorch -c nvidia

python -m pip install --upgrade pip
python -m pip install 'numpy==1.23.5' 'setuptools==80.9.0' \
    'yapf==0.40.1' 'opencv-python==4.8.1.78'
python -m pip install 'mmcv-full==1.7.1' --only-binary=mmcv-full \
    -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/index.html
python -m pip install 'mmdet==2.28.2' 'e2cnn==0.2.3' \
    'pycocotools==2.0.11' 'terminaltables==3.1.10'
python -m pip install -e . --no-deps

python - <<'PY'
import mmcv
import mmdet
import mmrotate
import torch
from mmcv.ops import box_iou_rotated

assert torch.__version__.startswith('1.13.1'), torch.__version__
assert torch.version.cuda == '11.7', torch.version.cuda
assert mmcv.__version__ == '1.7.1', mmcv.__version__
assert mmdet.__version__ == '2.28.2', mmdet.__version__
assert mmrotate.__version__ == '0.3.4', mmrotate.__version__
assert torch.cuda.is_available(), 'GPU is not available to PyTorch'
boxes = torch.tensor([[0., 0., 10., 2., 0.]], device='cuda')
overlap = box_iou_rotated(boxes, boxes)
assert torch.allclose(overlap, torch.ones_like(overlap)), overlap
conv = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1).cuda()
conv(torch.randn(1, 3, 64, 64, device='cuda')).sum().backward()
print('Environment OK:', torch.__version__, torch.version.cuda,
      mmcv.__version__, mmdet.__version__, mmrotate.__version__)
print('GPU:', torch.cuda.get_device_name(0), 'compute capability:',
      torch.cuda.get_device_capability(0))
PY
