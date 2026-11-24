# Reproduce the RIoU and original C01 experiments

The November official 36-epoch protocol and the frozen-classifier follow-ups have separate [configurations and commands](research/nwpu_c01/REPRODUCE.md) and a [results record](research/nwpu_c01/RESULTS_zh-CN.md).

The configurations use Rotated FCOS, ResNet-50 with an ImageNet initialization, FPN, 800×800 input, batch size 2, and 12 epochs. Use seed 42 and evaluate the epoch 12 checkpoint. The RIoU baseline uses `-log(RIoU)`. C01 adds a P3 feature comparison to BD with 5×5 sampling, weight 0.1, and a ground-truth aspect ratio threshold of 1.2. Its auxiliary gradient acts directly on the decoded angle.

## Environment

The experiments use Linux/WSL, Python 3.9, PyTorch 1.13.1 with CUDA 11.7, MMCV-full 1.7.1, MMDetection 2.28.2, and MMRotate 0.3.4. For a new Conda environment on a CUDA machine, run `bash tools/setup_dior_r_server_env.sh` from the repository root and activate `mmrotate034`. For an existing compatible environment, run `python -m pip install -e . --no-deps`.

## Datasets and configuration paths

| Dataset | RIoU | Original C01 |
|---|---|---|
| DIOR-R | `configs/dior_r/server_trainval/riou.py` | `configs/dior_r/server_trainval/bd_square_angle_content.py` |
| NWPU-R | `configs/nwpu_r/riou.py` | `configs/nwpu_r/c01.py` |

Obtain DIOR-R from its [dataset authors](https://gcheng-nwpu.github.io/#Datasets). Arrange the train/val images in `data/DIOR-R/JPEGImages-trainval/`, test images in `data/DIOR-R/JPEGImages-test/`, oriented annotation XMLs in `data/DIOR-R/Annotations/Oriented Bounding Boxes/`, and the `train.txt`, `val.txt`, `test.txt` ID lists in `data/DIOR-R/ImageSets/Main/`. The training split combines train and val.

Obtain the NWPU-R tiles from the [AOFS release](https://github.com/lalalayong/AOFS). Put them under `data/NWPU-R/`, with `training/{images,labelTxt}` and `evaluation/{images,labelTxt}`. The included dataset reader uses JPEG images with DOTA-style polygon labels. The tested split uses 650 training tiles and 339 evaluation tiles, including tiles with empty annotations. Evaluation is at tile level.

## Train and evaluate

Run from the repository root. Set `config` to either configuration in the table and `run_dir` to a separate output directory for each method.

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
config=configs/nwpu_r/c01.py
run_dir=work_dirs/nwpu_r/c01
python tools/train.py "$config" --work-dir "$run_dir" --seed 42 --no-validate
python tools/test.py "$config" "$run_dir/epoch_12.pth" \
  --out "$run_dir/predictions.pkl" --eval mAP --eval-options iou_thr=0.5
python tools/test.py "$config" "$run_dir/epoch_12.pth" \
  --eval mAP --eval-options iou_thr=0.75
```

The evaluator reports rotated VOC 2007 mAP at the requested IoU threshold. The NWPU-R RIoU/C01 comparison changes the complete box regression objective, including both the BD base loss and C01's auxiliary term.
