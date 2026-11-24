# NWPU-R C01 follow-up reproduction

The [experiment record](RESULTS_zh-CN.md) describes the main results. The [metric CSV](results/metrics.csv) stores AP as fractions and separates full training, frozen classification, and frozen ranking. The JSON files under [results](results/) contain the selected paired intervals, prediction counts, gradient audits, output swaps, and checkpoint provenance.

## Environment and inputs

The runs used Python 3.9, PyTorch 1.13.1+cu117, MMCV-full 1.7.1, and MMDetection 2.28.2. Use the repository setup and add the repository root to the Python path. Follow the [dataset preparation instructions](../../REPRODUCE.md) to place the 650 official training tiles and 339 official evaluation tiles under `data/NWPU-R`. These experiments use batch size 2, input size 800×800, seed 42 and rotated FCOS with a ResNet-50/FPN.

Place the shared complete initialization at `checkpoints/nwpu_r_shared_init.pth`. The [provenance file](results/provenance.json) records its SHA-256. The short runs require epoch-36 FIELD_P3 and FIELD_AGREE checkpoints from the full protocol. Exact replay depends on access to the archived initialization. This release contains source, configurations and aggregate results.

## Full training and shape evaluation

Each configuration under [c01_36](../../configs/nwpu_r/c01_36/) enables its objective from epoch 1. The saved comparison for FIELD_P3 was interrupted and continued from its epoch-18 checkpoint. The size coefficient in the published config comes from the original training-only calibration. Use the epoch-36 checkpoint for all reported evaluations.

```sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
method=FIELD_P3
cfg="configs/nwpu_r/c01_36/$method.py"
run="work_dirs/nwpu_r_c01/$method"
python tools/train.py "$cfg" --work-dir "$run" --seed 42 --deterministic --no-validate
python tools/test.py "$cfg" "$run/epoch_36.pth" --out "$run/predictions.pkl"
python -m research.nwpu_c01.score "$cfg" "$run/predictions.pkl" "$run/metrics.json"
```

The shape evaluator reports VOC07 rotated AP50 and AP75, class-level scores, and matched detection counts after NMS. Aspect bands use GT aspect ratio. The near-square cutoff is 1.2 and the elongated band begins above 2.0. GT outside the current band is treated as ignored in that band's comparison.

### Training objectives

| Configurations | Training difference |
|---|---|
| BD, C01 | Base BD and the original near-square P3 auxiliary respectively. |
| CONTENT_CHOICE, IOU_CHOICE | Angle candidates are selected using P3 or RIoU. |
| CONTENT_CHOICE_CENTER_SIZE | Content-selected angles with an IoU-weighted center and size branch. |
| NATIVE_ANGLE, SIZE_CHOICE | Separate native angle checks and log-size choices. |
| NATIVE_ANGLE_SIZE, NATIVE_IOU_ANGLE_SIZE | Same trial boxes, with content or pure RIoU selection. |
| FIELD_P3, FIELD_EDGE, FIELD_AGREE | Native IoU screens the trial step. The feature source is P3, fixed image edges, or P3 with edge-direction agreement. |
| BD_QUALITY, C01_QUALITY, C01_QUALITY_PERMUTED | Interpolate the centerness target toward matched RIoU. The permutation control reassigns it within centerness bins. |

## Frozen classifier and ranking

Run each command with a fresh output directory. The two arms of each command share the same augmented batches and frozen FIELD_P3 candidate boxes. Each arm uses three epochs and 975 steps. The runner verifies checkpoint epoch and iteration, the common start, and gradient isolation. It evaluates its two checkpoints on the official split after training. Add `--smoke` and use a separate output directory to check a single training batch.

```sh
p3=work_dirs/nwpu_r_c01/FIELD_P3/epoch_36.pth
agree=work_dirs/nwpu_r_c01/FIELD_AGREE/epoch_36.pth
python -m research.nwpu_c01.run_short classifier --p3 "$p3" --agree "$agree" \
  --work-dir work_dirs/nwpu_r_c01/frozen_classifier
python -m research.nwpu_c01.run_short rank --p3 "$p3" \
  --work-dir work_dirs/nwpu_r_c01/frozen_rank
```

Classification compares ordinary Focal Loss against Focal Loss plus teacher-probability BCE at selected locations. Ranking uses the log of the inference score product. The two ranking arms use equal classification-logit gradient budgets. The content arm upweights pairs where C01 feature residuals support the geometric ordering. See the [experiment record](RESULTS_zh-CN.md) for the comparison with GT-only and pointwise distillation.
