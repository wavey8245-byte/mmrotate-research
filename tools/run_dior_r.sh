#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# WSL cuDNN loads libcuda dynamically; torch.cuda.is_available alone is
# insufficient to detect this missing loader search path.
if [[ -d /usr/lib/wsl/lib ]]; then
    export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
action="${1:-check}"
variant="${2:-riou}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ $# -gt 0 ]]; then shift; fi
case "$variant" in
    riou) config=configs/dior_r/rotated_fcos_r50_fpn_1x_dior_r_le90.py ;;
    riou_bs2) config=configs/dior_r/rotated_fcos_r50_fpn_1x_dior_r_le90_bs2.py ;;
    riou_linear_bs2) config=configs/dior_r/rotated_fcos_r50_fpn_1x_dior_r_le90_riou_linear_bs2.py ;;
    smooth_l1) config=configs/dior_r/rotated_fcos_r50_fpn_1x_dior_r_le90_smooth_l1.py ;;
    smooth_l1_bs2) config=configs/dior_r/rotated_fcos_r50_fpn_1x_dior_r_le90_smooth_l1_bs2.py ;;
    wf_bs2) config=configs/dior_r/rotated_fcos_r50_fpn_1x_dior_r_le90_wf_bs2.py ;;
    *) echo "Unknown loss: $variant" >&2; exit 2 ;;
esac
case "$action" in
    check) exec python tools/check_dior_r.py --config "$config" \
        --report "work_dirs/dior_r/smoke_${variant}.json" "$@" ;;
    train) exec python tools/train.py "$config" --seed 42 "$@" ;;
    test) exec python tools/test.py "$config" "$@" --eval mAP ;;
    *) echo 'Usage: bash tools/run_dior_r.sh {check|train|test} {riou|riou_bs2|riou_linear_bs2|smooth_l1|smooth_l1_bs2|wf_bs2} [args]' >&2; exit 2 ;;
esac
