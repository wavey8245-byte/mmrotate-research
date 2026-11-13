#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
if [[ -d /usr/lib/wsl/lib ]]; then
    export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

resolve_variant() {
    case "$1" in
        riou) experiment_dir=work_dirs/dior_r_trainval/fcos_riou_bs2_800 ;;
        smooth_l1) experiment_dir=work_dirs/dior_r_trainval/fcos_smooth_l1_encoded_bs2_800 ;;
        wf) experiment_dir=work_dirs/dior_r_trainval/fcos_wf_bs2_800 ;;
        riou_linear) experiment_dir=work_dirs/dior_r_trainval/fcos_riou_linear_bs2_800 ;;
        *) echo "Unknown experiment: $1" >&2; exit 2 ;;
    esac
    config="configs/dior_r/server_trainval/$1.py"
}

train_one() {
    resolve_variant "$1"
    mkdir -p "$experiment_dir"
    if compgen -G "$experiment_dir/*.pth" > /dev/null; then
        echo "已有训练权重，停止以避免覆盖：$experiment_dir" >&2
        return 1
    fi
    printf '[%s] train:%s\n' "$(date -Is)" "$1"
    python tools/train.py "$config" --seed 42
    [[ -s "$experiment_dir/epoch_12.pth" ]]
}

test_one() {
    resolve_variant "$1"
    if [[ -e "$experiment_dir/test_final_predictions.pkl" ]]; then
        echo "已有 test 输出，停止以避免覆盖：$experiment_dir" >&2
        return 1
    fi
    printf '[%s] test:%s, checkpoint=epoch_12.pth\n' "$(date -Is)" "$1"
    python tools/test.py "$config" "$experiment_dir/epoch_12.pth" \
        --work-dir "$experiment_dir/test_final" \
        --out "$experiment_dir/test_final_predictions.pkl" --eval mAP
    python tools/summarize_dior_r_trainval.py
}

action="${1:-check}"
variant="${2:-riou}"
if [[ "$action" == train || "$action" == test || "$action" == all ]]; then
    mkdir -p work_dirs/dior_r_trainval
    # Hold one common lock for individual actions and the complete queue.
    exec 9>work_dirs/dior_r_trainval/server_queue.lock
    flock -n 9 || { echo '已有服务器实验任务运行中。' >&2; exit 1; }
fi
case "$action" in
    check) python tools/check_dior_r_trainval.py ;;
    train) train_one "$variant" ;;
    test) test_one "$variant" ;;
    all)
        mkdir -p work_dirs/dior_r_trainval
        python tools/check_dior_r_trainval.py
        for variant in riou smooth_l1 wf; do
            train_one "$variant"
            test_one "$variant"
        done
        printf '[%s] 三组训练与最终 test 已完成。\n' "$(date -Is)"
        ;;
    *) echo 'Usage: bash tools/run_dior_r_trainval.sh {check|train|test|all} [riou|smooth_l1|wf|riou_linear]' >&2; exit 2 ;;
esac
