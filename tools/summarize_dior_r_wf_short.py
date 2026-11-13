"""Verify cross-run initialization/input equality and summarize short controls."""
import json
from pathlib import Path

root = Path('work_dirs/dior_r_trainval')
control = json.loads((root / 'iou_short_control.json').read_text())
candidate = json.loads((root / 'wf_aux_short_control.json').read_text())
assert control['steps'] == candidate['steps']
assert control['initial_state_sha256'] == candidate['initial_state_sha256']
assert control['anchor_train_ids'] == candidate['anchor_train_ids']
reference = control['arms']['log_iou']['augmented_input_hashes']
for result in list(control['arms'].values()) + list(candidate['arms'].values()):
    assert result['augmented_input_hashes'] == reference
    assert result['identical_input_batches']
labels = {'log_iou': '官方对数 IoU', 'linear_iou': '官方线性 IoU',
          'original_wf': '原 WF 旋转扩展', 'wf_aux_log': '对数 IoU + 0.1 WF 辅助项'}
all_results = {**control['arms'], **candidate['arms']}
lines = [
    '# WF 处理方案：短程验证结果', '',
    f"每组 {control['steps']} 步，trainval=11,725，batch=2、800×800、SGD lr=0.0025、seed=42。",
    '初始化权重 SHA256 完全一致；三组主对照逐步核对图片顺序；'
    '候选与主对照每 50 步抽查的增强输入张量 SHA256 完全一致。'
    '各组重置优化器，并使用同样的 500 步 warmup。', '',
    '训练后在同一批固定增强的 16 张训练图片上检查。没有评估 test 调参。', '',
    '| 方法 | 正样本数 | 平均旋转 IoU | 平均规范角度误差 | 零宽高比例 |',
    '| --- | ---: | ---: | ---: | ---: |']
for key, value in all_results.items():
    metrics = value['anchor_metrics']
    lines.append(f"| {labels[key]} | {value['anchor_positives']} | "
                 f"{metrics['mean_iou']:.3f} | {metrics['mean_angle_error_degrees']:.2f}° | "
                 f"{metrics['zero_wh_fraction']*100:.2f}% |")
lines += ['', '**限制：这些是训练样本上的早期诊断，不是 mAP，不是独立验证结果。**'
          '500 步刚完成 warmup，不能证明最终收敛或精度提升，也未复现完整训练 7% 的所有原因。', '',
          '原 WF 的错误角度零损失反例仍然成立。短程一次运行没有崩溃，不能排除后续训练'
          '进入错误低损失区域，也不能证明它是本次完整失败的唯一原因。', '',
          '候选损失为 -log(RotatedIoU) + 0.1×L_WF，是本项目的新变体。'
          '它通过已知反例的非零损失与角度恢复梯度检查，辅助系数为零时与官方基线完全一致，'
          '并通过 centerness 加权检查。不能标为论文原式，也不能据此保证超过 60.83%。', '',
          '推荐保留原完整结果，先延长受控训练并检查定位收敛，再决定是否对候选做完整 12 轮实验。'
          '所有完整结果仍固定报告 epoch_12.pth。', '',
          '详细处理说明：configs/dior_r/server_trainval/wf_recovery.md。', '']
destination = root / 'wf_short_control_summary.md'
destination.write_text('\n'.join(lines), encoding='utf-8')
print('\n'.join(lines))
