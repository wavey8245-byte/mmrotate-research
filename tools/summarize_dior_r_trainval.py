"""Summarize fixed-epoch test results for the server trainval protocol."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = {
    'riou': ('Rotated IoU (-log)', 'fcos_riou_bs2_800'),
    'smooth_l1': ('编码参数 Smooth L1', 'fcos_smooth_l1_encoded_bs2_800'),
    'wf': ('WF 旋转扩展', 'fcos_wf_bs2_800'),
    'riou_linear': ('线性 IoU（可选消融）', 'fcos_riou_linear_bs2_800'),
}


def main():
    directory = ROOT / 'work_dirs/dior_r_trainval'
    directory.mkdir(parents=True, exist_ok=True)
    results = {}
    lines = ['# 服务器 trainval / test 实验结果', '',
             '训练：train+val=11,725；评估：test=11,738。',
             '共同设置：单卡 batch 2，800×800，SGD lr=0.0025，seed=42，12 轮。',
             '按预先确定的 epoch_12.pth 报告结果，不按 test 选择最佳轮数。', '',
             '| 实验 | 权重 | 完整 test mAP |', '| --- | --- | ---: |']
    for variant, (label, folder) in EXPERIMENTS.items():
        experiment = directory / folder
        files = sorted((experiment / 'test_final').glob('eval_*.json'),
                       key=lambda p: p.stat().st_mtime)
        if not files:
            if variant != 'riou_linear':
                lines.append(f'| {label} | epoch_12.pth | 待完成 |')
            continue
        score = float(json.loads(files[-1].read_text())['metric']['mAP'])
        if not math.isfinite(score):
            raise RuntimeError(f'{variant} test mAP 无效：{files[-1]}')
        results[variant] = dict(test_mAP=score, checkpoint=str(experiment / 'epoch_12.pth'),
                                metrics=str(files[-1]))
        lines.append(f'| {label} | epoch_12.pth | {score * 100:.2f}% |')
    lines += ['', '此协议没有独立验证集。test 也参与逐轮评估。',
              '这些结果不能直接与 train=5,862 / val=5,863 的旧协议比较。',
              '数据划分对齐 MMRotate 1.x DIOR 配置；模型及实现为本项目的 MMRotate 0.3.4 FCOS。', '']
    (directory / 'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
    (directory / 'results.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
