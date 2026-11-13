"""Summarize matched training-only paper-loss controls without ranking test AP."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    if report['status'] != 'completed_short_controls':
        raise RuntimeError('The matched controls have not all completed')
    arms = report['arms']
    steps_per_arm = report['steps_per_arm']
    arm_count = len(arms)
    experiment_date = report['created_at'][:10]
    resume_step = report.get('resume_step')
    first = next(iter(arms.values()))
    for name, arm in arms.items():
        if arm['status'] != 'completed_short_probe':
            raise RuntimeError('Incomplete arm: ' + name)
        if arm['augmented_batch_sha256'] != first['augmented_batch_sha256']:
            raise RuntimeError('Augmented inputs differ: ' + name)
        if arm['anchors_before'] != first['anchors_before']:
            raise RuntimeError('Initial localization differs: ' + name)
    destination = args.report.parent / 'results.md'
    figure_path = args.report.parent / 'diagnostics.png'
    if destination.exists() or figure_path.exists():
        raise FileExistsError('Existing summaries will not be overwritten')
    names = dict(riou='原 RIoU', dcagpl='DCAGPL', bd='普通 BD', bd_agbb='AGBB + BD')
    colors = dict(riou='#64748b', dcagpl='#087e83', bd='#c36b26', bd_agbb='#744aa4')
    english = dict(riou='RIoU', dcagpl='DCAGPL', bd='BD', bd_agbb='AGBB + BD')
    lines = ['# DIOR-R / Rotated FCOS：论文损失短训对照', '',
             f'日期：{experiment_date}。{arm_count} 组从相同初始化分别训练 '
             f"{report['steps_per_arm']} 步，800×800、batch 2、SGD，沿用 500 步线性 warmup。",
             '', '本报告只用于接入和初期学习验证。固定图片来自训练集，没有读取 test 成绩；'
             '训练图片上的诊断 AP 不能当作测试集 mAP，也不能据此宣称论文方法取得精度提升。', '',
             '## 主要结果', '',
             '| 损失 | 固定图片加权 IoU：训练前 → 后 | 固定图片 IoU≥0.5 比例 | '
             '固定图片零边长比例 | 末100步梯度中位数 / P95 | 原始角度绝对值 P95（末批次） |',
             '| --- | --- | --- | --- | --- | --- |']
    for name, arm in arms.items():
        before, after = arm['anchors_before'], arm['anchors_after']
        lines.append(f"| {names[name]} | {before['weighted_iou']:.4f} → "
                     f"{after['weighted_iou']:.4f} | {100*after['iou_ge_half_fraction']:.2f}% | "
                     f"{100*after['zero_wh_fraction']:.2f}% | "
                     f"{arm['last100_gradient_median']:.2f} / {arm['last100_gradient_p95']:.2f} | "
                     f"{arm['history'][-1]['angle_p95']:.3f} rad |")
    if resume_step is not None:
        lines += ['', f'## 从 {resume_step} 步继续训练的变化', '',
                  '| 损失 | 续训前加权 IoU | 续训后加权 IoU | 续训前 IoU≥0.5 比例 | 续训后 IoU≥0.5 比例 |',
                  '| --- | --- | --- | --- | --- |']
        for name, arm in arms.items():
            before, after = arm['anchors_at_resume'], arm['anchors_after']
            lines.append(f"| {names[name]} | {before['weighted_iou']:.4f} | "
                         f"{after['weighted_iou']:.4f} | {100*before['iou_ge_half_fraction']:.2f}% | "
                         f"{100*after['iou_ge_half_fraction']:.2f}% |")
        lines += ['', f'从各自 {resume_step} 步权重恢复模型和优化器，'
                  f'逐批重放并核对前 {resume_step} 批图像哈希；恢复时固定图片定位指标及诊断 AP '
                  '与原实验一致。续训沿用全局步数设置学习率，没有重新 warmup。']
    lines += ['', '加权 IoU 使用原 FCOS 的真值 centerness 权重。IoU≥0.5 比例按正样本数统计。'
              '角度 P95 是网络原始周期角输出的幅度，角度幅度较大不直接等于几何误差较大，'
              '应结合解码 IoU 判断。不同损失的数值尺度不同，不以回归损失的绝对值排名。', '',
              '## 训练图片诊断 AP 与资源', '',
              '| 损失 | 16张训练图片 AP50：前 → 后 | 时长（秒） | 峰值张量显存（MiB） |',
              '| --- | --- | --- | --- |']
    for name, arm in arms.items():
        lines.append(f"| {names[name]} | {100*arm['train_ap_before']:.2f}% → "
                     f"{100*arm['train_ap_after']:.2f}% | {arm['elapsed_seconds']:.1f} | "
                     f"{arm['peak_memory_allocated_mib']:.1f} |")
    lines += ['', '诊断 AP 的类别覆盖和样本数很小，只用于发现完全未学习或明显异常。'
              f'{steps_per_arm} 步还不足以完成训练或比较最终精度。峰值显存为 PyTorch 已分配张量，'
              '不包括显示占用、驱动及所有缓存；与 nvidia-smi 总占用口径不同。', '',
              '## 对照一致性', '',
              f'- {arm_count} 组具有相同的初始化状态，状态 SHA256：`' + report['initial_state_sha256'] + '`。',
              f'- {arm_count} 组逐批增强后图像的 SHA256 全部一致；固定诊断图片一致。',
              '- 数据、模型结构、优化器、warmup、梯度裁剪与基线对照通过。',
              '- 运行期间已记录的源码哈希未变化：' + str(report['sources_unchanged']) + '。',
              f'- 每组保存诊断权重，文件名为 `<arm>_step_{steps_per_arm}.pth`，不是第12轮最终权重。',
              '- 详细配置、历史指标、图像标识、初始状态及源码 SHA256 见 `report.json` 和 `snapshot/`。', '',
              '## 公式与适配边界', '',
              '本次实现是按论文编写的独立移植，不是作者官方代码。BD 的 alpha=3；AGBB 的 delta=5，'
              '近正方形阈值 1.1 是显式适配选择。DCAGPL 保留原式中心项和负对数 KFIoU。'
              '没有改动公式来追求短训结果。三组回归权重均沿用当前 FCOS 的 1。', '',
              'FrSIoU 的语义特征和历史机制仍未接入；CRL 原式的指数项/概率范围问题仍未解决，'
              '此次没有将 EVCS 的猜测修正版加入训练。', '',
              '## 诊断图', '', '![短训诊断](diagnostics.png)', '']
    if resume_step is not None:
        lines += ['', '资源统计说明：时长为原短训与本次续训两段之和，包含数据重放和诊断；'
                  '显存数值是本次续训段的峰值，不是两次进程的合并峰值。', '']
    destination.write_text('\n'.join(lines), encoding='utf-8')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.grid': True,
                         'grid.alpha': .18, 'figure.facecolor': 'white'})
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), constrained_layout=True)
    for name, arm in arms.items():
        steps = [r['step'] for r in arm['history']]
        axes[0, 0].plot(steps, [r['weighted_iou'] for r in arm['history']],
                        color=colors[name], label=english[name], linewidth=1.8)
        axes[0, 1].plot(steps, [r['grad_norm'] for r in arm['history']],
                        color=colors[name], linewidth=1.8)
        axes[1, 0].plot(steps, [r['angle_p95'] for r in arm['history']],
                        color=colors[name], linewidth=1.8)
    axes[0, 0].set(title='Decoded IoU on matched training batches',
                    xlabel='Optimizer step', ylabel='Centerness-weighted IoU', ylim=(0, 1))
    axes[0, 0].legend(frameon=False)
    axes[0, 1].set(title='Gradient norm before clipping', xlabel='Optimizer step',
                    ylabel='L2 norm (log scale)', yscale='log')
    axes[0, 1].axhline(35, color='#111827', ls='--', lw=1, label='Clip threshold 35')
    axes[0, 1].legend(frameon=False)
    axes[1, 0].set(title='Raw angle magnitude (not angular error)',
                    xlabel='Optimizer step', ylabel='Absolute angle P95 (rad, log scale)',
                    yscale='log')
    x = np.arange(len(arms))
    width = .24 if resume_step else .34
    axes[1, 1].bar(x-width/2 if not resume_step else x-width,
                   [a['anchors_before']['weighted_iou'] for a in arms.values()],
                   width=width, color='#cbd5e1', label='Step 0')
    if resume_step:
        axes[1, 1].bar(x, [a['anchors_at_resume']['weighted_iou'] for a in arms.values()],
                       width=width, color='#94a3b8', label=f'Step {resume_step}')
        for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
            ax.axvline(resume_step, color='#94a3b8', ls=':', lw=1)
    axes[1, 1].bar(x+width/2 if not resume_step else x+width,
                   [a['anchors_after']['weighted_iou'] for a in arms.values()],
                   width=width, color=[colors[n] for n in arms], label=f'Step {steps_per_arm}')
    axes[1, 1].set(title='Fixed training anchors: progression',
                    ylabel='Centerness-weighted IoU', ylim=(0, 1), xticks=x,
                    xticklabels=[english[n] for n in arms])
    axes[1, 1].legend(frameon=False)
    fig.suptitle(f'DIOR-R / Rotated FCOS: {steps_per_arm}-step matched controls\n'
                 'Training-image diagnostics only; no test-set performance claims', fontsize=15)
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    print(destination)
    print(figure_path)


if __name__ == '__main__':
    main()
