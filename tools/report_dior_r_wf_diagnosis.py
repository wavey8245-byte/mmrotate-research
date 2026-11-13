"""Audit saved configs and report the WF angle counterexample without training."""
import json
from pathlib import Path

import mmcv
import torch

from mmrotate.models.losses.rotated_wf_iou_loss import (
    rotated_wf_geometry, wf_weighting_factor)

torch.set_num_threads(2)
root = Path('work_dirs/dior_r_trainval')
destination = root / 'wf_diagnosis.json'
report = json.loads(destination.read_text())
wf = mmcv.Config.fromfile(str(root / 'fcos_wf_bs2_800/wf.py'))
base = mmcv.Config.fromfile(str(root / 'fcos_riou_bs2_800/riou.py'))
keys = ('data', 'optimizer', 'optimizer_config', 'lr_config', 'runner',
        'evaluation', 'seed', 'load_from', 'resume_from')
report['saved_configuration_audit'] = {key: wf[key] == base[key] for key in keys}
wf_model = dict(wf.model)
base_model = dict(base.model)
wf_model['bbox_head'] = dict(wf.model.bbox_head)
base_model['bbox_head'] = dict(base.model.bbox_head)
wf_model['bbox_head'].pop('loss_bbox')
base_model['bbox_head'].pop('loss_bbox')
report['saved_configuration_audit']['model_except_bbox_loss'] = wf_model == base_model
assert all(report['saved_configuration_audit'].values())

target = torch.tensor([[0., 0., 10., 2., 0.]])
checks = []
for degrees in (25, 30, 35, 40, 43, 44, 44.5, 45, 46, 50):
    pred = target.clone()
    pred[:, 4] = degrees * torch.pi / 180
    pred.requires_grad_()
    iou, ratio = rotated_wf_geometry(pred, target)
    factor = wf_weighting_factor(iou, ratio)
    loss = (1 - factor*iou).clamp(min=0)
    derivative = torch.autograd.grad(loss.sum(), pred)[0][0, 4]
    checks.append(dict(angle_degrees=degrees, iou=float(iou), ratio=float(ratio),
                       weight=float(factor), loss=float(loss),
                       loss_angle_derivative_per_radian=float(derivative)))
report['angle_gradient_counterexample'] = checks
destination.write_text(json.dumps(report, indent=2) + '\n')

lines = [
    '# DIOR-R trainval：WF 失败诊断', '',
    '完整 test：对数 Rotated IoU 60.83%，编码 Smooth L1 49.76%，WF 7.00%。', '',
    '已核对实际保存的 WF 与基线配置：数据、优化器、学习率调度、轮数、评估、种子、'
    '恢复设置和除回归损失外的模型结构完全一致。', '',
    '## 相同真实训练样本的检查', '',
    '使用固定种子抽取 16 张 trainval 图片、关闭翻转，得到 298 个正样本。'
    '各权重检查同一批图片，没有优化器更新。结果只代表此次抽样，不是全数据集统计。', '',
    '| 权重 | 平均旋转 IoU | 平均规范角度误差 | 零宽高框比例 |',
    '| --- | ---: | ---: | ---: |',
]
labels = {'trainval_log_iou_epoch12': '本次基线 epoch12',
          'old_split_wf_epoch12': '旧划分 WF epoch12',
          'trainval_wf_epoch1': '本次 WF epoch1',
          'trainval_wf_epoch4': '本次 WF epoch4',
          'trainval_wf_epoch12': '本次 WF epoch12'}
for key, label in labels.items():
    value = report['arms'][key]
    lines.append(f"| {label} | {value['mean_iou']:.3f} | "
                 f"{value['mean_canonical_angle_error_degrees']:.2f}° | "
                 f"{value['zero_wh_fraction']*100:.2f}% |")
lines += ['', '旋转 IoU 与 MMCV 参考算子的最大误差小于 1e-6；抽样未发现零宽高框。'
          '这不排除其它样本或训练阶段的数值问题。', '',
          '## 可复现的错误角度低损失', '',
          '两个中心重合、尺寸均为 10×2 的旋转框，只改变预测框角度：', '',
          '| 角度误差 | IoU | WF 损失 | 损失对角度导数（每弧度） |',
          '| --- | ---: | ---: | ---: |']
for value in checks:
    lines.append(f"| {value['angle_degrees']}° | {value['iou']:.4f} | "
                 f"{value['loss']:.4f} | {value['loss_angle_derivative_per_radian']:.4f} |")
lines += ['', '当 IoU<0.3 时，当前公式使用 min(6,1/R) 放大 IoU。'
          '旋转扩展的 R=d²/c² 可能比 IoU 下降更快，因此放大的 IoU 反而增大，'
          '损失随角度误差增大而降低。约 45° 附近出现较差定位的局部低损失区域。', '',
          '论文式(9)–(11)与当前权重分支相符，论文没有给出旋转实现；'
          '本项目定义的旋转几何扩展不能据此视为经过验证。', '',
          '该反例与本次抽样约 46° 的角度误差吻合，是训练失败的重要机制证据；'
          '尚未通过改变单一因素的受控训练证明它是全部 mAP 下降的唯一原因。', '',
          '## 后续验证', '',
          '先进行短程、相同初始权重与相同 batch 的对照，比较原 WF、无 WF 线性 IoU，'
          '以及明确标记为新变体的梯度处理/几何定义。关注实际 IoU、角度误差和梯度方向，'
          '通过后才重跑完整训练。不能把改动后的损失冒充论文原式。', '',
          '本次只添加诊断工具与报告，没有修改生产损失、配置或已有训练权重，'
          '也没有启动新的完整实验。', '']
(root / 'wf_diagnosis.md').write_text('\n'.join(lines), encoding='utf-8')
print(json.dumps(report['saved_configuration_audit']))
print(json.dumps(checks))
print(root / 'wf_diagnosis.md')
