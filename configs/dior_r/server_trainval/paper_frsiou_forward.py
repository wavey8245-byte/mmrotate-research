# 当前新版 FrSIoU 配置入口：继承纯 SIoU 的语义设置，再增加历史调制。
# 逐框当前分数 + 冻结组历史属于本项目适配，不宣称作者完整复现。
_base_ = ['./paper_semantic_siou.py']

# Audit 2026-11-01: per-box current q; no group-current weighting or tangent.
# Eq.(12) resolves the inconsistent recurrence printed as Eq.(11).
# Frozen history: previous visits of image/GT/FPN groups, zero at cold start.
# nu=.5, M=2, FPN/5x5/1.5 context and history identity are adaptation choices.
# This is NOT a verified reproduction of the paper's Eq.(20) or author code.
custom_imports = dict(imports=[
    'mmrotate.models.losses.fractional_semantic_iou_loss',
    'mmrotate.models.dense_heads.fractional_rotated_fcos_head',
    'mmrotate.core.frsiou_memory_hook'], allow_failed_imports=False)
# 新版类在构造时锁定 per_box + values；nu=0.5、历史长度 M=2。
# 每组最近两次访问的 q 均值参与调制；最多保存 200000 个组。
model = dict(bbox_head=dict(
    type='FractionalRotatedFCOSHead',
    loss_bbox=dict(type='FCOSFrSIoUForwardLoss', fractional_order=.5,
                   memory_steps=2, modulation_lambda=.3,
                   max_history_keys=200000)))
# 优化器更新后提交历史；保存每轮模型时另存匹配历史文件。
custom_hooks = [dict(type='FrSIoUMemoryHook', priority='BELOW_NORMAL')]
# 这两个 None 表示不加载/续接检测器检查点；主干仍按基础配置加载预训练权重。
load_from = None
resume_from = None
seed = 42
# 这是入口默认工作目录；已完成的 12 轮运行器另行设置并冻结了 fresh12 目录。
work_dir = 'work_dirs/dior_r_trainval/fcos_frsiou_forward_20261101'
