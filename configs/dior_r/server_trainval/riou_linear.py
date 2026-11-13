_base_ = ['./riou.py']

# 可选 WF 权重消融：与 WF 使用相同的线性 IoU 基础项。
model = dict(bbox_head=dict(loss_bbox=dict(mode='linear')))
work_dir = 'work_dirs/dior_r_trainval/fcos_riou_linear_bs2_800'
