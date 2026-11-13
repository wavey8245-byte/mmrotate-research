_base_ = ['./rotated_fcos_r50_fpn_1x_dior_r_le90_bs2.py']

# 可选 WF 消融基线：使用论文重塑前的 1-IoU，而不是原基线的 -log(IoU)。
model = dict(bbox_head=dict(loss_bbox=dict(mode='linear')))
work_dir = 'work_dirs/dior_r/fcos_riou_linear_bs2_800'
