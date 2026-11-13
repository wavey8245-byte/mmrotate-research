_base_ = ['./rotated_fcos_r50_fpn_1x_dior_r_le90_bs2.py']

# 论文的 reshaped-IoU 公式，采用明确的旋转几何扩展，详见 README。
model = dict(bbox_head=dict(loss_bbox=dict(
    _delete_=True, type='RotatedWFIoULoss', tau=0.7, eps=1e-7,
    loss_weight=1.0)))
work_dir = 'work_dirs/dior_r/fcos_wf_bs2_800'
