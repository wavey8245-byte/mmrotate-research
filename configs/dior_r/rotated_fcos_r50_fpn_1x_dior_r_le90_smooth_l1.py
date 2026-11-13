_base_ = ['./rotated_fcos_r50_fpn_1x_dior_r_le90.py']

# FCOS passes decoded (cx, cy, w, h, angle), not encoded distances.
# Legacy decoded residuals; distances are stride-normalized in training.
# Kept unchanged for historical experiments, without periodic-angle correction.
model = dict(bbox_head=dict(loss_bbox=dict(
    _delete_=True, type='RotatedSmoothL1Loss', beta=1.0, loss_weight=1.0)))
work_dir = 'work_dirs/dior_r/fcos_smooth_l1'
