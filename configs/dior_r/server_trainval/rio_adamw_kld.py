_base_ = ['./rio_adamw.py']

# Matched component ablation; retain the same FCOS head and optimizer.
model = dict(bbox_head=dict(loss_bbox=dict(spatial_weight=0., angle_weight=0.)))
work_dir = 'work_dirs/dior_r_trainval/fcos_rio_adamw_kld_20261018'
