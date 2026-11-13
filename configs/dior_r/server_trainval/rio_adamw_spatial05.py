_base_ = ['./rio_adamw.py']

# Lower only the spatial auxiliary term; preserve all three losses.
model = dict(bbox_head=dict(loss_bbox=dict(spatial_weight=.5)))
work_dir = 'work_dirs/dior_r_trainval/fcos_rio_adamw_spatial05_20261022'
