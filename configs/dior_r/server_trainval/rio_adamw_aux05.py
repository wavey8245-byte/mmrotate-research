_base_ = ['./rio_adamw.py']

# Joint low-auxiliary-weight arm of the 2x2 comparison, not a tuned optimum.
model = dict(bbox_head=dict(loss_bbox=dict(spatial_weight=.5, angle_weight=.5)))
work_dir = 'work_dirs/dior_r_trainval/fcos_rio_adamw_aux05_20261022'
