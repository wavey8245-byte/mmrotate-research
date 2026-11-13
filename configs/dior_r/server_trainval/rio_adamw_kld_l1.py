_base_ = ['./rio_adamw.py']

# Relative to the three-term candidate, remove only the SP-L1 contribution.
model = dict(bbox_head=dict(loss_bbox=dict(angle_weight=0.)))
work_dir = 'work_dirs/dior_r_trainval/fcos_rio_adamw_kld_l1_20261018'
