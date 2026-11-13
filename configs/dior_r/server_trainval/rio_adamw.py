import math

_base_ = ['./riou_adamw.py']

# Preserve all three terms and the original FCOS angle branch. SP-L1 is the
# paper-formula adaptation; the public repository uses ordinary angular L1.
# This candidate needs learning validation, not just finite-loss checks.
model = dict(bbox_head=dict(loss_bbox=dict(
    _delete_=True, type='RotatedRiOLoss',
    kld_weight=5., spatial_weight=5., angle_weight=5., loss_weight=1.,
    angle_normalizer=math.pi, kld_normalized=True,
    covariance_epsilon=1e-8, min_size=1e-8)))
work_dir = 'work_dirs/dior_r_trainval/fcos_rio_adamw_bs2_800_20261018'
