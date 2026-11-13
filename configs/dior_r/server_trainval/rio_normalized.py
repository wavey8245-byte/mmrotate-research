import math

_base_ = ['./rio.py']

# Candidate derived from the publicly documented normalized box variables.
# Full RiO SP-L1 code is not public; this is explicitly an FCOS adaptation.
# Keep the original FCOS angle branch; do not use the failed sigmoid mapping.
model = dict(bbox_head=dict(loss_bbox=dict(
    angle_normalizer=math.pi,
    kld_normalized=True,
    covariance_epsilon=1e-8,
    min_size=1e-8)))
work_dir = 'work_dirs/dior_r_trainval/fcos_rio_normalized_bs2_800_20261016'
