_base_ = ['./riou.py']

# Loss-only RiO-DETR adaptation to FCOS. Spatial L1 is image-normalized;
# angle SP-L1 is in radians. Both are in the L1 group weighted by 5.
model = dict(bbox_head=dict(loss_bbox=dict(
    _delete_=True,
    type='RotatedRiOLoss',
    kld_weight=5.0,
    spatial_weight=5.0,
    angle_weight=5.0,
    min_size=1e-4,
    loss_weight=1.0)))

work_dir = 'work_dirs/dior_r_trainval/fcos_rio_bs2_800_20261015'
