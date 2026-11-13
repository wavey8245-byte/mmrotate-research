_base_ = ['./riou.py']

custom_imports = dict(
    imports=['mmrotate.models.losses.paper_rotated_losses'],
    allow_failed_imports=False)
# Whole replacement: Lc from Eq.(11) + Lkf from Eq.(10). Not KFLoss + Lc.
model = dict(bbox_head=dict(loss_bbox=dict(
    _delete_=True, type='PaperDCAGPLLoss', eps=1e-6,
    min_size=1e-4, loss_weight=1.)))
work_dir = 'work_dirs/dior_r_trainval/fcos_paper_dcagpl_bs2_800_20261018'
