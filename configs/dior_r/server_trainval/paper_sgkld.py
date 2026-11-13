_base_ = ['./riou.py']

# Li et al. TGRS 2023: n=4, directed GT||prediction KL, Eq. (17).
# Retain the current FCOS training protocol; this is a loss port, not a
# reproduction of the paper's RetinaNet/R3Det/RoI-Transformer detectors.
custom_imports = dict(
    imports=['mmrotate.models.losses.paper_sgkld_loss'],
    allow_failed_imports=False)
model = dict(bbox_head=dict(loss_bbox=dict(
    _delete_=True, type='PaperSGKLDLoss', min_size=1e-7, loss_weight=1.)))
work_dir = 'work_dirs/dior_r_trainval/fcos_paper_sgkld_bs2_800_20261103'
