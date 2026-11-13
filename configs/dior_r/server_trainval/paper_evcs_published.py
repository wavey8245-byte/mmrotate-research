_base_ = ['./riou.py']
custom_imports = dict(
    imports=['mmrotate.models.losses.paper_evcs_loss'],
    allow_failed_imports=False)
# KFIoU log form, centers in the current FCOS decoded training coordinates.
# This coordinate choice is a transfer setting, not an author CRL config.
model = dict(bbox_head=dict(loss_bbox=dict(
    _delete_=True, type='PaperEVCSPublishedLoss', fun='ln', beta=1./9.,
    eps=1e-6, min_size=1e-7, loss_weight=1., evcs_weight=0.006)))
work_dir = 'work_dirs/dior_r_trainval/fcos_evcs_published_20261021'
