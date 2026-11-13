_base_ = ['./riou.py']
custom_imports = dict(
    imports=['mmrotate.models.losses.paper_evcs_abs_loss'],
    allow_failed_imports=False)
# ENGINEERING CORRECTION: absolute length difference in the exponent.
# Not the published EVCS Eq.(4); other FCOS transfer settings unchanged.
model = dict(bbox_head=dict(loss_bbox=dict(
    _delete_=True, type='EVCSAbsCorrectionLoss', fun='ln', beta=1./9.,
    eps=1e-6, min_size=1e-7, loss_weight=1., evcs_weight=0.006)))
work_dir = 'work_dirs/dior_r_trainval/fcos_evcs_abs_correction_20261021'
