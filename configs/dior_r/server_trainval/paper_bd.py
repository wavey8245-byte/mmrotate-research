_base_ = ['./riou.py']

# Independent formula port, not authors' code. Keep FCOS/data/SGD unchanged.
custom_imports = dict(
    imports=['mmrotate.models.losses.paper_rotated_losses'],
    allow_failed_imports=False)
model = dict(bbox_head=dict(loss_bbox=dict(
    _delete_=True, type='PaperBDLoss', alpha=3., agbb=False,
    min_size=1e-4, loss_weight=1.)))
work_dir = 'work_dirs/dior_r_trainval/fcos_paper_bd_bs2_800_20261018'
