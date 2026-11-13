_base_ = ['./rio_adamw.py']

# Scope selected by the user: retain FCOS outputs and point assignment;
# use v2 VFL + KLD + spatial/periodic L1, with uniform positive weighting.
# No FDR, Hungarian matching, decoder auxiliaries, or distillation.
custom_imports = dict(
    imports=['mmrotate.models.dense_heads.rio_v2_fcos_head'],
    allow_failed_imports=False)
model = dict(bbox_head=dict(
    type='RiOV2FCOSHead',
    loss_cls=dict(_delete_=True, type='RiOV2VarifocalLoss', use_sigmoid=True,
                  alpha=.75, gamma=2., loss_weight=1.)))
work_dir = 'work_dirs/dior_r_trainval/fcos_rio_v2_adamw_bs2_800_20261023'
