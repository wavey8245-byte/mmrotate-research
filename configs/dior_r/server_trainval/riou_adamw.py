_base_ = ['./riou.py']

# Matched optimizer control for the RiO-loss adaptation. Keep FCOS's head,
# data, batch size, initialization, and 12-epoch schedule. This is not the
# published RT-DETR-OBB training recipe.
optimizer = dict(
    _delete_=True, type='AdamW', lr=1e-4, betas=(0.9, 0.999),
    weight_decay=1e-4, paramwise_cfg=dict(norm_decay_mult=0.))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
work_dir = 'work_dirs/dior_r_trainval/fcos_riou_adamw_bs2_800_20261018'
