_base_ = ['./bd_square_angle_content.py']
custom_imports = dict(imports=[
    'mmrotate.models.losses.bd_near_square_context_angle_content_loss',
    'mmrotate.models.dense_heads.bd_content_rotated_fcos_head'],allow_failed_imports=False)
model = dict(bbox_head=dict(loss_bbox=dict(type='BDNearSquareContextAngleContentLoss',
    intra_tau=1.,intra_beta=.5,context_scale=1.5,cosine_eps=1e-3)))
custom_hooks = [dict(type='NearSquareContextDiagnosticsHook', priority='BELOW_NORMAL')]
work_dir = 'work_dirs/dior_r_trainval/fcos_bd_square_context_angle_content_fresh12_20261106'
