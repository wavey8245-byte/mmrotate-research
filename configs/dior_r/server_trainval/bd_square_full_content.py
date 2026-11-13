_base_ = ['./bd_square_angle_content.py']

# The only objective change is removing content stop-gradients for cx,cy,w,h.
custom_imports = dict(imports=[
    'mmrotate.models.losses.bd_near_square_full_content_loss',
    'mmrotate.models.dense_heads.bd_content_rotated_fcos_head'],
    allow_failed_imports=False)
model = dict(bbox_head=dict(loss_bbox=dict(type='BDNearSquareFullContentLoss')))
custom_hooks = [dict(type='NearSquareFullContentDiagnosticsHook', priority='BELOW_NORMAL')]
work_dir = 'work_dirs/dior_r_trainval/fcos_bd_square_full_content_fresh12_20261105'
