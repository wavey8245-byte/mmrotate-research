_base_ = ['./paper_bd.py']

# Fixed candidate selected before full training; no test-driven adaptation.
custom_imports = dict(imports=[
    'mmrotate.models.losses.bd_near_square_assigned_angle_content_loss',
    'mmrotate.models.dense_heads.bd_assigned_content_rotated_fcos_head'],
    allow_failed_imports=False)
model = dict(bbox_head=dict(
    type='BDAssignedContentRotatedFCOSHead',
    loss_bbox=dict(_delete_=True, type='BDNearSquareAssignedAngleContentLoss',
        alpha=3., semantic_weight=.1, square_ratio=1.2, grid_size=5,
        chunk_size=128, min_size=1e-4, eps=1e-6,
        loss_weight=1.)))
custom_hooks = [dict(type='AssignedAngleContentDiagnosticsHook', priority='BELOW_NORMAL')]
work_dir = 'work_dirs/dior_r_trainval/fcos_bd_square_assigned_angle_content_fresh12_20261107'
