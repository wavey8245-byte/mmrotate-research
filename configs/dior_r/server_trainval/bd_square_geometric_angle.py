_base_ = ['./paper_bd.py']

# Fixed candidate selected before full training; no test-driven adaptation.
custom_imports = dict(imports=[
    'mmrotate.models.losses.bd_near_square_geometric_angle_loss',
    'mmrotate.models.dense_heads.bd_content_rotated_fcos_head'],
    allow_failed_imports=False)
model = dict(bbox_head=dict(
    type='BDContentRotatedFCOSHead',
    loss_bbox=dict(_delete_=True, type='BDNearSquareGeometricAngleLoss',
        alpha=3., angle_weight=0.007965899687385276, square_ratio=1.2,
        min_size=1e-4, eps=1e-6,
        loss_weight=1.)))
custom_hooks = [dict(type='GeometricAngleDiagnosticsHook', priority='BELOW_NORMAL')]
work_dir = 'work_dirs/dior_r_trainval/fcos_bd_square_geometric_angle_fresh12_20261109'
