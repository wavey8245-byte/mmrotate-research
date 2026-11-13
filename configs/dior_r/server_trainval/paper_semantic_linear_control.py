_base_=['./riou.py']
custom_imports=dict(imports=['mmrotate.models.losses.semantic_iou_loss',
                            'mmrotate.models.dense_heads.semantic_rotated_fcos_head'],
                    allow_failed_imports=False)
# Eq.(3)-(9) semantic IoU only. No fractional history.
# FPN source, 5x5 samples and 1.5x AABB context are explicit adapter choices.
model=dict(bbox_head=dict(type='SemanticRotatedFCOSHead',loss_bbox=dict(
    _delete_=True,type='SemanticIoULoss',alpha=.3,beta=.5,grid_size=5,
    context_scale=1.5,min_size=1e-4,cosine_eps=1e-8,
    semantic_enabled=False,loss_weight=1.)))
work_dir='work_dirs/dior_r_trainval/fcos_semantic_linear_control_20261021'
