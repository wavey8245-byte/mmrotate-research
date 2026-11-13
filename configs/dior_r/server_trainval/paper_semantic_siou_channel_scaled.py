_base_=['./paper_semantic_siou.py']
# Engineering scale adaptation. This is NOT verified as the paper's setting.
# Per FPN level f -> f/sqrt(C); with C=256 the fixed divisor is 16.
custom_imports=dict(imports=['mmrotate.models.losses.semantic_iou_loss',
    'mmrotate.models.losses.semantic_iou_channel_scaled_loss',
    'mmrotate.models.dense_heads.semantic_rotated_fcos_head'],allow_failed_imports=False)
model=dict(bbox_head=dict(loss_bbox=dict(type='ChannelScaledSemanticIoULoss')))
work_dir='work_dirs/dior_r_trainval/fcos_semantic_siou_channel_scaled_20261021'
