_base_=['./paper_semantic_siou.py']
# ENGINEERING INTERPRETATION, not verified author code.
# Eq.(12) GL coefficients; nu=.5 and M=2 are explicit choices, lambda=.3.
# Raw FPN semantics retained; no feature channel scaling in this experiment.
custom_imports=dict(imports=['mmrotate.models.losses.fractional_semantic_iou_loss',
    'mmrotate.models.dense_heads.fractional_rotated_fcos_head',
    'mmrotate.core.frsiou_memory_hook'],allow_failed_imports=False)
model=dict(bbox_head=dict(type='FractionalRotatedFCOSHead',loss_bbox=dict(
    type='FCOSFrSIoUEngineeringLoss',fractional_order=.5,memory_steps=2,
    modulation_lambda=.3,history_gradient='box_tangent',max_history_keys=200000)))
custom_hooks=[dict(type='FrSIoUMemoryHook',priority='BELOW_NORMAL')]
work_dir='work_dirs/dior_r_trainval/fcos_frsiou_box_tangent_20261022'
