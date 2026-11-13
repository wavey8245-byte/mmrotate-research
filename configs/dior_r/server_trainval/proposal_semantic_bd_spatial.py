# Experimental BD + feature alignment. No performance claim or automatic training.
_base_=['./riou.py']
custom_imports=dict(imports=['mmrotate.models.losses.spatial_semantic_bd_loss',
                            'mmrotate.models.dense_heads.semantic_rotated_fcos_head'],
                    allow_failed_imports=False)
model=dict(bbox_head=dict(type='SemanticRotatedFCOSHead',loss_bbox=dict(
    _delete_=True,type='SpatialSemanticBDLoss',alpha=3.,semantic_weight=.1,
    semantic_mode='spatial',grid_size=5,min_size=1e-4,cosine_eps=1e-6,loss_weight=1.)))
work_dir='work_dirs/dior_r_trainval/semantic_bd_spatial_proposal_20261104'
