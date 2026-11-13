# 纯 SIoU 对照配置：继承同一 FCOS/DIOR-R 基础设置，只替换语义头与框损失。
_base_=['./riou.py']
# 导入模块完成注册，使配置中的类名能被 MMRotate 构建器识别。
custom_imports=dict(imports=['mmrotate.models.losses.semantic_iou_loss',
                            'mmrotate.models.dense_heads.semantic_rotated_fcos_head'],
                    allow_failed_imports=False)
# Eq.(3)-(9) semantic IoU only. No fractional history.
# FPN source, 5x5 samples and 1.5x AABB context are explicit adapter choices.
# _delete_=True 完整替换继承的框损失；不叠加原 RIoU 损失。
# alpha/beta 控制语义分数；grid_size/context_scale 为本项目采样约定。
model=dict(bbox_head=dict(type='SemanticRotatedFCOSHead',loss_bbox=dict(
    _delete_=True,type='SemanticIoULoss',alpha=.3,beta=.5,grid_size=5,
    context_scale=1.5,min_size=1e-4,cosine_eps=1e-8,
    semantic_enabled=True,loss_weight=1.)))
work_dir='work_dirs/dior_r_trainval/fcos_semantic_siou_20261021'
