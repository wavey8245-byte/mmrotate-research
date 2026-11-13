_base_ = ['./riou.py']

# 使用已修正的编码参数 Smooth L1，不恢复旧失败实验的检测器权重。
model = dict(bbox_head=dict(loss_bbox=dict(
    _delete_=True, type='EncodedRotatedSmoothL1Loss', beta=1.0, loss_weight=1.0)))
work_dir = 'work_dirs/dior_r_trainval/fcos_smooth_l1_encoded_bs2_800'
