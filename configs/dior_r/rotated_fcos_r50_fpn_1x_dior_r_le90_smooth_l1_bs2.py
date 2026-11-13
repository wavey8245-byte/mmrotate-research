_base_ = ['./rotated_fcos_r50_fpn_1x_dior_r_le90_bs2.py']

# 800×800、bs=2、lr=0.0025；对步长归一化的四边距离做 Smooth L1。
# 五个参数直接回归真值的同一 le90 规范表示，不单独绕回角度残差。
# 绕过旋转解码，避免角度/中心梯度耦合。
# 训练时监督 clamp 前的距离，使负值预测仍可获得恢复梯度；推理照常 clamp。
# 这是编码距离对照，回归表示也与 IoU 组不同，须在实验说明中注明。
model = dict(bbox_head=dict(loss_bbox=dict(
    _delete_=True, type='EncodedRotatedSmoothL1Loss', beta=1.0, loss_weight=1.0)))
work_dir = 'work_dirs/dior_r/fcos_smooth_l1_encoded_bs2_800'
