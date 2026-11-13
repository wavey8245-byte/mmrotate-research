_base_ = ['../rotated_fcos_r50_fpn_1x_dior_r_le90_bs2.py']

# MMRotate 1.x DIOR 数据协议：train + val 训练，test 用于评估。
# MMDetection 2.x 会将 ann_file 列表构建成 ConcatDataset，无须生成新划分文件。
data_root = 'data/DIOR-R/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(ann_file=[
        data_root + 'ImageSets/Main/train.txt',
        data_root + 'ImageSets/Main/val.txt']),
    val=dict(
        ann_file=data_root + 'ImageSets/Main/test.txt',
        img_prefix=data_root + 'JPEGImages-test/'))

# test 参与逐轮评估，不把它作为独立 val 选最优模型。
# 固定使用第 12 轮权重报告最终 test 结果。
evaluation = dict(interval=1, metric='mAP', iou_thr=0.5,
                  use_07_metric=True, save_best=None)
load_from = None
resume_from = None
work_dir = 'work_dirs/dior_r_trainval/fcos_riou_bs2_800'
