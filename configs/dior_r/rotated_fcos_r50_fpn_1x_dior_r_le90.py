_base_ = ['../rotated_fcos/rotated_fcos_r50_fpn_1x_dota_le90.py']

data_root = 'data/DIOR-R/'
model = dict(bbox_head=dict(num_classes=20))

# Keep the inherited FCOS pipelines; replace all DOTA data sources.
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type='DIORDataset',
        ann_file=data_root + 'ImageSets/Main/train.txt',
        ann_prefix=data_root + 'Annotations/Oriented Bounding Boxes/',
        img_prefix=data_root + 'JPEGImages-trainval/'),
    val=dict(
        type='DIORDataset',
        ann_file=data_root + 'ImageSets/Main/val.txt',
        ann_prefix=data_root + 'Annotations/Oriented Bounding Boxes/',
        img_prefix=data_root + 'JPEGImages-trainval/'),
    test=dict(
        type='DIORDataset',
        ann_file=data_root + 'ImageSets/Main/test.txt',
        ann_prefix=data_root + 'Annotations/Oriented Bounding Boxes/',
        img_prefix=data_root + 'JPEGImages-test/'))

# Single 6 GiB GPU: halve the inherited learning rate for batch size one.
optimizer = dict(lr=0.00125)
evaluation = dict(interval=1, metric='mAP', iou_thr=0.5,
                  use_07_metric=True, save_best='mAP')
work_dir = 'work_dirs/dior_r/fcos_riou'
seed = 42
