_base_ = ['./riou.py']

model = dict(bbox_head=dict(loss_bbox=dict(
    _delete_=True, type='RotatedWFIoULoss', tau=0.7, eps=1e-7, loss_weight=1.0)))
work_dir = 'work_dirs/dior_r_trainval/fcos_wf_bs2_800'
