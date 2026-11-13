_base_ = ['./rio.py']

# Recovery experiment: all three RiO regression terms and weights stay 5/5/5.
# The failed unbounded-angle trial is retained in rio.py's original work_dir.
model = dict(bbox_head=dict(scale_angle=False, bounded_angle=True))
work_dir = 'work_dirs/dior_r_trainval/fcos_rio_bounded_bs2_800_20261016'
