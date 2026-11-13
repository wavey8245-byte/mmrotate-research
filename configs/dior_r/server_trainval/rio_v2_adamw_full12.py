_base_ = ['./rio_v2_adamw.py']

# Fresh full run of the checked FCOS/v2 loss adaptation. Fixed epoch-12 test;
# the launcher disables intermediate validation and never selects on test AP.
runner = dict(type='EpochBasedRunner', max_epochs=12)
load_from = None
resume_from = None
work_dir = 'work_dirs/dior_r_trainval/fcos_rio_v2_adamw_full12_20261026'
