_base_ = ['./wf.py']

# Reproduce original WF on trainval from scratch for one epoch.
# Keep the original warmup and step schedule; no decay occurs in epoch 1.
runner = dict(max_epochs=1)
load_from = None
resume_from = None
work_dir = 'work_dirs/dior_r_trainval/fcos_wf_retry_1epoch_20261007'
