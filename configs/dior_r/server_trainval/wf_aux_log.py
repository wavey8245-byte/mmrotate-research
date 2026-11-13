_base_ = ['./riou.py']

# Experimental project variant, not the paper's reshaped loss.
# Keep official log-IoU supervision; add WF with a small preset coefficient.
model = dict(bbox_head=dict(loss_bbox=dict(
    _delete_=True, type='RotatedIoUWithWFAuxLoss', wf_weight=0.1,
    tau=0.7, eps=1e-6, loss_weight=1.0)))
work_dir = 'work_dirs/dior_r_trainval/fcos_wf_aux_log_bs2_800'
