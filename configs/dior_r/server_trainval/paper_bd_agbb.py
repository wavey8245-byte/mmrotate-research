_base_ = ['./paper_bd.py']

# Port choice: the paper does not give a numerical square-like threshold.
# Switch BOTH boxes according to GT max(w/h, h/w), never prediction alone.
# delta=5 follows the text (the PDF calls this sigma in one sentence).
model = dict(bbox_head=dict(loss_bbox=dict(
    agbb=True, square_ratio_threshold=1.1, delta=5.)))
work_dir = 'work_dirs/dior_r_trainval/fcos_paper_bd_agbb110_bs2_800_20261018'
