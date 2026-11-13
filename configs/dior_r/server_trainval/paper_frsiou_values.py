_base_=['./paper_frsiou_box_tangent.py']
model=dict(bbox_head=dict(loss_bbox=dict(history_gradient='values')))
work_dir='work_dirs/dior_r_trainval/fcos_frsiou_values_20261022'
