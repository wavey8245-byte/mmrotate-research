_base_=['./paper_frsiou_box_tangent.py']
# Same adapter and protocol, no fractional modulation or history hook.
model=dict(bbox_head=dict(loss_bbox=dict(_delete_=True,type='SemanticIoULoss',
    alpha=.3,beta=.5,grid_size=5,context_scale=1.5,min_size=1e-4,
    cosine_eps=1e-8,semantic_enabled=True,loss_weight=1.)))
custom_hooks=[]
work_dir='work_dirs/dior_r_trainval/fcos_frsiou_siou_control_20261022'
