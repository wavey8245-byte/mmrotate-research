_base_=['./proposal_semantic_bd_spatial.py']
model=dict(bbox_head=dict(loss_bbox=dict(semantic_weight=0.)))
work_dir='work_dirs/dior_r_trainval/semantic_bd_control_proposal_20261104'
