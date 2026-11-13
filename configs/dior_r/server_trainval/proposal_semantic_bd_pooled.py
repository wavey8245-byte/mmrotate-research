_base_=['./proposal_semantic_bd_spatial.py']
model=dict(bbox_head=dict(loss_bbox=dict(semantic_mode='pooled')))
work_dir='work_dirs/dior_r_trainval/semantic_bd_pooled_proposal_20261104'
