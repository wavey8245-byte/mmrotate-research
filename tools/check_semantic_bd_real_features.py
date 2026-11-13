"""Read-only feature and head-gradient check using an existing trained FCOS checkpoint."""
import copy
import json
import math
from pathlib import Path
import torch
import mmcv
from mmcv.parallel import collate,scatter
from mmcv.runner import load_checkpoint
from mmrotate.datasets import build_dataset
from mmrotate.models import build_detector
from mmrotate.models.builder import build_head
from mmrotate.models.losses.spatial_semantic_bd_loss import semantic_bd_components
from mmrotate.models.dense_heads.semantic_rotated_fcos_head import SemanticRotatedFCOSHead

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'work_dirs/dior_r_trainval/semantic_bd_validation_20261104'


def main():
    torch.set_num_threads(2)
    cfg=mmcv.Config.fromfile(str(ROOT/'configs/dior_r/server_trainval/proposal_semantic_bd_spatial.py'))
    old=mmcv.Config.fromfile(str(ROOT/'work_dirs/dior_r_trainval/fcos_paper_bd_recovery12_20261019/snapshot/experiment.py'))
    dataset_cfg=copy.deepcopy(old.custom_hooks[0]['dataset_cfg'])
    assert 'train.txt' in dataset_cfg['ann_file'] and 'test.txt' not in dataset_cfg['ann_file']
    ds=build_dataset(dataset_cfg)
    # Choose the most square-like real annotation from a bounded training-only pool.
    candidates=[]
    for idx in range(min(128,len(ds))):
        ann=ds.get_ann_info(idx)
        for j,b in enumerate(ann['bboxes']):
            if min(b[2],b[3])>=20:
                candidates.append((abs(math.log(float(b[2]/b[3]))),idx,j))
    _,idx,j=min(candidates)
    ann=ds.get_ann_info(idx)
    cfg.model.pretrained=None
    if 'pretrained' in cfg.model.backbone:cfg.model.backbone.pretrained=None
    if 'init_cfg' in cfg.model.backbone:cfg.model.backbone.init_cfg=None
    model=build_detector(cfg.model,train_cfg=cfg.get('train_cfg'),test_cfg=cfg.get('test_cfg'))
    checkpoint=ROOT/'work_dirs/dior_r_trainval/fcos_paper_sgkld_bs2_800_20261103/epoch_12.pth'
    load_checkpoint(model,str(checkpoint),map_location='cpu',strict=True)
    model.cuda().eval()
    batch=scatter(collate([ds[idx]],samples_per_gpu=1),[0])[0]
    image=batch['img'][0]
    metadata=batch['img_metas'][0]
    with torch.no_grad():features=model.extract_feat(image)
    gt=torch.tensor(ann['bboxes'],device='cuda',dtype=torch.float32)
    labels=torch.tensor(ann['labels'],device='cuda',dtype=torch.long)
    original=gt[j].clone()
    # Construct an exact square pair at a real object's location; artificial geometry.
    side=original[2:4].prod().sqrt()
    p=original[None].clone();t=p.clone()
    p[:,2:4]=side;t[:,2:4]=side
    p[:,4]=math.pi/8;t[:,4]=-math.pi/8
    p.requires_grad_(True)
    # Match the native FCOS regression ranges by scale for this diagnostic.
    half=float(side)/2
    level=0 if half<=64 else 1 if half<=128 else 2 if half<=256 else 3 if half<=512 else 4
    ids=torch.zeros(1,device='cuda',dtype=torch.long)
    levels=torch.full_like(ids,level)
    g,spatial=semantic_bd_components(p,t,features,ids,levels,model.bbox_head.strides)
    _,pooled=semantic_bd_components(p,t,features,ids,levels,model.bbox_head.strides,mode='pooled')
    gradient=torch.autograd.grad(spatial.sum(),p)[0]
    assert torch.isfinite(gradient).all()
    assert abs(float(g))<1e-6 and float(spatial)>1e-6
    # Native FCOS and pixel-box adapter must preserve the ordinary BD objective.
    native=build_head(copy.deepcopy(old.model.bbox_head)).cuda().train()
    native.load_state_dict(model.bbox_head.state_dict(),strict=True)
    head=model.bbox_head.train()
    outs=head(features)
    head.loss_bbox.semantic_weight=0.
    control=head.loss(*outs,[gt],[labels],metadata,semantic_features=features)
    native_loss=native.loss(*outs,[gt],[labels],metadata)
    for key in ['loss_bbox','loss_cls','loss_centerness']:
        assert torch.allclose(control[key],native_loss[key],rtol=1e-4,atol=1e-5),(key,float(control[key]),float(native_loss[key]))
    head.loss_bbox.semantic_weight=.1
    losses=head.loss(*outs,[gt],[labels],metadata,semantic_features=features)
    assert all(torch.isfinite(v) for v in losses.values())
    sum(losses.values()).backward()
    angle_gradient=head.conv_angle.weight.grad
    assert angle_gradient is not None and torch.isfinite(angle_gradient).all() and float(angle_gradient.norm())>0
    report=dict(status='passed',training_image_index=idx,training_image=ds.data_infos[idx].get('filename'),
                chosen_annotation_index=j,original_box=original.tolist(),diagnostic_square_side=float(side),
                assigned_diagnostic_level=level,synthetic_angles_degrees=[22.5,-22.5],
                BD=float(g),spatial_feature_distance=float(spatial),pooled_feature_distance=float(pooled),
                angle_sampling_gradient=float(gradient[0,4]),
                native_BD_control_loss=float(native_loss['loss_bbox']),pixel_adapter_BD_control_loss=float(control['loss_bbox']),
                head_losses={k:float(v.detach()) for k,v in losses.items()},head_diagnostics=head.loss_bbox.last_diagnostics,
                angle_head_gradient_norm=float(angle_gradient.norm()),optimizer_updates=0,
                feature_source=str(checkpoint),scope='One existing training image; frozen existing SGKLD-trained FPN features, artificial square pair and one loss/backward check; no training or accuracy claim.')
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'real_feature_head_check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
