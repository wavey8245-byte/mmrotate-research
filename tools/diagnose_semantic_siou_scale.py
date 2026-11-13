"""Read-only SIoU feature-scale diagnosis; never updates model weights."""
import copy
import fcntl
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import mmcv
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmcv.utils import get_logger
from mmrotate.datasets import build_dataset
from mmrotate.models import build_detector
from mmrotate.models.losses.semantic_iou_loss import sample_box_features, geometric_iou
from check_semantic_siou import seed

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT/'work_dirs/dior_r_trainval/semantic_siou_short_500_20261021'
OUTPUT = ROOT/'work_dirs/dior_r_trainval/semantic_siou_scale_diagnosis_20261021'


def stats(t):
    x = t.detach().double().cpu().flatten()
    if not len(x):
        return dict(count=0)
    if not torch.isfinite(x).all():
        raise FloatingPointError('Nonfinite diagnostic measurement')
    return dict(count=len(x), mean=float(x.mean()), min=float(x.min()),
                p05=float(x.quantile(.05)), median=float(x.median()),
                p95=float(x.quantile(.95)), max=float(x.max()),
                zero_fraction=float((x==0).double().mean()),
                below_1e6_fraction=float((x<1e-6).double().mean()))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def samples(pred,target,features,images,levels,strides):
    # Diagnostic only: comparisons are not installed into the training loss.
    result = {}
    vectors = {}
    for level,fmap in enumerate(features):
        for image in range(len(fmap)):
            ids=((images==image)&(levels==level)).nonzero().flatten()
            if not len(ids):
                continue
            f=fmap[image:image+1]
            p,t=pred[ids],target[ids]
            fp=sample_box_features(f,p,strides[level],5)
            fg=sample_box_features(f,t,strides[level],5)
            cp=sample_box_features(f,p,strides[level],5,1.5)
            cg=sample_box_features(f,t,strides[level],5,1.5)
            d=(fp-fg).square().sum(1).mean((1,2))
            nd=(F.normalize(fp,dim=1)-F.normalize(fg,dim=1)).square().sum(1).mean((1,2))
            s2=(F.cosine_similarity(cp,cg,dim=1).mean((1,2))+1)/2
            values=dict(distance=d, per_channel_distance=d/f.shape[1],
                        pred_feature_norm=fp.norm(dim=1).mean((1,2)),
                        gt_feature_norm=fg.norm(dim=1).mean((1,2)),
                        raw_s1=torch.exp(-.3*d), channel_mean_s1=torch.exp(-.3*d/f.shape[1]),
                        unit_l2_s1=torch.exp(-.3*nd), context_s2=s2,
                        feature_rms=f.square().mean().sqrt().expand(len(ids)))
            # All-ones maps distinguish padded/outside samples from feature values.
            ones=f.new_ones((1,1,*f.shape[-2:]))
            values['pred_padding_affected_fraction']=(sample_box_features(ones,p,strides[level],5)<.999).float().mean((1,2,3))
            values['gt_padding_affected_fraction']=(sample_box_features(ones,t,strides[level],5)<.999).float().mean((1,2,3))
            for offset in (0.,.125,.25,.5,1.):
                shifted=t.clone()
                shifted[:,0]+=offset*strides[level]
                shifted_f=sample_box_features(f,shifted,strides[level],5)
                shift_d=(shifted_f-fg).square().sum(1).mean((1,2))
                values['gt_shift_'+str(offset)+'_stride_raw_s1']=torch.exp(-.3*shift_d)
                values['gt_shift_'+str(offset)+'_stride_distance']=shift_d
            for key,value in values.items():
                if key not in vectors:
                    vectors[key]=pred.new_zeros(len(pred))
                vectors[key]=vectors[key].index_copy(0,ids,value)
    return vectors


def gradient_probe(pred,target,features,images,levels,strides,weight):
    # Gradient decomposition is on detached leaf boxes/features from the same
    # real model output, isolating objective paths without backpropagating twice
    # through the network. It is not an optimizer update.
    p=pred.detach().clone().requires_grad_(True)
    fs=[f.detach().requires_grad_(True) for f in features]
    v=samples(p,target.detach(),fs,images,levels,strides)
    u=geometric_iou(p,target.detach())
    denom=weight.sum().clamp_min(1e-6)
    output={}
    for variant,key in [('raw','raw_s1'),('channel_mean','channel_mean_s1'),('unit_l2','unit_l2_s1')]:
        s1,s2=v[key],v['context_s2']
        s=.5*(s1+s2)
        losses={
          'intra_semantic':(-.5*s1*(1-u.detach())*weight).sum()/denom,
          'context_semantic':(-.5*s2*(1-u.detach())*weight).sum()/denom,
          'gated_geometry':((1-s.detach())*(1-u)*weight).sum()/denom,
          'full':((1-s)*(1-u)*weight).sum()/denom}
        item={}
        saved=[]
        for label,loss in losses.items():
            grads=torch.autograd.grad(loss,[p]+fs,retain_graph=True,allow_unused=True)
            g=grads[0]
            if not torch.isfinite(g).all():
                raise FloatingPointError('Nonfinite box gradient')
            feature_norm=math.sqrt(sum(float(x.detach().double().square().sum()) for x in grads[1:] if x is not None))
            item[label]=dict(box_gradient_l2=float(g.norm()),
                             center_gradient_l2=float(g[:,:2].norm()),
                             size_gradient_l2=float(g[:,2:4].norm()),
                             angle_gradient_l2=float(g[:,4].norm()),
                             fpn_gradient_l2=feature_norm)
            saved.append(g.detach())
        item['chain_rule_residual_linf']=float((saved[0]+saved[1]+saved[2]-saved[3]).abs().max())
        output[variant]=item
    return output


def main():
    os.chdir(ROOT)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    get_logger('mmcv').setLevel(logging.ERROR)
    with (ROOT/'work_dirs/dior_r_trainval/server_queue.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        OUTPUT.mkdir(exist_ok=False)
        previous=json.loads((PREVIOUS/'report.json').read_text())
        paths=[ROOT/'mmrotate/models/losses/semantic_iou_loss.py',
               ROOT/'mmrotate/models/dense_heads/semantic_rotated_fcos_head.py',
               ROOT/'configs/dior_r/server_trainval/paper_semantic_siou.py',
               PREVIOUS/'semantic_siou_step_500.pth',PREVIOUS/'linear_iou_step_500.pth']
        hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}
        cfg=mmcv.Config.fromfile('configs/dior_r/server_trainval/paper_semantic_siou.py')
        seed(42)
        train=build_dataset(cfg.data.train)
        seed(314159)
        ids=previous['anchor_train_indices']
        anchors=[train[int(i)] for i in ids]
        names=[x['img_metas'].data['ori_filename'] for x in anchors]
        if names!=previous['anchor_image_names']:
            raise RuntimeError('Anchor image mismatch')
        seed(42)
        model=build_detector(cfg.model).cuda()
        model.init_weights()
        state_hash=hashlib.sha256()
        for name,tensor in model.state_dict().items():
            state_hash.update(name.encode())
            state_hash.update(tensor.detach().cpu().numpy().tobytes())
        if state_hash.hexdigest()!=previous['initial_state_sha256']:
            raise RuntimeError('Initial model state differs from matched controls')
        result=dict(scope='Fixed 16 augmented training images; no optimizer steps and no test images.',
                    status='running',anchor_train_indices=ids,anchor_names=names,
                    initial_state_verified=True,source_and_checkpoint_sha256=hashes,
                    alternatives='Offline measurements only. channel_mean and unit_l2 are engineering changes, not verified paper settings.',
                    models={})
        original=model.bbox_head.loss_bbox.forward
        captured={}
        def capture(pred,target,*args,**kw):
            captured.clear()
            captured.update(pred=pred.detach().clone(),target=target.detach().clone(),
                            weight=kw['weight'].detach().clone(),
                            features=[f.detach() for f in kw['feature_maps']],
                            images=kw['image_indices'].detach(), levels=kw['level_indices'].detach(),
                            strides=kw['feature_strides'])
            return original(pred,target,*args,**kw)
        model.bbox_head.loss_bbox.forward=capture
        for name in ('initial','linear_iou_step_500','semantic_siou_step_500'):
            if name!='initial':
                load_checkpoint(model,str(PREVIOUS/(name+'.pth')),map_location='cpu',strict=True)
            model.train()  # Same train-mode forward as the existing fixed-anchor diagnostic.
            pooled={}
            level_vectors={}
            gradient=None
            for offset in range(0,len(anchors),2):
                batch=scatter(collate(anchors[offset:offset+2],samples_per_gpu=2),[0])[0]
                with torch.no_grad():
                    model(return_loss=True,**batch)
                    p=captured['pred']
                    p[:,2:4].clamp_(min=1e-4)
                    vectors=samples(p,captured['target'],captured['features'],captured['images'],captured['levels'],captured['strides'])
                    for key,value in vectors.items():
                        pooled.setdefault(key,[]).append(value.cpu())
                    for level in captured['levels'].unique().tolist():
                        for key,value in vectors.items():
                            level_vectors.setdefault(int(level),{}).setdefault(key,[]).append(value[captured['levels']==level].cpu())
                if gradient is None and len(p):
                    gradient=gradient_probe(p,captured['target'],captured['features'],captured['images'],captured['levels'],captured['strides'],captured['weight'])
                del batch
            metrics={key:stats(torch.cat(value)) for key,value in pooled.items()}
            levels={str(level):{key:stats(torch.cat(value)) for key,value in items.items()} for level,items in level_vectors.items()}
            result['models'][name]=dict(positives=metrics['distance']['count'],metrics=metrics,fpn_levels=levels,
                                       gradient_first_anchor_batch=gradient)
            (OUTPUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
            print(name,json.dumps({k:metrics[k] for k in ('distance','raw_s1','channel_mean_s1','unit_l2_s1','context_s2','pred_padding_affected_fraction')}),flush=True)
            torch.cuda.empty_cache()
        result['sources_and_checkpoints_unchanged']=all(sha(p)==hashes[str(p.relative_to(ROOT))] for p in paths)
        if not result['sources_and_checkpoints_unchanged']:
            raise RuntimeError('Inputs changed during diagnosis')
        result['status']='completed_read_only_diagnosis'
        (OUTPUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print('COMPLETED',OUTPUT,flush=True)


if __name__=='__main__':
    main()
