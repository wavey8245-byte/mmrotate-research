"""Frozen-branch classification ranking short training."""
import copy
import json

from . import classifier as shared
from .runtime import now, save, sha, checkpoint_path, reference_state, reference_records

MODES = ('RIoU_RANK','C01_WEIGHTED_RANK')
EPOCHS, SEED = shared.EPOCHS, shared.SEED
GRADIENT_FRACTION = .15
TOP_CANDIDATES = 128
CONTENT_WEIGHT_RANGE = .5

def flatten(levels):
    import torch

    return torch.cat([tensor.permute(0,2,3,1).reshape(-1,tensor.shape[1]) for tensor in levels])


def mine_pairs(head, features, data, raw, priors, labels):
    import torch
    import torch.nn.functional as F
    from mmcv.ops import box_iou_rotated
    from mmrotate.models.losses.bd_near_square_angle_content_loss import angle_content_values

    count = len(data['gt_bboxes'])
    points = torch.cat([grid.repeat(count,1) for grid in priors])
    image_ids = torch.cat([torch.arange(count,device=points.device).repeat_interleave(len(p)) for p in priors])
    encoded = torch.cat((flatten(raw[1]),flatten(raw[2])),dim=1)
    centers = F.logsigmoid(flatten(raw[3])).flatten()
    scores = F.logsigmoid(flatten(raw[0])) + centers[:,None]
    box = head.bbox_coder.decode(points,encoded)
    all_pairs = []
    all_weights = []
    gt_count = 0
    for image_idx,(targets,classes) in enumerate(zip(data['gt_bboxes'],data['gt_labels'])):
        ids = (image_ids==image_idx).nonzero().flatten()
        field = features[0][image_idx:image_idx+1]
        field = field/(field.square().sum(1,keepdim=True)+1e-12).sqrt()
        for cls in classes.unique().tolist():
            gt = targets[classes==cls].contiguous()
            top = scores[ids,cls].topk(min(TOP_CANDIDATES,len(ids))).indices
            local = torch.cat((ids[top],ids[labels[ids]==cls])).unique()
            candidates = box[local].contiguous()
            iou = box_iou_rotated(candidates,gt)
            quality, match = iou.max(1)
            ignored = data.get('gt_bboxes_ignore')
            eligible = torch.ones_like(quality,dtype=torch.bool)
            if ignored is not None and len(ignored[image_idx]):
                ignore_overlap = box_iou_rotated(candidates,ignored[image_idx].contiguous()).amax(1)
                eligible &= (ignore_overlap<.5) | (quality>=.5)
            for j in range(len(gt)):
                use = (match==j)&eligible
                positive = (use&(quality>=.5)).nonzero().flatten()
                if not len(positive):
                    continue
                # Select reliable positives by IoU and difficult negatives by score.
                positive = positive[quality[positive].topk(min(2,len(positive))).indices]
                negative = (use&(quality<.5)).nonzero().flatten()
                negative = negative[scores[local[negative],cls].topk(min(4,len(negative))).indices]
                high = positive[quality[positive]>=.75]
                middle = (use&(quality>=.5)&(quality<.75)).nonzero().flatten()
                middle = middle[scores[local[middle],cls].topk(min(2,len(middle))).indices]
                pos = torch.cat((positive.repeat_interleave(len(negative)), high.repeat_interleave(len(middle))))
                neg = torch.cat((negative.repeat(len(positive)),middle.repeat(len(high))))
                if not len(pos):
                    continue
                gt_count += 1
                all_pairs.append(torch.stack((local[pos],local[neg],torch.full_like(pos,cls)),dim=1))
                weight = quality.new_ones(len(pos))
                comparable = quality[neg] >= .10
                if comparable.any():
                    p,n = pos[comparable],neg[comparable]
                    unique, inverse = torch.cat((p,n)).unique(return_inverse=True)
                    cost = angle_content_values(candidates[unique], gt[j:j+1].expand(len(unique),-1),
                                                field, stride=head.strides[0],grid_size=5)
                    preferred_cost, other_cost = cost[inverse].split(len(p))
                    agreement = (other_cost-preferred_cost).clamp_min(0)/(other_cost+preferred_cost+1e-6)
                    weight[comparable] += CONTENT_WEIGHT_RANGE*agreement
                all_weights.append(weight)
    pairs = torch.cat(all_pairs) if all_pairs else labels.new_empty((0,3))
    weights = torch.cat(all_weights) if all_weights else centers.new_empty(0)
    return pairs, weights, centers, gt_count


def train(config, state, destination, checkpoints, smoke=False):
    import torch
    import torch.nn.functional as F
    from mmcv import Config
    from mmcv.parallel import scatter
    from mmcv.utils import import_modules_from_strings
    from mmdet.apis import set_random_seed
    from mmdet.datasets import build_dataloader
    from mmrotate.datasets import build_dataset
    from mmrotate.models import build_detector

    torch.set_num_threads(2)
    set_random_seed(SEED,deterministic=True)
    source = reference_records(checkpoints)['FIELD_P3']
    assert (source['status'],source['epoch'],source['iter'])==('completed',36,11700)
    assert sha(source['checkpoint'])==source['checkpoint_sha256']
    cfg = Config.fromfile(str(config))
    import_modules_from_strings(**cfg.custom_imports)
    model = build_detector(cfg.model).eval().requires_grad_(False)
    model.load_state_dict(reference_state(source['checkpoint']),strict=True)
    model.cuda()
    head = model.bbox_head
    assert head.centerness_on_reg and head.norm_on_bbox
    dataset = build_dataset(cfg.data.train)
    loader = build_dataloader(dataset,samples_per_gpu=2,workers_per_gpu=2,dist=False,
                              shuffle=True,seed=SEED)
    assert len(dataset)==650 and len(loader)==325
    branches = {name: (copy.deepcopy(head.cls_convs).requires_grad_(True).train(),
                       copy.deepcopy(head.conv_cls).requires_grad_(True).train()) for name in MODES}
    params = {name: [p for module in branch for p in module.parameters()] for name,branch in branches.items()}
    optimizers = {name:torch.optim.SGD(p,lr=.0025,momentum=.9,weight_decay=.0001) for name,p in params.items()}
    counts = dict(pairs=0,gt_with_pairs=0,upweighted=0,batches_with_pairs=0)
    with (destination/'training.jsonl').open('w',encoding='utf8') as log:
        for epoch in range(1,EPOCHS+1):
            for i,packed in enumerate(loader,1):
                data = scatter(packed,[0])[0]
                step = (epoch-1)*len(loader)+i
                with torch.no_grad():
                    features = model.extract_feat(data['img'])
                    raw = head(features)
                    priors = head.prior_generator.grid_priors([f.shape[-2:] for f in features],
                                                              device=features[0].device,dtype=features[0].dtype)
                    labels = torch.cat(head.get_targets(priors,data['gt_bboxes'],data['gt_labels'])[0])
                    pairs, weights, centers, gt_count = mine_pairs(head,features,data,raw,priors,labels)
                    pos,neg,cls = pairs.unbind(1)
                    norm = max(1,int(((labels>=0)&(labels<head.num_classes)).sum()))
                    reference = flatten(raw[0])
                    counts['pairs'] += len(pairs)
                    counts['gt_with_pairs'] += gt_count
                    counts['upweighted'] += int((weights>1).sum())
                    counts['batches_with_pairs'] += bool(len(pairs))
                    if step==1:
                        for branch in branches.values():
                            assert torch.equal(reference,flatten(shared.logits_for(features,*branch)))
                        for grid, distance in zip(priors,raw[1]):
                            assert len(grid)==distance.shape[-2]*distance.shape[-1]
                reference.requires_grad_(True)
                reference_loss = head.loss_cls(reference,labels,avg_factor=norm)
                focal_norm = torch.autograd.grad(reference_loss,reference)[0].norm().detach()
                target = GRADIENT_FRACTION*focal_norm
                lr = .0025*min(1.,.4+.6*step/100)
                stats = {}
                for name in MODES:
                    optimizer = optimizers[name]
                    optimizer.param_groups[0]['lr'] = lr
                    optimizer.zero_grad(set_to_none=True)
                    logits = flatten(shared.logits_for(features,*branches[name]))
                    focal = head.loss_cls(logits,labels,avg_factor=norm)
                    beta = logits.new_zeros(())
                    ranking = beta
                    realized = beta
                    if len(pairs):
                        s = F.logsigmoid(torch.cat((logits[pos,cls],logits[neg,cls])))
                        positive, negative = s.split(len(pairs))
                        each = F.softplus(negative+centers[neg]-positive-centers[pos])
                        ranking = each.mean() if name==MODES[0] else (each*weights).sum()/weights.sum()
                        rank_norm = torch.autograd.grad(ranking,logits,retain_graph=True)[0].norm().detach()
                        beta = (target/rank_norm.clamp_min(1e-8)).clamp_max(5.)
                        realized = beta*rank_norm
                    objective = focal+beta*ranking
                    assert torch.isfinite(objective)
                    objective.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(params[name],35.,error_if_nonfinite=True)
                    optimizer.step()
                    stats[name] = dict(focal=float(focal.detach()),rank=float(ranking.detach()),
                                       beta=float(beta),classifier_grad_norm=float(grad_norm),
                                       rank_to_reference_focal_grad=float(realized/focal_norm.clamp_min(1e-8)))
                if step==1:
                    assert len(pairs)>0 and torch.isfinite(weights).all()
                    assert all(p.grad is None for p in model.parameters())
                    save(destination/'preflight.json',dict(at=now(),initial_logits_identical=True,
                         frozen_gradients_absent=True,source_sha256=source['checkpoint_sha256'],
                         candidate_pairs=len(pairs),upweighted=int((weights>1).sum()),first_step=stats))
                if smoke:
                    return
                if step==1 or i%25==0:
                    entry = dict(epoch=epoch,step=step,total_steps=EPOCHS*len(loader),pairs=len(pairs),
                                 accumulated=counts.copy(),losses=stats)
                    log.write(json.dumps(entry)+'\n')
                    log.flush()
                    state.update(status='training',updated_at=now(),**entry)
                    save(destination/'status.json',state)
    assert counts['upweighted']>0 and counts['batches_with_pairs']>0
    original = model.state_dict()
    checkpoints = {}
    for name,branch in branches.items():
        params_by_prefix = dict(original)
        for prefix,module in zip(('bbox_head.cls_convs.','bbox_head.conv_cls.'),branch):
            params_by_prefix.update((prefix+key,value) for key,value in module.state_dict().items())
        assert params_by_prefix.keys()==original.keys()
        assert all(k.startswith(('bbox_head.cls_convs.','bbox_head.conv_cls.')) or v is original[k]
                   for k,v in params_by_prefix.items())
        assert any(v is not original[k] and not torch.equal(v,original[k]) for k,v in params_by_prefix.items())
        path = checkpoint_path(name,destination)
        path.parent.mkdir(exist_ok=True)
        pending = path.with_suffix('.pending')
        torch.save(dict(state_dict=params_by_prefix,meta=dict(CLASSES=dataset.CLASSES,epoch=EPOCHS,iter=975)),pending)
        pending.replace(path)
        checkpoints[name] = sha(path)
    save(destination/'training_complete.json',dict(at=now(),steps=975,counts=counts,checkpoints=checkpoints))
