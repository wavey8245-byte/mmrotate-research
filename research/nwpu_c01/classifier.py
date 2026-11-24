"""Frozen-branch classification short training."""
import copy
import json
from pathlib import Path

from .runtime import now, save, sha, checkpoint_path, reference_state, reference_records

MODES = ('GT_ONLY', 'GT_PLUS_FIELD_AGREE')
EPOCHS, SEED, DISTILL_WEIGHT = 3, 42, .5

def frozen_models(config, status):
    import torch
    from mmcv import Config
    from mmcv.utils import import_modules_from_strings
    from mmrotate.models import build_detector

    cfg = Config.fromfile(str(config))
    import_modules_from_strings(**cfg.custom_imports)
    models = {}
    for name in ('FIELD_P3', 'FIELD_AGREE'):
        job = status[name]
        assert job['status']=='completed' and (job['epoch'],job['iter'])==(36,11700)
        path = Path(job['checkpoint'])
        assert sha(path) == job['checkpoint_sha256']
        model = build_detector(cfg.model).eval().requires_grad_(False)
        model.load_state_dict(reference_state(path),strict=True)
        assert model.bbox_head.centerness_on_reg
        models[name] = model.cuda()
    return cfg, models['FIELD_P3'], models['FIELD_AGREE']


def logits_for(features, tower, predictor):
    outputs = []
    for feature in features:
        for conv in tower:
            feature = conv(feature)
        outputs.append(predictor(feature))
    return outputs


def train(config, state, destination, checkpoints, smoke=False):
    import torch
    import torch.nn.functional as F
    from mmcv.parallel import scatter
    from mmdet.apis import set_random_seed
    from mmdet.datasets import build_dataloader
    from mmrotate.datasets import build_dataset

    torch.set_num_threads(2)
    set_random_seed(SEED,deterministic=True)
    source_status = reference_records(checkpoints)
    cfg, student, teacher = frozen_models(config, source_status)
    dataset = build_dataset(cfg.data.train)
    assert len(dataset)==650
    loader = build_dataloader(dataset,samples_per_gpu=2,workers_per_gpu=2,dist=False,
                              shuffle=True,seed=SEED)
    assert len(loader)==325
    base_head, teacher_head = student.bbox_head, teacher.bbox_head
    branches = {name: (copy.deepcopy(base_head.cls_convs).requires_grad_(True).train(),
                       copy.deepcopy(base_head.conv_cls).requires_grad_(True).train()) for name in MODES}
    parameters = {name: [p for module in branch for p in module.parameters()] for name,branch in branches.items()}
    optimizers = {name: torch.optim.SGD(values, lr=.0025, momentum=.9, weight_decay=.0001)
                  for name,values in parameters.items()}
    freeze_keys = ('bbox_head.cls_convs.', 'bbox_head.conv_cls.')
    def flatten(outputs):
        return torch.cat([v.permute(0,2,3,1).reshape(-1,base_head.cls_out_channels) for v in outputs])

    record = destination/'training.jsonl'
    with record.open('w',encoding='utf8') as log:
        for epoch in range(1,EPOCHS+1):
            for i,packed in enumerate(loader,1):
                batch_data = scatter(packed,[0])[0]
                image = batch_data['img']
                step = (epoch-1)*len(loader)+i
                with torch.no_grad():
                    features = student.extract_feat(image)
                    teacher_logits = flatten(logits_for(teacher.extract_feat(image),
                                                       teacher_head.cls_convs,teacher_head.conv_cls))
                    points = base_head.prior_generator.grid_priors(
                        [f.shape[-2:] for f in features],device=image.device,dtype=features[0].dtype)
                    labels = torch.cat(base_head.get_targets(points,batch_data['gt_bboxes'],
                                                            batch_data['gt_labels'])[0])
                    positives = ((labels>=0)&(labels<base_head.num_classes)).nonzero().flatten()
                    negatives = (labels==base_head.num_classes).nonzero().flatten()
                    k = min(len(negatives),3*max(len(positives),1))
                    hard = teacher_logits[negatives].amax(1).topk(k).indices
                    selected = torch.cat((positives,negatives[hard]))
                    teacher_soft = teacher_logits[selected].sigmoid()
                    assert selected.numel() > 0
                    if step==1:
                        original = base_head(features)
                        for branch in branches.values():
                            assert torch.equal(flatten(logits_for(features,*branch)),flatten(original[0]))
                        for model in (student,teacher):
                            assert not any(p.requires_grad for p in model.parameters())
                lr = .0025 * min(1.,.4+.6*step/100)
                scores = {}
                for name in MODES:
                    optimizer = optimizers[name]
                    optimizer.param_groups[0]['lr'] = lr
                    optimizer.zero_grad(set_to_none=True)
                    logits = flatten(logits_for(features,*branches[name]))
                    focal = base_head.loss_cls(logits,labels,avg_factor=max(1,len(positives)))
                    distill = F.binary_cross_entropy_with_logits(logits[selected],teacher_soft)
                    objective = focal if name=='GT_ONLY' else focal+DISTILL_WEIGHT*distill
                    assert torch.isfinite(objective)
                    objective.backward()
                    norm = torch.nn.utils.clip_grad_norm_(parameters[name],35.,error_if_nonfinite=True)
                    assert all(p.grad is not None for p in parameters[name])
                    optimizer.step()
                    scores[name] = dict(gt_loss=float(focal.detach()),soft_bce=float(distill.detach()),
                                        cls_grad_norm=float(norm))
                if step==1:
                    assert all(p.grad is None for net in (student,teacher) for p in net.parameters())
                    save(destination/'preflight.json',dict(at=now(),identical_initial_logits=True,
                         finite_first_step_losses=scores,frozen_models_have_no_gradients=True,
                         source_checkpoints={k:source_status[k]['checkpoint_sha256']
                                             for k in ('FIELD_P3','FIELD_AGREE')}))
                if smoke:
                    return
                if step==1 or i%25==0:
                    entry = dict(epoch=epoch,step=step,total_steps=EPOCHS*len(loader),lr=lr,
                                 positives=len(positives),distill_positions=len(selected),losses=scores)
                    log.write(json.dumps(entry)+'\n')
                    log.flush()
                    state.update(status='training',updated_at=now(),**entry)
                    save(destination/'status.json',state)
    fixed = student.state_dict()
    for name, branch in branches.items():
        state_dict = dict(fixed)
        for prefix, module in zip(freeze_keys,branch):
            state_dict.update((prefix+key,value) for key,value in module.state_dict().items())
        assert state_dict.keys()==fixed.keys()
        assert all(state_dict[key] is value for key,value in fixed.items() if not key.startswith(freeze_keys))
        assert any(not torch.equal(state_dict[k],v) for k,v in fixed.items() if k.startswith(freeze_keys))
        assert all(torch.isfinite(state_dict[k]).all() for k in state_dict if k.startswith(freeze_keys))
        path = checkpoint_path(name,destination)
        path.parent.mkdir(exist_ok=True)
        temporary = path.with_suffix('.pending')
        torch.save(dict(state_dict=state_dict,meta=dict(CLASSES=dataset.CLASSES,epoch=EPOCHS,iter=975)),temporary)
        temporary.replace(path)
    save(destination/'training_complete.json',dict(at=now(),steps=975,frozen_parameter_keys_verified=True,
                                            checkpoints={name:sha(checkpoint_path(name,destination)) for name in MODES}))
