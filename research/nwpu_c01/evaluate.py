"""NWPU-R official VOC07 evaluation by ground-truth shape."""
import math

GROUPS = ['all','near_square','medium','elongated']
THRESHOLDS = [.5,.75]


def worker(task):
    import numpy as np
    import torch
    from mmcv.ops import box_iou_rotated
    torch.set_num_threads(1)
    image_id,pred,ann=task
    result=[]
    for cls,det in enumerate(pred):
        gt=ann['bboxes'][ann['labels']==cls]
        ign=ann['bboxes_ignore'][ann['labels_ignore']==cls]
        ng=len(gt);nd=len(det)
        ratios=np.maximum(gt[:,2],gt[:,3])/np.maximum(1e-12,np.minimum(gt[:,2],gt[:,3]))
        masks=[np.ones(ng,dtype=bool),ratios<=1.2,(ratios>1.2)&(ratios<=2),ratios>2]
        counts=[int(m.sum()) for m in masks]
        tp=np.zeros((4,2,nd),dtype=np.uint8);fp=tp.copy()
        matches=[]
        if ng+len(ign)==0:
            fp[:]=1
        elif nd:
            fullgt=np.concatenate([gt,ign])
            ious=box_iou_rotated(torch.from_numpy(det[:,:5]).float(),torch.from_numpy(fullgt).float()).numpy()
            order=np.argsort(-det[:,-1])
            for group,m in enumerate(masks):
                # Same as moving excluded normal GT into ignored GT, preserving
                # included GT first, original ignores second, out-of-group last.
                cols=np.concatenate([np.where(m)[0],np.arange(ng,ng+len(ign)),np.where(~m)[0]])
                scores=ious[:,cols]
                best=scores.argmax(1);quality=scores.max(1)
                included=int(m.sum())
                for tid,threshold in enumerate(THRESHOLDS):
                    covered=np.zeros(len(cols),dtype=bool)
                    for i in order:
                        if quality[i]>=threshold:
                            j=best[i]
                            if j<included:
                                if not covered[j]:
                                    tp[group,tid,i]=1;covered[j]=True
                                    if group==0 and tid==0:
                                        gi=int(cols[j]);g=gt[gi];p=det[i]
                                        delta=(float(p[4]-g[4])+math.pi/4)%(math.pi/2)-math.pi/4
                                        matches.append(dict(key=f'{image_id}:{cls}:{gi}',cls=cls,ratio=float(ratios[gi]),
                                            iou=float(quality[i]),axis_angle_error_degrees=abs(delta)*180/math.pi,
                                            center_error=float(np.linalg.norm(p[:2]-g[:2])/np.linalg.norm(g[2:4]))))
                                else:fp[group,tid,i]=1
                            # A threshold match to ignored GT contributes neither TP nor FP.
                        else:fp[group,tid,i]=1
        result.append(dict(scores=det[:,-1],tp=tp,fp=fp,counts=counts,matches=matches))
    return result


def evaluate(config, cache):
    import mmcv
    import torch
    from mmcv.utils import import_modules_from_strings
    from mmrotate.datasets import build_dataset

    torch.set_num_threads(2)
    cfg = mmcv.Config.fromfile(str(config))
    import_modules_from_strings(**cfg.custom_imports)
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    predictions = mmcv.load(str(cache))
    return evaluate_predictions(dataset, predictions, check_native=True)


def evaluate_predictions(dataset, predictions, check_native=False):
    import numpy as np
    import torch
    from mmdet.core import average_precision

    if len(predictions) != len(dataset) or len(dataset) != 339:
        raise ValueError('Expected 339 evaluation predictions')
    if not all(len(image) == 10 and all(boxes.ndim == 2 and boxes.shape[1] == 6
               and np.isfinite(boxes).all() for boxes in image) for image in predictions):
        raise ValueError('Invalid prediction layout or values')
    annotations = [dataset.get_ann_info(i) for i in range(len(dataset))]
    threads = torch.get_num_threads()
    try:
        matched = [worker(task) for task in zip(dataset.img_ids, predictions, annotations)]
    finally:
        torch.set_num_threads(threads)

    per_class = {group: [] for group in GROUPS}
    for label, name in enumerate(dataset.CLASSES):
        rows = [image[label] for image in matched]
        counts = np.sum([row['counts'] for row in rows], axis=0)
        order = np.argsort(-np.concatenate([row['scores'] for row in rows]))
        tp = np.cumsum(np.concatenate([row['tp'] for row in rows], axis=2)[:, :, order], axis=2)
        fp = np.cumsum(np.concatenate([row['fp'] for row in rows], axis=2)[:, :, order], axis=2)
        for g, group in enumerate(GROUPS):
            if not counts[g]:
                continue
            precision = tp[g] / np.maximum(tp[g] + fp[g], np.finfo(np.float32).eps)
            recall = tp[g].astype(np.float64) / counts[g]
            ap = [float(average_precision(recall[t], precision[t], '11points'))
                  for t in range(len(THRESHOLDS))]
            per_class[group].append(dict(cls=name, gt=int(counts[g]), AP50=ap[0], AP75=ap[1]))

    groups = {group: dict(gt=sum(row['gt'] for row in rows), classes=len(rows),
              **{key: float(np.mean([row[key] for row in rows])) for key in ('AP50', 'AP75')})
              for group, rows in per_class.items()}
    if groups['all']['gt'] != 2114 or groups['near_square']['gt'] != 1013:
        raise ValueError('Unexpected evaluation ground-truth counts')
    if check_native:
        for threshold, key in zip(THRESHOLDS, ('AP50', 'AP75')):
            native = float(dataset.evaluate(predictions, iou_thr=threshold, nproc=1, logger='silent')['mAP'])
            if abs(native - groups['all'][key]) > 1e-7:
                raise ValueError(f'Overall {key} does not match native evaluator')
            groups['all'][key] = native
    hit_counts = {group:{k:sum(int(row['tp'][i,t].sum()) for image in matched for row in image)
                          for t,k in enumerate(('AP50','AP75'))}
                  for i,group in enumerate(GROUPS)}
    return dict(evaluation_images=len(dataset), groups=groups, per_class=per_class,
                post_nms_matches=hit_counts)
