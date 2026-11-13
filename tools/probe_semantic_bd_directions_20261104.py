"""Paired training-image directional diagnostic, fixed P3, no network updates."""
import argparse,ast,hashlib,json,math,sys,time
from pathlib import Path
import numpy as np
import torch,mmcv
from mmcv.runner import load_checkpoint
from mmcv.utils import import_modules_from_strings
from mmrotate.models import build_detector
from mmrotate.models.losses.spatial_semantic_bd_loss import semantic_bd_components


def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
 return h.hexdigest()


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--reference-dir',required=True);args=parser.parse_args()
 ref=Path(args.reference_dir);sys.path.insert(0,str(ref))
 from prototype import decode_local,align_by_geometry,_phi
 root=Path(__file__).resolve().parents[1]
 out=root/'work_dirs/dior_r_trainval/semantic_bd_direction_probe_20261104'
 if (out/'protocol.json').exists():raise RuntimeError('Refusing to overwrite existing diagnostic')
 out.mkdir(parents=True,exist_ok=True)
 prev=ref/'real_probe_polygon_results'
 protocol=json.loads((prev/'protocol.json').read_text())
 meta=json.loads((prev/'gt_calibration.json').read_text())
 oldrows=[json.loads(x) for x in (prev/'paired_rows.jsonl').read_text().splitlines()]
 byimage={};bygt={}
 for m in meta:byimage.setdefault(m['image'],[]).append(m)
 for z in oldrows:bygt.setdefault((z['image'],z['object']),[]).append(z)
 assert len(meta)==268 and len(oldrows)==5853
 torch.set_num_threads(4);torch.manual_seed(20261104)
 checkpoint=Path(protocol['checkpoint']);assert sha(checkpoint)==protocol['checkpoint_sha256']
 cfg=mmcv.Config.fromfile(protocol['config'])
 if cfg.get('custom_imports'):import_modules_from_strings(**cfg.custom_imports)
 cfg.model.pretrained=None
 if 'init_cfg' in cfg.model.backbone:cfg.model.backbone.init_cfg=None
 model=build_detector(cfg.model,train_cfg=None,test_cfg=cfg.get('test_cfg'))
 load_checkpoint(model,str(checkpoint),map_location='cpu',strict=True);model.cuda().eval()
 for par in model.parameters():par.requires_grad_(False)
 # Independently evaluated double-precision polygon IoU, reusing audited source.
 polysource=ref.parent/'bd_spatial_content_20261030/verify_design.py'
 tree=ast.parse(polysource.read_text());ns={'math':math}
 exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'polygon','area','riou'}],type_ignores=[]),str(polysource),'exec'),ns)
 def ious(p,t):return np.array([ns['riou'](a,b) for a,b in zip(p.detach().tolist(),t.detach().tolist())])
 sys.path.insert(0,str(ref/'diagnostic_deps'))
 from shapely.geometry import Polygon
 errors=[];rng=np.random.default_rng(123)
 for theta in [0.,math.pi/2,-math.pi/2,.2]:
  target=torch.tensor([[300.,300.,200.,100.,theta]],dtype=torch.float64)
  for dim in range(4):
   e=torch.zeros((1,5),dtype=torch.float64);e[0,dim]=.05
   expected=.95/1.05 if dim<2 else math.exp(-.05)
   errors.append(abs(ious(decode_local(e,target),target)[0]-expected))
 for _ in range(100):
  t=[400.,400.,float(rng.uniform(5,300)),float(rng.uniform(5,100)),float(rng.uniform(-1.57,1.57))]
  p=np.array(t)+rng.normal(size=5)*np.array([5,5,1,1,.05]);p[2:4]=np.maximum(p[2:4],1.)
  pa,pb=Polygon(ns['polygon'](p)),Polygon(ns['polygon'](t))
  errors.append(abs(ns['riou'](p,t)-pa.intersection(pb).area/pa.union(pb).area))
 assert max(errors)<1e-8
 (out/'metric_validation.json').write_text(json.dumps(dict(checks=len(errors),max_error=max(errors),reference='analytic cases + Shapely double polygons')))
 methods=['bd','legacy_content','pooled_cosine','spatial_cosine','bd_pooled_cosine','bd_spatial_cosine'];steps=[.001,.002,.005]
 settings=dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),image_ids=protocol['image_ids'],images=96,gt=268,cases=5853,
  support='Identical GT boxes, object indices, perturbations and normalized coordinates from the supplied LICC diagnostic',
  reference_dir=str(prev),feature='Frozen SGKLD P3; eval mode; same image preprocessing; gradients CPU float64',
  methods=methods,steps=steps,primary_step=.002,semantic_weight=.1,BD_alpha=3.,grid_size=5,
  new_semantics='Current prototype: per-grid cosine AFTER interpolation, C4 minimum from content',
  legacy_content='Supplied ordinary content: pre-normalized feature field, squared residual, geometry-selected C4 branch',
  update='L2-normalized gradient in GT-relative center/logsize/quarter-turn coordinates; free and changed-coordinate-only',
  no_training=True,no_tuning=True,metric='independent double-precision polygon clipping, validated against analytic and Shapely results',
  source_sha256={str(root/'mmrotate/models/losses/spatial_semantic_bd_loss.py'):sha(root/'mmrotate/models/losses/spatial_semantic_bd_loss.py'),str(Path(__file__)):sha(Path(__file__))})
 (out/'protocol.json').write_text(json.dumps(settings,ensure_ascii=False,indent=2))
 rows=[];start=time.time();legacy_error=0.;initial_error=0.
 data=root/'data/DIOR-R'
 with (out/'paired_rows.jsonl').open('w') as stream:
  for ni,image_id in enumerate(protocol['image_ids']):
   image=mmcv.imread(str(data/'JPEGImages-trainval'/f'{image_id}.jpg'))
   image=mmcv.imrescale(image,(800,800));image=mmcv.imnormalize(image.astype(np.float32),np.array(cfg.img_norm_cfg['mean']),np.array(cfg.img_norm_cfg['std']),to_rgb=True);image=mmcv.impad_to_multiple(image,32)
   tensor=torch.from_numpy(image.transpose(2,0,1).copy())[None].cuda()
   with torch.no_grad():feature=model.extract_feat(tensor)[0].cpu().double()
   del tensor
   field=feature/(feature.square().sum(1,keepdim=True)+1e-12).sqrt()
   for item in byimage.get(image_id,[]):
    previous=bygt[(image_id,item['object'])];t=torch.tensor([item['box']],dtype=torch.float64).expand(len(previous),-1)
    e=torch.zeros((len(previous),5),dtype=torch.float64)
    for i,z in enumerate(previous):e[i,z['dim']]=z['amplitude']
    e.requires_grad_();p=decode_local(e,t);ids=torch.zeros(len(p),dtype=torch.long)
    bd,spatial=semantic_bd_components(p,t,[feature],ids,ids,[8])
    _,pooled=semantic_bd_components(p,t,[feature],ids,ids,[8],mode='pooled')
    branch=align_by_geometry(p,t)[0]
    legacy=(_phi(field,p,8.,5,branch)-_phi(field,t,8.,5)).square().sum(-1)
    gb=torch.autograd.grad(bd.sum(),e,retain_graph=True)[0]
    gs=torch.autograd.grad(spatial.sum(),e,retain_graph=True)[0]
    gp=torch.autograd.grad(pooled.sum(),e,retain_graph=True)[0]
    gl=torch.autograd.grad(legacy.sum(),e)[0]
    gradients=dict(bd=gb,legacy_content=gl,pooled_cosine=gp,spatial_cosine=gs,bd_pooled_cosine=gb+.1*gp,bd_spatial_cosine=gb+.1*gs)
    initial=ious(p,t);initial_error=max(initial_error,float(np.max(np.abs(initial-np.array([z['initial_iou'] for z in previous])))))
    updates={};norms={}
    for name,g in gradients.items():
     assert torch.isfinite(g).all(),name
     norms[name]=g.norm(dim=1).tolist();updates[name]={}
     for mode in ['free','restricted']:
      grad=g.clone()
      if mode=='restricted':
       mask=torch.zeros_like(grad)
       for i,z in enumerate(previous):mask[i,z['dim']]=1
       grad*=mask
      direction=grad/grad.norm(dim=1,keepdim=True).clamp_min(1e-12)
      updates[name][mode]=[ (ious(decode_local(e.detach()-step*direction,t),t)-initial).tolist() for step in steps]
    legacy_error=max(legacy_error,float(np.max(np.abs(np.array(updates['legacy_content']['free']).T-np.array([z['raw_free'] for z in previous])))))
    for i,z in enumerate(previous):
     row={k:z[k] for k in ['image','object','cls','dim','kind','amplitude','aspect','area','initial_iou']}
     row['grad_norm']={name:norms[name][i] for name in methods}
     row['updates']={name:{mode:[arr[i] for arr in updates[name][mode]] for mode in ['free','restricted']} for name in methods}
     rows.append(row);stream.write(json.dumps(row)+'\n')
   stream.flush()
   progress=dict(images_done=ni+1,cases=len(rows),seconds=round(time.time()-start,1))
   (out/'progress.json').write_text(json.dumps(progress));print(json.dumps(progress),flush=True)
 assert len(rows)==5853 and initial_error<1e-8 and legacy_error<1e-7,(len(rows),initial_error,legacy_error)
 groups={'all':rows}
 for kind in ['translation','scale','angle']:groups[kind]=[z for z in rows if z['kind']==kind]
 for name,fun in [('near_square',lambda z:z['aspect']<=1.2),('slender',lambda z:z['aspect']>=3),('small',lambda z:z['area']<32**2)]:groups[name]=[z for z in rows if fun(z)]
 def summarize(part):
  if not part:return dict(n=0)
  result=dict(n=len(part),images=len(set(z['image'] for z in part)),methods={})
  for name in methods:
   arr=np.array([z['updates'][name]['free'] for z in part]);restricted=np.array([z['updates'][name]['restricted'] for z in part])
   delta=arr[:,1]-np.array([z['updates']['bd']['free'][1] for z in part])
   image_ids=sorted(set(z['image'] for z in part));clusters=[np.array([delta[i] for i,z in enumerate(part) if z['image']==im]) for im in image_ids]
   sums=np.array([x.sum() for x in clusters]);counts=np.array([len(x) for x in clusters]);idx=np.random.default_rng(42).integers(len(clusters),size=(2000,len(clusters)))
   boot=sums[idx].sum(1)/counts[idx].sum(1)
   result['methods'][name]=dict(mean=arr.mean(0).tolist(),median=np.median(arr,axis=0).tolist(),improve=(arr>1e-6).mean(0).tolist(),harm=(arr< -1e-6).mean(0).tolist(),restricted_mean=restricted.mean(0).tolist(),primary_delta_vs_BD=float(delta.mean()),primary_delta_vs_BD_cluster_CI95=np.percentile(boot,[2.5,97.5]).tolist(),primary_win_vs_BD=float((delta>1e-6).mean()))
  return result
 summary=dict(status='completed',protocol=settings,gt_count=len(meta),case_count=len(rows),elapsed=time.time()-start,initial_IoU_reproduction_max_error=initial_error,legacy_content_update_reproduction_max_error=legacy_error,groups={k:summarize(v) for k,v in groups.items()})
 (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
 print('COMPLETED '+json.dumps({k:{name:v['mean'][1] for name,v in group.get('methods',{}).items()} for k,group in summary['groups'].items()}),flush=True)


if __name__=='__main__':main()
