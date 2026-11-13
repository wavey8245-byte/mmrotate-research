"""Analytical/gradient validation of the semantic-BD proposal; no training run."""
import json
import math
from pathlib import Path
import unittest
import torch
from mmcv.ops import box_iou_rotated
from mmrotate.models.losses.paper_rotated_losses import bd_values, PaperBDLoss
from mmrotate.models.losses.spatial_semantic_bd_loss import SpatialSemanticBDLoss,semantic_bd_components

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'work_dirs/dior_r_trainval/semantic_bd_validation_20261104'
OBS={}


def maps(constant=False,requires_grad=False,device='cpu'):
    axis=torch.arange(96,device=device,dtype=torch.float64)+.5
    yy,xx=torch.meshgrid(axis,axis,indexing='ij')
    xx=(xx-48)/32; yy=(yy-48)/32
    f=torch.stack([torch.ones_like(xx),xx,yy,xx*0+.5])[None]
    if constant: f=torch.ones_like(f)
    return [f.requires_grad_(requires_grad)]


def box(angle=0.,w=40.,h=40.,requires_grad=False,device='cpu'):
    return torch.tensor([[48.,48.,w,h,angle]],dtype=torch.float64,device=device,requires_grad=requires_grad)


def components(p,t,features=None,mode='spatial'):
    ids=torch.zeros(len(p),dtype=torch.long,device=p.device)
    return semantic_bd_components(p,t,maps(device=p.device) if features is None else features,ids,ids,[1],mode=mode)


class Checks(unittest.TestCase):
    def test_square_blind_spot_and_spatial_signal(self):
        p,t=box(math.pi/8),box(-math.pi/8)
        g,sem=components(p,t);_,pooled=components(p,t,mode='pooled')
        iou=float(box_iou_rotated(p.float(),t.float())[0,0])
        self.assertLess(abs(float(g)),1e-7)
        self.assertGreater(float(sem),.01)
        self.assertLess(float(pooled),1e-10)
        self.assertAlmostEqual(iou,1/math.sqrt(2),places=5)
        OBS['square_counterexample']=dict(BD=float(g),spatial_semantic=float(sem),pooled_semantic=float(pooled),true_rotated_IoU=iou)

    def test_identity_zero(self):
        p=box(.23,w=42,h=24);g,s=components(p,p)
        self.assertLess(abs(float(g)),1e-7);self.assertLess(float(s),1e-12)

    def test_pi_equivalence(self):
        _,s=components(box(.23+math.pi,w=42,h=24),box(.23,w=42,h=24))
        self.assertLess(float(s),1e-12)

    def test_width_height_exchange(self):
        _,s=components(box(.23+math.pi/2,w=24,h=42),box(.23,w=42,h=24))
        self.assertLess(float(s),1e-12)

    def test_square_90_equivalence(self):
        _,s=components(box(.23+math.pi/2),box(.23))
        self.assertLess(float(s),1e-12)

    def test_constant_features_supply_no_direction(self):
        _,s=components(box(.31),box(0),maps(constant=True))
        self.assertLess(float(s),1e-12)

    def test_detached_features_target_live_box_grad(self):
        f=maps(requires_grad=True);p=box(.28,requires_grad=True);t=box(0,requires_grad=True)
        _,s=components(p,t,f);s.sum().backward()
        self.assertIsNone(f[0].grad);self.assertIsNone(t.grad)
        self.assertGreater(abs(float(p.grad[0,4])),1e-4)
        self.assertTrue(torch.isfinite(p.grad).all())
        OBS['coordinate_gradient']=dict(angle_gradient=float(p.grad[0,4]),feature_gradient_detached=True,target_gradient_detached=True)

    def test_angle_finite_difference(self):
        p=box(.28,requires_grad=True);_,s=components(p,box(0));g=torch.autograd.grad(s.sum(),p)[0][0,4]
        eps=1e-5;plus=components(box(.28+eps),box(0))[1];minus=components(box(.28-eps),box(0))[1]
        numerical=(plus-minus)/(2*eps)
        self.assertAlmostEqual(float(g),float(numerical),places=6)

    def test_short_angle_optimization(self):
        p=box(.30,requires_grad=True);t=box(0);start=float(components(p,t)[1]);initial_angle=float(p[0,4].detach())
        for _ in range(25):
            g,s=components(p,t);loss=g+.1*s
            gradient=torch.autograd.grad(loss.sum(),p)[0]
            with torch.no_grad():p[0,4]-=2.*gradient[0,4]
        final=float(components(p,t)[1]);angle=float(p[0,4].detach())
        self.assertLess(final,start*.9);self.assertLess(abs(angle),abs(initial_angle))
        OBS['synthetic_angle_updates']=dict(steps=25,initial_angle_deg=initial_angle*180/math.pi,final_angle_deg=angle*180/math.pi,initial_semantic=start,final_semantic=final,note='Synthetic spatial features, not real-image training.')

    def test_plain_bd_control_unchanged(self):
        p=torch.cat([box(.2,w=30,h=20),box(.3,w=40,h=25)]);t=torch.cat([box(0,w=35,h=24),box(.1,w=35,h=23)])
        w=torch.tensor([.4,.7],dtype=p.dtype)
        old=PaperBDLoss(alpha=3.)(p,t,weight=w,avg_factor=w.sum())
        control=SpatialSemanticBDLoss(semantic_weight=0.)(p,t,weight=w,avg_factor=w.sum())
        self.assertTrue(torch.allclose(old,control,atol=1e-12,rtol=1e-12))

    def test_empty_and_zero_weight(self):
        loss=SpatialSemanticBDLoss();p=box(.2,requires_grad=True);t=box(0)
        z=loss(p,t,weight=torch.zeros(1,dtype=p.dtype));z.backward();self.assertEqual(float(z),0.)
        self.assertEqual(float(loss(p[:0],t[:0])),0.)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_cpu_agree(self):
        cpu=components(box(.28),box(0));gpu=components(box(.28,device='cuda'),box(0,device='cuda'))
        for a,b in zip(cpu,gpu):self.assertTrue(torch.allclose(a,b.cpu(),atol=1e-9,rtol=1e-9))


if __name__=='__main__':
    REPORT.mkdir(parents=True,exist_ok=True)
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    out=dict(status='passed' if result.wasSuccessful() else 'failed',tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skips=len(result.skipped),observations=OBS,scope='Synthetic spatial feature identifiability and gradient tests only; no real semantic accuracy or AP claim.',real_training_started=False)
    (REPORT/'result.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
