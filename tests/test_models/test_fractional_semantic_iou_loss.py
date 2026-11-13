"""Analytic recurrence, explicit surrogate gradients, identity and cache integrity."""
import copy
import math
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
import mmcv
from mmrotate.models.builder import build_head
from mmrotate.models.losses.fractional_semantic_iou_loss import (
    FCOSFrSIoUEngineeringLoss,gl_coefficients,fractional_terms,
    canonical_box_coordinates,object_history_keys)
from mmrotate.models.losses.semantic_iou_loss import SemanticIoULoss
from mmrotate.datasets.pipelines.transforms import RRandomFlip
from mmrotate.core.frsiou_memory_hook import FrSIoUMemoryHook,digest


class TestFractionalMemory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_gl_coefficients_match_expansion(self):
        self.assertEqual(gl_coefficients(.5,2),[1.,-.5,-.125])
        self.assertAlmostEqual(gl_coefficients(.3,4)[3],-.0595)
        for value in (0.,1.,-.2):
            with self.assertRaises(ValueError):
                gl_coefficients(value,2)

    def test_values_recurrence_forward_and_current_gradient(self):
        q=torch.tensor([.6],dtype=torch.double,requires_grad=True)
        z=torch.zeros(1,5,dtype=torch.double,requires_grad=True)
        group=torch.zeros(1,dtype=torch.long)
        lag=torch.tensor([[.5,.4]],dtype=torch.double)
        gradient=torch.zeros(1,2,5,dtype=torch.double)
        values,_,_,d,w=fractional_terms(q,z,group,torch.ones_like(q),lag,gradient,[1.,-.5,-.125],.3)
        self.assertAlmostEqual(float(d),.3)
        self.assertAlmostEqual(float(w),math.exp(-.09))
        self.assertAlmostEqual(float(values),math.exp(-.09)*.4)
        g,=torch.autograd.grad(values.sum(),q)
        self.assertAlmostEqual(float(g),-math.exp(-.09)*(1+.3*.4))
        self.assertTrue(torch.autograd.gradcheck(lambda x:fractional_terms(x,z,group,torch.ones_like(x),lag,gradient,[1.,-.5,-.125],.3)[0],(q,)))

    def test_history_tangent_has_distinct_analytic_gradient(self):
        angle=torch.tensor(.0,dtype=torch.double,requires_grad=True)
        q=(.6+.1*angle)[None]
        z=torch.stack([angle*0]*4+[angle])[None]
        group=torch.zeros(1,dtype=torch.long)
        lag=torch.tensor([[.5,.4]],dtype=torch.double)
        g=torch.zeros(1,2,5,dtype=torch.double)
        g[0,0,4]=.3
        args=(q,z,group,torch.ones_like(q),lag,g,[1.,-.5,-.125],.3)
        a=fractional_terms(*args,gradient_mode='values')[0]
        b=fractional_terms(*args,gradient_mode='box_tangent')[0]
        torch.testing.assert_close(a,b)
        ga,=torch.autograd.grad(a.sum(),angle,retain_graph=True)
        gb,=torch.autograd.grad(b.sum(),angle)
        expected=math.exp(-.09)*(-.1-.3*.4*(.1-.5*.3))
        self.assertAlmostEqual(float(gb),expected)
        self.assertGreater(abs(float(gb-ga)),1e-3)
        # Numerical test is specifically of the frozen-reference local Taylor
        # objective, NOT a claim that moving-reference stale values have this
        # ordinary derivative under a fresh production forward each time.
        x=torch.tensor([[.1,.2,.3,.4,.5]],dtype=torch.double,requires_grad=True)
        q0=torch.tensor([.6],dtype=torch.double,requires_grad=True)
        ref=x.detach().clone()
        fn=lambda score,box:fractional_terms(score,box,group,torch.ones_like(score),lag,g,[1.,-.5,-.125],.3,'box_tangent',reference=ref)[0]
        self.assertTrue(torch.autograd.gradcheck(fn,(q0,x)))

    def test_group_weighting_matches_explicit_scalar(self):
        q=torch.tensor([.3,.7,.2],dtype=torch.double,requires_grad=True)
        z=torch.zeros(3,5,dtype=torch.double)
        group=torch.tensor([0,0,1])
        weights=torch.tensor([1.,3.,2.],dtype=torch.double)
        lag=torch.tensor([[.4,.2],[.8,.5]],dtype=torch.double)
        v,m,_,d,w=fractional_terms(q,z,group,weights,lag,torch.zeros(2,2,5,dtype=torch.double),[1.,-.5,-.125],.3)
        torch.testing.assert_close(m,torch.tensor([.6,.2],dtype=torch.double))
        expected=(math.exp(-.3*(.6-.2-.025))*(1*.7+3*.3)+math.exp(-.3*(.2-.4-.0625))*2*.8)/6
        self.assertAlmostEqual(float((v*weights).sum()/weights.sum()),float(expected),places=7)

    def test_canonical_box_coordinates_across_flips_and_scale(self):
        p=torch.tensor([[40.13,48.27,18.,10.,.31]],dtype=torch.double)
        t=torch.tensor([[43.19,45.31,21.,12.,.17]],dtype=torch.double)
        image=torch.zeros(1,dtype=torch.long)
        base,_=canonical_box_coordinates(p,t,image,[dict(scale_factor=[1.,1.,1.,1.])])
        transform=RRandomFlip(version='le90')
        for direction in ('horizontal','vertical','diagonal'):
            pp=transform.bbox_flip(p.numpy(),(128,128),direction)
            tt=transform.bbox_flip(t.numpy(),(128,128),direction)
            actual,_=canonical_box_coordinates(torch.from_numpy(pp),torch.from_numpy(tt),image,
                [dict(flip=True,flip_direction=direction,scale_factor=[1.,1.,1.,1.])])
            torch.testing.assert_close(base,actual,atol=1e-12,rtol=1e-12)
        pp=p.clone(); tt=t.clone()
        pp[:,:4]*=2; tt[:,:4]*=2
        actual,_=canonical_box_coordinates(pp,tt,image,[dict(scale_factor=[2.,2.,2.,2.])])
        torch.testing.assert_close(base,actual)

    def test_history_identity_uses_gt_and_separates_images_and_levels(self):
        gt=[torch.tensor([[10.,10.,8.,4.,.2],[20.,20.,7.,3.,-.3]]),torch.tensor([[10.,10.,8.,4.,.2]])]
        target=torch.stack([gt[0][1],gt[0][0],gt[0][1],gt[1][0]])
        images=torch.tensor([0,0,0,1]); levels=torch.tensor([0,0,1,0])
        keys,groups=object_history_keys(target,images,levels,[dict(ori_filename='a.jpg'),dict(ori_filename='b.jpg')],gt)
        self.assertEqual(keys,[('a.jpg',1,0),('a.jpg',0,0),('a.jpg',1,1),('b.jpg',0,0)])
        self.assertEqual(groups.tolist(),[0,1,2,3])

    def test_history_roundtrip_capacity_and_rejection(self):
        loss=FCOSFrSIoUEngineeringLoss(max_history_keys=2)
        for i in range(3):
            loss._pending=[(('image',i,0),.3+i*.1,[0.,0.,0.,0.,.1])]
            loss.commit_history()
        self.assertEqual(list(loss.history),[('image',1,0),('image',2,0)])
        restored=FCOSFrSIoUEngineeringLoss(max_history_keys=2)
        restored.import_history(loss.export_history())
        self.assertEqual(loss.export_history(),restored.export_history())
        with self.assertRaises(ValueError):
            FCOSFrSIoUEngineeringLoss(history_gradient='values').import_history(loss.export_history())

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA native IoU required')
    def test_lambda_zero_matches_siou_and_memory_is_transactional(self):
        y,x=torch.meshgrid(torch.linspace(.1,1.,64,device='cuda'),torch.linspace(.1,1.,64,device='cuda'),indexing='ij')
        feature=torch.stack([x,y,x*y,x*x+y*y])[None].requires_grad_()
        p=torch.tensor([[10.13,11.27,7.,5.,.23]],device='cuda',requires_grad=True)
        target=torch.tensor([[43.19,41.31,10.,8.,-.17]],device='cuda')
        idx=torch.zeros(1,dtype=torch.long,device='cuda')
        kw=dict(feature_maps=[feature],image_indices=idx,level_indices=idx,feature_strides=[1])
        control=SemanticIoULoss()(p,target,**kw)
        loss=FCOSFrSIoUEngineeringLoss(modulation_lambda=0.)
        loss.set_batch_context([dict(ori_filename='a.jpg',scale_factor=1.)],[target])
        value=loss(p,target,**kw)
        torch.testing.assert_close(value,control)
        ga=torch.autograd.grad(value,(p,feature),retain_graph=True)
        gb=torch.autograd.grad(control,(p,feature),retain_graph=True)
        for a,b in zip(ga,gb):
            torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)
        self.assertEqual(len(loss.history),0)
        self.assertIsNone(feature.grad)
        with self.assertRaises(RuntimeError):
            loss(p,target,**kw)
        value.backward()
        loss.commit_history()
        self.assertEqual(len(loss.history),1)
        self.assertIsNotNone(feature.grad)
        saved=copy.deepcopy(loss.export_history())
        with torch.no_grad():
            loss(p,target,**kw)
        self.assertEqual(saved,loss.export_history())
        self.assertEqual(loss.last_diagnostics['history_hit_groups'],1)
        loss.clear_batch_context()

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA head required')
    def test_actual_head_reuses_history_and_handles_empty_images(self):
        cfg=mmcv.Config.fromfile('configs/dior_r/server_trainval/paper_frsiou_box_tangent.py')
        head=build_head(cfg.model.bbox_head).cuda()
        head.init_weights()
        gt=[torch.tensor([[64.,64.,48.,24.,.2]],device='cuda')]
        labels=[torch.tensor([1],device='cuda')]
        meta=[dict(ori_filename='a.jpg',img_shape=(128,128,3),pad_shape=(128,128,3),scale_factor=1.)]
        for step in range(3):
            features=[torch.randn(1,256,k,k,device='cuda',requires_grad=True) for k in (16,8,4,2,1)]
            losses=head.loss(*head(features),gt,labels,meta,semantic_features=features)
            total=sum(losses.values())
            total.backward()
            self.assertTrue(torch.isfinite(total))
            self.assertIsNone(head.loss_bbox._context)
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in head.parameters() if p.grad is not None))
            self.assertGreater(float(head.conv_angle.weight.grad.abs().sum()),0)
            if step:
                self.assertGreater(head.loss_bbox.last_diagnostics['history_hit_fraction'],0)
                self.assertGreater(head.loss_bbox.last_diagnostics['transported_history_gradient_l2'],0)
            head.loss_bbox.commit_history()
            head.zero_grad(set_to_none=True)
        features=[f.detach().requires_grad_() for f in features]
        empty=head.loss(*head(features),[gt[0][:0]],[labels[0][:0]],meta,semantic_features=features)
        self.assertEqual(float(empty['loss_bbox']),0.)
        self.assertIsNone(head.loss_bbox._pending)

    def test_epoch_sidecar_restores_exact_history_and_rejects_changed_checkpoint(self):
        from types import SimpleNamespace
        loss=FCOSFrSIoUEngineeringLoss()
        loss._pending=[(('a.jpg',0,0),.6,[.1,0,0,0,.2])]
        loss.commit_history()
        with tempfile.TemporaryDirectory() as directory:
            cp=Path(directory)/'epoch_1.pth'
            torch.save(dict(meta=dict(epoch=1,iter=10)),str(cp))
            runner=SimpleNamespace(work_dir=directory,epoch=0,iter=10,
                model=SimpleNamespace(bbox_head=SimpleNamespace(loss_bbox=loss)),
                logger=SimpleNamespace(info=lambda *a:None))
            FrSIoUMemoryHook().after_train_epoch(runner)
            other=FCOSFrSIoUEngineeringLoss()
            runner.model.bbox_head.loss_bbox=other
            runner.epoch=1
            hook=FrSIoUMemoryHook(resume_checkpoint=str(cp))
            hook.before_run(runner)
            self.assertEqual(other.export_history(),loss.export_history())
            cp.write_bytes(b'changed')
            with self.assertRaises(RuntimeError):
                hook.before_run(runner)


    def test_native_runner_hook_order_and_epoch_checkpoint_counters(self):
        import logging
        from mmcv.runner import EpochBasedRunner,OptimizerHook,CheckpointHook
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.value=torch.nn.Parameter(torch.tensor(1.))
                self.bbox_head=torch.nn.Module()
                self.bbox_head.loss_bbox=FCOSFrSIoUEngineeringLoss()
            def train_step(self,batch,optimizer):
                self.bbox_head.loss_bbox._pending=[(('a.jpg',0,0),.6,[0.,0.,0.,0.,.1])]
                loss=self.value.square()
                return dict(loss=loss,log_vars=dict(loss=float(loss.detach())),num_samples=1)
        with tempfile.TemporaryDirectory() as directory:
            model=Tiny()
            optimizer=torch.optim.SGD(model.parameters(),lr=.1)
            runner=EpochBasedRunner(model=model,optimizer=optimizer,work_dir=directory,
                logger=logging.getLogger('frsiou_hook_test'),max_epochs=1)
            observations=[]
            original=model.bbox_head.loss_bbox.commit_history
            def commit():
                observations.append(float(model.value.detach()))
                original()
            model.bbox_head.loss_bbox.commit_history=commit
            runner.register_hook(OptimizerHook(),priority='ABOVE_NORMAL')
            runner.register_hook(CheckpointHook(interval=1),priority='NORMAL')
            runner.register_hook(FrSIoUMemoryHook(),priority='BELOW_NORMAL')
            runner.run([[None,None]],[('train',1)])
            self.assertAlmostEqual(observations[0],.8,places=6)
            self.assertAlmostEqual(observations[1],.64,places=6)
            cp=Path(directory)/'epoch_1.pth'
            native=torch.load(str(cp),map_location='cpu')
            side=torch.load(str(cp.with_suffix('.frsiou_history.pth')),map_location='cpu')
            self.assertEqual(native['meta']['epoch'],1)
            self.assertEqual(native['meta']['iter'],2)
            self.assertEqual(side['iteration'],2)
            self.assertEqual(side['checkpoint_sha256'],digest(cp))
            self.assertEqual(side['history']['committed_updates'],2)



if __name__=='__main__':
    unittest.main(verbosity=2)
