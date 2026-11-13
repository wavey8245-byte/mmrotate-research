"""Analytic sampling, box gradients, pixel decoding and real FCOS SIoU checks."""
import math, unittest
import mmcv
import torch
from mmcv.ops import box_iou_rotated
from mmrotate.models.builder import build_head
from mmrotate.models.losses.semantic_iou_loss import (
    SemanticIoULoss,sample_box_features,semantic_scores,geometric_iou)


class TestSemanticSIoU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def smooth(self):
        y,x=torch.meshgrid(torch.linspace(.1,1.,64,dtype=torch.float64),
                           torch.linspace(.1,1.,64,dtype=torch.float64),indexing='ij')
        return torch.stack([x,y,x*y,x.square()+2*y.square()])[None]

    def test_pixel_to_feature_alignment_and_packed_regions(self):
        stride=4
        y,x=torch.meshgrid((torch.arange(24,dtype=torch.float64)+.5)*stride,
                           (torch.arange(24,dtype=torch.float64)+.5)*stride,indexing='ij')
        feature=torch.stack([x,y])[None]
        boxes=torch.tensor([[30.13,40.27,11.,7.,.21],[55.19,51.31,9.,8.,-.32]],dtype=torch.float64,requires_grad=True)
        for context in (None,1.5):
            sampled=sample_box_features(feature,boxes,stride,5,context)
            torch.testing.assert_close(sampled.mean((2,3)),boxes[:,:2],atol=1e-11,rtol=1e-11)
            gx,=torch.autograd.grad(sampled[:,0].mean(),boxes,retain_graph=True)
            torch.testing.assert_close(gx[:,0],torch.full((2,),.5,dtype=gx.dtype))
        self.assertTrue(torch.autograd.gradcheck(lambda b:sample_box_features(feature,b,stride,5),(boxes,)))

    def test_exact_two_region_semantic_score(self):
        feature=torch.zeros(1,2,64,64,dtype=torch.float64)
        feature[:,0,:,:32]=1
        feature[:,1,:,32:]=1
        p=torch.tensor([[15.13,20.27,8.,6.,.21]],dtype=torch.float64)
        t=torch.tensor([[47.19,20.31,8.,6.,-.17]],dtype=torch.float64)
        indices=torch.zeros(1,dtype=torch.long)
        score,s1,s2=semantic_scores(p,t,[feature],indices,indices,[1])
        self.assertAlmostEqual(float(s1),math.exp(-.3*2))
        self.assertAlmostEqual(float(s2),.5)
        self.assertAlmostEqual(float(score),.5*math.exp(-.6)+.25)

    def test_semantic_gradients_reach_nonoverlapping_box_and_features(self):
        feature=self.smooth().requires_grad_()
        p=torch.tensor([[10.13,11.27,7.,5.,.23]],dtype=torch.float64,requires_grad=True)
        t=torch.tensor([[43.19,41.31,10.,8.,-.17]],dtype=torch.float64)
        idx=torch.zeros(1,dtype=torch.long)
        def objective(box):
            return 1-semantic_scores(box,t,[feature],idx,idx,[1])[0]
        value=objective(p)
        gp,gf=torch.autograd.grad(value.sum(),(p,feature),retain_graph=True)
        self.assertTrue(torch.isfinite(gp).all() and torch.isfinite(gf).all())
        self.assertGreater(float(gp[:,:2].abs().sum()),0)
        self.assertGreater(float(gp[:,4].abs().sum()),0)
        self.assertGreater(float(gf.abs().sum()),0)
        self.assertTrue(torch.autograd.gradcheck(objective,(p,),atol=1e-6,rtol=1e-4))

    @unittest.skipUnless(torch.cuda.is_available(),'Native differentiable IoU needs CUDA')
    def test_iou_reference_degenerate_and_reduction(self):
        device='cuda'
        t=torch.tensor([[0.,0.,9.,6.,.2],[0.,0.,8.,5.,-.3],[0.,0.,10.,7.,.1]],device=device)
        p=t.clone()
        p[0,:2]+=.4
        p[1,:2]+=40
        p[2,2:4]=0
        p.requires_grad_()
        iou=geometric_iou(p,t)
        ref=box_iou_rotated(p[:2].detach(),t[:2],aligned=True).reshape(-1)
        torch.testing.assert_close(iou[:2],ref,atol=2e-5,rtol=2e-5)
        loss=SemanticIoULoss(semantic_enabled=False)
        weights=torch.tensor([1.,.25,0.],device=device)
        raw=loss(p,t,reduction_override='none')
        value=loss(p,t,weight=weights,avg_factor=weights.sum())
        torch.testing.assert_close(value,(raw*weights).sum()/weights.sum())
        value.backward()
        self.assertTrue(torch.isfinite(p.grad).all())
        self.assertEqual(float(loss(p[:0],t[:0])),0.)
        self.assertEqual(float(loss(p,t,weight=weights*0,avg_factor=1e-6)),0.)
        same=loss(t,t)
        self.assertLess(abs(float(same)),1e-5)

    @unittest.skipUnless(torch.cuda.is_available(),'Native differentiable IoU needs CUDA')
    def test_full_loss_has_nonoverlap_semantic_gradient(self):
        feature=self.smooth().float().cuda().requires_grad_()
        p=torch.tensor([[10.13,11.27,7.,5.,.23]],device='cuda',requires_grad=True)
        t=torch.tensor([[43.19,41.31,10.,8.,-.17]],device='cuda')
        idx=torch.zeros(1,dtype=torch.long,device='cuda')
        control=SemanticIoULoss(semantic_enabled=False)(p,t)
        gc,=torch.autograd.grad(control,p,retain_graph=True)
        self.assertLess(float(gc.abs().max()),1e-6)
        value=SemanticIoULoss()(p,t,feature_maps=[feature],image_indices=idx,
                                level_indices=idx,feature_strides=[1])
        gp,gf=torch.autograd.grad(value,(p,feature))
        self.assertTrue(torch.isfinite(gp).all() and torch.isfinite(gf).all())
        self.assertGreater(float(gp[:,:2].abs().sum()),0)
        self.assertGreater(float(gp[:,4].abs().sum()),0)
        self.assertGreater(float(gf.abs().sum()),0)
        self.assertGreaterEqual(float(value),0)
        self.assertLessEqual(float(value),1)

    @unittest.skipUnless(torch.cuda.is_available(),'FCOS native IoU needs CUDA')
    def test_actual_head_matches_native_linear_control_and_decodes_pixels(self):
        cfg=mmcv.Config.fromfile('configs/dior_r/server_trainval/paper_semantic_linear_control.py')
        native_cfg=mmcv.Config.fromfile('configs/dior_r/server_trainval/riou.py')
        native_cfg.model.bbox_head.loss_bbox=dict(type='RotatedIoULoss',mode='linear',loss_weight=1.)
        torch.manual_seed(42)
        native=build_head(native_cfg.model.bbox_head).cuda()
        native.init_weights()
        control=build_head(cfg.model.bbox_head).cuda()
        control.load_state_dict(native.state_dict(),strict=True)
        features=[torch.randn(2,256,k,k,device='cuda') for k in (16,8,4,2,1)]
        boxes=[torch.tensor([[64.,64.,48.,24.,.2]],device='cuda'),
               torch.tensor([[52.,50.,40.,22.,-.17]],device='cuda')]
        labels=[torch.tensor([1],device='cuda'),torch.tensor([2],device='cuda')]
        meta=[dict(img_shape=(128,128,3),pad_shape=(128,128,3),scale_factor=1.)]*2
        actual=native.loss(*native(features),boxes,labels,meta)
        captured={}
        original=control.loss_bbox.forward
        def capture(pred,target,**kw):
            captured.update(target=target.detach(),image=kw['image_indices'])
            return original(pred,target,**kw)
        control.loss_bbox.forward=capture
        expected=control.loss(*control(features),boxes,labels,meta,semantic_features=features)
        for name in ('loss_cls','loss_centerness'):
            torch.testing.assert_close(actual[name],expected[name],atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(actual['loss_bbox'],expected['loss_bbox'],atol=2e-4,rtol=2e-4)
        for image in range(2):
            selected=captured['target'][captured['image']==image]
            torch.testing.assert_close(selected,boxes[image].expand_as(selected),atol=2e-5,rtol=2e-5)

    @unittest.skipUnless(torch.cuda.is_available(),'FCOS native IoU needs CUDA')
    def test_real_semantic_head_backward_and_empty(self):
        cfg=mmcv.Config.fromfile('configs/dior_r/server_trainval/paper_semantic_siou.py')
        torch.manual_seed(42)
        head=build_head(cfg.model.bbox_head).cuda()
        head.init_weights()
        features=[torch.randn(1,256,k,k,device='cuda',requires_grad=True) for k in (16,8,4,2,1)]
        boxes=[torch.tensor([[64.,64.,48.,24.,.2]],device='cuda')]
        labels=[torch.tensor([1],device='cuda')]
        meta=[dict(img_shape=(128,128,3),pad_shape=(128,128,3),scale_factor=1.)]
        losses=head.loss(*head(features),boxes,labels,meta,semantic_features=features)
        total=sum(losses.values())
        self.assertTrue(torch.isfinite(total))
        total.backward()
        for p in head.parameters():
            if p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all())
        self.assertGreater(float(head.conv_angle.weight.grad.abs().sum()),0)
        self.assertGreater(sum(float(f.grad.abs().sum()) for f in features if f.grad is not None),0)
        empty=head.loss(*head(features),[boxes[0][:0]],[labels[0][:0]],meta,semantic_features=features)
        self.assertEqual(float(empty['loss_bbox']),0.)

if __name__=='__main__':
    unittest.main(verbosity=2)
