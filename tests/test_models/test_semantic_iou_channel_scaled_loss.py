"""Scale invariance, analytic score and nonoverlap gradients of engineering SIoU."""
import math
import unittest
import torch
from mmrotate.models.losses.semantic_iou_channel_scaled_loss import ChannelScaledSemanticIoULoss
from mmrotate.models.losses.semantic_iou_loss import semantic_scores


class TestChannelScale(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_channel_replication_invariance(self):
        y,x=torch.meshgrid(torch.linspace(.1,1.,64,dtype=torch.double),torch.linspace(.1,1.,64,dtype=torch.double),indexing='ij')
        f=torch.stack([x,y,x*y,x*x+2*y*y])[None]
        p=torch.tensor([[10.13,11.27,7.,5.,.23]],dtype=torch.double,requires_grad=True)
        t=torch.tensor([[43.19,41.31,10.,8.,-.17]],dtype=torch.double)
        idx=torch.zeros(1,dtype=torch.long)
        a=semantic_scores(p,t,[f/math.sqrt(4)],idx,idx,[1])[0]
        b=semantic_scores(p,t,[f.repeat(1,4,1,1)/math.sqrt(16)],idx,idx,[1])[0]
        torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)
        ga,=torch.autograd.grad(a.sum(),p,retain_graph=True)
        gb,=torch.autograd.grad(b.sum(),p)
        torch.testing.assert_close(ga,gb,atol=1e-12,rtol=1e-12)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA IoU required')
    def test_full_loss_analytic_orthogonal_regions(self):
        f=torch.zeros(1,256,64,64,device='cuda')
        f[:,0,:,:32]=16
        f[:,1,:,32:]=16
        p=torch.tensor([[15.13,20.27,8.,6.,.21]],device='cuda')
        t=torch.tensor([[47.19,20.31,8.,6.,-.17]],device='cuda')
        idx=torch.zeros(1,device='cuda',dtype=torch.long)
        loss=ChannelScaledSemanticIoULoss()
        value=loss(p,t,feature_maps=[f],image_indices=idx,level_indices=idx,feature_strides=[1])
        self.assertAlmostEqual(float(value),1-(.5*math.exp(-.6)+.25),places=6)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA IoU required')
    def test_full_loss_nonoverlap_gradients_and_zero_features(self):
        y,x=torch.meshgrid(torch.linspace(.1,1.,64,device='cuda'),torch.linspace(.1,1.,64,device='cuda'),indexing='ij')
        f=torch.stack([x,y,x*y,x*x+2*y*y])[None].requires_grad_()
        p=torch.tensor([[10.13,11.27,7.,5.,.23]],device='cuda',requires_grad=True)
        t=torch.tensor([[43.19,41.31,10.,8.,-.17]],device='cuda')
        idx=torch.zeros(1,device='cuda',dtype=torch.long)
        loss=ChannelScaledSemanticIoULoss()
        value=loss(p,t,feature_maps=[f],image_indices=idx,level_indices=idx,feature_strides=[1])
        gp,gf=torch.autograd.grad(value,(p,f))
        self.assertTrue(torch.isfinite(gp).all() and torch.isfinite(gf).all())
        self.assertGreater(float(gp[:,:2].abs().sum()),0)
        self.assertGreater(float(gp[:,4].abs().sum()),0)
        self.assertGreater(float(gf.abs().sum()),0)
        zeros=torch.zeros_like(f,requires_grad=True)
        value=loss(p,t,feature_maps=[zeros],image_indices=idx,level_indices=idx,feature_strides=[1])
        gp,gf=torch.autograd.grad(value,(p,zeros))
        self.assertTrue(torch.isfinite(gp).all() and torch.isfinite(gf).all())


if __name__=='__main__':
    unittest.main()
