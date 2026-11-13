"""BD unchanged; near-square theta-only intra exponential + rotated exterior ring.

A fixed proposal inspired by semantic-IoU cues, not a paper reproduction.
"""
import torch
import torch.nn.functional as F
from torch import nn
from mmcv.runner import HOOKS, Hook
from mmdet.models.losses.utils import weight_reduce_loss
from ..builder import ROTATED_LOSSES
from .paper_rotated_losses import bd_values


def _corners(boxes):
    axis = boxes.new_tensor([-.5, .5])
    vv, uu = torch.meshgrid(axis, axis, indexing='ij')
    w, h, a = [boxes[:, k, None, None] for k in (2, 3, 4)]
    x, y = w*uu, h*vv
    xy = torch.stack((a.cos()*x-a.sin()*y, a.sin()*x+a.cos()*y), -1)
    return xy+boxes[:, None, None, :2]


@torch.no_grad()
def geometry_branch(pred, target):
    cp, ct = _corners(pred), _corners(target)
    scale = target[:, 2:4].square().sum(-1).clamp_min(1e-12)
    errors = torch.stack([
        (torch.rot90(cp, k, (1, 2))-ct).square().sum(-1).mean((1, 2))/scale
        for k in range(4)], -1)
    return errors.argmin(-1)


def _sample(field, boxes, stride, grid_size):
    axis = (torch.arange(grid_size, device=boxes.device, dtype=boxes.dtype)+.5)/grid_size-.5
    vv, uu = torch.meshgrid(axis, axis, indexing='ij')
    cx, cy, w, h, angle = [boxes[:, i, None, None] for i in range(5)]
    x, y = w*uu, h*vv
    gx = cx+angle.cos()*x-angle.sin()*y
    gy = cy+angle.sin()*x+angle.cos()*y
    height, width = field.shape[-2:]
    grid = torch.stack((2*gx/(stride*width)-1, 2*gy/(stride*height)-1), -1)
    packed = grid.reshape(1, len(boxes)*grid_size, grid_size, 2)
    sampled = F.grid_sample(field, packed, mode='bilinear',
                            padding_mode='zeros', align_corners=False)
    return sampled.reshape(field.shape[1], len(boxes), grid_size, grid_size).permute(1, 0, 2, 3)


def context_angle_content_values(pred, target, normalized_field, stride=8., grid_size=5,
        min_size=1e-4, intra_tau=1., intra_beta=.5, context_scale=1.5, cosine_eps=1e-3):
    """Same C4 branch for inner and exterior grids; fixed GT-valid cosine support.

    Center/size and field values stop gradient; only decoded theta is active.
    A near-zero GT context vector is ignored. If all GT context vectors are
    invalid, that box's context residual is zero (inner residual is retained).
    """
    if len(pred)==0:
        return pred[:,4]*0
    target=target.detach();field=normalized_field.detach()
    p=torch.cat((pred[:,:2].detach(),pred[:,2:4].detach().clamp_min(min_size),pred[:,4:5]),-1)
    branch=geometry_branch(p.detach(),target)
    def aligned_sample(boxes):
        z=_sample(field,boxes,stride,grid_size)
        variants=torch.stack([torch.rot90(z,k,(-2,-1)) for k in range(4)],1)
        return variants[torch.arange(len(p),device=p.device),branch]
    fp=aligned_sample(p)
    with torch.no_grad():fg=_sample(field,target,stride,grid_size)
    distance=.25*(fp-fg).square().sum(1).mean((-2,-1))
    inner=-torch.expm1(-distance/intra_tau)
    factor=p.new_tensor([1.,1.,context_scale,context_scale,1.])
    cp=aligned_sample(p*factor)
    with torch.no_grad():cg=_sample(field,target*factor,stride,grid_size)
    axis=(torch.arange(grid_size,device=p.device,dtype=p.dtype)+.5)/grid_size-.5
    vv,uu=torch.meshgrid(axis,axis,indexing='ij')
    exterior=(context_scale*uu.abs()>.5)|(context_scale*vv.abs()>.5)
    # Ring mask is C4 symmetric, so alignment preserves its positions.
    npred=cp.norm(dim=1).clamp_min(cosine_eps)
    ntarget=cg.norm(dim=1)
    valid=(exterior[None] & (ntarget>cosine_eps)).detach()
    cosine=(cp*cg).sum(1)/(npred*ntarget.clamp_min(cosine_eps))
    residual=.5*(1-cosine.clamp(-1.,1.))
    context=(residual*valid).sum((-2,-1))/valid.sum((-2,-1)).clamp_min(1)
    return intra_beta*inner+(1-intra_beta)*context


@ROTATED_LOSSES.register_module()
class BDNearSquareContextAngleContentLoss(nn.Module):
    def __init__(self, alpha=3., semantic_weight=.1, square_ratio=1.2,
                 grid_size=5, feature_level=0, chunk_size=128, min_size=1e-4,
                 eps=1e-6, reduction='mean', loss_weight=1., intra_tau=1.,
                 intra_beta=.5, context_scale=1.5, cosine_eps=1e-3):
        super().__init__()
        if alpha <= 0 or semantic_weight < 0 or square_ratio < 1 or grid_size < 2:
            raise ValueError('Invalid BD/content settings')
        if feature_level != 0 or chunk_size < 1 or min_size <= 0 or eps <= 0:
            raise ValueError('This candidate uses fixed P3, positive chunk/size/eps')
        if reduction not in ('none', 'mean', 'sum') or loss_weight < 0:
            raise ValueError('Invalid reduction or weight')
        self.alpha, self.semantic_weight, self.square_ratio = alpha, semantic_weight, square_ratio
        self.grid_size, self.feature_level, self.chunk_size = grid_size, feature_level, chunk_size
        self.min_size, self.eps = min_size, eps
        self.reduction, self.loss_weight = reduction, loss_weight
        self.last_diagnostics = {}

        if intra_tau<=0 or not 0<=intra_beta<=1 or context_scale<=1 or cosine_eps<=0:
            raise ValueError('Invalid intra/context settings')
        if grid_size!=5 or context_scale!=1.5:
            raise ValueError('Fixed candidate uses 5x5 and a 1.5 exterior ring')
        self.intra_tau,self.intra_beta=intra_tau,intra_beta
        self.context_scale,self.cosine_eps=context_scale,cosine_eps

    def forward(self, pred, target, weight=None, avg_factor=None,
                reduction_override=None, feature_maps=None, image_indices=None,
                level_indices=None, feature_strides=None, pixel_pred=None,
                pixel_target=None, **kwargs):
        if pred.shape != target.shape or pred.ndim != 2 or pred.shape[-1] != 5:
            raise ValueError('Expected paired Nx5 decoded pixel boxes')
        reduction = reduction_override or self.reduction
        if weight is not None and weight.ndim > 1:
            weight = weight.mean(-1)
        raw = pred.sum(-1)*0
        if len(pred) == 0:
            self.last_diagnostics = {}
            return raw if reduction == 'none' else raw.sum()
        active = torch.ones(len(pred), device=pred.device, dtype=torch.bool) if weight is None else weight > 0
        # Geometry receives the ORIGINAL stride-normalized decoded boxes.
        # Content alone uses separately decoded pixel boxes. Changing the BD
        # units would change its fixed min_size at collapsed initial boxes.
        content_pred = pred if pixel_pred is None else pixel_pred
        content_target = target if pixel_target is None else pixel_target
        if content_pred.shape != pred.shape or content_target.shape != target.shape:
            raise ValueError('Pixel boxes must align with original BD boxes')
        geometry, content = raw, pred[:, 4]*0
        eligible = torch.zeros(len(pred), device=pred.device, dtype=torch.bool)
        if active.any():
            ids = active.nonzero().flatten()
            p, t = pred[ids], target[ids].detach()
            geom = bd_values(p, t, alpha=self.alpha, min_size=self.min_size, agbb=False).to(pred.dtype)
            geometry = geometry.index_copy(0, ids, geom)
            pixel_t = content_target[ids].detach()
            ratio = pixel_t[:, 2:4].amax(-1)/pixel_t[:, 2:4].amin(-1).clamp_min(self.min_size)
            eligible[ids] = ratio <= self.square_ratio
            if self.semantic_weight > 0 and eligible.any():
                if feature_maps is None or image_indices is None or feature_strides is None:
                    raise ValueError('P3 map, image identities and feature strides are required')
                features = feature_maps[0]
                stride = float(feature_strides[0])
                for image in range(features.shape[0]):
                    selected = (eligible & (image_indices == image)).nonzero().flatten()
                    if selected.numel() == 0:
                        continue
                    field = features[image:image+1].detach().to(pred.dtype)
                    field = field/torch.sqrt(field.square().sum(1, keepdim=True)+self.eps**2)
                    for chunk in selected.split(self.chunk_size):
                        value = context_angle_content_values(content_pred[chunk], content_target[chunk], field,
                            stride, self.grid_size, self.min_size, self.intra_tau,
                            self.intra_beta, self.context_scale, self.cosine_eps)
                        content = content.index_copy(0, chunk, value)
        auxiliary = self.semantic_weight*content
        raw = geometry+auxiliary
        if not torch.isfinite(raw).all():
            raise FloatingPointError('Nonfinite BD or near-square angle content')
        # Both terms share the original ALL-positive GT-centerness denominator.
        if reduction != 'none':
            g = weight_reduce_loss(geometry.detach(), weight, reduction, avg_factor)
            a = weight_reduce_loss(auxiliary.detach(), weight, reduction, avg_factor)
            self.last_diagnostics = dict(bd_term=float(g), content_term=float(a),
                square_positive_frac=float(eligible.sum()/active.sum().clamp_min(1)),
                square_positive_count=int(eligible.sum()), positive_count=int(active.sum()))
        return self.loss_weight*weight_reduce_loss(raw, weight, reduction, avg_factor)


@HOOKS.register_module()
class NearSquareContextDiagnosticsHook(Hook):
    """Log detached objective components without adding new training losses."""
    def after_train_iter(self, runner):
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        diagnostic = getattr(model.bbox_head.loss_bbox, 'last_diagnostics', {})
        if diagnostic:
            runner.log_buffer.update(diagnostic, runner.outputs.get('num_samples', 1))
