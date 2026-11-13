"""C01 objective with paired content sampled at the prediction source FPN level.

BD units, GT near-square gate, theta-only content, 25 samples and lambda=.1
are unchanged. This is the uncalibrated assigned-level structural ablation.
"""
import torch
from torch import nn
from mmcv.runner import HOOKS, Hook
from mmdet.models.losses.utils import weight_reduce_loss
from ..builder import ROTATED_LOSSES
from .paper_rotated_losses import bd_values
from .bd_near_square_angle_content_loss import angle_content_values


@ROTATED_LOSSES.register_module()
class BDNearSquareAssignedAngleContentLoss(nn.Module):
    def __init__(self, alpha=3., semantic_weight=.1, square_ratio=1.2,
                 grid_size=5, chunk_size=128, min_size=1e-4,
                 eps=1e-6, reduction='mean', loss_weight=1.):
        super().__init__()
        if alpha <= 0 or semantic_weight < 0 or square_ratio < 1 or grid_size < 2:
            raise ValueError('Invalid BD/content settings')
        if chunk_size < 1 or min_size <= 0 or eps <= 0:
            raise ValueError('Positive chunk/size/eps required')
        if reduction not in ('none', 'mean', 'sum') or loss_weight < 0:
            raise ValueError('Invalid reduction or weight')
        self.alpha, self.semantic_weight, self.square_ratio = alpha, semantic_weight, square_ratio
        self.grid_size, self.chunk_size = grid_size, chunk_size
        self.min_size, self.eps = min_size, eps
        self.reduction, self.loss_weight = reduction, loss_weight
        self.last_diagnostics = {}

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
                if feature_maps is None or image_indices is None or feature_strides is None or level_indices is None:
                    raise ValueError('FPN maps, image identities, source levels and strides are required')
                if len(feature_maps) != len(feature_strides):
                    raise ValueError('One stride per FPN map is required')
                if image_indices.shape != (len(pred),) or level_indices.shape != (len(pred),):
                    raise ValueError('Per-positive image and source-level vectors are required')
                if level_indices.dtype != torch.long or image_indices.dtype != torch.long:
                    raise ValueError('Image and level indices must be int64')
                if ((level_indices < 0) | (level_indices >= len(feature_maps))).any():
                    raise ValueError('Source FPN level out of bounds')
                for level, features in enumerate(feature_maps):
                    stride = float(feature_strides[level])
                    if stride <= 0 or not torch.isfinite(pred.new_tensor(stride)):
                        raise ValueError('Finite positive FPN stride required')
                    level_selected = eligible & (level_indices == level)
                    if not level_selected.any():
                        continue
                    if ((image_indices[level_selected] < 0) | (image_indices[level_selected] >= features.shape[0])).any():
                        raise ValueError('Source image out of bounds')
                    for image in range(features.shape[0]):
                        selected = (level_selected & (image_indices == image)).nonzero().flatten()
                        if selected.numel() == 0:
                            continue
                        field = features[image:image+1].detach().to(pred.dtype)
                        field = field/torch.sqrt(field.square().sum(1, keepdim=True)+self.eps**2)
                        for chunk in selected.split(self.chunk_size):
                            value = angle_content_values(content_pred[chunk], content_target[chunk], field,
                                stride, self.grid_size, self.min_size)
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
class AssignedAngleContentDiagnosticsHook(Hook):
    """Log detached objective components without adding new training losses."""
    def after_train_iter(self, runner):
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        diagnostic = getattr(model.bbox_head.loss_bbox, 'last_diagnostics', {})
        if diagnostic:
            runner.log_buffer.update(diagnostic, runner.outputs.get('num_samples', 1))
