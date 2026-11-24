"""Use fixed image derivatives only in the C01 auxiliary loss."""
import torch
import torch.nn.functional as F
from mmrotate.models.builder import ROTATED_DETECTORS
from mmrotate.models.detectors.base import RotatedBaseDetector
from mmrotate.models.detectors.rotated_fcos import RotatedFCOS


@torch.no_grad()
def edge_texture_field(img, stride=8):
    """Signed derivatives and local contrast at the same pixel centers as P3."""
    image = img.detach().float()
    features = []
    for width in (1, 5, 13, 25):
        if width == 1:
            smooth = image
        else:
            half = width // 2
            horizontal = F.avg_pool2d(F.pad(image, (half, half, 0, 0), mode='replicate'),
                                      (1, width), stride=1)
            smooth = F.avg_pool2d(F.pad(horizontal, (0, 0, half, half), mode='replicate'),
                                  (width, 1), stride=1)
        pixels = F.pad(smooth, (1, 1, 1, 1), mode='replicate')
        dx = (pixels[:, :, 1:-1, 2:] - pixels[:, :, 1:-1, :-2]) * width / 2
        dy = (pixels[:, :, 2:, 1:-1] - pixels[:, :, :-2, 1:-1]) * width / 2
        moments = (dx, dy, dx.abs(), dy.abs(), (dx + dy).abs(), (dx - dy).abs())
        features.extend(F.avg_pool2d(component, stride, stride=stride) for component in moments)
    return torch.cat(features, dim=1)


@ROTATED_DETECTORS.register_module()
class EdgeTextureRotatedFCOS(RotatedFCOS):
    def forward_train(self, img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None):
        RotatedBaseDetector.forward_train(self, img, img_metas)
        features = self.extract_feat(img)
        field = edge_texture_field(img, self.bbox_head.strides[0])
        assert field.shape[-2:] == features[0].shape[-2:]
        return self.bbox_head.loss(*self.bbox_head(features), gt_bboxes, gt_labels, img_metas,
                                   gt_bboxes_ignore=gt_bboxes_ignore, semantic_features=(field,))
