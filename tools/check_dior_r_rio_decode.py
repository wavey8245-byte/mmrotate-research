"""Read-only checks of FCOS decoding and the RiO loss coordinate conversion."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import mmcv
import numpy as np
import torch

from mmrotate.core import poly2obb_np
from mmrotate.core.bbox.coder.distance_angle_point_coder import DistanceAnglePointCoder
from mmrotate.models.builder import build_head
from mmrotate.models.losses.rotated_rio_loss import RotatedRiOLoss, periodic_l1


ROOT = Path(__file__).resolve().parents[1]


def to_pixels(boxes, points, stride):
    return torch.cat((points + (boxes[:, :2] - points) * stride,
                      boxes[:, 2:4] * stride, boxes[:, 4:]), dim=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', default='work_dirs/dior_r/rio_decode_audit_20261018.json')
    args = parser.parse_args()
    destination = ROOT / args.report
    if destination.exists():
        raise RuntimeError(f'Existing audit will not be overwritten: {destination}')
    torch.set_num_threads(2)
    torch.manual_seed(63)
    coder = DistanceAnglePointCoder(angle_version='le90')
    report = dict(status='running', optimizer_updates=0, checks={})

    # Sample interior points in known rectangles, including the angle seam.
    for dtype in (torch.float32, torch.float64):
        boxes = torch.rand(10000, 5, dtype=dtype)
        boxes[:, :2] *= 800
        boxes[:, 2:4] = boxes[:, 2:4] * 599 + 1
        boxes[:, 4] = boxes[:, 4] * math.pi - math.pi / 2
        boxes[:4, 4] = torch.tensor([-math.pi/2, -math.pi/2+1e-5,
                                    math.pi/2-1e-5, 0.], dtype=dtype)
        local = (torch.rand(10000, 2, dtype=dtype) - .5) * boxes[:, 2:4] * .9
        c, s = boxes[:, 4].cos(), boxes[:, 4].sin()
        offset = torch.stack((c*local[:, 0]-s*local[:, 1],
                              s*local[:, 0]+c*local[:, 1]), -1)
        points = boxes[:, :2] + offset
        encoded = coder.encode(points, boxes)
        decoded = coder.decode(points, encoded)
        angular = periodic_l1(decoded[:, 4], boxes[:, 4])
        pixel_error = float((decoded[:, :4] - boxes[:, :4]).abs().max())
        tolerance = 2e-4 if dtype == torch.float32 else 1e-10
        assert pixel_error < tolerance
        assert float(angular.max()) < tolerance
        report['checks']['roundtrip_' + str(dtype)] = dict(
            boxes=10000, max_coordinate_error_pixels=pixel_error,
            max_periodic_angle_error_radians=float(angular.max()))

    # Compare current training-coordinate losses against decoding in pixels.
    loss = RotatedRiOLoss(kld_normalized=True, covariance_epsilon=1e-8,
                          min_size=1e-8, angle_normalizer=math.pi)
    conversions = []
    for height, width in ((800, 800), (640, 800)):
        for stride in (8, 16, 32, 64, 128):
            points = torch.rand(100, 2, dtype=torch.float64) * 800
            prediction = torch.rand(100, 5, dtype=torch.float64)
            prediction[:, :4] = prediction[:, :4] * 7 + .2
            prediction[:, 4] = prediction[:, 4] * 8*math.pi - 4*math.pi
            prediction.requires_grad_()
            target = torch.rand(100, 5, dtype=torch.float64)
            target[:, :4] = target[:, :4] * 7 + .2
            target[:, 4] = target[:, 4] * math.pi - math.pi/2
            train_pred, train_target = coder.decode(points, prediction), coder.decode(points, target)
            multiplier = prediction.new_tensor([stride]*4+[1])
            pixel_pred = coder.decode(points, prediction * multiplier)
            pixel_target = coder.decode(points, target * multiplier)
            torch.testing.assert_close(to_pixels(train_target, points, stride), pixel_target,
                                       rtol=1e-10, atol=1e-10)
            scale = prediction.new_tensor([[1/width, 1/height]*2]).expand(100, -1)
            weights = torch.linspace(.1, 1., 100, dtype=torch.float64)
            train_parts = loss(train_pred, train_target, weight=weights,
                avg_factor=weights.sum(), spatial_scale=scale*stride, return_components=True)
            pixel_parts = loss(pixel_pred, pixel_target, weight=weights,
                avg_factor=weights.sum(), spatial_scale=scale, return_components=True)
            errors, gradient_errors = {}, {}
            for name in train_parts:
                torch.testing.assert_close(train_parts[name], pixel_parts[name], rtol=1e-10, atol=1e-10)
                a = torch.autograd.grad(train_parts[name], prediction, retain_graph=True)[0]
                b = torch.autograd.grad(pixel_parts[name], prediction, retain_graph=True)[0]
                torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-9)
                errors[name] = abs(float(train_parts[name]-pixel_parts[name]))
                gradient_errors[name] = float((a-b).abs().max())
            conversions.append(dict(stride=stride, height=height, width=width,
                component_value_errors=errors, component_gradient_errors=gradient_errors))
    report['checks']['loss_coordinate_equivalence'] = conversions

    cfg = mmcv.Config.fromfile(str(ROOT/'configs/dior_r/server_trainval/rio_adamw.py'))
    head = build_head(cfg.model.bbox_head)
    # Check the actual train/eval distance-stride branch using identical features.
    forward_errors = []
    for level, stride in enumerate(head.strides):
        feature = torch.randn(2, 256, 4, 4)
        with torch.no_grad():
            head.train()
            train = head.forward_single(feature, head.scales[level], stride)
            head.eval()
            infer = head.forward_single(feature, head.scales[level], stride)
        torch.testing.assert_close(infer[1], train[1]*stride)
        torch.testing.assert_close(infer[2], train[2])
        forward_errors.append(dict(stride=stride,
            distance_error=float((infer[1]-train[1]*stride).abs().max()),
            angle_error=float((infer[2]-train[2]).abs().max())))
    report['checks']['head_train_inference_strides'] = forward_errors

    # Real DIOR-R annotations through actual FCOS assignment at all FPN levels.
    image_ids = (ROOT/'data/DIOR-R/ImageSets/Main/train.txt').read_text().split()
    chosen = np.random.RandomState(63).choice(len(image_ids), 32, replace=False)
    points = head.prior_generator.grid_priors([(100,100),(50,50),(25,25),(13,13),(7,7)],
                                              dtype=torch.float32, device='cpu')
    coordinates = ('x_left_top','y_left_top','x_right_top','y_right_top',
                   'x_right_bottom','y_right_bottom','x_left_bottom','y_left_bottom')
    real_boxes, positives, max_box_error = 0, 0, 0.
    for index in chosen:
        image_id = image_ids[int(index)]
        xml = ROOT/'data/DIOR-R/Annotations/Oriented Bounding Boxes'/f'{image_id}.xml'
        boxes = []
        for obj in ET.parse(xml).getroot().findall('object'):
            if int(obj.findtext('difficult', '0')):
                continue
            polygon = np.array([float(obj.find('robndbox').findtext(k)) for k in coordinates], dtype=np.float32)
            box = poly2obb_np(polygon, 'le90')
            if box is None:
                raise RuntimeError(f'Use the dataset tiny-box fallback for this audit image: {image_id}')
            boxes.append(box)
        gt = torch.tensor(np.array(boxes), dtype=torch.float32).reshape(-1, 5)
        real_boxes += len(gt)
        if not len(gt):
            continue
        labels, distances, angles = head.get_targets(points, [gt], [torch.zeros(len(gt), dtype=torch.long)])
        for stride, p, label, distance, angle in zip(head.strides, points, labels, distances, angles):
            positive = label < head.num_classes
            if not positive.any():
                continue
            encoded = torch.cat((distance[positive]*stride, angle[positive]), -1)
            recovered = coder.decode(p[positive], encoded)
            error = (recovered[:, None, :4]-gt[None, :, :4]).abs().amax(-1)
            angular = periodic_l1(recovered[:, None, 4], gt[None, :, 4])
            matched = torch.where(angular < 1e-5, error, float('inf')).min(-1).values
            max_box_error = max(max_box_error, float(matched.max()))
            assert float(matched.max()) < 2e-4
            positives += len(recovered)
    report['checks']['actual_target_assignment'] = dict(images=32, gt_boxes=real_boxes,
        positive_points=positives, max_coordinate_error_pixels=max_box_error)

    # Angle periodicity alone is not an equivalence of the complete encoding.
    point = torch.tensor([[100.,100.]], dtype=torch.float64)
    target = torch.tensor([[120.,110.,80.,30.,math.pi/6]], dtype=torch.float64)
    encoded = coder.encode(point, target)
    changed = encoded.clone()
    changed[:, 4] += math.pi
    wrong = coder.decode(point, changed)
    repaired = changed[:, [2,3,0,1,4]]
    right = coder.decode(point, repaired)
    torch.testing.assert_close(right, target)
    report['checks']['pi_shift_requires_distance_swap'] = dict(
        target=target.tolist(), decoded_with_only_angle_pi_shift=wrong.tolist(),
        decoded_after_distance_swap=right.tolist(),
        unchanged_distances_center_error_pixels=float((wrong[:,:2]-target[:,:2]).norm()),
        sp_l1_radians=float(periodic_l1(wrong[:,4], target[:,4])))
    report['source_sha256'] = {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in (
        'mmrotate/models/dense_heads/rotated_fcos_head.py',
        'mmrotate/core/bbox/coder/distance_angle_point_coder.py',
        'mmrotate/models/losses/rotated_rio_loss.py',
        'tools/check_dior_r_rio_decode.py')}
    report['status'] = 'passed'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
