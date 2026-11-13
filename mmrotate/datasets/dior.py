# Copyright (c) OpenMMLab. All rights reserved.
"""DIOR-R XML annotations with the official image-id splits."""
import os.path as osp
import warnings
import xml.etree.ElementTree as ET

import cv2
import mmcv
import numpy as np
from mmdet.datasets.custom import CustomDataset

from mmrotate.core import eval_rbbox_map, poly2obb_np
from .builder import ROTATED_DATASETS


@ROTATED_DATASETS.register_module()
class DIORDataset(CustomDataset):
    CLASSES = ('airplane', 'airport', 'baseballfield', 'basketballcourt',
               'bridge', 'chimney', 'dam', 'Expressway-Service-area',
               'Expressway-toll-station', 'golffield', 'groundtrackfield',
               'harbor', 'overpass', 'ship', 'stadium', 'storagetank',
               'tenniscourt', 'trainstation', 'vehicle', 'windmill')

    def __init__(self, ann_file, pipeline, ann_prefix, version='le90',
                 **kwargs):
        self.ann_prefix = ann_prefix
        self.version = version
        super().__init__(ann_file, pipeline, **kwargs)

    def load_annotations(self, ann_file):
        class_map = {name.lower(): i for i, name in enumerate(self.CLASSES)}
        data_infos = []
        self.invalid_annotations = []
        self.img_ids = mmcv.list_from_file(ann_file)
        coordinates = ('x_left_top', 'y_left_top', 'x_right_top',
                       'y_right_top', 'x_right_bottom', 'y_right_bottom',
                       'x_left_bottom', 'y_left_bottom')
        for image_id in self.img_ids:
            xml_path = osp.join(self.ann_prefix, image_id + '.xml')
            root = ET.parse(xml_path).getroot()
            size = root.find('size')
            filename = root.findtext('filename', image_id + '.jpg')
            if not osp.isfile(osp.join(self.img_prefix, filename)):
                raise FileNotFoundError(osp.join(self.img_prefix, filename))
            boxes, labels, polygons = [], [], []
            ignored_boxes, ignored_labels, ignored_polygons = [], [], []
            for obj in root.findall('object'):
                name = obj.findtext('name').strip().lower()
                if name not in class_map:
                    raise ValueError(f'{xml_path}: unknown class {name}')
                rbbox = obj.find('robndbox')
                poly = np.array([float(rbbox.findtext(k))
                                 for k in coordinates], dtype=np.float32)
                box = poly2obb_np(poly, self.version)
                # The upstream DOTA converter drops sides shorter than 2 px.
                # DIOR-R has valid tiny vehicles; preserve them for le90.
                if box is None and self.version == 'le90':
                    (x, y), (w, h), angle = cv2.minAreaRect(poly.reshape(4, 2))
                    if w > 0 and h > 0:
                        angle = np.deg2rad(angle)
                        if w < h:
                            w, h = h, w
                            angle += np.pi / 2
                        angle = (angle + np.pi / 2) % np.pi - np.pi / 2
                        box = (x, y, w, h, angle)
                if box is None:
                    self.invalid_annotations.append(dict(image_id=image_id,
                                                         polygon=poly.tolist()))
                    continue
                if not np.isfinite(box).all():
                    raise ValueError(f'{xml_path}: invalid polygon {poly}')
                if int(obj.findtext('difficult', '0')):
                    ignored_boxes.append(box)
                    ignored_labels.append(class_map[name])
                    ignored_polygons.append(poly)
                else:
                    boxes.append(box)
                    labels.append(class_map[name])
                    polygons.append(poly)
            ann = dict(
                bboxes=np.array(boxes, dtype=np.float32).reshape(-1, 5),
                labels=np.array(labels, dtype=np.int64),
                polygons=np.array(polygons, dtype=np.float32).reshape(-1, 8),
                bboxes_ignore=np.array(ignored_boxes, dtype=np.float32).reshape(-1, 5),
                labels_ignore=np.array(ignored_labels, dtype=np.int64),
                polygons_ignore=np.array(ignored_polygons, dtype=np.float32).reshape(-1, 8))
            data_infos.append(dict(
                id=image_id, filename=filename,
                width=int(size.findtext('width')),
                height=int(size.findtext('height')), ann=ann))
        if self.invalid_annotations:
            warnings.warn(f'{ann_file}: excluded {len(self.invalid_annotations)} '
                          'zero-area/unsupported boxes; details in '
                          'dataset.invalid_annotations')
        return data_infos

    def evaluate(self, results, metric='mAP', logger=None, iou_thr=0.5,
                 scale_ranges=None, nproc=4, use_07_metric=True, **kwargs):
        if metric not in ('mAP', ['mAP']):
            raise KeyError(f'Unsupported metric: {metric}')
        if len(results) != len(self):
            raise ValueError('One prediction result is required per image')
        mean_ap, _ = eval_rbbox_map(
            results, [self.get_ann_info(i) for i in range(len(self))],
            scale_ranges=scale_ranges, iou_thr=iou_thr,
            use_07_metric=use_07_metric, dataset=self.CLASSES,
            logger=logger, nproc=nproc)
        return dict(mAP=mean_ap)
