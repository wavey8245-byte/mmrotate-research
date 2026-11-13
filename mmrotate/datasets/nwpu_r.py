"""Read the JPEG tiles and DOTA-format labels in the NWPU-R release."""
from pathlib import Path

from mmrotate.datasets.builder import ROTATED_DATASETS
from mmrotate.datasets.dota import DOTADataset


@ROTATED_DATASETS.register_module()
class NWPURDataset(DOTADataset):
    CLASSES = ('airplane', 'ship', 'storage-tank', 'baseball-diamond', 'tennis-court',
               'basketball-court', 'ground-track-field', 'harbor', 'bridge', 'vehicle')

    def load_annotations(self, ann_folder):
        infos = super().load_annotations(ann_folder)
        infos.sort(key=lambda info: info['filename'])
        for info in infos:
            info['filename'] = str(Path(info['filename']).with_suffix('.jpg'))
        self.img_ids = [Path(info['filename']).stem for info in infos]
        return infos
