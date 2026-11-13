"""Compare the two completed WF experiments and historical source manifests."""
import hashlib
import json
import tarfile
from pathlib import Path

import mmcv

root = Path('work_dirs/dior_r_trainval')
old = mmcv.Config.fromfile('work_dirs/dior_r/fcos_wf_bs2_800/'
                          'rotated_fcos_r50_fpn_1x_dior_r_le90_wf_bs2.py')
new = mmcv.Config.fromfile(str(root / 'fcos_wf_bs2_800/wf.py'))
keys = ('model', 'optimizer', 'optimizer_config', 'lr_config', 'runner',
        'seed', 'load_from', 'resume_from', 'workflow')
report = dict(common_settings_equal={key: old[key] == new[key] for key in keys},
              train_pipeline_equal=old.data.train.pipeline == new.data.train.pipeline,
              test_config_equal=old.data.test == new.data.test,
              samples_per_gpu_equal=old.data.samples_per_gpu == new.data.samples_per_gpu,
              workers_per_gpu_equal=old.data.workers_per_gpu == new.data.workers_per_gpu,
              old_training_ids=old.data.train.ann_file,
              new_training_ids=new.data.train.ann_file,
              old_epoch_evaluation_ids=old.data.val.ann_file,
              new_epoch_evaluation_ids=new.data.val.ann_file,
              old_evaluation=dict(old.evaluation), new_evaluation=dict(new.evaluation),
              old_steps_per_epoch=2932, new_steps_per_epoch=5863,
              old_total_updates=2932*12, new_total_updates=5863*12,
              old_updates_before_first_decay=2932*8,
              new_updates_before_first_decay=5863*8)
for label, directory in [('old', Path('work_dirs/dior_r/fcos_wf_bs2_800')),
                         ('new', root / 'fcos_wf_bs2_800')]:
    logfile = next(directory.glob('*.log.json'))
    first = json.loads(logfile.read_text().splitlines()[0])
    report[label+'_environment'] = first['env_info']
    report[label+'_logged_seed'] = first['seed']
    report[label+'_random_seed_log'] = [line for line in
        next(directory.glob('20*.log')).read_text().splitlines()
        if 'Set random seed' in line]
    report[label+'_val_history'] = [
        dict(epoch=record['epoch'], mAP=record['mAP'])
        for line in logfile.read_text().splitlines()[1:]
        for record in [json.loads(line)] if record.get('mode') == 'val']

manifest = json.loads(Path('work_dirs/dior_r/auto_queue_fixed/manifest.json').read_text())
with tarfile.open('dist/mmrotate034_dior_trainval_source.tar.gz') as archive:
    packaged = json.load(archive.extractfile('mmrotate034-src/SERVER_SOURCE_MANIFEST.json'))
report['old_manifest_keys'] = list(manifest)
report['new_package_manifest_keys'] = list(packaged)
report['historical_source_records'] = {}
paths = ('mmrotate/models/losses/rotated_wf_iou_loss.py',
         'mmrotate/models/dense_heads/rotated_fcos_head.py',
         'mmrotate/datasets/dior.py', 'tools/train.py')

def search(value, path):
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).endswith(path):
                return child
            if isinstance(child, dict) and str(child.get('path', '')).endswith(path):
                return child
            found = search(child, path)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = search(child, path)
            if found is not None:
                return found
    return None

for path in paths:
    report['historical_source_records'][path] = dict(
        old=search(manifest, path), before_new_training=search(packaged, path),
        current_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
destination = root / 'wf_two_runs_audit.json'
destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({key: value for key, value in report.items()
                  if key not in ('old_environment', 'new_environment',
                                 'old_val_history', 'new_val_history')},
                 ensure_ascii=False, indent=2))
