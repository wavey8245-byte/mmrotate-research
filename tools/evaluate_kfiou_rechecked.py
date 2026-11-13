"""Evaluate the saved standard KFIoU epoch after an unchanged fresh-process health recheck."""
import argparse
from datetime import datetime
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('rio_evaluation_helpers', ROOT / 'tools/run_dior_r_rio.py')
HELPERS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPERS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    folder = (ROOT / args.work_dir).resolve()
    output = folder / 'test_epoch_1'
    checkpoint = folder / 'epoch_1.pth'
    frozen = folder / 'snapshot/experiment.py'
    lock_path = ROOT / 'work_dirs/dior_r_trainval/server_queue.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        training = json.loads((folder / 'run.json').read_text())
        if training['status'] != 'failed' or training.get('training_returncode') != 1:
            raise RuntimeError('Expected preserved original post-epoch health failure')
        recheck_path = folder / 'fresh_process_recheck_v2/result.json'
        recheck = json.loads(recheck_path.read_text())
        if recheck['status'] != 'passed' or recheck['checkpoint_epoch'] != 1 or recheck['checkpoint_iteration'] != 5863:
            raise RuntimeError('Standard epoch-1 checkpoint must pass unchanged fresh-process health check')
        if not recheck['same_original_hook'] or recheck['same_threshold'] != .05 or not recheck['weights_finite']:
            raise RuntimeError('Recheck changed the hook/threshold or found nonfinite weights')
        if len(recheck['training_health']) != 1 or recheck['training_health'][0]['status'] != 'passed' or len(recheck['sample_ids']) != 128:
            raise RuntimeError('Expected successful original 128-image training health check')
        if HELPERS.hashes([str(checkpoint)])[str(checkpoint)] != recheck['checkpoint_sha256']:
            raise RuntimeError('Checkpoint changed after successful recheck')
        if recheck['frozen_config_sha256'] != training['frozen_config_sha256']:
            raise RuntimeError('Recheck used a different configuration')
        if HELPERS.hashes(training['source_sha256']) != training['source_sha256']:
            raise RuntimeError('Probe sources changed after training')
        if HELPERS.hashes([str(frozen)])[str(frozen)] != training['frozen_config_sha256']:
            raise RuntimeError('Frozen training configuration changed')
        if output.exists() and any(output.iterdir()):
            raise RuntimeError(f'Existing test output will not be overwritten: {output}')
        sources = [str(frozen), str(checkpoint), 'tools/test.py',
                   'tools/evaluate_kfiou_rechecked.py', str(recheck_path), str(folder / 'run.json'),
                   'mmrotate/models/losses/paper_evcs_loss.py',
                   'mmrotate/models/losses/kf_iou_loss.py',
                   'mmrotate/models/dense_heads/rotated_fcos_head.py',
                   'mmrotate/models/losses/rotated_rio_loss.py',
                   'mmrotate/models/losses/rotated_iou_loss.py', 'mmrotate/datasets/dior.py']
        before = HELPERS.hashes(sources)
        output.mkdir(parents=True, exist_ok=True)
        (output / 'evaluator.py').write_bytes(Path(__file__).read_bytes())
        state = dict(status='starting', created_at=datetime.now().astimezone().isoformat(),
                     checkpoint=str(checkpoint), checkpoint_epoch=1, config=str(frozen),
                     test_images=11738, iou_thr=.5, use_07_metric=True,
                     source_sha256=before,
                     scope='Full test at epoch 1 after successful unchanged health recheck; original online health failure preserved.',
                     original_training_status=training['status'], original_training_returncode=1,
                     original_training_health=training['training_health'],
                     fresh_process_training_health=recheck['training_health'][0],
                     checkpoint_sha256=recheck['checkpoint_sha256'], loss=training['loss'])
        HELPERS.DIRECTORY = output
        HELPERS.write_json(output / 'run.json', state)
        try:
            HELPERS.run_child([sys.executable, str(ROOT / 'tools/test.py'), str(frozen),
                               str(checkpoint), '--work-dir', str(output),
                               '--out', str(output / 'predictions.pkl'), '--eval', 'mAP',
                               '--eval-options', 'iou_thr=0.5', 'use_07_metric=True'],
                              output / 'test.log', state, 'testing', lock.fileno())
            evaluations = list(output.glob('eval_*.json'))
            if len(evaluations) != 1:
                raise RuntimeError('Expected exactly one full test evaluation')
            metric = json.loads(evaluations[0].read_text())['metric']
            score = float(metric['mAP'])
            if not math.isfinite(score):
                raise RuntimeError('Nonfinite test AP')
            if HELPERS.hashes(sources) != before:
                raise RuntimeError('Evaluation inputs changed while testing')
            state.update(status='completed', finished_at=datetime.now().astimezone().isoformat(),
                         metric=metric, mAP_percent=100 * score,
                         evaluation_source=str(evaluations[0]))
            comparisons = []
            for label, relative in [
                ('EVCS-abs', 'fcos_evcs_abs_probe1_20261021'),
                ('ordinary BD', 'fcos_paper_bd_probe1_20261019'),
                ('DCAGPL', 'fcos_paper_dcagpl_probe1_20261020')]:
                source = ROOT / 'work_dirs/dior_r_trainval' / relative / 'test_epoch_1/result.json'
                reference = json.loads(source.read_text())
                reference_score = float(reference['metric']['mAP'])
                comparisons.append(dict(method=label, test_mAP=reference_score,
                                        kfiou_minus_reference_percentage_points=100 * (score-reference_score),
                                        source=str(source)))
            state.update(epoch=1, test_mAP=score, comparisons=comparisons,
                         evcs_abs_minus_kfiou_percentage_points=100*(comparisons[0]['test_mAP']-score))
            HELPERS.write_json(folder / 'results.json', state)
            lines = ['# 纯 KFIoU 第 1 轮：复查后完整测试', '',
                     f'完整 test mAP50：**{100*score:.4f}%**。', '',
                     '原训练完成 5863 次更新，但在线训练图片检查报零检测而中断。',
                     '同一标准检查点在新进程中通过未修改的 128 张训练图片检查（38.4206%），随后完成本次完整 test。',
                     '原始失败日志与 run.json 均保留；在线检查产生空结果的具体原因尚未定位。', '',
                     '| 方法 | 第 1 轮 test mAP50 | 纯 KFIoU 相对差值（百分点） |',
                     '|---|---:|---:|', f'| 纯 KFIoU | {100*score:.4f}% | 0 |']
            for row in comparisons:
                lines.append(f"| {row['method']} | {100*row['test_mAP']:.4f}% | {row['kfiou_minus_reference_percentage_points']:+.4f} |")
            lines.extend(['', '固定第 1 轮；test=11,738，IoU=0.5，VOC2007 11 点 AP。',
                          '纯 KFIoU 与 EVCS-abs 的基项适配和训练设置一致，仅去除 EVCS 附加项。',
                          '当前属于单种子第 1 轮对照，不能推断第 12 轮或稳定统计结论。', ''])
            (folder / 'results.md').write_text('\n'.join(lines), encoding='utf-8')
            (output / 'results.md').write_text('\n'.join(lines), encoding='utf-8')
            HELPERS.write_json(output / 'result.json', state)
            HELPERS.write_json(output / 'run.json', state)
            print(json.dumps(dict(mAP_percent=100 * score, result=str(output / 'result.json'))), flush=True)
        except Exception as error:
            state.update(status='failed', error=str(error),
                         finished_at=datetime.now().astimezone().isoformat())
            HELPERS.write_json(output / 'run.json', state)
            raise


if __name__ == '__main__':
    main()
