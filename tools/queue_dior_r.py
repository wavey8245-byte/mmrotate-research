"""Persistent single-GPU DIOR-R queue; never interrupts an existing trainer.

Start from the mmrotate034 environment:
  python tools/queue_dior_r.py start --baseline-pid PID
Or reuse a completed, tested baseline in a new queue:
  python tools/queue_dior_r.py start --completed-baseline WORK_DIR --queue-dir NEW_DIR
Status / cancel pending stages:
  python tools/queue_dior_r.py status
  python tools/queue_dior_r.py stop
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUEUE = ROOT / 'work_dirs/dior_r/auto_queue'
CONFIGS = {
    'riou_bs2': 'rotated_fcos_r50_fpn_1x_dior_r_le90_bs2.py',
    'smooth_l1_bs2': 'rotated_fcos_r50_fpn_1x_dior_r_le90_smooth_l1_bs2.py',
    'wf_bs2': 'rotated_fcos_r50_fpn_1x_dior_r_le90_wf_bs2.py',
}
SOURCES = [
    'mmrotate/datasets/dior.py',
    'mmrotate/models/losses/rotated_iou_loss.py',
    'mmrotate/models/losses/rotated_smooth_l1_loss.py',
    'mmrotate/models/losses/rotated_wf_iou_loss.py',
    'mmrotate/models/dense_heads/rotated_fcos_head.py',
    'tools/check_dior_r.py', 'tools/train.py', 'tools/test.py',
]


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def process_identity(pid):
    """PID plus kernel start ticks avoids mistaking a reused PID for training."""
    try:
        directory = Path('/proc') / str(pid)
        fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        command = (directory / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
        return dict(pid=int(pid), start_ticks=fields[19], command=command)
    except (OSError, ValueError, IndexError, UnicodeError):
        return None


def identity_alive(expected):
    actual = process_identity(expected['pid'])
    return actual is not None and actual['start_ticks'] == expected['start_ticks']


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def completed_training(experiment):
    """Require final checkpoint AND final validation; checkpoint alone is early."""
    directory = Path(experiment['work_dir'])
    final = directory / f"epoch_{experiment['max_epochs']}.pth"
    if not final.is_file() or final.stat().st_size < 1024:
        raise RuntimeError(f'最终训练权重缺失或不完整：{final}')
    logs = sorted(directory.glob('*.log.json'), key=lambda p: p.stat().st_mtime)
    if not logs:
        raise RuntimeError(f'没有训练指标日志：{directory}')
    records = []
    for line in logs[-1].read_text().splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get('mode') == 'val' and 'mAP' in entry:
            records.append(entry)
    final_records = [r for r in records if r.get('epoch') == experiment['max_epochs']]
    if not final_records or not math.isfinite(float(final_records[-1]['mAP'])):
        raise RuntimeError('训练未完成最后一轮验证，队列停止；不会提前测试或继续训练')
    if not any(math.isfinite(float(r['mAP'])) and float(r['mAP']) > 0
               for r in records):
        raise RuntimeError('所有轮次验证 mAP 均为 0；训练结果异常，'
                           '请先排查定位回归，不自动选择最终权重测试')
    candidates = list(directory.glob('best_mAP_epoch_*.pth'))
    if len(candidates) != 1:
        raise RuntimeError(f'最佳权重数量应为 1，实际为 {len(candidates)}：{directory}')
    checkpoint = candidates[0]
    if checkpoint.stat().st_size < 1024:
        raise RuntimeError(f'最佳权重文件不完整：{checkpoint}')
    epoch = int(re.search(r'epoch_(\d+)\.pth$', checkpoint.name).group(1))
    selected = [r for r in records if r.get('epoch') == epoch]
    if not selected or not math.isfinite(float(selected[-1]['mAP'])):
        raise RuntimeError('最佳权重缺少对应的有效验证指标')
    score = float(selected[-1]['mAP'])
    maximum = max(float(r['mAP']) for r in records if math.isfinite(float(r['mAP'])))
    # JSON logger rounds metrics; permit only rounding-level differences.
    if score < maximum - 1e-5:
        raise RuntimeError('最佳权重与验证日志不一致，停止自动测试')
    return dict(checkpoint=str(checkpoint), best_epoch=epoch, val_mAP=score,
                validation_log=str(logs[-1]))


def read_test_metric(directory):
    files = sorted(Path(directory).glob('eval_*.json'), key=lambda p: p.stat().st_mtime)
    if not files:
        raise RuntimeError('测试没有生成评估结果文件')
    metric = json.loads(files[-1].read_text())['metric']['mAP']
    if not math.isfinite(float(metric)):
        raise RuntimeError('测试 mAP 不是有效数值')
    return dict(test_mAP=float(metric), test_metrics=str(files[-1]))


class QueueCancelled(Exception):
    pass


class Worker:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / 'manifest.json').read_text())
        self.state = json.loads((self.directory / 'state.json').read_text())
        self.env = os.environ.copy()
        self.env['LD_LIBRARY_PATH'] = '/usr/lib/wsl/lib' + (
            ':' + self.env['LD_LIBRARY_PATH'] if self.env.get('LD_LIBRARY_PATH') else '')
        self.env['PYTHONUNBUFFERED'] = '1'
        self.python = self.manifest['python']

    def save(self):
        self.state['updated_at'] = now()
        atomic_json(self.directory / 'state.json', self.state)
        atomic_json(self.directory / 'results.json', self.state['results'])
        self.write_report()

    def write_report(self):
        labels = dict(riou_bs2='Rotated IoU 基线', smooth_l1_bs2='Smooth L1', wf_bs2='WF 旋转扩展')
        lines = ['# DIOR-R 自动实验结果', '',
                 '共同设置：800×800、单卡 batch 2、SGD lr=0.0025、seed=42、12 轮。', '',
                 f"队列状态：{self.state['status']}；阶段：{self.state.get('stage', '准备')}。", '',
                 '| 实验 | 最佳轮数 | val mAP | test mAP |',
                 '| --- | ---: | ---: | ---: |']
        for variant in CONFIGS:
            result = self.state['results'].get(variant, {})
            val = f"{result['val_mAP'] * 100:.2f}%" if 'val_mAP' in result else '待完成'
            test = f"{result['test_mAP'] * 100:.2f}%" if 'test_mAP' in result else '待完成'
            lines.append(f"| {labels[variant]} | {result.get('best_epoch', '—')} | {val} | {test} |")
        lines += ['', '按 val 选择最佳权重，每组在完整 test 集上评估一次。',
                  'WF 是论文公式的旋转扩展；其基于线性 IoU，与对数 IoU 基线的差异不只包含权重。', '']
        if self.state.get('error'):
            lines += ['队列错误：' + self.state['error'], '']
        (self.directory / 'results.md').write_text('\n'.join(lines))

    def cancelled(self):
        if (self.directory / 'STOP').exists():
            raise QueueCancelled('已停止后续任务；不会中断当前基线或已开始的任务')

    def check_sources(self):
        for name, expected in self.manifest['source_sha256'].items():
            if sha256(ROOT / name) != expected:
                raise RuntimeError(f'实验代码已改变，停止避免组间不一致：{name}')
        for experiment in self.manifest['experiments'].values():
            if sha256(experiment['snapshot']) != experiment['snapshot_sha256']:
                raise RuntimeError('冻结的实验配置已改变，队列停止')

    def stage(self, name, status='running', **extra):
        self.state.update(stage=name, status=status, child_pid=None, **extra)
        self.save()
        print(f'[{now()}] {name}: {status}', flush=True)

    def wait_gpu_idle(self):
        while True:
            self.cancelled()
            result = subprocess.run(
                ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                text=True, capture_output=True, timeout=20)
            if result.returncode:
                raise RuntimeError('无法查询 GPU 进程：' + result.stderr.strip())
            pids = [int(x.strip()) for x in result.stdout.splitlines() if x.strip().isdigit()]
            pids = [pid for pid in pids if process_identity(pid) is not None]
            if not pids:
                self.state.pop('waiting_gpu_pids', None)
                return
            if self.state.get('waiting_gpu_pids') != pids:
                self.state['waiting_gpu_pids'] = pids
                self.save()
                print(f'[{now()}] 等待其他 GPU 任务退出：{pids}', flush=True)
            time.sleep(self.manifest['poll_seconds'])

    def command(self, name, args, log_path):
        self.cancelled()
        self.check_sources()
        self.stage(name)
        self.wait_gpu_idle()
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open('x') as log:
            child = subprocess.Popen(args, cwd=ROOT, env=self.env,
                                     stdout=log, stderr=subprocess.STDOUT)
            self.state.update(child_pid=child.pid, child_process=process_identity(child.pid),
                              command=args, stage_log=str(log_path))
            self.save()
            returncode = child.wait()
        self.state.update(child_pid=None, returncode=returncode)
        self.save()
        if returncode:
            raise RuntimeError(f'{name} 失败，退出码 {returncode}。日志：{log_path}')
        self.state['completed_stages'].append(name)
        self.save()

    def evaluate(self, variant):
        experiment = self.manifest['experiments'][variant]
        completion = completed_training(experiment)
        self.state['results'][variant] = completion
        self.save()
        directory = Path(experiment['work_dir'])
        metrics = directory / 'test_auto'
        predictions = directory / 'auto_test_predictions.pkl'
        if predictions.exists() or (metrics.exists() and list(metrics.iterdir())):
            raise RuntimeError(f'自动测试输出已经存在，停止避免覆盖：{directory}')
        self.command(f'test:{variant}', [
            self.python, str(ROOT / 'tools/test.py'), experiment['snapshot'],
            completion['checkpoint'], '--work-dir', str(metrics),
            '--out', str(predictions), '--eval', 'mAP'], directory / 'auto_test.log')
        self.state['results'][variant].update(read_test_metric(metrics))
        self.state['results'][variant]['predictions'] = str(predictions)
        self.save()

    def run(self):
        os.chdir(ROOT)
        with (self.directory / 'worker.lock').open('a') as lock:
            # A second worker must not overwrite the first worker's state.
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                if self.state['status'] != 'starting':
                    raise RuntimeError('队列已经运行过；请先检查状态，不自动重跑实验')
                self.state['worker'] = process_identity(os.getpid())
                baseline = self.manifest['baseline_process']
                if baseline is not None:
                    self.stage('wait:riou_bs2', 'waiting')
                    while identity_alive(baseline):
                        self.cancelled()
                        time.sleep(self.manifest['poll_seconds'])
                self.check_sources()
                self.cancelled()
                if self.manifest.get('baseline_result') is not None:
                    completed_training(self.manifest['experiments']['riou_bs2'])
                    self.state['results']['riou_bs2'] = self.manifest['baseline_result']
                    self.state['completed_stages'].append('reuse:riou_bs2')
                    self.stage('reuse:riou_bs2')
                else:
                    self.evaluate('riou_bs2')
                for variant in ('smooth_l1_bs2', 'wf_bs2'):
                    self.cancelled()
                    experiment = self.manifest['experiments'][variant]
                    work_dir = Path(experiment['work_dir'])
                    if work_dir.exists() and list(work_dir.glob('*.pth')):
                        raise RuntimeError(f'新实验目录已有训练权重，停止避免覆盖：{work_dir}')
                    self.command(f'check:{variant}', [
                        self.python, str(ROOT / 'tools/check_dior_r.py'),
                        '--config', experiment['snapshot'], '--report',
                        str(self.directory / f'check_{variant}.json')],
                        self.directory / f'check_{variant}.log')
                    self.command(f'train:{variant}', [
                        self.python, str(ROOT / 'tools/train.py'),
                        experiment['snapshot'], '--seed', '42'], work_dir / 'auto_train.log')
                    self.evaluate(variant)
                self.stage('complete', 'completed')
            except QueueCancelled as error:
                self.stage('cancelled', 'cancelled', error=str(error))
            except Exception as error:
                self.stage('failed', 'failed', error=str(error))
                raise


def start_queue(args):
    from mmcv import Config

    os.chdir(ROOT)
    directory = Path(args.queue_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (directory / 'state.json').exists():
            raise RuntimeError('队列已经存在，请用 status 查看；不会创建重复训练任务')
        baseline = None
        if args.baseline_pid is not None:
            baseline = process_identity(args.baseline_pid)
            if baseline is None or 'tools/train.py' not in baseline['command'] or CONFIGS['riou_bs2'] not in baseline['command']:
                raise RuntimeError('指定 PID 不是当前 batch 2 基线的有效训练进程')
        if args.poll_seconds < 5:
            raise ValueError('poll_seconds must be at least 5')
        snapshots = directory / 'configs'
        snapshots.mkdir(exist_ok=True)
        manifest = dict(created_at=now(), python=sys.executable,
                        baseline_process=baseline, poll_seconds=args.poll_seconds,
                        source_sha256={name: sha256(ROOT / name) for name in SOURCES},
                        experiments={})
        baseline_cfg = None
        for variant, filename in CONFIGS.items():
            source = ROOT / 'configs/dior_r' / filename
            cfg = Config.fromfile(str(source))
            if variant == 'riou_bs2':
                baseline_directory = (ROOT / args.completed_baseline).resolve() if (
                    args.completed_baseline is not None) else ROOT / cfg.work_dir
                saved = baseline_directory / filename
                if not saved.exists():
                    raise RuntimeError('找不到当前基线实际使用的保存配置')
                cfg = Config.fromfile(str(saved))
                if args.completed_baseline is not None:
                    cfg.work_dir = str(baseline_directory)
                baseline_cfg = cfg
            else:
                for field in ('data', 'optimizer', 'runner', 'lr_config', 'evaluation', 'seed'):
                    if cfg[field] != baseline_cfg[field]:
                        raise RuntimeError(f'{variant} 的 {field} 与当前基线不一致')
            if cfg.data.samples_per_gpu != 2 or cfg.optimizer.lr != .0025:
                raise RuntimeError('当前配置不是预期的 batch 2 / lr=0.0025')
            if cfg.data.train.pipeline[2].img_scale != (800, 800):
                raise RuntimeError('训练尺寸不是 800×800')
            cfg.work_dir = str((ROOT / cfg.work_dir).resolve())
            cfg.load_from = None
            cfg.resume_from = None
            snapshot = snapshots / filename
            cfg.dump(str(snapshot))
            manifest['experiments'][variant] = dict(
                original_config=str(source), snapshot=str(snapshot),
                snapshot_sha256=sha256(snapshot), work_dir=cfg.work_dir,
                max_epochs=cfg.runner.max_epochs)
        if args.completed_baseline is not None:
            experiment = manifest['experiments']['riou_bs2']
            completion = completed_training(experiment)
            baseline_directory = Path(experiment['work_dir'])
            completion.update(read_test_metric(baseline_directory / 'test_auto'))
            test_record = json.loads(Path(completion['test_metrics']).read_text())
            tested_cfg = Config.fromfile(test_record['config'])
            for field in ('model', 'data'):
                if tested_cfg[field] != baseline_cfg[field]:
                    raise RuntimeError('已有基线测试配置与基线不一致，不能复用')
            predictions = baseline_directory / 'auto_test_predictions.pkl'
            if not predictions.is_file():
                raise RuntimeError('已有基线测试缺少预测文件，不能复用')
            completion['predictions'] = str(predictions)
            manifest['baseline_result'] = completion
        if args.dry_run:
            print(json.dumps(dict(validated=True, worker_started=False,
                baseline_reused=args.completed_baseline is not None,
                work_dirs={k: v['work_dir'] for k, v in manifest['experiments'].items()}),
                ensure_ascii=False, indent=2))
            return
        atomic_json(directory / 'manifest.json', manifest)
        state = dict(status='starting', created_at=now(), updated_at=now(),
                     stage='starting', completed_stages=[], results={})
        atomic_json(directory / 'state.json', state)
        with (directory / 'worker.log').open('x') as log:
            child = subprocess.Popen([
                sys.executable, str(Path(__file__).resolve()), 'run',
                '--queue-dir', str(directory)], cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps(dict(worker_pid=child.pid, queue_dir=str(directory)), ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    for name in ('start', 'run', 'status', 'stop'):
        command = commands.add_parser(name)
        command.add_argument('--queue-dir', default=str(DEFAULT_QUEUE))
        if name == 'start':
            baseline_options = command.add_mutually_exclusive_group(required=True)
            baseline_options.add_argument('--baseline-pid', type=int)
            baseline_options.add_argument('--completed-baseline',
                help='Reuse completed baseline and its existing test_auto results')
            command.add_argument('--poll-seconds', type=int, default=30)
            command.add_argument('--dry-run', action='store_true',
                help='Validate configurations and baseline without launching a worker')
    args = parser.parse_args()
    directory = Path(args.queue_dir)
    if args.action == 'start':
        start_queue(args)
    elif args.action == 'run':
        Worker(directory).run()
    elif args.action == 'status':
        state = json.loads((directory / 'state.json').read_text())
        state['worker_alive'] = bool(state.get('worker') and identity_alive(state['worker']))
        print(json.dumps(state, ensure_ascii=False, indent=2))
    elif args.action == 'stop':
        (directory / 'STOP').write_text(now() + '\n')
        print('已请求停止后续任务；当前正在训练或测试的任务会自然完成。')


if __name__ == '__main__':
    main()
