"""Training-image localization checks for opt-in RiO transfer experiments."""
import copy
import json
import math
from pathlib import Path

from mmcv.runner import HOOKS, Hook
import numpy as np
import torch


def learning_failure(score, best_score, minimum=.05, minimum_fraction=.5):
    """Use training-image learning, never test AP, as a failure safeguard."""
    if not math.isfinite(score):
        return 'Nonfinite training-image mAP'
    if score < minimum:
        return f'Training-image mAP {score:.6f} is below {minimum}'
    if best_score > 0 and score < best_score * minimum_fraction:
        return f'Training-image mAP collapsed from {best_score:.6f} to {score:.6f}'
    return None


@HOOKS.register_module()
class RiOTrainingHealthHook(Hook):
    """Check a fixed training subset after each epoch, before test evaluation.

    This changes no gradient, parameter, optimizer state, or test-set decision.
    Configure priority LOW so checkpoints exist before a failure is raised.
    """

    def __init__(self, dataset_cfg, images=128, seed=42, minimum_map=.05,
                 minimum_fraction=.5, diagnostic_iterations=()):
        if 'test.txt' in str(dataset_cfg.get('ann_file', '')):
            raise ValueError('Training health must not use the test split')
        self.dataset_cfg = copy.deepcopy(dataset_cfg)
        self.images = images
        self.seed = seed
        self.minimum_map = minimum_map
        self.minimum_fraction = minimum_fraction
        self.dataset = None
        self.best_score = 0.
        self.history = []
        self.diagnostic_iterations = set(diagnostic_iterations)

    def after_train_iter(self, runner):
        iteration = runner.iter + 1
        if iteration in self.diagnostic_iterations:
            from mmcv.runner import save_checkpoint
            save_checkpoint(runner.model,
                str(Path(runner.work_dir) / f'diagnostic_iter_{iteration}.pth'),
                meta=dict(iter=iteration, diagnostic_only=True,
                          note='Weights for inspection, not a scheduled-epoch resume.'))

    def after_train_epoch(self, runner):
        from mmcv.parallel import collate, scatter
        from mmrotate.datasets import build_dataset

        if self.dataset is None:
            self.dataset = build_dataset(self.dataset_cfg)
            indices = np.random.RandomState(self.seed).permutation(len(self.dataset))[
                :self.images]
            self.dataset.data_infos = [self.dataset.data_infos[i] for i in indices]
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        was_training = model.training
        device = next(model.parameters()).device
        target = device.index if device.type == 'cuda' else -1
        responses = []

        def capture(module, inputs, output):
            if len(responses) < 2:
                responses.append(torch.cat([
                    value.detach().reshape(-1) for levels in output[1:3]
                    for value in levels]).cpu())

        handle = model.bbox_head.register_forward_hook(capture)
        model.eval()
        try:
            results = []
            with torch.no_grad():
                for index in range(len(self.dataset)):
                    batch = scatter(collate([self.dataset[index]], samples_per_gpu=1),
                                    [target])[0]
                    results.extend(model(return_loss=False, rescale=True, **batch))
            score = float(self.dataset.evaluate(results, nproc=2, use_07_metric=True)['mAP'])
        finally:
            handle.remove()
            model.train(was_training)
        failure = learning_failure(score, self.best_score, self.minimum_map,
                                   self.minimum_fraction)
        record = dict(epoch=runner.epoch + 1, training_images=len(self.dataset),
                      training_image_mAP=score, previous_best=self.best_score,
                      minimum_map=self.minimum_map,
                      status='failed' if failure else 'passed', error=failure,
                      ann_file=self.dataset_cfg['ann_file'])
        if len(responses) == 2:
            record['regression_image_response'] = float(
                (responses[1] - responses[0]).abs().mean()
                / (responses[0].abs().mean() + 1e-6))
        self.history.append(record)
        path = Path(runner.work_dir) / 'training_health.json'
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(self.history, indent=2) + '\n')
        temporary.replace(path)
        runner.logger.info('RiO training-image health: %s', record)
        if failure:
            raise RuntimeError('RiO learning safeguard: ' + failure)
        self.best_score = max(self.best_score, score)
