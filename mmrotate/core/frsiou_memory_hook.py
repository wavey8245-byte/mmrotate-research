"""Commit fractional memory after SGD, save/verify matching checkpoint sidecars."""
import hashlib
from pathlib import Path
import torch
from mmcv.runner import HOOKS, Hook


# 分块计算模型文件 SHA-256，用于确认独立历史文件属于同一个检查点。
def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):
            value.update(block)
    return value.hexdigest()


# 兼容有/无 module 包装的模型，从检测头中取出实际回归损失对象。
def loss_of(runner):
    model=runner.model.module if hasattr(runner.model,'module') else runner.model
    return model.bbox_head.loss_bbox


# 配置需指定 BELOW_NORMAL：迭代时在优化器之后提交，轮末在模型保存后写历史。
# 当前工程缓存仅支持单 GPU；历史不包含在普通模型 state_dict 中。
@HOOKS.register_module()
class FrSIoUMemoryHook(Hook):
    """Register priority BELOW_NORMAL: after optimizer/checkpoint, before health."""
    def before_run(self,runner):
        if torch.distributed.is_initialized() and torch.distributed.get_world_size()!=1:
            raise RuntimeError('FrSIoU engineering memory currently supports one GPU only')
        loss=loss_of(runner)
        if not hasattr(loss,'commit_history'):
            raise TypeError('FrSIoUMemoryHook requires fractional loss')
        resume=getattr(runner,'_frsiou_resume_path',None)
        # Native runner resume does not expose its path; the config supplies it
        # through the explicit hook constructor on a resumed experiment.
        # 续训必须显式指定匹配检查点；恢复历史前同时核对文件哈希、epoch 和 iteration。
        if self.resume_checkpoint is not None:
            cp=Path(self.resume_checkpoint)
            sidecar=cp.with_suffix('.frsiou_history.pth')
            saved=torch.load(str(sidecar),map_location='cpu')
            if saved['checkpoint_sha256']!=digest(cp) or saved['epoch']!=runner.epoch or saved['iteration']!=runner.iter:
                raise RuntimeError('FrSIoU sidecar does not match resumed checkpoint/counters')
            loss.import_history(saved['history'])
        # 已有训练进度却未恢复历史时直接报错，避免静默把续训变成历史冷启动。
        elif runner.epoch or runner.iter:
            raise RuntimeError('Resumed fractional experiment requires explicit history sidecar')

    def __init__(self,resume_checkpoint=None):
        self.resume_checkpoint=resume_checkpoint

    # 此时 optimizer.step 已完成；提交的是本批次前向时暂存的组平均分数。
    def after_train_iter(self,runner):
        loss_of(runner).commit_history()

    # 每轮保存 epoch_N.frsiou_history.pth，与 epoch_N.pth 一一配对。
    def after_train_epoch(self,runner):
        cp=Path(runner.work_dir)/('epoch_'+str(runner.epoch+1)+'.pth')
        if not cp.is_file():
            raise RuntimeError('Save native epoch checkpoint before FrSIoU memory hook')
        # 历史文件携带进度与模型哈希；损失导出的状态还包含配置签名和缓存记录。
        saved=dict(epoch=runner.epoch+1,iteration=runner.iter,
                   checkpoint_sha256=digest(cp),history=loss_of(runner).export_history())
        # EpochBasedRunner increments iter in the inner loop, before epoch hooks.
        dest=cp.with_suffix('.frsiou_history.pth')
        # 先写临时文件再替换，避免正常保存过程中留下半写入的目标文件。
        temporary=dest.with_suffix(dest.suffix+'.tmp')
        torch.save(saved,str(temporary))
        temporary.replace(dest)
        runner.logger.info('FrSIoU memory: epoch=%d cached_groups=%d updates=%d total_history_hits=%d',
            saved['epoch'],len(loss_of(runner).history),loss_of(runner).committed_updates,loss_of(runner).total_history_hits)
