from torch.utils.data.dataloader import DataLoader
from runtime import Runtime
from optim import adjust_learning_rate, build_optimizer, get_loss
from model import build_model
from dataset import build_dataset
from typing import Dict, Tuple
import torch
from torch.utils.data import DistributedSampler
from torch import nn, Tensor
import os
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import SyncBatchNorm as SyncBN
from comm import reduce_loss_watch, cuda_preload_count, get_pg_for_syncbn, reduce_tensor
import math


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __float__(self):
        return self.avg


def rmse_accumulate(pred: Tensor, ref: Tensor) -> Tuple[Tensor, Tensor]:
    # pred, ref: [batch, numpt * 2]
    diff = pred - ref
    inv_numpt = torch.empty((), dtype=torch.float32).fill_(2. / ref.size(1))
    batch = torch.empty((), dtype=torch.long).fill_(ref.size(0))
    rmse_batch = torch.sqrt(inv_numpt * (diff * diff).sum(axis=1))
    rmse_acc = rmse_batch.sum()
    return rmse_acc, batch


def module_state_dict(m: nn.Module) -> Dict[str, Tensor]:
    if isinstance(m, DDP):
        return m.module.state_dict()
    else:
        return m.state_dict()


def module_load_state_dict(m: nn.Module, w: Dict[str, Tensor]) -> None:
    if isinstance(m, DDP):
        return m.module.load_state_dict(w)
    else:
        return m.load_state_dict(w)


class KeyPointTraining:

    def __init__(self, ckpt_path: str, pretrain: str = None, training: bool = True):
        from config import cfg

        self.ckpt_path = ckpt_path
        self.pretrain = pretrain

        self.model = build_model().cuda()
        pg_syncbn = get_pg_for_syncbn(Runtime.world_size, Runtime.rank, Runtime.local_rank)
        self.model = DDP(SyncBN.convert_sync_batchnorm(self.model, pg_syncbn), device_ids=[Runtime.local_rank])
        self.optim = build_optimizer(self.model)
        self.loss = get_loss().cuda()
        self.dataset = build_dataset(True) if training else None
        self.val_dataset = build_dataset(False)

        if training:
            train_sampler = DistributedSampler(self.dataset, Runtime.world_size, Runtime.rank, shuffle=True)
            self.loader = DataLoader(self.dataset, cfg.TRAIN.BATCH_SIZE,
                                     sampler=train_sampler, pin_memory=True,
                                     drop_last=True, num_workers=Runtime.num_workers)
        else:
            self.loader = None

        if len(self.val_dataset) > 0:
            val_sampler = DistributedSampler(self.val_dataset, Runtime.world_size, Runtime.rank, shuffle=False)
            self.val_loader = DataLoader(self.val_dataset, cfg.TRAIN.BATCH_SIZE,
                                         sampler=val_sampler, pin_memory=True,
                                         drop_last=False, num_workers=Runtime.num_workers)
        else:
            self.val_loader = None

        self.loss_tracker = AverageMeter()
        self.train_rmse_tracker = AverageMeter()
        self.epoch = 0

        self._init()

    def _init(self):
        if self.pretrain is not None:
            new_weights = torch.load(self.pretrain, map_location='cpu')
            cur_weights = module_state_dict(self.model)
            for key in cur_weights:
                if key not in new_weights:
                    continue
                if new_weights[key].size() != cur_weights[key].size():
                    print(f'skip {key}: expect {cur_weights[key].size()} got {new_weights[key].size()}')
                    continue
                cur_weights[key] = new_weights[key].to(cur_weights[key].device)
            module_load_state_dict(cur_weights)

        # if os.path.exists(self.ckpt_path) and os.path.isdir(self.ckpt_path):
        # we just try and fall back to ignoring ckpts in case any error
        try:
            import re
            # resume from latest checkpoint
            ckpt_pat = re.compile(r'epoch_([0-9]+).pth')
            ckpt_names = {int(ckpt_pat.match(filename).groups()[0]): filename
                          for filename in os.listdir(self.ckpt_path)
                          if ckpt_pat.match(filename) is not None}
            latest_ckpt = torch.load(self.ckpt_path + '/' +
                                     ckpt_names[max(ckpt_names.keys())],
                                     map_location='cpu')
            self.optim.load_state_dict(latest_ckpt['optim'])
            self.loss.load_state_dict(latest_ckpt['loss'])
            module_load_state_dict(self.model, latest_ckpt['model'])
            self.epoch = latest_ckpt['epoch']

        except (FileNotFoundError, NotADirectoryError, ValueError) as e:
            if Runtime.local_rank == 0:
                print(f'FAIL TO LOAD CHECKPOINT DUE TO: {str(e)}')
            if os.path.exists(self.ckpt_path) and os.path.isdir(self.ckpt_path):
                raise
            elif os.path.exists(self.ckpt_path):
                os.remove(self.ckpt_path)
            os.makedirs(self.ckpt_path, exist_ok=True)

    def _train_step(self, batch: Dict[str, Tensor]):
        self.optim.zero_grad()
        pred = self.model(batch['image'])
        loss = self.loss(pred, batch['pts'])
        loss.backward()
        self.optim.step()

        with torch.no_grad():
            rmse_acc, count = rmse_accumulate(pred, batch['pts'])
        loss, watch = reduce_loss_watch(loss, {'rmse': rmse_acc / count.float()})
        self.loss_tracker.update(loss.item())
        self.train_rmse_tracker.update(watch['rmse'].item())

    def snapshot(self):
        if Runtime.rank != 0:
            return

        ckpt = {
            'optim': self.optim.state_dict(),
            'loss': self.loss.state_dict(),
            'model': module_state_dict(self.model),
            'epoch': self.epoch
        }
        torch.save(ckpt, self.ckpt_path + f'/epoch_{self.epoch}.pth')

    def train_epoch(self):

        from config import cfg

        step_per_epoch = len(self.loader)
        self.model.train()
        for step, batch in enumerate(cuda_preload_count(self.loader)):
            cur_lr = adjust_learning_rate(self.optim, self.epoch, step,
                                          cfg.TRAIN.EPOCH, step_per_epoch)
            self._train_step(batch)

            if (Runtime.local_rank == 0 and (step + 1) % Runtime.print_freq == 0):
                print(f'epoch {self.epoch} step {step + 1}, lr = {cur_lr}, '
                      f'loss = {float(self.loss_tracker)}, '
                      f'rmse = {float(self.train_rmse_tracker)}')
                self.loss_tracker.reset()
                self.train_rmse_tracker.reset()

        self.epoch += 1
        self.snapshot()
        val_rmse = self.validation()
        if Runtime.local_rank == 0 and not math.isnan(val_rmse):
            print(f'epoch {self.epoch} validation rmse = {val_rmse}')

    def validation(self) -> float:
        import torch.distributed as dist

        if self.val_loader is None:
            return float('nan')

        common_step = torch.empty((), dtype=torch.long, device='cuda').fill_(
            len(self.val_loader))
        dist.all_reduce(common_step, op=dist.ReduceOp.MIN)
        common_step_val = common_step.item()

        self.model.eval()

        rmse_sum = torch.zeros((), dtype=torch.float32, device='cuda')
        count = torch.zeros((), dtype=torch.long, device='cuda')
        with torch.no_grad():
            for step, batch in enumerate(cuda_preload_count(self.val_loader)):
                if step < common_step_val and step % 10 == 0:
                    dist.barrier()
                pred = self.model(batch['image'])
                ref = batch['pts']
                rmse_cur, count_cur = rmse_accumulate(pred, ref)
                rmse_sum += rmse_cur
                count += count_cur
                del rmse_cur, count_cur

            rmse_sum = reduce_tensor(rmse_sum, mean=False)
            count = reduce_tensor(count.float(), mean=False)
            return (rmse_sum / count).item()

    @property
    def should_stop(self) -> bool:
        from config import cfg
        return self.epoch >= cfg.TRAIN.EPOCH
