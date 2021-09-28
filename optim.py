from torch.optim.optimizer import Optimizer
from torch.optim import SGD, Adam
from torch.optim.adagrad import Adagrad
from torch.optim.adadelta import Adadelta
from torch.optim.asgd import ASGD
from torch import nn
import math


OPTIMIZERS = {
    'sgd': lambda lr, wd, params: SGD(params, lr, momentum=0.9, weight_decay=wd),
    'adam': lambda lr, wd, params: Adam(params, lr, weight_decay=wd),
    'adagrad': lambda lr, wd, params: Adagrad(params, lr, weight_decay=wd),
    'adadelta': lambda lr, wd, params: Adadelta(params, lr, weight_decay=wd),
    'asgd': lambda lr, wd, params: ASGD(params, lr, weight_decay=wd),
}


def build_optimizer(model: nn.Module) -> Optimizer:
    from config import cfg
    from runtime import Runtime
    return OPTIMIZERS[cfg.TRAIN.OPTIMIZER](
        cfg.TRAIN.LR * Runtime.lr_scale,
        cfg.TRAIN.WD, model.parameters(),
    )


LOSSES = {
    'smoothl1': nn.SmoothL1Loss(reduction='mean'),
    'l2': nn.MSELoss(reduction='mean'),
    'l1': nn.L1Loss(reduction='mean'),
}


def get_loss() -> nn.Module:
    from config import cfg
    return LOSSES[cfg.TRAIN.LOSS]


def adjust_learning_rate(
    optimzer: Optimizer, epoch: int, step: int,
    end_epoch: int, step_per_epoch: int,
) -> float:

    from config import cfg
    from runtime import Runtime

    equiv_epoch = epoch + float(step) / step_per_epoch
    coeff = 0.01 + 0.99 * 0.5 * (
        (1 + math.cos(math.pi * equiv_epoch / end_epoch))
    )  # cosine schedule with minimum lr to be 0.01 * LR
    lr = coeff * cfg.TRAIN.LR * Runtime.lr_scale

    for param_group in optimzer.param_groups:
        param_group['lr'] = lr

    return lr


if __name__ == '__main__':
    from model import build_model
    from config import merge_from_file
    import sys

    merge_from_file(sys.argv[1])

    model = build_model()
    optim = build_optimizer(model)
    loss = get_loss()

    print(optim)
    print(loss)

    adjust_learning_rate(optim, 30, 0, 120, 1)
    print(optim)
